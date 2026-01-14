import os
import argparse
import logging
import torch
import numpy as np
from glob import glob
import re

# ==============================================================================
# Monkey Patching 区域
# ==============================================================================
import dataset.constants

# 1. 注册数据路径
dataset.constants.DATA_PATH["HubeiDown_Train"] = "data/HubeiDown_Train"
dataset.constants.CLASS_NAMES["HubeiDown_Train"] = ["road"]
dataset.constants.DOMAINS["HubeiDown_Train"] = "Industrial"

# 2. 注册真实名称 (修正为纯名词)
dataset.constants.REAL_NAMES["HubeiDown_Train"] = {
    "road": "asphalt road surface"
}

# 3. 定制混合 Prompt (针对道路补丁优化)
dataset.constants.PROMPTS = {
    "prompt_normal": [
        "{}", "a {}", "smooth {}", "flat {}", "clean {}",
        "newly paved {}", "repaired {}", "road surface with patches"
    ],
    "prompt_abnormal": [
        "damaged {}", "broken {}", "cracked {}", "fissured {}",
        "{} with pothole", "{} with gap", "{} with fracture"
    ],
    "prompt_templates": [
        "a photo of {}.", "a close-up photo of {}.", "a top-down view of {}."
    ]
}

# 4. [新增] 修复 Gradient Checkpointing 的 Bug
from model.transformer import Transformer
from torch.utils.checkpoint import checkpoint
from typing import Optional


def fixed_transformer_forward(
        self,
        x: torch.Tensor,
        out_layers: list = [3, 6, 9],
        attn_mask: Optional[torch.Tensor] = None,
):
    idx = 0
    out_tokens = []
    for r in self.resblocks:
        idx += 1
        if self.grad_checkpointing and not torch.jit.is_scripting():
            # [关键修复] 解包 checkpoint 返回的元组 (x, attn)
            x, _ = checkpoint(r, x, None, None, attn_mask)
        else:
            if idx == 12:
                x, attn = r(x, attn_mask=attn_mask)
            else:
                x, attn_tmp = r(x, attn_mask=attn_mask)

        if idx in out_layers:
            out_tokens.append(x)
    return x, out_tokens


Transformer.forward = fixed_transformer_forward
# ==============================================================================

from train import train_text_adapter, train_image_adapter
from utils import setup_seed
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset
from forward_utils import get_adapted_text_embedding


def find_latest_image_checkpoint(save_path):
    files = glob(os.path.join(save_path, "image_adapter_*.pth"))
    if not files:
        return None, 0

    def extract_epoch(f):
        match = re.search(r'image_adapter_(\d+).pth', f)
        return int(match.group(1)) if match else -1

    latest_file = max(files, key=extract_epoch)
    return latest_file, extract_epoch(latest_file)


def main():
    parser = argparse.ArgumentParser(description="Training AnyModel Final Fix")
    # OOM 优化设置
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=336)
    parser.add_argument("--image_batch_size", type=int, default=1)

    parser.add_argument("--surgery_until_layer", type=int, default=20)
    parser.add_argument("--relu", action="store_true")
    parser.add_argument("--dataset", type=str, default="HubeiDown_Train")
    parser.add_argument("--save_path", type=str, default="ckpt/HubeiDown_Model")
    parser.add_argument("--training_mode", type=str, default="full_shot", choices=["few_shot", "full_shot"])
    parser.add_argument("--shot", type=int, default=32)
    parser.add_argument("--text_batch_size", type=int, default=16)
    parser.add_argument("--text_epoch", type=int, default=5)
    parser.add_argument("--image_epoch", type=int, default=20)
    parser.add_argument("--text_lr", type=float, default=0.00001)
    parser.add_argument("--image_lr", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)

    args = parser.parse_args()
    setup_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    logger = logging.getLogger(__name__)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "train.log"),
        encoding="utf-8",
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    logger.info("Fixed Script Started (with Grad Checkpoint Patch).")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load Model
    clip_surgery = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_surgery.eval()
    clip_surgery.visual.DAPM_replace(DPAM_layer=args.surgery_until_layer)

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
        cache_dir="./model",
    )
    clip_model.eval()

    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
    ).to(device)

    # 开启梯度检查点 (现在是安全的)
    model.clipmodel.set_grad_checkpointing(True)
    model.eval()

    # Optimizer
    text_optimizer = torch.optim.Adam(model.text_adapter.parameters(), lr=args.text_lr, betas=(0.5, 0.999))
    image_optimizer = torch.optim.Adam(model.image_adapter.parameters(), lr=args.image_lr, betas=(0.5, 0.999))
    image_scheduler = MultiStepLR(image_optimizer, milestones=[16000, 32000], gamma=0.5)

    # Resume Logic
    text_ckpt_path = os.path.join(args.save_path, "text_adapter.pth")
    text_start_epoch = 0
    run_text_training = True

    if os.path.exists(text_ckpt_path):
        logger.info(f"Found Text Adapter Checkpoint: {text_ckpt_path}")
        checkpoint = torch.load(text_ckpt_path, map_location=device)
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        text_optimizer.load_state_dict(checkpoint["text_optimizer"])
        text_start_epoch = checkpoint["epoch"]
        if text_start_epoch >= args.text_epoch:
            logger.info("Text Adapter training completed. Skipping.")
            run_text_training = False

    latest_img_ckpt, img_last_epoch = find_latest_image_checkpoint(args.save_path)
    image_start_epoch = 0
    if latest_img_ckpt:
        logger.info(f"Resuming Image Adapter from epoch {img_last_epoch}")
        checkpoint = torch.load(latest_img_ckpt, map_location=device)
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
        image_optimizer.load_state_dict(checkpoint["image_optimizer"])
        image_start_epoch = checkpoint["epoch"]

    # Load Dataset
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    text_dataset, image_dataset = get_dataset(
        args.dataset, args.img_size, args.training_mode, args.shot, "train", logger=logger
    )
    text_dataloader = DataLoader(text_dataset, batch_size=args.text_batch_size, shuffle=True, **kwargs)
    image_dataloader = DataLoader(image_dataset, batch_size=args.image_batch_size, shuffle=True, **kwargs)

    # Training
    if run_text_training:
        logger.info("Starting Text Adapter Training...")
        model = train_text_adapter(
            adapted_model=model,
            clip_surgery=clip_surgery,
            text_norm_weight=args.text_norm_weight,
            train_loader=text_dataloader,
            optimizer=text_optimizer,
            device=device,
            start_epoch=text_start_epoch,
            dataset_name=args.dataset,
            save_path=args.save_path,
            text_epoch=args.text_epoch,
            img_size=args.img_size,
            logger=logger,
        )

    del text_dataloader, text_dataset, clip_surgery, text_optimizer
    torch.cuda.empty_cache()

    with torch.no_grad():
        logger.info("Generating Text Embeddings...")
        text_embeddings = get_adapted_text_embedding(model, args.dataset, device)

    if image_start_epoch < args.image_epoch:
        logger.info("Starting Image Adapter Training...")
        model = train_image_adapter(
            model=model,
            text_embeddings=text_embeddings,
            train_loader=image_dataloader,
            optimizer=image_optimizer,
            scheduler=image_scheduler,
            device=device,
            start_epoch=image_start_epoch,
            save_path=args.save_path,
            image_epoch=args.image_epoch,
            img_size=args.img_size,
            logger=logger,
        )

    logger.info("All Training Finished.")


if __name__ == "__main__":
    main()