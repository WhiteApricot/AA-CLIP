import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR
from PIL import Image
from torchvision import transforms
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler

# ==============================================================================
# 1. 深度修复与优化区域
# ==============================================================================
from model.transformer import Transformer
from model.adapter import AdaptedCLIP


# --- 修复 1: 强制冻结 CLIP 骨干网络 ---
def freeze_backbone(model):
    for param in model.clipmodel.parameters():
        param.requires_grad = False
    print(">>> CLIP Backbone Frozen Explicitly.")


# --- 修复 2: 为 Stage 2 重写无 Bug 的 Forward 函数 ---
# 这个函数只在 Stage 2 (Image Adapter 训练) 时被调用
def optimized_adapted_clip_forward(self, x):
    # Part 1: Embedding
    x = self.image_encoder.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    x = torch.cat(
        [
            self.image_encoder.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ],
        dim=1,
    )
    x = x + self.image_encoder.positional_embedding.to(x.dtype)
    x = self.image_encoder.patch_dropout(x)
    x = self.image_encoder.ln_pre(x)
    x = x.permute(1, 0, 2)

    # Part 2: Loop with Correct Checkpointing
    tokens = []
    for i in range(24):
        # 如果开启了 checkpointing，使用正确的调用方式（解包 tuple）
        if self.image_encoder.transformer.grad_checkpointing:
            def run_layer(layer, x):
                return layer(x, attn_mask=None)[0]  # [关键修复] 只取 x，丢弃 attn

            x = checkpoint(run_layer, self.image_encoder.transformer.resblocks[i], x)
        else:
            x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)

        # Adapter Logic
        if i < self.image_adapt_until:
            adapt_out = self.image_adapter["layer_adapters"][i](x)
            adapt_out = (
                    adapt_out * x.norm(dim=-1, keepdim=True) / adapt_out.norm(dim=-1, keepdim=True)
            )
            x = self.i_w * adapt_out + (1 - self.i_w) * x
        if i + 1 in self.levels:
            tokens.append(x[1:, :, :])

    # Part 3: Projection
    x = x.permute(1, 0, 2)
    tokens = [t.permute(1, 0, 2) for t in tokens]
    tokens = [self.image_encoder.ln_post(t) for t in tokens]
    seg_tokens = [self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)]
    seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]
    det_token = self.image_adapter["det_proj"](tokens[-1])
    det_token = F.normalize(det_token, dim=-1).mean(1)
    return seg_tokens, det_token


# 应用 Patch
AdaptedCLIP.forward = optimized_adapted_clip_forward

# ==============================================================================
# 基础导入
# ==============================================================================
from utils import setup_seed
from model.clip import create_model
from forward_utils import (
    get_adapted_text_embedding,
    get_adapted_single_class_text_embedding,
    calculate_similarity_map,
    calculate_seg_loss,
)
import dataset.constants as constants
import warnings

warnings.filterwarnings("ignore")

# ================= 配置环境 =================
cpu_num = 4
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)

# ================= 注册 DeepCrack 信息 =================
DATASET_NAME = "DeepCrack"
CLASS_NAME_KEY = "DeepCrack"
REAL_SEMANTIC_NAME = "asphalt road surface"

constants.CLASS_NAMES[DATASET_NAME] = [CLASS_NAME_KEY]
constants.REAL_NAMES[DATASET_NAME] = {CLASS_NAME_KEY: REAL_SEMANTIC_NAME}
constants.DATA_PATH[DATASET_NAME] = "./DeepCrack"

constants.PROMPTS = {
    "prompt_normal": [
        "{}", "a {}", "smooth {}", "flat {}", "clean {}",
        "intact {}", "road surface", "pavement"
    ],
    "prompt_abnormal": [
        "damaged {}", "broken {}", "cracked {}", "fissured {}",
        "{} with pothole", "{} with gap", "{} with line crack", "{} with fracture"
    ],
    "prompt_templates": [
        "a photo of {}.", "a close-up photo of {}.", "a view of {}."
    ]
}


# ================= Dataset 类 =================
class DeepCrackDataset(Dataset):
    def __init__(self, root_dir, img_size, is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train
        if self.is_train:
            self.img_dir = os.path.join(root_dir, "train_img")
            self.lab_dir = os.path.join(root_dir, "train_lab")
        else:
            self.img_dir = os.path.join(root_dir, "test_img")
            self.lab_dir = os.path.join(root_dir, "test_lab")
        self.image_files = sorted(glob(os.path.join(self.img_dir, "*")))

        self.transform_x = transforms.Compose([
            transforms.Resize((img_size, img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((img_size, img_size), Image.NEAREST),
            transforms.ToTensor(),
        ])
        print(f"[{'Train' if is_train else 'Test'}] Loaded {len(self.image_files)} images.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        file_name = os.path.basename(img_path)
        mask_path = os.path.join(self.lab_dir, file_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transform_x(img)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros([1, self.img_size, self.img_size])
        label = 1 if mask.sum() > 0 else 0
        return {"image": img, "mask": mask, "label": torch.tensor(label).long(), "file_name": file_name,
                "class_name": CLASS_NAME_KEY}


# ================= 训练函数 =================
def train_stage(model, dataloader, optimizer, scaler, device, epoch, max_epoch, stage_name, accumulation_steps,
                text_embeddings=None, clip_surgery=None, img_size=336, text_norm_weight=0.1, logger=None):
    model.train()
    logger.info(f"Training {stage_name} Epoch {epoch + 1}/{max_epoch}")
    loss_list = []
    optimizer.zero_grad()

    pbar = tqdm(dataloader)
    for i, input_data in enumerate(pbar):
        image = input_data["image"].to(device)
        mask = input_data["mask"].to(device)
        class_names = input_data["class_name"]

        with autocast():
            loss = 0.0
            if stage_name == "Text Adapter":
                # Stage 1: Text Feature Alignment
                epoch_text_feature_dict = {}
                for c_name in list(set(class_names)):
                    text_embedding = get_adapted_single_class_text_embedding(model, DATASET_NAME, c_name, device)
                    epoch_text_feature_dict[c_name] = text_embedding
                epoch_text_feature = torch.stack([epoch_text_feature_dict[c_name] for c_name in class_names], dim=0)

                with torch.no_grad():
                    # 注意：clip_surgery 此时不应开启 checkpointing
                    _, patch_features = clip_surgery.encode_image(image, [6, 12, 18, 24])
                    cls_token, _ = model.clipmodel.encode_image(image, [])
                    cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
                    patch_features = [clip_surgery.visual.ln_post(t[:, 1:, :]) for t in patch_features]
                    patch_features = [t @ clip_surgery.visual.proj for t in patch_features]
                    patch_features = [t / t.norm(dim=-1, keepdim=True) for t in patch_features]
                    patch_features = [t + cls_token.unsqueeze(1) for t in patch_features]

                for f in patch_features:
                    patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                    loss += calculate_seg_loss(patch_preds, mask)
                    orthogonal_loss = ((epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1]).sum(1).mean()) ** 2
                    loss += orthogonal_loss * text_norm_weight
            else:
                # Stage 2: Image Adapter Training
                label = input_data["label"].to(device)
                epoch_text_feature = torch.stack([text_embeddings[c_name] for c_name in class_names], dim=0)

                # 这里会调用我们重写的 optimized_adapted_clip_forward
                patch_features, det_feature = model(image)

                det_feature = det_feature.unsqueeze(1)
                cls_preds = torch.matmul(det_feature, epoch_text_feature)[:, 0]
                loss += F.cross_entropy(cls_preds, label)

                for f in patch_features:
                    patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                    loss += calculate_seg_loss(patch_preds, mask)

            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_val = loss.item() * accumulation_steps
        loss_list.append(loss_val)
        pbar.set_description(f"Loss: {loss_val:.4f}")

    avg_loss = np.mean(loss_list)
    logger.info(f"Epoch {epoch + 1} Mean Loss: {avg_loss:.4f}")
    return model


# ================= 主函数 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="ckpt/DeepCrack")
    parser.add_argument("--img_size", type=int, default=336, help="Reduced for 8GB VRAM")
    parser.add_argument("--text_epoch", type=int, default=5)
    parser.add_argument("--image_epoch", type=int, default=20)
    parser.add_argument("--text_batch_size", type=int, default=8)
    parser.add_argument("--image_batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--text_lr", type=float, default=1e-5)
    parser.add_argument("--image_lr", type=float, default=5e-4)
    args = parser.parse_args()

    setup_seed(111)
    os.makedirs(args.save_path, exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(f"Args: {vars(args)}")
    logger.info(">>> Mode: 8GB VRAM Optimization (336px + Fixed Checkpointing)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()

    # 1. Models
    # 手动处理 patch_dropout
    clip_surgery = create_model("ViT-L-14-336", args.img_size, device=device, pretrained="openai",
                                require_pretrained=True)
    if hasattr(clip_surgery.visual, 'patch_dropout'): clip_surgery.visual.patch_dropout = nn.Identity()
    clip_surgery.eval()
    clip_surgery.visual.DAPM_replace(DPAM_layer=20)

    clip_model = create_model("ViT-L-14-336", args.img_size, device=device, pretrained="openai",
                              require_pretrained=True, cache_dir="./model")
    if hasattr(clip_model.visual, 'patch_dropout'): clip_model.visual.patch_dropout = nn.Identity()
    clip_model.eval()

    model = AdaptedCLIP(clip_model=clip_model, text_adapt_weight=0.1, image_adapt_weight=0.1, text_adapt_until=3,
                        image_adapt_until=6).to(device)
    freeze_backbone(model)
    model.eval()

    # 2. Data
    train_dataset = DeepCrackDataset(constants.DATA_PATH[DATASET_NAME], args.img_size, is_train=True)
    text_loader = DataLoader(train_dataset, batch_size=args.text_batch_size, shuffle=True, num_workers=4,
                             pin_memory=True)
    image_loader = DataLoader(train_dataset, batch_size=args.image_batch_size, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)

    text_optimizer = torch.optim.Adam(model.text_adapter.parameters(), lr=args.text_lr, betas=(0.5, 0.999))
    image_optimizer = torch.optim.Adam(model.image_adapter.parameters(), lr=args.image_lr, betas=(0.5, 0.999))
    image_scheduler = MultiStepLR(image_optimizer, milestones=[1000], gamma=0.5)

    # 3. Stage 1: Text Adapter
    text_ckpt_path = os.path.join(args.save_path, "text_adapter.pth")
    if os.path.exists(text_ckpt_path):
        logger.info(f"Loading existing Text Adapter from {text_ckpt_path}")
        ckpt = torch.load(text_ckpt_path)
        model.text_adapter.load_state_dict(ckpt["text_adapter"])
        text_optimizer.load_state_dict(ckpt["text_optimizer"])
        logger.info(">>> Skipping Stage 1 (Loaded from checkpoint)")
    elif args.text_epoch > 0:
        logger.info(">>> Start Stage 1: Text Adapter")
        # [关键] Stage 1 强制关闭 checkpointing，避免触发原生 Bug
        model.image_encoder.transformer.grad_checkpointing = False
        for epoch in range(args.text_epoch):
            model = train_stage(model, text_loader, text_optimizer, scaler, device, epoch, args.text_epoch,
                                "Text Adapter", 1, clip_surgery=clip_surgery, img_size=args.img_size, logger=logger)
            torch.save({"text_adapter": model.text_adapter.state_dict(), "epoch": epoch,
                        "text_optimizer": text_optimizer.state_dict()}, text_ckpt_path)

    # 4. Stage 2: Image Adapter
    logger.info("Computing text embeddings...")
    with torch.no_grad():
        text_embeddings = get_adapted_text_embedding(model, DATASET_NAME, device)

    logger.info(">>> Start Stage 2: Image Adapter")
    # [关键] Stage 2 开启 checkpointing，利用我们重写的 Optimized Forward
    model.image_encoder.transformer.grad_checkpointing = True

    for epoch in range(args.image_epoch):
        model = train_stage(model, image_loader, image_optimizer, scaler, device, epoch, args.image_epoch,
                            "Image Adapter", args.accumulation_steps, text_embeddings=text_embeddings,
                            img_size=args.img_size, logger=logger)
        image_scheduler.step()
        torch.save({"image_adapter": model.image_adapter.state_dict(), "epoch": epoch},
                   os.path.join(args.save_path, "image_adapter.pth"))


if __name__ == "__main__":
    main()