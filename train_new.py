import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import ipdb
from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset
from forward_utils import (
    get_adapted_text_embedding,
    get_adapted_single_class_text_embedding,
    calculate_similarity_map,
    calculate_seg_loss,
)
import warnings

warnings.filterwarnings("ignore")

cpu_num = 4

os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =================================================================================
# 新增模块：高分辨率 Std Head (直接处理 518x518 的原图)
# 作用：提取高频物理边缘，输出高分辨率的方差图，指导 Aseg Loss 精细收缩
# =================================================================================
class HighResStdHead(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, x):
        # x shape: [B, 3, 518, 518]
        # return shape: [B, 1, 518, 518] (对数方差)
        return self.net(x)


# =================================================================================


def train_text_adapter(
        adapted_model: nn.Module,
        clip_surgery: nn.Module,
        text_norm_weight: float,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str,
        start_epoch: int,
        save_path: str,
        text_epoch: int,
        dataset_name: str,
        img_size: int,
        logger: logging.Logger,
):
    for epoch in range(start_epoch, text_epoch):
        logger.info(f"training text epoch {epoch}:")

        loss_list = []
        for input_data in tqdm(train_loader):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            class_names = input_data["class_name"]

            # forward text
            epoch_text_feature_dict = {}
            for class_name in list(set(class_names)):
                text_embedding = get_adapted_single_class_text_embedding(
                    adapted_model, dataset_name, class_name, device
                )
                epoch_text_feature_dict[class_name] = text_embedding
            epoch_text_feature = torch.stack(
                [epoch_text_feature_dict[class_name] for class_name in class_names],
                dim=0,
            )  # bs,768,2

            # forward image
            with torch.no_grad():
                _, patch_features = clip_surgery.encode_image(image, [6, 12, 18, 24])
                cls_token, _ = adapted_model.clipmodel.encode_image(image, [])
                cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
                patch_features = [
                    clip_surgery.visual.ln_post(t[:, 1:, :]) for t in patch_features
                ]
                patch_features = [t @ clip_surgery.visual.proj for t in patch_features]
                patch_features = [
                    t / t.norm(dim=-1, keepdim=True) for t in patch_features
                ]
                patch_features = [t + cls_token.unsqueeze(1) for t in patch_features]

            # calculate similarity and get prediction
            for f in patch_features:
                # bs,patch_num,768
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                loss = calculate_seg_loss(patch_preds, mask)
                orthogonal_loss = (
                                      (epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1])
                                      .sum(1)
                                      .mean()
                                  ) ** 2
                loss += orthogonal_loss * text_norm_weight

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        logger.info(f"loss: {np.mean(loss_list)}")
        # save checkpoint
        ckp_path = os.path.join(save_path, "text_adapter.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "text_adapter": adapted_model.text_adapter.state_dict(),
                "text_optimizer": optimizer.state_dict(),
            },
            ckp_path,
        )
    return adapted_model


def train_image_adapter(
        model: nn.Module,
        std_head: nn.Module,  # <--- 新增
        text_embeddings: torch.Tensor,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: str,
        start_epoch: int,
        save_path: str,
        image_epoch: int,
        img_size: int,
        logger: logging.Logger,
):
    std_head.train()

    for epoch in range(start_epoch, image_epoch):
        logger.info(f"training image epoch {epoch}:")
        loss_list = []
        for input_data in tqdm(train_loader):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            label = input_data["label"].to(device)
            B, C, H, W = image.shape

            # forward text
            class_names = input_data["class_name"]
            epoch_text_feature = torch.stack(
                [text_embeddings[class_name] for class_name in class_names], dim=0
            )

            # forward image
            patch_features, det_feature = model(image)

            # calculate similarity and get prediction
            loss = 0.0
            det_feature = det_feature.unsqueeze(1)
            cls_preds = torch.matmul(det_feature, epoch_text_feature)[:, 0]
            loss += F.cross_entropy(cls_preds, label)

            # =================================================================
            # 提取原图级别的高分辨率方差
            # 因为它只依赖原图，所以提取到 patch_features 循环外部以节省算力
            # =================================================================
            log_variance = std_head(image)  # [B, 1, 518, 518]
            variance = torch.exp(log_variance)
            lambda_val = 0.3
            mask_f = mask.unsqueeze(1).float() if mask.dim() == 3 else mask.float()

            for f in patch_features:
                # text-image alignment
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                orig_seg_loss = calculate_seg_loss(patch_preds, mask)

                # 计算 Aseg Loss (高分辨率逐像素约束)
                prob = torch.softmax(patch_preds, dim=1)[:, 1:2, ...]  # 取异常类概率
                sq_error = (prob - mask_f) ** 2
                aseg_loss = ((sq_error + variance) / (lambda_val + variance)).mean()

                # 将两种 Loss 叠加约束
                loss += (orig_seg_loss + aseg_loss)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            scheduler.step()

        logger.info(f"loss: {np.mean(loss_list)}")

        # save checkpoint (连带 std_head)
        model_dict = {
            "epoch": epoch + 1,
            "image_adapter": model.image_adapter.state_dict(),
            "std_head": std_head.state_dict(),
            "image_optimizer": optimizer.state_dict(),
        }
        torch.save(model_dict, os.path.join(save_path, "image_adapter.pth"))
        if (epoch + 1) % 1 == 0:
            ckp_path = os.path.join(save_path, f"image_adapter_{epoch + 1}.pth")
            torch.save(
                model_dict,
                ckp_path,
            )
    return model


def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="clip model to use (default: ViT-L-14-336)",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--surgery_until_layer", type=int, default=20)
    parser.add_argument("--relu", action="store_true", help="use relu after projection")

    parser.add_argument(
        "--dataset", type=str, default="DeepCrack_Train"
    )

    parser.add_argument(
        "--training_mode",
        type=str,
        default="few_shot",
        choices=["few_shot", "full_shot"],
    )
    parser.add_argument("--shot", type=int, default=32, help="number of shots (0 means full shot)")
    parser.add_argument("--text_batch_size", type=int, default=16)
    parser.add_argument("--image_batch_size", type=int, default=1)
    parser.add_argument("--text_epoch", type=int, default=5, help="epochs for stage1")
    parser.add_argument("--image_epoch", type=int, default=20, help="epochs for stage2")
    parser.add_argument("--text_lr", type=float, default=0.00001, help="learning rate for stage1")
    parser.add_argument("--image_lr", type=float, default=0.0005, help="learning rate for stage2")
    parser.add_argument(
        "--criterion", type=str, default=["dice_loss", "focal_loss"], nargs="+"
    )
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)

    args = parser.parse_args()
    setup_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "train.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("args: %s", vars(args))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

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
    model.eval()

    # 实例化高分辨率 Std Head
    std_head = HighResStdHead().to(device)

    text_optimizer = torch.optim.Adam(
        model.text_adapter.parameters(),
        lr=args.text_lr,
        betas=(0.5, 0.999),
    )

    # 将 std_head 的参数加入联合优化
    image_optimizer = torch.optim.Adam(
        list(model.image_adapter.parameters()) + list(std_head.parameters()),
        lr=args.image_lr,
        betas=(0.5, 0.999),
    )
    image_scheduler = MultiStepLR(image_optimizer, milestones=[16000, 32000], gamma=0.5)

    text_file = glob(args.save_path + "/text_adapter.pth")
    if len(text_file) > 0:
        checkpoint = torch.load(text_file[0])
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        text_optimizer.load_state_dict(checkpoint["text_optimizer"])
        text_start_epoch = checkpoint["epoch"]
        adapt_text = not (text_start_epoch == (args.text_epoch - 1))
    elif args.text_epoch == 0:
        adapt_text = False
    else:
        text_start_epoch = 0
        adapt_text = True

    file = glob(args.save_path + "/image_adapter.pth")
    if len(file) > 0:
        checkpoint = torch.load(file[0])
        image_start_epoch = checkpoint["epoch"]
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])

        # 恢复 std_head 权重
        if "std_head" in checkpoint:
            std_head.load_state_dict(checkpoint["std_head"])

        image_optimizer.load_state_dict(checkpoint["image_optimizer"])
    else:
        image_start_epoch = 0

    if args.training_mode == "full_shot":
        args.shot = -1
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    logger.info("loading dataset ...")

    text_dataset, image_dataset = get_dataset(
        args.dataset,
        args.img_size,
        None,
        args.shot,
        "train",
        logger=logger
    )

    text_dataloader = torch.utils.data.DataLoader(
        text_dataset, batch_size=args.text_batch_size, shuffle=True, **kwargs
    )
    logger.info("loading image adaptation dataset ...")
    image_dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=args.image_batch_size, shuffle=True, **kwargs
    )

    if adapt_text:
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
        if args.text_epoch == 0:
            text_embeddings = get_adapted_text_embedding(
                clip_model, args.dataset, device
            )
        else:
            text_embeddings = get_adapted_text_embedding(model, args.dataset, device)

    model = train_image_adapter(
        model=model,
        std_head=std_head,  # 传入初始化好的高分辨率头
        text_embeddings=text_embeddings,
        image_epoch=args.image_epoch,
        train_loader=image_dataloader,
        optimizer=image_optimizer,
        scheduler=image_scheduler,
        device=device,
        start_epoch=image_start_epoch,
        save_path=args.save_path,
        img_size=args.img_size,
        logger=logger,
    )


if __name__ == "__main__":
    main()