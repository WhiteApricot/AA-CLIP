import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ==============================================================================
# Monkey Patching 区域
# ==============================================================================
import dataset.constants

# 1. 注册数据路径
dataset.constants.DATA_PATH["HubeiDown_Test"] = "data/HubeiDown_Test"
# 2. 注册类别名称
dataset.constants.CLASS_NAMES["HubeiDown_Test"] = ["road"]
# 3. 注册真实名称
dataset.constants.REAL_NAMES["HubeiDown_Test"] = {"road": "road"}
# 4. 注册领域
if not hasattr(dataset.constants, "DOMAINS"):
    dataset.constants.DOMAINS = {}
dataset.constants.DOMAINS["HubeiDown_Test"] = "Industrial"
# ==============================================================================

from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import DOMAINS  # 保留 DOMAINS 引用
# 注意：我们跳过 get_dataset，自己实现数据加载
from forward_utils import get_adapted_text_embedding
from test import get_predictions, add_header_to_image


class DirectFolderDataset(Dataset):
    """
    自定义数据集类：直接扫描文件夹，不依赖 jsonl 元数据
    """

    def __init__(self, root_dir, class_name, img_size):
        self.root_dir = root_dir
        self.class_name = class_name

        # 直接使用 glob 扫描文件夹中的图片
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(glob(os.path.join(root_dir, ext)))
            # 兼容大小写
            self.image_files.extend(glob(os.path.join(root_dir, ext.upper())))

        self.image_files = sorted(list(set(self.image_files)))  # 去重并排序

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        # 标准 CLIP 预处理 transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        file_name = os.path.basename(img_path)

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回一个全黑的 dummy image 防止程序崩溃
            image = torch.zeros((3, 336, 336))

        # 构造伪造的 mask 和 label (因为是推理模式，没有 GT)
        # 形状需要符合 collate_fn 的预期
        mask = torch.zeros((1, image.shape[1], image.shape[2]))
        label = torch.tensor(0)

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "file_name": file_name,  # get_predictions 需要这个
            "class_name": self.class_name  # get_predictions 需要这个
        }


def visualize_custom_hubei(
        pixel_preds: np.ndarray,
        file_names: list[str],
        save_path: str,
        dataset_name: str,
        class_name: str,
):
    """
    可视化函数 (保持全局归一化逻辑)
    """
    save_dir = os.path.join("results", "DeepCrack")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving visualization results to: {save_dir} ...")

    # === 核心：全局归一化 ===
    # 统计所有测试图像的 Min 和 Max，统一进行归一化
    # 这意味着异常程度最高的图片会接近 1 (红色)，异常程度低的图片可能整体偏蓝
    p_min, p_max = pixel_preds.min(), pixel_preds.max()
    print(f"Global Stats - Min: {p_min:.4f}, Max: {p_max:.4f}")

    if p_max != p_min:
        pixel_preds_norm = (pixel_preds - p_min) / (p_max - p_min)
    else:
        pixel_preds_norm = pixel_preds

    for idx, file in enumerate(tqdm(file_names, desc="Saving Images")):
        # 1. 读取原图
        image_full_path = os.path.join(dataset.constants.DATA_PATH[dataset_name], file)
        original_image = cv2.imread(image_full_path)

        if original_image is None:
            continue

        h, w, c = original_image.shape

        # 2. 处理热力图
        pred = pixel_preds_norm[idx]
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h))

        pred_vis = (pred * 255).astype(np.uint8)
        pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)

        # 3. 处理预测掩码 (0.5 阈值)
        mask_pred = (pred > 0.5).astype(np.uint8) * 255
        mask_pred_vis = cv2.cvtColor(mask_pred, cv2.COLOR_GRAY2BGR)

        # 4. 添加标题
        original_vis = add_header_to_image(original_image, "Original")
        heatmap_vis = add_header_to_image(pred_vis, "Global Heatmap")
        mask_vis = add_header_to_image(mask_pred_vis, "Predicted Mask")

        # 5. 保存
        combined_image = np.hstack([original_vis, heatmap_vis, mask_vis])
        safe_filename = os.path.basename(file).split('.')[0] + ".png"
        cv2.imwrite(os.path.join(save_dir, safe_filename), combined_image)


def main():
    parser = argparse.ArgumentParser(description="Predicting AnyModel Fixed")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=336)
    parser.add_argument("--relu", action="store_true")
    parser.add_argument("--dataset", type=str, default="HubeiDown_Test")
    parser.add_argument("--save_path", type=str, default="ckpt/my_road_crack_model")

    # 这里的 shot 参数已经不再重要，因为我们不使用 jsonl
    parser.add_argument("--shot", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)  # 可以适当调大，因为不是实时显示
    parser.add_argument("--visualize", action="store_true", default=True)

    # 保持默认参数
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)
    parser.add_argument("--seed", type=int, default=111)

    args = parser.parse_args()
    setup_seed(args.seed)

    # Logger
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logging.basicConfig(filename=os.path.join(args.save_path, "predict.log"), level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # 1. Load Model
    logger.info(f"Loading model {args.model_name}...")
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

    # 2. Load Checkpoints
    text_file = glob(args.save_path + "/text_adapter.pth")
    if len(text_file) > 0:
        logger.info(f"Loading Text Adapter: {text_file[0]}")
        checkpoint = torch.load(text_file[0], map_location=device)
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        adapt_text = True
    else:
        adapt_text = False

    all_files = sorted(glob(args.save_path + "/image_adapter_*.pth"))
    if len(all_files) > 0:
        try:
            all_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        except:
            pass
        checkpoint_path = all_files[-1]
        logger.info(f"Loading Image Adapter checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
    else:
        logger.error("No image adapter checkpoint found!")
        return

    # 3. [修改点] 直接加载数据集，不通过 get_dataset 和 jsonl
    data_dir = dataset.constants.DATA_PATH[args.dataset]
    logger.info(f"Scanning images directly from: {data_dir}")

    test_dataset = DirectFolderDataset(
        root_dir=data_dir,
        class_name="road",
        img_size=args.img_size
    )
    # 包装成 dict 以适配原有逻辑
    image_datasets = {"road": test_dataset}
    logger.info(f"Found {len(test_dataset)} images.")

    # 4. Text Embeddings
    with torch.no_grad():
        if adapt_text:
            text_embeddings = get_adapted_text_embedding(model, args.dataset, device)
        else:
            text_embeddings = get_adapted_text_embedding(clip_model, args.dataset, device)

    # 5. Inference Loop
    for class_name, image_dataset in image_datasets.items():
        logger.info(f"Processing class: {class_name}")

        image_dataloader = DataLoader(
            image_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4 if use_cuda else 0
        )

        with torch.no_grad():
            class_text_embeddings = text_embeddings[class_name]

            # 使用原来的 get_predictions 获取所有结果
            # 这会遍历整个 DataLoader 并收集所有结果到列表中
            masks, labels, preds, preds_image, file_names = get_predictions(
                model=model,
                class_text_embeddings=class_text_embeddings,
                test_loader=image_dataloader,
                device=device,
                img_size=args.img_size,
                dataset=args.dataset,
            )

        # 6. 批量可视化 (包含全局归一化)
        if args.visualize:
            visualize_custom_hubei(
                preds,  # 所有图像的预测结果数组
                file_names,  # 所有文件名
                args.save_path,
                args.dataset,
                class_name
            )

        logger.info(f"Finished processing {class_name}")

    print("Done!")


if __name__ == "__main__":
    main()