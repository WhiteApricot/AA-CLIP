import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
import math

# ==============================================================================
# Monkey Patching 区域
# ==============================================================================
import dataset.constants

dataset.constants.DATA_PATH["MyRoadCrack_Test"] = "data/MyRoadCrack_Test/images"
dataset.constants.CLASS_NAMES["MyRoadCrack_Test"] = ["road_crack"]
dataset.constants.REAL_NAMES["MyRoadCrack_Test"] = {
    "road_crack": "a photo of a road with a crack"
}
if not hasattr(dataset.constants, "DOMAINS"):
    dataset.constants.DOMAINS = {}
dataset.constants.DOMAINS["MyRoadCrack_Test"] = "Industrial"
# ==============================================================================

from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
from forward_utils import get_adapted_text_embedding, calculate_similarity_map
from test import add_header_to_image


def get_gaussian_2d(size: int, sigma_scale: float = 1.0 / 4.0) -> torch.Tensor:
    """
    生成二维高斯核，用于平滑窗口拼接
    """
    tmp = np.linspace(-1, 1, size)
    x, y = np.meshgrid(tmp, tmp)
    d = np.sqrt(x * x + y * y)
    sigma = sigma_scale
    g = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
    g = (g - g.min()) / (g.max() - g.min())
    return torch.from_numpy(g).float()


def visualize_and_save_immediate(
        pixel_pred_raw: np.ndarray,
        file_path: str,
        save_dir: str
):
    """
    单张图片可视化并立即保存
    包含：原图 | 预测掩码(Mask) | 热力图
    """
    # 1. 读取原图
    original_image = cv2.imread(file_path)
    if original_image is None:
        return

    h, w = original_image.shape[:2]

    # 2. 尺寸对齐 (防止计算误差导致 1px 偏差)
    if pixel_pred_raw.shape != (h, w):
        pixel_pred_raw = cv2.resize(pixel_pred_raw, (w, h))

    # 3. 单图归一化 (Local Normalization)
    # 因为是立即输出，无法得知全局 Max，只能根据当前图的最强响应来归一化
    p_min, p_max = pixel_pred_raw.min(), pixel_pred_raw.max()
    if p_max - p_min < 1e-6:
        pixel_pred_norm = pixel_pred_raw
    else:
        pixel_pred_norm = (pixel_pred_raw - p_min) / (p_max - p_min)

    pixel_pred_norm = np.clip(pixel_pred_norm, 0, 1)

    # 4. 生成热力图 (Heatmap)
    pred_vis = (pixel_pred_norm * 255).astype(np.uint8)
    pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)

    # 5. 生成二值化掩码 (Binary Mask)，阈值设为 0.5
    # 如果归一化后大部分区域都很亮，这个阈值可能需要根据实际情况调整，或者使用自适应阈值
    mask_pred = (pixel_pred_norm > 0.5).astype(np.uint8) * 255
    mask_pred_vis = cv2.cvtColor(mask_pred, cv2.COLOR_GRAY2BGR)

    # 6. 添加标题栏
    original_vis = add_header_to_image(original_image, "Original")
    mask_vis = add_header_to_image(mask_pred_vis, "Predicted Mask (>0.5)")
    heatmap_vis = add_header_to_image(pred_vis, "Heatmap")

    # 7. 拼接：原图 | 掩码 | 热力图
    combined_image = np.hstack([original_vis, mask_vis, heatmap_vis])

    # 8. 保存
    base_name = os.path.basename(file_path).split('.')[0]
    save_filename = f"{base_name}.png"
    save_full_path = os.path.join(save_dir, save_filename)
    cv2.imwrite(save_full_path, combined_image)


def sliding_window_inference(
        model,
        image_path,
        text_features,
        patch_transform,
        device,
        window_size=518,
        stride=259,
        dataset_name="MyRoadCrack_Test"
):
    """
    带高斯加权的滑动窗口推理
    """
    img_pil = Image.open(image_path).convert("RGB")
    w, h = img_pil.size

    # Fallback: 图片小于窗口
    if w < window_size or h < window_size:
        resize_transform = transforms.Compose([
            transforms.Resize((window_size, window_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))
        ])
        input_tensor = resize_transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            patch_features_list, _ = model(input_tensor)
            patch_preds = []
            for f in patch_features_list:
                patch_pred = calculate_similarity_map(
                    f, text_features, img_size=window_size, test=True,
                    domain=dataset.constants.DOMAINS[dataset_name]
                )
                patch_preds.append(patch_pred)
            pred = torch.cat(patch_preds, dim=1).sum(1).squeeze(0).cpu().numpy()
            pred = cv2.resize(pred, (w, h))
        return pred

    # 滑动窗口
    sum_map = torch.zeros((h, w), dtype=torch.float32, device='cpu')
    weight_map = torch.zeros((h, w), dtype=torch.float32, device='cpu')
    gaussian_weight = get_gaussian_2d(window_size).to('cpu')

    h_steps = math.ceil((h - window_size) / stride) + 1
    w_steps = math.ceil((w - window_size) / stride) + 1

    for i in range(h_steps):
        for j in range(w_steps):
            y = i * stride
            x = j * stride
            if y + window_size > h: y = h - window_size
            if x + window_size > w: x = w - window_size

            crop = img_pil.crop((x, y, x + window_size, y + window_size))
            input_tensor = patch_transform(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                patch_features_list, _ = model(input_tensor)
                patch_preds = []
                for f in patch_features_list:
                    patch_pred = calculate_similarity_map(
                        f, text_features, img_size=window_size, test=True,
                        domain=dataset.constants.DOMAINS[dataset_name]
                    )
                    patch_preds.append(patch_pred)

                pred_patch = torch.cat(patch_preds, dim=1).sum(1).squeeze(0).cpu()

            sum_map[y:y + window_size, x:x + window_size] += pred_patch * gaussian_weight
            weight_map[y:y + window_size, x:x + window_size] += gaussian_weight

    final_pred = sum_map / (weight_map + 1e-6)
    return final_pred.numpy()


def main():
    parser = argparse.ArgumentParser(description="Sliding Window Immediate Viz")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    parser.add_argument("--dataset", type=str, default="MyRoadCrack_Test")
    parser.add_argument("--save_path", type=str, default="ckpt/my_road_crack_model")
    parser.add_argument("--stride", type=int, default=259, help="Overlap stride")

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
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting Gaussian Sliding Window for {args.dataset}...")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # 1. Load Models
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

    # Load Weights
    text_file = glob(args.save_path + "/text_adapter.pth")
    if len(text_file) > 0:
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
        logger.info(f"Loading Image Adapter: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
    else:
        logger.error("No image adapter checkpoint found!")
        return

    # Text Embeddings
    with torch.no_grad():
        if adapt_text:
            text_embeddings = get_adapted_text_embedding(model, args.dataset, device)
        else:
            text_embeddings = get_adapted_text_embedding(clip_model, args.dataset, device)

    # Transform (No Resize)
    patch_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])

    # Find Files
    data_root = dataset.constants.DATA_PATH[args.dataset]
    class_name = "road_crack"
    class_text_emb = text_embeddings[class_name]

    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in exts:
        image_files.extend(glob(os.path.join(data_root, ext)))
        image_files.extend(glob(os.path.join(data_root, "**", ext), recursive=True))

    image_files = sorted(list(set(image_files)))
    if len(image_files) == 0:
        logger.error("No images found.")
        return

    # === 结果保存路径 ===
    save_dir = os.path.join("results", "Crack_GaussianWindow")
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {save_dir}")

    # === 主循环：推理并立即保存 ===
    for file_path in tqdm(image_files, desc="Processing & Saving"):
        # 1. 推理
        raw_pred = sliding_window_inference(
            model=model,
            image_path=file_path,
            text_features=class_text_emb,
            patch_transform=patch_transform,
            device=device,
            window_size=args.img_size,
            stride=args.stride,
            dataset_name=args.dataset
        )

        # 2. 立即可视化并保存
        visualize_and_save_immediate(
            pixel_pred_raw=raw_pred,
            file_path=file_path,
            save_dir=save_dir
        )

    print(f"\nAll Done! Please check {save_dir}")


if __name__ == "__main__":
    main()