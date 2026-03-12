import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
import torch
import cv2
from PIL import Image
from torchvision import transforms
import math

# ==============================================================================
# Monkey Patching 区域
# ==============================================================================
import dataset.constants

dataset.constants.DATA_PATH["HubeiDown_Test"] = "data/HubeiDown_Test"
dataset.constants.CLASS_NAMES["HubeiDown_Test"] = ["road"]
dataset.constants.REAL_NAMES["HubeiDown_Test"] = {"road": "road"}
if not hasattr(dataset.constants, "DOMAINS"):
    dataset.constants.DOMAINS = {}
dataset.constants.DOMAINS["HubeiDown_Test"] = "Industrial"

# 针对道路裂缝优化的高级 Prompt (必须保留，否则对细微裂缝效果不好)
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
# ==============================================================================

from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
from forward_utils import get_adapted_text_embedding, calculate_similarity_map
from test import add_header_to_image


class SlidingWindowInferencer:
    def __init__(self, model, transform, img_size, stride_rate=0.5, device="cuda"):
        self.model = model
        self.transform = transform
        self.img_size = img_size
        self.stride = int(img_size * stride_rate)  # 滑动步长，0.5表示50%重叠
        self.device = device

    def predict_large_image(self, image_path, class_text_embeddings, dataset_name):
        """
        对单张大图进行滑窗推理
        """
        # 1. 读取原图 (OpenCV 读取的是 BGR, HxWxC)
        original_img_cv = cv2.imread(image_path)
        if original_img_cv is None:
            print(f"Error: Cannot read {image_path}")
            return None, None

        h_orig, w_orig = original_img_cv.shape[:2]

        # 2. 初始化全尺寸热力图累加器和计数器 (用于重叠平均)
        full_heatmap = np.zeros((h_orig, w_orig), dtype=np.float32)
        count_map = np.zeros((h_orig, w_orig), dtype=np.float32)

        # 3. 生成滑窗坐标
        # 我们需要在原图上滑动。如果原图不够切，需要 Pad 吗？
        # 简单的策略：只在有效区域滑动，边缘部分如果不满一个窗口，强行回退对齐边缘。

        y_steps = self._get_steps(h_orig)
        x_steps = self._get_steps(w_orig)

        patches_batch = []
        coords_batch = []

        # 4. 遍历所有滑窗
        # 为了速度，我们可以攒一个 batch 再推，或者逐个推
        # 这里为了显存安全，逐个推（或小 batch）

        total_patches = len(y_steps) * len(x_steps)
        # 如果图特别大，不要打印太频繁

        with torch.no_grad():
            for y in y_steps:
                for x in x_steps:
                    # 裁剪 (注意 OpenCV 是 BGR，Model 需要 RGB)
                    patch_cv = original_img_cv[y:y + self.img_size, x:x + self.img_size]

                    # 转 PIL RGB 并预处理
                    patch_pil = Image.fromarray(cv2.cvtColor(patch_cv, cv2.COLOR_BGR2RGB))
                    patch_tensor = self.transform(patch_pil).unsqueeze(0).to(self.device)

                    # --- 模型推理核心 ---
                    # 这里的 patch_features 是 list of tensors
                    patch_features, det_feature = self.model(patch_tensor)

                    # 计算局部热力图
                    patch_preds = []
                    for f in patch_features:
                        # calculate_similarity_map 返回的是 [B, 1, H, W]
                        # 这里的 img_size 传入的是窗口大小 (336)
                        patch_pred = calculate_similarity_map(
                            f,
                            class_text_embeddings,
                            self.img_size,
                            test=True,
                            domain=dataset.constants.DOMAINS[dataset_name]
                        )
                        patch_preds.append(patch_pred)

                    # 聚合多尺度特征 -> [1, 336, 336]
                    pred_patch_map = torch.cat(patch_preds, dim=1).sum(1).cpu().numpy()[0]

                    # --- 填充回大图 ---
                    full_heatmap[y:y + self.img_size, x:x + self.img_size] += pred_patch_map
                    count_map[y:y + self.img_size, x:x + self.img_size] += 1.0

        # 5. 计算平均值 (处理重叠区域)
        # 避免除以 0 (理论上 slide 覆盖全图不会有 0)
        np.place(count_map, count_map == 0, 1.0)
        final_heatmap = full_heatmap / count_map

        return original_img_cv, final_heatmap

    def _get_steps(self, length):
        """计算滑窗的起始坐标列表"""
        if length <= self.img_size:
            return [0]

        steps = []
        curr = 0
        while curr + self.img_size <= length:
            steps.append(curr)
            curr += self.stride

        # 确保覆盖最后一个边缘：如果最后一步没到头，添加一个从末尾往回数的窗口
        if steps[-1] + self.img_size < length:
            steps.append(length - self.img_size)

        return steps


def visualize_large_image(original_img, pred_score_map, file_name, save_dir):
    """
    保存全尺寸结果
    """
    h, w = original_img.shape[:2]

    # 归一化 (基于全图)
    p_min, p_max = pred_score_map.min(), pred_score_map.max()
    if p_max != p_min:
        pred_norm = (pred_score_map - p_min) / (p_max - p_min)
    else:
        pred_norm = pred_score_map

    # 生成热力图
    pred_vis = (pred_norm * 255).astype(np.uint8)
    pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)

    # 生成掩码 (0.5 阈值)
    mask_pred = (pred_norm > 0.5).astype(np.uint8) * 255
    mask_pred_vis = cv2.cvtColor(mask_pred, cv2.COLOR_GRAY2BGR)

    # 考虑到大图拼接可能太宽，我们这次不横向拼接3张，而是分开保存，或者只保存一张合成图
    # 这里为了看清细节，我们生成一张 "原图 + 半透明热力图" 的覆盖图 (Overlay)
    overlay = cv2.addWeighted(original_img, 0.6, pred_vis, 0.4, 0)

    # 还是拼接一下吧，为了统一
    # 给它们加上标题
    original_vis = add_header_to_image(original_img, "Original High-Res")
    heatmap_vis = add_header_to_image(pred_vis, "Detailed Heatmap")
    overlay_vis = add_header_to_image(overlay, "Overlay")

    # 3张并排 (注意：如果是4K图，并排后是12K宽，文件会很大)
    combined_image = np.hstack([original_vis, heatmap_vis, overlay_vis])

    safe_filename = os.path.basename(file_name).split('.')[0] + "_sliding.jpg"  # 存jpg省空间
    # 使用 cv2.imwrite 保存，JPEG质量设高点
    cv2.imwrite(os.path.join(save_dir, safe_filename), combined_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def main():
    parser = argparse.ArgumentParser(description="Sliding Window Prediction")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=336, help="这是滑窗的大小，不是缩放大小")
    parser.add_argument("--relu", action="store_true")
    parser.add_argument("--dataset", type=str, default="HubeiDown_Test")
    parser.add_argument("--save_path", type=str, default="ckpt/HubeiDown_Model")

    # 滑窗参数
    parser.add_argument("--stride_rate", type=float, default=0.5, help="滑窗重叠率，0.5代表重叠一半")

    # 参数需与训练时一致
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
    logging.basicConfig(filename=os.path.join(args.save_path, "predict_sliding.log"), level=logging.INFO)
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

    # 2. Load Checkpoints (Text & Image)
    text_file = glob(args.save_path + "/text_adapter.pth")
    if text_file:
        checkpoint = torch.load(text_file[0], map_location=device)
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        adapt_text = True
    else:
        adapt_text = False

    all_files = sorted(glob(args.save_path + "/image_adapter_*.pth"))
    if all_files:
        try:
            all_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        except:
            pass
        checkpoint = torch.load(all_files[-1], map_location=device)
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
        logger.info(f"Loaded Image Adapter: {all_files[-1]}")
    else:
        logger.error("No image adapter found!")
        return

    # 3. 准备变换 (Transforms) - 注意这里没有 Resize!
    # 只有 ToTensor 和 Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # 4. 初始化滑窗推理器
    inferencer = SlidingWindowInferencer(
        model,
        transform,
        img_size=args.img_size,
        stride_rate=args.stride_rate,
        device=device
    )

    # 5. 准备 Text Embedding
    with torch.no_grad():
        if adapt_text:
            text_embeddings = get_adapted_text_embedding(model, args.dataset, device)
        else:
            text_embeddings = get_adapted_text_embedding(clip_model, args.dataset, device)

    class_text_embedding = text_embeddings["road"]

    # 6. 扫描图片并处理
    data_dir = dataset.constants.DATA_PATH[args.dataset]
    image_files = glob(os.path.join(data_dir, "*.jpg")) + glob(os.path.join(data_dir, "*.png"))
    image_files = sorted(list(set(image_files)))

    save_dir = os.path.join("results", "HubeiDown_Sliding")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results (Full Resolution) will be saved to: {save_dir}")

    for img_path in tqdm(image_files, desc="Processing Large Images"):
        file_name = os.path.basename(img_path)

        # 执行滑窗推理
        original_img, heatmap = inferencer.predict_large_image(
            img_path,
            class_text_embedding,
            args.dataset
        )

        if original_img is not None:
            # 可视化并保存
            visualize_large_image(original_img, heatmap, file_name, save_dir)

    print("Done! Check results in 'results/HubeiDown_Sliding'")


if __name__ == "__main__":
    main()