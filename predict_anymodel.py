import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
from pandas import DataFrame, Series
import torch
import cv2

# ==============================================================================
# Monkey Patching 区域：必须在导入其他模块之前完成
# ==============================================================================
import dataset.constants

# 1. 注册数据路径
dataset.constants.DATA_PATH["HubeiDown_Test"] = "data/HubeiDown_Test"

# 2. 注册类别名称
dataset.constants.CLASS_NAMES["HubeiDown_Test"] = ["road"]

# 3. 【修复点】注册真实名称映射 (REAL_NAMES)
# 缺少这一步会导致 get_adapted_text_embedding 报错
dataset.constants.REAL_NAMES["HubeiDown_Test"] = {"road": "road"}

# 4. 【修复点】注册领域信息 (DOMAINS)
# 缺少这一步会导致 metrics_eval 计算相似度时报错
if not hasattr(dataset.constants, "DOMAINS"):
    dataset.constants.DOMAINS = {}
dataset.constants.DOMAINS["HubeiDown_Test"] = "general"
# ==============================================================================

from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset, DOMAINS
from forward_utils import get_adapted_text_embedding, calculate_similarity_map
# 注意：我们不再导入 metrics_eval，因为无 GT 的情况下无法计算指标
from test import get_predictions, add_header_to_image


def visualize_custom_hubei(
        pixel_preds: np.ndarray,
        file_names: list[str],
        save_path: str,
        dataset_name: str,
        class_name: str,
):
    """
    修改后的可视化函数
    输出: [原图 | 异常分数热力图 | 像素级异常掩码图 (预测)]
    """
    # 结果保存路径
    save_dir = os.path.join("results", "HubeiDown")  # 强制指定输出到 results/HubeiDown
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving visualization results to: {save_dir} ...")

    # 归一化预测值到 0-1
    if pixel_preds.max() != pixel_preds.min():
        pixel_preds_norm = (pixel_preds - pixel_preds.min()) / (pixel_preds.max() - pixel_preds.min())
    else:
        pixel_preds_norm = pixel_preds

    for idx, file in enumerate(tqdm(file_names, desc="Saving Images")):
        # 1. 读取原图
        image_full_path = os.path.join(dataset.constants.DATA_PATH[dataset_name], file)
        original_image = cv2.imread(image_full_path)

        if original_image is None:
            continue

        h, w, c = original_image.shape

        # 2. 处理热力图 (Heatmap)
        pred = pixel_preds_norm[idx]
        # 调整大小以匹配原图
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h))

        pred_vis = (pred * 255).astype(np.uint8)
        pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)

        # 3. 处理预测掩码 (Pixel-level Anomaly Mask)
        # 这里采用 0.5 阈值 (基于归一化后的图)
        mask_pred = (pred > 0.5).astype(np.uint8) * 255
        mask_pred_vis = cv2.cvtColor(mask_pred, cv2.COLOR_GRAY2BGR)

        # 4. 添加标题
        original_vis = add_header_to_image(original_image, "Original")
        heatmap_vis = add_header_to_image(pred_vis, "Anomaly Heatmap")
        mask_vis = add_header_to_image(mask_pred_vis, "Predicted Mask")

        # 5. 拼接
        combined_image = np.hstack([original_vis, heatmap_vis, mask_vis])

        # 6. 保存
        # 处理文件名，防止路径分隔符问题
        safe_filename = os.path.basename(file).split('.')[0] + ".png"
        cv2.imwrite(os.path.join(save_dir, safe_filename), combined_image)


def main():
    parser = argparse.ArgumentParser(description="Predicting AnyModel Fixed")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    # 确保这里的 img_size 与训练时一致 (推荐 336)
    parser.add_argument("--img_size", type=int, default=336)
    parser.add_argument("--relu", action="store_true")

    parser.add_argument("--dataset", type=str, default="HubeiDown_Test")
    parser.add_argument("--save_path", type=str, default="ckpt/HubeiDown_Model")

    parser.add_argument("--shot", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--visualize", action="store_true", default=True)

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
    # 清理旧 handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "predict.log"),
        encoding="utf-8",
        level=logging.INFO
    )
    # 添加控制台输出
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load Model
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

    # Load Text Adapter
    text_file = glob(args.save_path + "/text_adapter.pth")
    if len(text_file) > 0:
        logger.info(f"Loading Text Adapter: {text_file[0]}")
        checkpoint = torch.load(text_file[0], map_location=device)
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        adapt_text = True
    else:
        logger.warning("No Text Adapter found, using zero-shot CLIP text features.")
        adapt_text = False

    # Load Image Adapter (Last Epoch)
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
        logger.error("No image adapter checkpoint found! Cannot proceed.")
        return

    # Load Dataset
    logger.info(f"Loading dataset {args.dataset}...")
    image_datasets = get_dataset(args.dataset, args.img_size, None, args.shot, "test", logger=logger)

    # Text Embeddings
    with torch.no_grad():
        if adapt_text:
            text_embeddings = get_adapted_text_embedding(model, args.dataset, device)
        else:
            text_embeddings = get_adapted_text_embedding(clip_model, args.dataset, device)

    # Inference Loop
    for class_name, image_dataset in image_datasets.items():
        logger.info(f"Processing class: {class_name}")
        image_dataloader = torch.utils.data.DataLoader(
            image_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4 if use_cuda else 0
        )

        with torch.no_grad():
            class_text_embeddings = text_embeddings[class_name]
            masks, labels, preds, preds_image, file_names = get_predictions(
                model=model,
                class_text_embeddings=class_text_embeddings,
                test_loader=image_dataloader,
                device=device,
                img_size=args.img_size,
                dataset=args.dataset,
            )

        # 可视化
        visualize_custom_hubei(
            preds,  # 使用像素级预测热力图
            file_names,
            args.save_path,
            args.dataset,
            class_name
        )

        logger.info(f"Inference and visualization finished for class {class_name}.")
        print(f"Done! Results saved to results/HubeiDown")


if __name__ == "__main__":
    main()