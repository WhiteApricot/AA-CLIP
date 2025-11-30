import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
from pandas import DataFrame, Series
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2  # 引入 opencv

from utils import setup_seed, cos_sim
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset, DOMAINS
from dataset.constants import DATA_PATH  # 引入 DATA_PATH 以读取原始图像
from forward_utils import (
    get_adapted_text_embedding,
    calculate_similarity_map,
    metrics_eval,
    # visualize, # 不再使用原有的 visualize
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


def add_header_to_image(image, text):
    """
    辅助函数：在图像顶部添加一个白色标题栏并写入文字
    """
    h, w, c = image.shape
    header_height = 40
    header = np.ones((header_height, w, c), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_color = (0, 0, 0)

    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (header_height + text_size[1]) // 2

    cv2.putText(header, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return np.vstack([header, image])


def visualize_custom(
        pixel_label: np.ndarray,
        pixel_preds: np.ndarray,
        file_names: list[str],
        save_path: str,  # 保留参数签名
        dataset_name: str,
        class_name: str,
):
    """
    自定义可视化函数：保存 [原始图像 | 真值Mask | 预测热力图] 并带有标题
    保存路径固定为项目根目录下的 results/
    """
    # -------------------------------------------------------------------------
    # 路径修改: 结果保存到项目根目录下的 results 文件夹
    save_dir = os.path.join("results", dataset_name, class_name)
    # -------------------------------------------------------------------------

    os.makedirs(save_dir, exist_ok=True)
    print(f"正在保存可视化结果到: {save_dir} ...")

    if pixel_preds.max() != pixel_preds.min():
        pixel_preds_norm = (pixel_preds - pixel_preds.min()) / (pixel_preds.max() - pixel_preds.min())
    else:
        pixel_preds_norm = pixel_preds

    for idx, file in enumerate(tqdm(file_names, desc="Saving Images")):
        # 1. 读取原始图像
        image_full_path = os.path.join(DATA_PATH[dataset_name], file)
        original_image = cv2.imread(image_full_path)

        if original_image is None:
            image_full_path_alt = os.path.join(DATA_PATH[dataset_name], os.path.basename(file))
            original_image = cv2.imread(image_full_path_alt)
            if original_image is None:
                continue

                # 2. 获取并处理 Mask (真值)
        mask = pixel_label[idx]
        if mask.ndim == 3: mask = mask.squeeze()

        h, w = mask.shape
        # 统一尺寸
        if original_image.shape[:2] != (h, w):
            original_image = cv2.resize(original_image, (w, h))

        mask_vis = (mask * 255).astype(np.uint8)
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

        # 3. 获取并处理预测结果 (热力图)
        pred = pixel_preds_norm[idx]
        pred_vis = (pred * 255).astype(np.uint8)
        pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)

        # 4. 添加标题栏
        original_vis = add_header_to_image(original_image, "Original Image")
        mask_vis = add_header_to_image(mask_vis, "Ground Truth")
        pred_vis = add_header_to_image(pred_vis, "Prediction Heatmap")

        # 5. 组合图像
        combined_image = np.hstack([original_vis, mask_vis, pred_vis])

        # 6. 保存
        safe_filename = file.replace("/", "_").replace("\\", "_")
        if not safe_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            safe_filename += ".png"

        save_file_path = os.path.join(save_dir, safe_filename)
        cv2.imwrite(save_file_path, combined_image)


def get_support_features(model, support_loader, device):
    all_features = []
    for input_data in support_loader:
        image = input_data[0].to(device)
        patch_tokens = model(image)
        patch_tokens = [t.reshape(-1, 768) for t in patch_tokens]
        all_features.append(patch_tokens)
    support_features = [
        torch.cat([all_features[j][i] for j in range(len(all_features))], dim=0)
        for i in range(len(all_features[0]))
    ]
    return support_features


def get_predictions(
        model: nn.Module,
        class_text_embeddings: torch.Tensor,
        test_loader: DataLoader,
        device: str,
        img_size: int,
        dataset: str = "MVTec",
):
    masks = []
    labels = []
    preds = []
    preds_image = []
    file_names = []
    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        mask = input_data["mask"].cpu().numpy()
        label = input_data["label"].cpu().numpy()
        file_name = input_data["file_name"]

        class_name = input_data["class_name"]
        assert len(set(class_name)) == 1, "mixed class not supported"
        masks.append(mask)
        labels.append(label)
        file_names.extend(file_name)

        epoch_text_feature = class_text_embeddings
        # forward image
        patch_features, det_feature = model(image)

        pred = det_feature @ epoch_text_feature
        pred = (pred[:, 1] + 1) / 2
        preds_image.append(pred.cpu().numpy())
        patch_preds = []
        for f in patch_features:
            patch_pred = calculate_similarity_map(
                f, epoch_text_feature, img_size, test=True, domain=DOMAINS[dataset]
            )
            patch_preds.append(patch_pred)
        patch_preds = torch.cat(patch_preds, dim=1).sum(1).cpu().numpy()
        preds.append(patch_preds)
    masks = np.concatenate(masks, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    preds_image = np.concatenate(preds_image, axis=0)
    return masks, labels, preds, preds_image, file_names


def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="ViT-B-16-plus-240, ViT-L-14-336",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    # testing
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--shot", type=int, default=4)
    # -------------------------------------------------------------------------
    # 修改点: 默认 batch_size 设为 1，以防 OOM
    parser.add_argument("--batch_size", type=int, default=1)
    # -------------------------------------------------------------------------
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)

    args = parser.parse_args()
    # ========================================================
    setup_seed(args.seed)
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "test.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("args: %s", vars(args))
    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # ========================================================
    # load model
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
        cache_dir="./model",  # 确保这里保留了本地路径
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

    # load checkpoints
    text_file = glob(args.save_path + "/text_adapter.pth")
    assert len(text_file) >= 0, "text adapter checkpoint not found"
    if len(text_file) > 0:
        checkpoint = torch.load(text_file[0])
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        adapt_text = True
    else:
        adapt_text = False

    # -------------------------------------------------------------------------
    # 修改点: 核心逻辑修改 - 只选择最后一个 epoch 的模型文件
    all_files = sorted(glob(args.save_path + "/image_adapter_*.pth"))
    if len(all_files) > 0:
        # 按照文件名中的数字大小排序，确保取到的是最大的那个数字 (epoch)
        # 假设文件名格式类似 .../image_adapter_20.pth
        try:
            all_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        except ValueError:
            # 如果解析数字失败，回退到默认排序，取最后一个
            pass
        files = [all_files[-1]]  # 列表只包含最后一个文件
        logger.info(f"Selected last checkpoint for testing: {files[0]}")
    else:
        files = []
    # -------------------------------------------------------------------------

    assert len(files) > 0, "image adapter checkpoint not found"

    # 这里的循环现在只会执行 1 次
    for file in files:
        checkpoint = torch.load(file)
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
        test_epoch = checkpoint["epoch"]
        logger.info("-----------------------------------------------")
        logger.info("load model from epoch %d", test_epoch)
        logger.info("-----------------------------------------------")

        # load dataset
        kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
        image_datasets = get_dataset(
            args.dataset,
            args.img_size,
            None,
            args.shot,
            "test",
            logger=logger,
        )
        with torch.no_grad():
            if adapt_text:
                text_embeddings = get_adapted_text_embedding(
                    model, args.dataset, device
                )
            else:
                text_embeddings = get_adapted_text_embedding(
                    clip_model, args.dataset, device
                )

        df = DataFrame(
            columns=[
                "class name",
                "pixel AUC",
                "pixel AP",
                "image AUC",
                "image AP",
            ]
        )
        for class_name, image_dataset in image_datasets.items():
            image_dataloader = torch.utils.data.DataLoader(
                image_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
            )

            # testing
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

            if args.visualize:
                # 调用自定义可视化
                visualize_custom(
                    masks,
                    preds,
                    file_names,
                    args.save_path,
                    args.dataset,
                    class_name=class_name,
                )

            class_result_dict = metrics_eval(
                masks,
                labels,
                preds,
                preds_image,
                class_name,
                domain=DOMAINS[args.dataset],
            )
            df.loc[len(df)] = Series(class_result_dict)

        # -------------------------------------------------------------------------
        # 修改点: 修复 Pandas 报错
        mean_metrics = df.mean(numeric_only=True)
        df.loc[len(df)] = mean_metrics
        df.loc[len(df) - 1, "class name"] = "Average"
        # -------------------------------------------------------------------------

        # 最终输出性能指标
        print("\n" + "=" * 50)
        print("Final Evaluation Results (Last Epoch):")
        print(df.to_string(index=False, justify="center"))
        print("=" * 50 + "\n")
        logger.info("final results:\n%s", df.to_string(index=False, justify="center"))


if __name__ == "__main__":
    main()