import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ==============================================================================
# Monkey Patching 区域
# ==============================================================================
import dataset.constants

# 1. 注册数据路径
dataset.constants.DATA_PATH["HubeiDown_Test"] = "data/G45_crack_mask/test_image"
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
from dataset import DOMAINS
from forward_utils import get_adapted_text_embedding, calculate_similarity_map
from test import add_header_to_image


class DirectFolderDataset(Dataset):
    """
    修改后的数据集类：直接将图像统一缩放到 518x518 送入模型
    """

    def __init__(self, root_dir, class_name, img_size=518):
        self.root_dir = root_dir
        self.class_name = class_name
        self.img_size = img_size

        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(glob(os.path.join(root_dir, ext)))
            self.image_files.extend(glob(os.path.join(root_dir, ext.upper())))

        self.image_files = sorted(list(set(self.image_files)))

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        # 直接 Resize 到 518x518
        self.base_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
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
            orig_W, orig_H = image.size
            # 直接转换为 518x518 的 Tensor
            image_tensor = self.base_transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image_tensor = torch.zeros((3, self.img_size, self.img_size))
            orig_W, orig_H = self.img_size, self.img_size

        return {
            "image": image_tensor,
            "orig_W": orig_W,
            "orig_H": orig_H,
            "file_name": file_name,
            "class_name": self.class_name
        }


def get_single_prediction_with_erosion(model, class_text_embeddings, image_tensor, orig_H, orig_W, img_size=518):
    """
    仅输入 518x518 进行预测，缩放回原图大小后，进行 3 次形态学腐蚀
    """
    # 1. 前向传播提取特征
    features, _ = model(image_tensor)
    patch_preds = 0
    for f in features:
        patch_preds += calculate_similarity_map(f, class_text_embeddings, img_size)
    patch_preds = patch_preds / len(features)

    # 2. 提取异常概率，shape: [518, 518]
    prob = torch.softmax(patch_preds, dim=1)[:, 1, :, :].squeeze(0).cpu().numpy()

    # 3. 恢复到原图分辨率
    prob_resized = cv2.resize(prob, (orig_W, orig_H))

    # 4. 形态学腐蚀 (与 CrackCLIP 保持一致：3x3 核，3 次迭代)
    # cv2.erode 支持对 float32 的概率图操作，相当于局部最小值滤波，能有效收缩高概率的裂缝区域
    kernel = np.ones((3, 3), np.uint8)
    prob_eroded = cv2.erode(prob_resized, kernel, iterations=0)

    return prob_eroded


def visualize_custom_hubei(
        pixel_preds: list,
        file_names: list[str],
        save_path: str,
        dataset_name: str,
        class_name: str,
):
    """
    可视化逻辑保持不变
    """
    save_dir = os.path.join("results", "Hubei_mask_eroded")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving visualization results to: {save_dir} ...")

    p_min = min([p.min() for p in pixel_preds])
    p_max = max([p.max() for p in pixel_preds])
    print(f"Global Stats - Min: {p_min:.4f}, Max: {p_max:.4f}")

    for idx, file in enumerate(tqdm(file_names, desc="Saving Images")):
        image_full_path = os.path.join(dataset.constants.DATA_PATH[dataset_name], file)
        original_image = cv2.imread(image_full_path)

        if original_image is None:
            continue

        h, w, c = original_image.shape

        pred = pixel_preds[idx]
        if p_max != p_min:
            pred = (pred - p_min) / (p_max - p_min)

        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h))

        pred_vis = (pred * 255).astype(np.uint8)
        pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)

        mask_pred = (pred > 0.5).astype(np.uint8) * 255
        mask_pred_vis = cv2.cvtColor(mask_pred, cv2.COLOR_GRAY2BGR)

        original_vis = add_header_to_image(original_image, "Original")
        heatmap_vis = add_header_to_image(pred_vis, "Eroded Heatmap")
        mask_vis = add_header_to_image(mask_pred_vis, "Predicted Mask")

        combined_image = np.hstack([original_vis, heatmap_vis, mask_vis])
        safe_filename = os.path.basename(file).split('.')[0] + ".png"
        cv2.imwrite(os.path.join(save_dir, safe_filename), combined_image)


def main():
    parser = argparse.ArgumentParser(description="Predicting AnyModel Fixed")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    parser.add_argument("--dataset", type=str, default="HubeiDown_Test")
    parser.add_argument("--save_path", type=str, default="ckpt/G45_mask")

    parser.add_argument("--shot", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--visualize", action="store_true", default=True)

    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)
    parser.add_argument("--seed", type=int, default=111)

    args = parser.parse_args()
    setup_seed(args.seed)

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

    data_dir = dataset.constants.DATA_PATH[args.dataset]
    logger.info(f"Scanning images directly from: {data_dir}")

    # 数据集直接 resize 到 518x518
    test_dataset = DirectFolderDataset(
        root_dir=data_dir,
        class_name="road",
        img_size=args.img_size
    )
    image_datasets = {"road": test_dataset}
    logger.info(f"Found {len(test_dataset)} images.")

    with torch.no_grad():
        if adapt_text:
            text_embeddings = get_adapted_text_embedding(model, args.dataset, device)
        else:
            text_embeddings = get_adapted_text_embedding(clip_model, args.dataset, device)

    for class_name, image_dataset in image_datasets.items():
        logger.info(f"Processing class: {class_name}")

        image_dataloader = DataLoader(
            image_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4 if use_cuda else 0
        )

        all_preds = []
        all_filenames = []

        with torch.no_grad():
            class_text_embeddings = text_embeddings[class_name]

            # 改为简单的单次前向推理遍历
            for batch in tqdm(image_dataloader, desc="Predicting (518x518 + Erosion)"):
                image_tensor = batch["image"].to(device)
                orig_W = batch["orig_W"][0].item()
                orig_H = batch["orig_H"][0].item()
                file_name = batch["file_name"][0]

                # 直接调用包含腐蚀操作的单次预测函数
                prob_map_final = get_single_prediction_with_erosion(
                    model,
                    class_text_embeddings,
                    image_tensor,
                    orig_H,
                    orig_W,
                    img_size=args.img_size
                )

                all_preds.append(prob_map_final)
                all_filenames.append(file_name)

        if args.visualize:
            visualize_custom_hubei(
                all_preds,
                all_filenames,
                args.save_path,
                args.dataset,
                class_name
            )

        logger.info(f"Finished processing {class_name}")

    print("Done!")

if __name__ == "__main__":
    main()