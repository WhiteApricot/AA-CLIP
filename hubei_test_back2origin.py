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
dataset.constants.DATA_PATH["HubeiDown_Test"] = "data/G45_Test"
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
    自定义数据集类：读取图像，如果宽高小于1036，则插值到至少1036
    """

    def __init__(self, root_dir, class_name):
        self.root_dir = root_dir
        self.class_name = class_name
        self.min_size = 1036

        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(glob(os.path.join(root_dir, ext)))
            self.image_files.extend(glob(os.path.join(root_dir, ext.upper())))

        self.image_files = sorted(list(set(self.image_files)))

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        # 仅保留基础的 ToTensor 和 Normalize，Resize 动态处理
        self.base_transform = transforms.Compose([
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

            # 保证宽高至少为 1036
            new_W = max(orig_W, self.min_size)
            new_H = max(orig_H, self.min_size)

            if new_W != orig_W or new_H != orig_H:
                image = image.resize((new_W, new_H), Image.BICUBIC)

            image_tensor = self.base_transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image_tensor = torch.zeros((3, self.min_size, self.min_size))
            orig_W, orig_H, new_W, new_H = self.min_size, self.min_size, self.min_size, self.min_size

        return {
            "image": image_tensor,
            "orig_W": orig_W,
            "orig_H": orig_H,
            "file_name": file_name,
            "class_name": self.class_name
        }


def get_sliding_window_predictions(model, class_text_embeddings, image_tensor, img_size=518, stride=259):
    """
    核心滑动窗口逻辑 (带 50% 重叠率)
    """
    # image_tensor shape: [1, 3, H, W]
    B, C, H, W = image_tensor.shape
    device = image_tensor.device

    # 累加图与计数图 (用于重叠区域求平均)
    prob_map = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)

    # 计算滑动步长节点
    h_steps = list(range(0, H - img_size + 1, stride))
    if H - img_size > 0 and h_steps[-1] != H - img_size:
        h_steps.append(H - img_size)

    w_steps = list(range(0, W - img_size + 1, stride))
    if W - img_size > 0 and w_steps[-1] != W - img_size:
        w_steps.append(W - img_size)

    for y in h_steps:
        for x in w_steps:
            # 截取 518x518 的物理 Patch
            patch = image_tensor[:, :, y:y + img_size, x:x + img_size]

            # 模型推理
            patch_features, _ = model(patch)
            patch_preds = 0
            for f in patch_features:
                patch_preds += calculate_similarity_map(f, class_text_embeddings, img_size)
            patch_preds = patch_preds / len(patch_features)

            # 提取异常类的概率
            prob = torch.softmax(patch_preds, dim=1)[:, 1, :, :].squeeze(0)  # shape: [518, 518]

            # 累加到整图 map
            prob_map[y:y + img_size, x:x + img_size] += prob
            count_map[y:y + img_size, x:x + img_size] += 1

    # 求平均以消除拼缝边界
    prob_map = prob_map / torch.clamp(count_map, min=1.0)
    return prob_map.cpu().numpy()


def visualize_custom_hubei(
        pixel_preds: list,  # 注意：由于原图尺寸不同，这里必须用 list 接收
        file_names: list[str],
        save_path: str,
        dataset_name: str,
        class_name: str,
):
    save_dir = os.path.join("results", "G45_new")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving visualization results to: {save_dir} ...")

    # 全局归一化逻辑适配 List
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
        heatmap_vis = add_header_to_image(pred_vis, "Global Heatmap")
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
    parser.add_argument("--save_path", type=str, default="ckpt/G45_Std")

    parser.add_argument("--shot", type=int, default=32)
    # 因为图片尺寸不一，滑动窗口必须强制 batch_size = 1
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

    test_dataset = DirectFolderDataset(
        root_dir=data_dir,
        class_name="road"
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
            batch_size=args.batch_size,  # 强制为 1
            shuffle=False,
            num_workers=4 if use_cuda else 0
        )

        all_preds = []
        all_filenames = []

        with torch.no_grad():
            class_text_embeddings = text_embeddings[class_name]

            for batch in tqdm(image_dataloader, desc="Sliding Window Predicting"):
                image_tensor = batch["image"].to(device)
                orig_W = batch["orig_W"][0].item()
                orig_H = batch["orig_H"][0].item()
                file_name = batch["file_name"][0]

                # 步长设为 259 (518 的一半，50%重叠率)
                prob_map = get_sliding_window_predictions(
                    model,
                    class_text_embeddings,
                    image_tensor,
                    img_size=args.img_size,
                    stride=args.img_size // 2
                )

                # 恢复回原本极其精细的真实物理分辨率
                if prob_map.shape != (orig_H, orig_W):
                    prob_map = cv2.resize(prob_map, (orig_W, orig_H))

                all_preds.append(prob_map)
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