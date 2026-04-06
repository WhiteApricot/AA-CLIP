import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms
from glob import glob

# ==============================================================================
# Monkey Patching 区域 (保持数据注册设置)
# ==============================================================================
import dataset.constants

dataset.constants.DATA_PATH["videoTrain"] = "data/videoTrain"
dataset.constants.CLASS_NAMES["videoTrain"] = ["road"]
dataset.constants.REAL_NAMES["videoTrain"] = {"road": "road"}
if not hasattr(dataset.constants, "DOMAINS"):
    dataset.constants.DOMAINS = {}
dataset.constants.DOMAINS["videoTrain"] = "Industrial"
# ==============================================================================

from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
from forward_utils import get_adapted_text_embedding, calculate_similarity_map

def add_header(image, text):
    """为可视化的图像添加标题头部"""
    header = np.zeros((50, image.shape[1], 3), dtype=np.uint8)
    header[:] = (50, 50, 50)  # 深灰色背景
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (50 + text_size[1]) // 2
    cv2.putText(header, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
    return np.vstack((header, image))

def get_sliding_window_predictions(model, class_text_embeddings, image_tensor, img_size=518, stride=259):
    """原始核心滑动窗口逻辑"""
    B, C, H, W = image_tensor.shape
    device = image_tensor.device

    prob_map = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)

    h_steps = list(range(0, H - img_size + 1, stride))
    if H - img_size > 0 and h_steps[-1] != H - img_size: h_steps.append(H - img_size)

    w_steps = list(range(0, W - img_size + 1, stride))
    if W - img_size > 0 and w_steps[-1] != W - img_size: w_steps.append(W - img_size)

    for y in h_steps:
        for x in w_steps:
            patch = image_tensor[:, :, y:y + img_size, x:x + img_size]
            patch_features, _ = model(patch)
            patch_preds = sum(calculate_similarity_map(f, class_text_embeddings, img_size) for f in patch_features) / len(patch_features)
            prob = torch.softmax(patch_preds, dim=1)[:, 1, :, :].squeeze(0)
            prob_map[y:y + img_size, x:x + img_size] += prob
            count_map[y:y + img_size, x:x + img_size] += 1

    return (prob_map / torch.clamp(count_map, min=1.0)).cpu().numpy()

def get_global_prediction(model, class_text_embeddings, image_tensor, img_size=518):
    """原始全局预测逻辑"""
    global_tensor = F.interpolate(image_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)
    global_features, _ = model(global_tensor)
    global_preds = sum(calculate_similarity_map(f, class_text_embeddings, img_size) for f in global_features) / len(global_features)
    prob = torch.softmax(global_preds, dim=1)[:, 1, :, :].squeeze(0)
    return prob.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Test Single Frames from Video")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    parser.add_argument("--dataset", type=str, default="videoTrain")
    parser.add_argument("--save_path", type=str, default="ckpt/videoTrain") 
    
    # 视频与输出配置
    parser.add_argument("--video_path", type=str, default="crack-data/5/5crackwaican.mp4")
    parser.add_argument("--output_dir", type=str, default="results/videoTest")
    parser.add_argument("--num_frames", type=int, default=40, help="随机抽取的帧数")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    setup_seed(args.seed)

    # 1. 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory initialized at: {args.output_dir}")

    # 2. 初始化模型与设备
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    print(f"Loading model {args.model_name}...")
    clip_model = create_model(
        model_name=args.model_name, img_size=args.img_size, device=device,
        pretrained="openai", require_pretrained=True, cache_dir="./model",
    )
    clip_model.eval()

    model = AdaptedCLIP(
        clip_model=clip_model, text_adapt_weight=0.1, image_adapt_weight=0.1,
        text_adapt_until=3, image_adapt_until=6, relu=args.relu,
    ).to(device)
    model.eval()

    # 3. 加载权重
    text_file = glob(args.save_path + "/text_adapter.pth")
    adapt_text = False
    if len(text_file) > 0:
        print(f"Loading Text Adapter: {text_file[0]}")
        checkpoint = torch.load(text_file[0], map_location=device)
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        adapt_text = True

    all_files = sorted(glob(args.save_path + "/image_adapter*.pth"))
    if len(all_files) > 0:
        checkpoint_path = all_files[-1]
        print(f"Loading Image Adapter: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
    else:
        print("Error: No image adapter checkpoint found in", args.save_path)
        return

    # 4. 提取文本特征
    with torch.no_grad():
        if adapt_text:
            text_embeddings = get_adapted_text_embedding(model, args.dataset, device)
        else:
            text_embeddings = get_adapted_text_embedding(clip_model, args.dataset, device)
    class_text_embeddings = text_embeddings["road"]

    # 5. 读取视频并抽取随机帧
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    
    # 确保抽取数量不超过总帧数
    num_samples = min(args.num_frames, total_frames)
    sampled_indices = sorted(random.sample(range(total_frames), num_samples))
    print(f"Randomly selected {num_samples} frames for testing.")

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    min_size = 1036

    # 6. 单帧推理与可视化
    with torch.no_grad():
        for idx in tqdm(sampled_indices, desc="Testing Single Frames"):
            # 跳转到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            orig_H, orig_W = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # 缩放逻辑 (保证最小边满足要求)
            new_W = max(orig_W, min_size)
            new_H = max(orig_H, min_size)
            if new_W != orig_W or new_H != orig_H:
                pil_img = pil_img.resize((new_W, new_H), Image.BICUBIC)
                
            image_tensor = base_transform(pil_img).unsqueeze(0).to(device)

            # --- 核心推理 (单图逻辑) ---
            prob_map_local = get_sliding_window_predictions(model, class_text_embeddings, image_tensor, img_size=args.img_size, stride=args.img_size // 2)
            if prob_map_local.shape != (orig_H, orig_W):
                prob_map_local = cv2.resize(prob_map_local, (orig_W, orig_H))

            prob_map_global = get_global_prediction(model, class_text_embeddings, image_tensor, img_size=args.img_size)
            if prob_map_global.shape != (orig_H, orig_W):
                prob_map_global = cv2.resize(prob_map_global, (orig_W, orig_H))

            # 融合概率图
            prob_map_final = np.sqrt(prob_map_local * prob_map_global)

            # --- 可视化渲染 (保持原有三联图风格) ---
            # 1. 热力图
            heatmap = (prob_map_final * 255).astype(np.uint8)
            heatmap_vis = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # 2. 二值化掩膜图
            binary_mask = (prob_map_final > 0.5).astype(np.uint8) * 255
            mask_vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

            # 添加头部标签
            orig_vis_with_header = add_header(frame, "Original Frame")
            heatmap_vis_with_header = add_header(heatmap_vis, "Global Heatmap")
            mask_vis_with_header = add_header(mask_vis, "Predicted Mask")

            # 左右拼接
            combined_image = np.hstack([orig_vis_with_header, heatmap_vis_with_header, mask_vis_with_header])
            
            # 保存图像
            save_name = os.path.join(args.output_dir, f"frame_{idx:05d}.png")
            cv2.imwrite(save_name, combined_image)

    cap.release()
    print(f"\nDone! All test frames saved to {args.output_dir}")

if __name__ == "__main__":
    main()