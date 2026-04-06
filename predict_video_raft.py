import os
import argparse
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms
from glob import glob

# 引入 PyTorch 官方 SOTA 光流模型 RAFT
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

# ==============================================================================
# Monkey Patching 区域
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

def get_sliding_window_predictions(model, class_text_embeddings, image_tensor, img_size=518, stride=259):
    """原始滑动窗口逻辑"""
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

def warp_probability_map(prob_map, flow, device):
    """
    核心：使用 RAFT 密集光流对上一帧概率图进行像素级扭曲(Warping)
    """
    _, H, W = flow.shape
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid = torch.stack((x, y), dim=0).float()
    grid = grid + flow
    
    grid[0, :, :] = 2.0 * grid[0, :, :] / max(W - 1, 1) - 1.0
    grid[1, :, :] = 2.0 * grid[1, :, :] / max(H - 1, 1) - 1.0
    grid = grid.permute(1, 2, 0).unsqueeze(0)
    
    prob_tensor = torch.from_numpy(prob_map).float().unsqueeze(0).unsqueeze(0).to(device)
    warped_prob = F.grid_sample(prob_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return warped_prob.squeeze().cpu().numpy()

def main():
    setup_seed(111)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    save_path = "ckpt/videoTrain"
    output_dir = "results/videoTest"
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. 加载你的 CLIP 裂缝检测模型 (修复参数传入方式)
    # ---------------------------------------------------------
    print("Loading CLIP Adapted Model...")
    
    # 【修复处1】: 严格使用关键字参数
    clip_model = create_model(
        model_name="ViT-L-14-336",
        img_size=518,
        device=device,
        pretrained="openai",
        require_pretrained=True,
        cache_dir="./model"
    )
    clip_model.eval()

    # 【修复处2】: 严格使用关键字参数
    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=0.1,
        image_adapt_weight=0.1,
        text_adapt_until=3,
        image_adapt_until=6,
        relu=False
    ).to(device)
    model.eval()

    # 加载权重
    text_file = glob(save_path + "/text_adapter.pth")
    if text_file:
        model.text_adapter.load_state_dict(torch.load(text_file[0], map_location=device)["text_adapter"])
    model.image_adapter.load_state_dict(torch.load(sorted(glob(save_path + "/image_adapter*.pth"))[-1], map_location=device)["image_adapter"])

    with torch.no_grad():
        class_text_embeddings = get_adapted_text_embedding(model if text_file else clip_model, "videoTrain", device)["road"]

    # ---------------------------------------------------------
    # 2. 加载 RAFT 光流模型 (SOTA Motion Estimation)
    # ---------------------------------------------------------
    print("Loading RAFT Optical Flow Model...")
    raft_weights = Raft_Small_Weights.DEFAULT
    raft_transforms = raft_weights.transforms()
    raft_model = raft_small(weights=raft_weights, progress=False).to(device)
    raft_model.eval()

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    min_size = 1036

    # ---------------------------------------------------------
    # 3. 视频测试任务配置
    # ---------------------------------------------------------
    video_tasks = [
        {"path": "crack-data/5/5crackwaican.mp4", "start_s": 150, "end_s": 170, "name": "5crackwaican"},
        {"path": "crack-data/3/3crack.mp4", "start_s": 188, "end_s": 211, "name": "3crack"}
    ]

    for task in video_tasks:
        print(f"\n[{task['name']}] Processing {task['start_s']}s to {task['end_s']}s with RAFT SOTA...")
        cap = cv2.VideoCapture(task["path"])
        if not cap.isOpened():
            print(f"Failed to open {task['path']}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(task["start_s"] * fps)
        end_frame = int(task["end_s"] * fps)
        
        # RAFT 需要图像尺寸是 8 的倍数
        raft_W = (orig_W // 8) * 8
        raft_H = (orig_H // 8) * 8
        
        out_path = os.path.join(output_dir, f"{task['name']}_RAFT_SOTA.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_W * 2, orig_H))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        prev_raft_img = None
        prev_prob_map = None

        with torch.no_grad():
            for i in tqdm(range(end_frame - start_frame), desc=f"Inferencing {task['name']}"):
                ret, frame = cap.read()
                if not ret: break
                
                # 图像准备 (CLIP模型输入)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                
                new_W, new_H = max(orig_W, min_size), max(orig_H, min_size)
                if new_W != orig_W or new_H != orig_H:
                    pil_img = pil_img.resize((new_W, new_H), Image.BICUBIC)
                image_tensor = base_transform(pil_img).unsqueeze(0).to(device)

                # A. 提取当前帧原始预测
                prob_map_local = get_sliding_window_predictions(model, class_text_embeddings, image_tensor, img_size=518, stride=259)
                prob_map_global = get_global_prediction(model, class_text_embeddings, image_tensor, img_size=518)
                curr_prob_map = np.sqrt(
                    cv2.resize(prob_map_local, (orig_W, orig_H)) * cv2.resize(prob_map_global, (orig_W, orig_H))
                )

                # 图像准备 (RAFT 光流输入)
                raft_frame = cv2.resize(img_rgb, (raft_W, raft_H))
                curr_raft_img = torch.from_numpy(raft_frame).permute(2, 0, 1).float()

                # B. SOTA 光流融合逻辑
                if prev_raft_img is None or prev_prob_map is None:
                    prob_map_final = curr_prob_map
                else:
                    # 计算光流
                    img1, img2 = raft_transforms(curr_raft_img, prev_raft_img)
                    list_of_flows = raft_model(img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device))
                    flow = list_of_flows[-1][0]
                    
                    # 放缩流场以匹配原图
                    flow_resized = F.interpolate(flow.unsqueeze(0), size=(orig_H, orig_W), mode="bilinear", align_corners=False)[0]
                    flow_resized[0] *= (orig_W / raft_W)
                    flow_resized[1] *= (orig_H / raft_H)

                    # 亚像素级扭曲与历史融合
                    aligned_prev_prob = warp_probability_map(prev_prob_map, flow_resized, device)
                    prob_map_final = 0.7 * curr_prob_map + 0.3 * aligned_prev_prob

                # 缓存更新
                prev_raft_img = curr_raft_img.clone()
                prev_prob_map = prob_map_final.copy()

                # C. 渲染叠加与保存
                binary_mask = (prob_map_final > 0.5).astype(np.uint8) * 255
                color_mask = np.zeros_like(frame)
                color_mask[:, :, 2] = binary_mask 
                
                overlaid_frame = cv2.addWeighted(frame, 1.0, color_mask, 0.5, 0)
                out.write(np.hstack((frame, overlaid_frame)))

        cap.release()
        out.release()
    print("\n[Done] All SOTA videos generated successfully in results/videoTest/")

if __name__ == "__main__":
    main()