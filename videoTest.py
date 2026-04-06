import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms
from collections import deque

# ==============================================================================
# Monkey Patching 区域 (保持你原有的数据注册设置)
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
    """
    保留你的原始核心滑动窗口逻辑 (带 50% 重叠率)
    """
    B, C, H, W = image_tensor.shape
    device = image_tensor.device

    prob_map = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)

    h_steps = list(range(0, H - img_size + 1, stride))
    if H - img_size > 0 and h_steps[-1] != H - img_size:
        h_steps.append(H - img_size)

    w_steps = list(range(0, W - img_size + 1, stride))
    if W - img_size > 0 and w_steps[-1] != W - img_size:
        w_steps.append(W - img_size)

    for y in h_steps:
        for x in w_steps:
            patch = image_tensor[:, :, y:y + img_size, x:x + img_size]

            patch_features, _ = model(patch)
            patch_preds = 0
            for f in patch_features:
                patch_preds += calculate_similarity_map(f, class_text_embeddings, img_size)
            patch_preds = patch_preds / len(patch_features)

            prob = torch.softmax(patch_preds, dim=1)[:, 1, :, :].squeeze(0)

            prob_map[y:y + img_size, x:x + img_size] += prob
            count_map[y:y + img_size, x:x + img_size] += 1

    prob_map = prob_map / torch.clamp(count_map, min=1.0)
    return prob_map.cpu().numpy()


def get_global_prediction(model, class_text_embeddings, image_tensor, img_size=518):
    """
    保留你的原始全局预测逻辑
    """
    global_tensor = F.interpolate(image_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)

    global_features, _ = model(global_tensor)
    global_preds = 0
    for f in global_features:
        global_preds += calculate_similarity_map(f, class_text_embeddings, img_size)
    global_preds = global_preds / len(global_features)

    prob = torch.softmax(global_preds, dim=1)[:, 1, :, :].squeeze(0)
    return prob.cpu().numpy()


def process_video():
    parser = argparse.ArgumentParser(description="Predicting Video with Temporal Context")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    # 注意：这里使用你训练好模型的名称作为 dataset 参数来加载正确的文本特征
    parser.add_argument("--dataset", type=str, default="videoTrain") 
    parser.add_argument("--save_path", type=str, default="ckpt/videoTrain") # 你刚刚训练保存的路径
    
    # 时序融合窗口
    parser.add_argument("--temporal_window", type=int, default=5, help="融合过去 N 帧的历史信息")
    # 输入视频与输出路径
    parser.add_argument("--video_path", type=str, default="crack-data/5/5crackwaican.mp4")
    parser.add_argument("--output_dir", type=str, default="results")
    
    args = parser.parse_args()
    setup_seed(111)

    # 1. 初始化模型
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

    # 2. 加载权重
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
        print("Error: No image adapter checkpoint found!")
        return

    # 3. 提取文本特征
    with torch.no_grad():
        if adapt_text:
            text_embeddings = get_adapted_text_embedding(model, args.dataset, device)
        else:
            text_embeddings = get_adapted_text_embedding(clip_model, args.dataset, device)
    class_text_embeddings = text_embeddings["road"]

    # 4. 准备视频读取
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_for_30s = int(10 * fps)
    
    # 随机选择开始帧 (确保有至少 30 秒)
    start_frame = 0
    if total_frames > frames_for_30s:
        start_frame = random.randint(0, total_frames - frames_for_30s)
        start_frame = 160 * fps
    else:
        frames_for_30s = total_frames # 视频不足 30 秒，全量处理

    print(f"Video total frames: {total_frames}, FPS: {fps}")
    print(f"Randomly selected start frame: {start_frame}, processing {frames_for_30s} frames (30s).")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 5. 准备视频写入
    os.makedirs(args.output_dir, exist_ok=True)
    out_video_path = os.path.join(args.output_dir, "predicted_30s_video.mp4")
    
    orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 输出视频将是左右拼接，所以宽度乘以 2
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (orig_W * 2, orig_H))

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    # 初始化时空队列
    temporal_queue = deque(maxlen=args.temporal_window)
    min_size = 1036

    # 6. 开始逐帧推理
    with torch.no_grad():
        for i in tqdm(range(frames_for_30s), desc="Processing Video Frames"):
            ret, frame = cap.read()
            if not ret:
                break
                
            # 预处理图像
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            new_W = max(orig_W, min_size)
            new_H = max(orig_H, min_size)
            if new_W != orig_W or new_H != orig_H:
                pil_img = pil_img.resize((new_W, new_H), Image.BICUBIC)
                
            image_tensor = base_transform(pil_img).unsqueeze(0).to(device)

            # --- 空间预测 (Local + Global) ---
            prob_map_local = get_sliding_window_predictions(model, class_text_embeddings, image_tensor, img_size=args.img_size, stride=args.img_size // 2)
            if prob_map_local.shape != (orig_H, orig_W):
                prob_map_local = cv2.resize(prob_map_local, (orig_W, orig_H))

            prob_map_global = get_global_prediction(model, class_text_embeddings, image_tensor, img_size=args.img_size)
            if prob_map_global.shape != (orig_H, orig_W):
                prob_map_global = cv2.resize(prob_map_global, (orig_W, orig_H))

            # 空间融合
            prob_map_spatial = np.sqrt(prob_map_local * prob_map_global)

            # --- 时空管融合 (Temporal Context Fusion) ---
            temporal_queue.append(prob_map_spatial)
            current_queue_len = len(temporal_queue)
            
            if current_queue_len == 1:
                prob_map_final = prob_map_spatial
            else:
                weights = np.linspace(0.5, 1.0, current_queue_len)
                weights = weights / np.sum(weights)
                spatio_temporal_volume = np.stack(list(temporal_queue), axis=0)
                prob_map_final = np.tensordot(weights, spatio_temporal_volume, axes=1)

            # --- 可视化掩膜渲染 ---
            # 阈值判断，生成二值掩膜 (你可以根据实际情况调整 0.5 的阈值)
            binary_mask = (prob_map_final > 0.5).astype(np.uint8) * 255
            
            # 创建红色掩膜层以叠加在原视频上
            color_mask = np.zeros_like(frame)
            color_mask[:, :, 2] = binary_mask # 红色通道赋最大值
            
            # 叠加掩膜，透明度 alpha 设为 0.5
            alpha = 0.5
            overlaid_frame = cv2.addWeighted(frame, 1.0, color_mask, alpha, 0)

            # 左右拼接: 原图 | 掩膜叠加图
            combined_frame = np.hstack((frame, overlaid_frame))
            
            # 将处理后的帧写入视频文件
            out.write(combined_frame)

    cap.release()
    out.release()
    print(f"\nDone! Output video saved to: {out_video_path}")

if __name__ == "__main__":
    process_video()import os
import argparse
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms
from collections import deque
from glob import glob

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

def align_prob_map_homography(curr_gray, prev_gray, prev_prob_map):
    """使用 ORB 特征和全局单应性矩阵进行对齐"""
    orb = cv2.ORB_create(1000) # 提取最多1000个特征点
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    
    # 如果特征点太少，降级为不对齐（直接返回上一帧）
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return prev_prob_map
        
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 计算单应性矩阵 (RANSAC 剔除误匹配)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            h, w = prev_prob_map.shape
            aligned_prob = cv2.warpPerspective(
                prev_prob_map, M, (w, h), 
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            return aligned_prob
            
    return prev_prob_map

def main():
    parser = argparse.ArgumentParser(description="Compare Video Processing Methods")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    parser.add_argument("--dataset", type=str, default="videoTrain")
    parser.add_argument("--save_path", type=str, default="ckpt/videoTrain")
    parser.add_argument("--output_dir", type=str, default="results/videoTest")
    args = parser.parse_args()
    
    setup_seed(111)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 初始化模型与加载权重
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

    text_file = glob(args.save_path + "/text_adapter.pth")
    adapt_text = False
    if len(text_file) > 0:
        checkpoint = torch.load(text_file[0], map_location=device)
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        adapt_text = True

    all_files = sorted(glob(args.save_path + "/image_adapter*.pth"))
    if len(all_files) > 0:
        checkpoint_path = all_files[-1]
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
    else:
        print("Error: No image adapter checkpoint found!")
        return

    with torch.no_grad():
        if adapt_text: text_embeddings = get_adapted_text_embedding(model, args.dataset, device)
        else: text_embeddings = get_adapted_text_embedding(clip_model, args.dataset, device)
    class_text_embeddings = text_embeddings["road"]

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    min_size = 1036

    # 2. 定义测试视频任务
    video_tasks = [
        {"path": "crack-data/5/5crackwaican.mp4", "start_s": 150, "end_s": 170, "name": "5crackwaican"},
        {"path": "crack-data/3/3crack.mp4", "start_s": 188, "end_s": 211, "name": "3crack"}
    ]
    
    methods = ["Baseline", "TemporalQueue", "Homography"]

    # 3. 嵌套循环测试
    time_records = []

    for task in video_tasks:
        video_path = task["path"]
        v_name = task["name"]
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}. Skipping...")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(task["start_s"] * fps)
        end_frame = int(task["end_s"] * fps)
        total_frames = end_frame - start_frame
        
        print(f"\n{'='*50}")
        print(f"Processing Video: {v_name} ({task['start_s']}s - {task['end_s']}s)")
        print(f"Resolution: {orig_W}x{orig_H}, FPS: {fps}, Total Frames: {total_frames}")
        print(f"{'='*50}")

        for method in methods:
            print(f"\n--- Running Method: {method} ---")
            
            out_path = os.path.join(args.output_dir, f"{v_name}_{method}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (orig_W * 2, orig_H))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 状态缓存初始化
            temporal_queue = deque(maxlen=5) if method == "TemporalQueue" else None
            prev_gray = None
            prev_prob_map = None
            
            start_time = time.time()
            
            with torch.no_grad():
                for i in tqdm(range(total_frames), desc=f"{v_name} - {method}"):
                    ret, frame = cap.read()
                    if not ret: break
                        
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    
                    new_W, new_H = max(orig_W, min_size), max(orig_H, min_size)
                    if new_W != orig_W or new_H != orig_H:
                        pil_img = pil_img.resize((new_W, new_H), Image.BICUBIC)
                        
                    image_tensor = base_transform(pil_img).unsqueeze(0).to(device)

                    # 基础推理 (空间)
                    prob_map_local = get_sliding_window_predictions(model, class_text_embeddings, image_tensor, img_size=args.img_size, stride=args.img_size // 2)
                    if prob_map_local.shape != (orig_H, orig_W): prob_map_local = cv2.resize(prob_map_local, (orig_W, orig_H))
                    
                    prob_map_global = get_global_prediction(model, class_text_embeddings, image_tensor, img_size=args.img_size)
                    if prob_map_global.shape != (orig_H, orig_W): prob_map_global = cv2.resize(prob_map_global, (orig_W, orig_H))
                    
                    curr_prob_map = np.sqrt(prob_map_local * prob_map_global)

                    # ---------------------------------------------------------
                    # 算法分支逻辑
                    # ---------------------------------------------------------
                    if method == "Baseline":
                        prob_map_final = curr_prob_map
                        
                    elif method == "TemporalQueue":
                        temporal_queue.append(curr_prob_map)
                        q_len = len(temporal_queue)
                        if q_len == 1:
                            prob_map_final = curr_prob_map
                        else:
                            weights = np.linspace(0.5, 1.0, q_len)
                            weights = weights / np.sum(weights)
                            vol = np.stack(list(temporal_queue), axis=0)
                            prob_map_final = np.tensordot(weights, vol, axes=1)
                            
                    elif method == "Homography":
                        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if prev_gray is None or prev_prob_map is None:
                            prob_map_final = curr_prob_map
                        else:
                            # 1. 获取对齐后的上一帧概率图
                            aligned_prev = align_prob_map_homography(curr_gray, prev_gray, prev_prob_map)
                            # 2. 软融合: 70% 信任当前帧，30% 信任对齐后的历史帧
                            prob_map_final = 0.7 * curr_prob_map + 0.3 * aligned_prev
                            
                        # 更新缓存
                        prev_gray = curr_gray.copy()
                        prev_prob_map = prob_map_final.copy()
                    
                    # ---------------------------------------------------------
                    # 掩膜渲染与保存
                    # ---------------------------------------------------------
                    binary_mask = (prob_map_final > 0.5).astype(np.uint8) * 255
                    color_mask = np.zeros_like(frame)
                    color_mask[:, :, 2] = binary_mask # 红色通道
                    
                    overlaid_frame = cv2.addWeighted(frame, 1.0, color_mask, 0.5, 0)
                    combined_frame = np.hstack((frame, overlaid_frame))
                    out.write(combined_frame)
                    
            end_time = time.time()
            elapsed = end_time - start_time
            out.release()
            
            time_records.append(f"[{v_name}] Method: {method:15s} | Time: {elapsed:.2f} s | Avg FPS: {total_frames/elapsed:.2f}")

        cap.release()

    # 输出性能报告
    print("\n" + "="*50)
    print("Execution Time Report")
    print("="*50)
    for record in time_records:
        print(record)
    print("="*50)

if __name__ == "__main__":
    main()