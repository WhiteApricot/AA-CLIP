import os
import shutil
import random
import json
import cv2
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 源数据路径 (请确保这些路径下有图片)
SOURCE_CROPS_DIR = "hubei/crops"
SOURCE_ALL_IMGS_DIR = "hubei/all_Imgs"

# 目标数据目录
DATA_ROOT = "data"
TRAIN_DIR = os.path.join(DATA_ROOT, "HubeiDown_Train")
TEST_DIR = os.path.join(DATA_ROOT, "HubeiDown_Test")

# 目标元数据目录
META_ROOT = "dataset/metadata"
TRAIN_META_DIR = os.path.join(META_ROOT, "HubeiDown_Train")
TEST_META_DIR = os.path.join(META_ROOT, "HubeiDown_Test")

# 类别名称
CLASS_NAME = "road"

# 图像增强参数
# amount: 锐化强度 (1.0 ~ 2.0 通常比较合适，过高会引入噪点)
# sigma: 高斯模糊半径，控制增强的频率范围
ENHANCE_AMOUNT = 2.0
ENHANCE_SIGMA = 2.0


# ===========================================

def setup_directories():
    """创建必要的目录"""
    for p in [TRAIN_DIR, TEST_DIR, TRAIN_META_DIR, TEST_META_DIR]:
        if not os.path.exists(p):
            os.makedirs(p)
            print(f"Created directory: {p}")


def get_image_files(directory):
    """获取目录下所有图片文件"""
    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        return []
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    return [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in extensions]


def enhance_and_save(src_path, dst_path):
    """
    读取图像，应用高频增强 (Unsharp Masking)，然后保存
    公式: Sharpened = Original + (Original - Blurred) * Amount
    """
    try:
        # 1. 读取图像
        img = cv2.imread(src_path)
        if img is None:
            print(f"Warning: Failed to read {src_path}")
            return False

        # 2. 高频增强处理
        # 高斯模糊，获取低频分量
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=ENHANCE_SIGMA)

        # 计算锐化后的图像： 原图 * (1 + amount) + 模糊图 * (-amount)
        # 例如 amount=1.5 -> 原图 * 2.5 - 模糊图 * 1.5
        sharpened = cv2.addWeighted(img, 1.0 + ENHANCE_AMOUNT, blurred, -ENHANCE_AMOUNT, 0)

        # 3. 保存图像
        cv2.imwrite(dst_path, sharpened)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def create_jsonl(data_list, output_path, is_train=True):
    """生成 JSONL 元数据文件"""
    with open(output_path, 'w') as f:
        for filename in data_list:
            # 构造 JSONL 行
            # 注意：此处假设没有 Ground Truth Mask，label 设为 0
            entry = {
                "image_path": filename,
                "mask_path": None,
                "label": 0,
                "class_name": CLASS_NAME
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Generated metadata: {output_path} with {len(data_list)} samples.")


def main():
    setup_directories()

    # ==========================================
    # 1. 处理训练集 (Crops) - 增强并保存
    # ==========================================
    print(f"\nProcessing Training Data (Crops) with High-Frequency Enhancement...")
    print(f"Enhancement Params: Amount={ENHANCE_AMOUNT}, Sigma={ENHANCE_SIGMA}")

    crop_files = get_image_files(SOURCE_CROPS_DIR)
    valid_train_files = []
    exclude_filenames = set()

    for f in tqdm(crop_files, desc="Enhancing Train Imgs"):
        src = os.path.join(SOURCE_CROPS_DIR, f)
        dst = os.path.join(TRAIN_DIR, f)

        # 执行增强并保存
        if enhance_and_save(src, dst):
            valid_train_files.append(f)

            # 解析原图文件名以供测试集排除使用
            # 假设文件名格式为 xxx_crop.jpg -> 原图为 xxx.jpg
            original_name = f.replace("_crop", "")
            exclude_filenames.add(original_name)

    # 生成训练集元数据
    create_jsonl(valid_train_files, os.path.join(TRAIN_META_DIR, "full-shot.jsonl"))
    # 复制一份作为 few-shot 兼容
    shutil.copyfile(os.path.join(TRAIN_META_DIR, "full-shot.jsonl"),
                    os.path.join(TRAIN_META_DIR, "32-shot.jsonl"))

    # ==========================================
    # 2. 处理测试集 (All Imgs) - 筛选、增强并保存
    # ==========================================
    print("\nProcessing Test Data (Random Selection with Enhancement)...")
    all_img_files = get_image_files(SOURCE_ALL_IMGS_DIR)

    # 筛选候选图片（排除已经在训练集中的原图）
    candidates = [f for f in all_img_files if f not in exclude_filenames]

    print(f"Total images: {len(all_img_files)}, Excluded: {len(exclude_filenames)}, Candidates: {len(candidates)}")

    # 随机选择 300 张 (如果不足则全选)
    if len(candidates) < 300:
        print(f"Warning: Only {len(candidates)} images available for test, selecting all.")
        selected_test_files = candidates
    else:
        selected_test_files = random.sample(candidates, 300)

    valid_test_files = []
    for f in tqdm(selected_test_files, desc="Enhancing Test Imgs"):
        src = os.path.join(SOURCE_ALL_IMGS_DIR, f)
        dst = os.path.join(TEST_DIR, f)

        # 执行增强并保存
        if enhance_and_save(src, dst):
            valid_test_files.append(f)

    # 生成测试集元数据
    create_jsonl(valid_test_files, os.path.join(TEST_META_DIR, "full-shot.jsonl"), is_train=False)

    print("\nDone! Data preparation with enhancement complete.")
    print(f"Training images saved to: {TRAIN_DIR}")
    print(f"Testing images saved to: {TEST_DIR}")


if __name__ == "__main__":
    main()