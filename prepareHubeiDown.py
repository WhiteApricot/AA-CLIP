import os
import shutil
import random
import json
import cv2
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 源数据路径
SOURCE_CROPS_DIR = "hubei/crops"
SOURCE_ALL_IMGS_DIR = "hubei/all_Imgs"
SOURCE_CRACK_TEST_DIR = "hubei/CrackTest"  # [新增] 指定新的测试集源文件夹

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
ENHANCE_AMOUNT = 2.0
ENHANCE_SIGMA = 2.0

# 对比度增强参数
CONTRAST_ALPHA = 2.0
CONTRAST_BETA = 128 * (1 - CONTRAST_ALPHA)


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
    读取图像，应用高频增强 (Unsharp Masking)，大幅增加对比度，然后保存
    """
    try:
        # 1. 读取图像
        img = cv2.imread(src_path)
        if img is None:
            print(f"Warning: Failed to read {src_path}")
            return False

        # # 2. 高频增强处理 (锐化)
        # blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=ENHANCE_SIGMA)
        # sharpened = cv2.addWeighted(img, 1.0 + ENHANCE_AMOUNT, blurred, -ENHANCE_AMOUNT, 0)
        #
        # # 3. 大幅增加对比度
        # high_contrast_img = cv2.convertScaleAbs(sharpened, alpha=CONTRAST_ALPHA, beta=CONTRAST_BETA)

        # 4. 保存图像
        high_contrast_img = img
        cv2.imwrite(dst_path, high_contrast_img)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def create_jsonl(data_list, output_path, is_train=True):
    """生成 JSONL 元数据文件"""
    with open(output_path, 'w') as f:
        for filename in data_list:
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
    print(f"\nProcessing Training Data (Crops) with High-Frequency Enhancement & High Contrast...")
    print(f"Enhancement Params: Amount={ENHANCE_AMOUNT}, Sigma={ENHANCE_SIGMA}")
    print(f"Contrast Params: Alpha={CONTRAST_ALPHA}, Beta={CONTRAST_BETA}")

    crop_files = get_image_files(SOURCE_CROPS_DIR)
    valid_train_files = []
    # exclude_filenames = set() # [修改] 原逻辑需要，现暂时注释掉不需要的部分

    for f in tqdm(crop_files, desc="Enhancing Train Imgs"):
        src = os.path.join(SOURCE_CROPS_DIR, f)
        dst = os.path.join(TRAIN_DIR, f)

        if enhance_and_save(src, dst):
            valid_train_files.append(f)

            # [修改] 原逻辑：用于记录原图名以防止测试集数据泄露
            # original_name = f.replace("_crop", "")
            # exclude_filenames.add(original_name)

    # 生成训练集元数据
    create_jsonl(valid_train_files, os.path.join(TRAIN_META_DIR, "full-shot.jsonl"))
    shutil.copyfile(os.path.join(TRAIN_META_DIR, "full-shot.jsonl"),
                    os.path.join(TRAIN_META_DIR, "32-shot.jsonl"))

    # ==========================================
    # 2. 处理测试集 (CrackTest) - 全量增强并保存
    # ==========================================
    print("\nProcessing Test Data (From hubei/CrackTest)...")

    # ------------------ [原逻辑开始：已注释] ------------------
    # print("\nProcessing Test Data (Random Selection with Enhancement)...")
    # all_img_files = get_image_files(SOURCE_ALL_IMGS_DIR)
    #
    # # 筛选候选图片（排除已经在训练集中的原图）
    # candidates = [f for f in all_img_files if f not in exclude_filenames]
    #
    # print(f"Total images: {len(all_img_files)}, Excluded: {len(exclude_filenames)}, Candidates: {len(candidates)}")
    #
    # # 随机选择 300 张 (如果不足则全选)
    # if len(candidates) < 300:
    #     print(f"Warning: Only {len(candidates)} images available for test, selecting all.")
    #     selected_test_files = candidates
    # else:
    #     selected_test_files = random.sample(candidates, 300)
    # ------------------ [原逻辑结束] ------------------

    # ------------------ [新逻辑开始] ------------------
    # 直接获取 CrackTest 目录下的所有图片
    selected_test_files = get_image_files(SOURCE_CRACK_TEST_DIR)
    print(f"Found {len(selected_test_files)} images in {SOURCE_CRACK_TEST_DIR} for testing.")
    # ------------------ [新逻辑结束] ------------------

    valid_test_files = []
    for f in tqdm(selected_test_files, desc="Enhancing Test Imgs"):
        # [修改] 源路径改为 CrackTest
        # src = os.path.join(SOURCE_ALL_IMGS_DIR, f) # 原路径
        src = os.path.join(SOURCE_CRACK_TEST_DIR, f)  # 新路径

        dst = os.path.join(TEST_DIR, f)

        # 执行增强并保存 (保持相同的处理逻辑)
        if enhance_and_save(src, dst):
            valid_test_files.append(f)

    # 生成测试集元数据
    create_jsonl(valid_test_files, os.path.join(TEST_META_DIR, "full-shot.jsonl"), is_train=False)

    print("\nDone! Data preparation with enhancement and high contrast complete.")
    print(f"Training images saved to: {TRAIN_DIR}")
    print(f"Testing images saved to: {TEST_DIR}")


if __name__ == "__main__":
    main()