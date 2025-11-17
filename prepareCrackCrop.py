import os
import cv2
import numpy as np
import shutil
import random
import json
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1. 复用 prepareCrackCrop.py 中的核心函数
# ---------------------------------------------------------------------------
def find_largest_pure_rectangle(mask_path):
    """
    (来自 prepareCrackCrop.py)
    加载掩码 (0=正常, 255=裂缝)，并使用 "最大矩形直方图" 算法
    找到全局最大的、100%纯净（全0）的矩形。

    该区域面积必须不小于原图的40%。
    """

    # 1. 以灰度模式加载掩码
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"警告: 无法加载掩码 {mask_path}。跳过此文件。")
        return None

    # 2. 获取原图总面积和40%阈值
    height, width = mask.shape
    total_area = height * width
    min_required_area = total_area * 0.4

    # 3. 反转掩码，使 0=裂缝, 1=正常
    # 我们寻找 "0" (正常) 区域，所以将 0 变为 1
    normal_mask = (mask == 0).astype(np.uint8)

    # 4. 基于 "最大矩形直方图" 的动态规划

    # hist (histogram) 存储从 (y, x) 向上连续为 1 (正常) 的像素数
    hist = np.zeros_like(normal_mask)
    max_area = 0
    best_rect = None  # (x, y, w, h)

    # --- 4a. 构建直方图 (动态规划) ---
    for y in range(height):
        for x in range(width):
            if normal_mask[y, x] == 1:
                hist[y, x] = hist[y - 1, x] + 1 if y > 0 else 1

    # --- 4b. 计算每一行的最大矩形 ---
    # (标准算法: https://www.geeksforgeeks.org/largest-rectangle-under-histogram/)
    for y in range(height):
        # 维护一个 (高度, 索引) 的递增栈
        stack = []  # (index, height)

        for x, h in enumerate(hist[y]):
            start_x = x
            while stack and stack[-1][1] > h:
                prev_x, prev_h = stack.pop()
                current_w = x - prev_x
                current_area = prev_h * current_w

                if current_area > max_area:
                    max_area = current_area
                    # 矩形坐标: (x1, y1, x2, y2)
                    # x1 = prev_x
                    # y1 = y - prev_h + 1
                    # x2 = x - 1
                    # y2 = y
                    best_rect = (prev_x, y - prev_h + 1, current_w, prev_h)

            if not stack or stack[-1][1] < h:
                stack.append((start_x, h))

        # --- 4c. 处理栈中剩余元素 ---
        while stack:
            prev_x, prev_h = stack.pop()
            current_w = width - prev_x
            current_area = prev_h * current_w

            if current_area > max_area:
                max_area = current_area
                best_rect = (prev_x, y - prev_h + 1, current_w, prev_h)

    # 5. 验证面积是否达标
    if best_rect and max_area >= min_required_area:
        return best_rect  # (x, y, w, h)
    else:
        # print(f"未找到足够大的纯净区域 (最大: {max_area}, 需要: {min_required_area})")
        return None


# ---------------------------------------------------------------------------
# 2. AA-CLIP 数据集处理主函数
# ---------------------------------------------------------------------------
def process_dataset(source_root, aa_data_root, aa_meta_root):
    """
    处理 CRACK500FINAL 数据集并创建两个 AA-CLIP 兼容的数据集：
    1. MyRoadCrack_Train (用于训练)
    2. MyRoadCrack_Test (用于测试)
    """

    # --- 1. 定义常量和路径 ---
    CLASS_NAME = "road_crack"
    TRAIN_DS_NAME = "MyRoadCrack_Train"
    TEST_DS_NAME = "MyRoadCrack_Test"

    SOURCE_IMG_DIR = os.path.join(source_root, "JPEGImages")
    SOURCE_MASK_DIR = os.path.join(source_root, "SegmentationClass")

    # 训练集输出路径
    TRAIN_IMG_DIR = os.path.join(aa_data_root, TRAIN_DS_NAME, "images")
    TRAIN_GT_DIR = os.path.join(aa_data_root, TRAIN_DS_NAME, "ground_truth")
    TRAIN_META_DIR = os.path.join(aa_meta_root, TRAIN_DS_NAME)
    TRAIN_JSONL_PATH = os.path.join(TRAIN_META_DIR, "full-shot.jsonl")

    # 测试集输出路径
    TEST_IMG_DIR = os.path.join(aa_data_root, TEST_DS_NAME, "images")
    TEST_GT_DIR = os.path.join(aa_data_root, TEST_DS_NAME, "ground_truth")
    TEST_META_DIR = os.path.join(aa_meta_root, TEST_DS_NAME)
    TEST_JSONL_PATH = os.path.join(TEST_META_DIR, "full-shot.jsonl")

    # --- 2. 创建输出目录 ---
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_GT_DIR, exist_ok=True)
    os.makedirs(TRAIN_META_DIR, exist_ok=True)
    os.makedirs(TEST_IMG_DIR, exist_ok=True)
    os.makedirs(TEST_GT_DIR, exist_ok=True)
    os.makedirs(TEST_META_DIR, exist_ok=True)

    print("输出目录创建完毕。")

    # --- 3. 拆分文件 ---
    # 查找所有源图像文件
    try:
        all_files = [
            f for f in os.listdir(SOURCE_IMG_DIR)
            if f.endswith(('.jpg', '.png'))
        ]
        random.shuffle(all_files)
    except FileNotFoundError:
        print(f"错误：未在 {SOURCE_IMG_DIR} 找到源图像。请检查 SOURCE_ROOT 路径。")
        return

    total_count = len(all_files)
    if total_count == 0:
        print("错误：源图像目录为空。")
        return

    split_size = int(total_count * 0.1)
    if split_size == 0:
        print("错误：数据集太小，无法进行10%拆分。")
        return

    # 按 10% 拆分为4个不重叠的组
    train_abnormal_files = all_files[0: split_size]
    train_normal_src_files = all_files[split_size: 2 * split_size]
    test_abnormal_files = all_files[2 * split_size: 3 * split_size]
    test_normal_src_files = all_files[3 * split_size: 4 * split_size]

    print(f"数据集拆分完毕 (每组 {split_size} 个源文件):")
    print(f"  - 训练集 (异常): {len(train_abnormal_files)}")
    print(f"  - 训练集 (正常源): {len(train_normal_src_files)}")
    print(f"  - 测试集 (异常): {len(test_abnormal_files)}")
    print(f"  - 测试集 (正常源): {len(test_normal_src_files)}")
    print("-" * 30)

    # --- 4. 处理数据并写入 JSONL ---

    # 打开 JSONL 文件写入器
    with open(TRAIN_JSONL_PATH, 'w') as f_train_json, \
            open(TEST_JSONL_PATH, 'w') as f_test_json:

        # A. 处理 训练集-异常
        print("处理 训练集 (异常)...")
        for i, basename in enumerate(tqdm(train_abnormal_files)):
            name, ext = os.path.splitext(basename)
            src_img_path = os.path.join(SOURCE_IMG_DIR, basename)
            src_mask_path = os.path.join(SOURCE_MASK_DIR, name + ".png")  # 掩码通常是 png

            new_basename_img = f"train_abnormal_{i:04d}{ext}"
            new_basename_mask = f"train_abnormal_{i:04d}.png"

            dest_img_path = os.path.join(TRAIN_IMG_DIR, new_basename_img)
            dest_mask_path = os.path.join(TRAIN_GT_DIR, new_basename_mask)

            shutil.copy(src_img_path, dest_img_path)
            shutil.copy(src_mask_path, dest_mask_path)

            record = {
                "image_path": f"images/{new_basename_img}",
                "label": 1.0,
                "class_name": CLASS_NAME,
                "mask_path": f"ground_truth/{new_basename_mask}"
            }
            f_train_json.write(json.dumps(record) + '\n')

        # B. 处理 训练集-正常 (裁剪)
        print("处理 训练集 (正常-裁剪)...")
        normal_count = 0
        for i, basename in enumerate(tqdm(train_normal_src_files)):
            name, ext = os.path.splitext(basename)
            src_img_path = os.path.join(SOURCE_IMG_DIR, basename)
            src_mask_path = os.path.join(SOURCE_MASK_DIR, name + ".png")

            rect = find_largest_pure_rectangle(src_mask_path)
            if rect:
                x, y, w, h = rect
                img = cv2.imread(src_img_path)
                if img is None: continue

                crop = img[y:y + h, x:x + w]

                new_basename_img = f"train_normal_{normal_count:04d}{ext}"
                dest_img_path = os.path.join(TRAIN_IMG_DIR, new_basename_img)
                cv2.imwrite(dest_img_path, crop)

                record = {
                    "image_path": f"images/{new_basename_img}",
                    "label": 0.0,
                    "class_name": CLASS_NAME
                }
                f_train_json.write(json.dumps(record) + '\n')
                normal_count += 1
        print(f"成功生成 {normal_count} 张 训练集-正常 图像。")

        # C. 处理 测试集-异常
        print("处理 测试集 (异常)...")
        for i, basename in enumerate(tqdm(test_abnormal_files)):
            name, ext = os.path.splitext(basename)
            src_img_path = os.path.join(SOURCE_IMG_DIR, basename)
            src_mask_path = os.path.join(SOURCE_MASK_DIR, name + ".png")

            new_basename_img = f"test_abnormal_{i:04d}{ext}"
            new_basename_mask = f"test_abnormal_{i:04d}.png"

            dest_img_path = os.path.join(TEST_IMG_DIR, new_basename_img)
            dest_mask_path = os.path.join(TEST_GT_DIR, new_basename_mask)

            shutil.copy(src_img_path, dest_img_path)
            shutil.copy(src_mask_path, dest_mask_path)

            record = {
                "image_path": f"images/{new_basename_img}",
                "label": 1.0,
                "class_name": CLASS_NAME,
                "mask_path": f"ground_truth/{new_basename_mask}"
            }
            f_test_json.write(json.dumps(record) + '\n')

        # D. 处理 测试集-正常 (裁剪)
        print("处理 测试集 (正常-裁剪)...")
        normal_count = 0
        for i, basename in enumerate(tqdm(test_normal_src_files)):
            name, ext = os.path.splitext(basename)
            src_img_path = os.path.join(SOURCE_IMG_DIR, basename)
            src_mask_path = os.path.join(SOURCE_MASK_DIR, name + ".png")

            rect = find_largest_pure_rectangle(src_mask_path)
            if rect:
                x, y, w, h = rect
                img = cv2.imread(src_img_path)
                if img is None: continue

                crop = img[y:y + h, x:x + w]

                new_basename_img = f"test_normal_{normal_count:04d}{ext}"
                dest_img_path = os.path.join(TEST_IMG_DIR, new_basename_img)
                cv2.imwrite(dest_img_path, crop)

                record = {
                    "image_path": f"images/{new_basename_img}",
                    "label": 0.0,
                    "class_name": CLASS_NAME
                }
                f_test_json.write(json.dumps(record) + '\n')
                normal_count += 1
        print(f"成功生成 {normal_count} 张 测试集-正常 图像。")

    print("\n预处理完成！")
    print(f"训练集元数据: {TRAIN_JSONL_PATH}")
    print(f"测试集元数据: {TEST_JSONL_PATH}")


if __name__ == "__main__":
    # 假设 'CRACK500FINAL' 在当前目录下
    SOURCE_ROOT = "CRACK500FINAL"

    # AA-CLIP 期望的数据和元数据目录
    AA_DATA_ROOT = "./data"
    AA_META_ROOT = "./dataset/metadata"

    process_dataset(SOURCE_ROOT, AA_DATA_ROOT, AA_META_ROOT)