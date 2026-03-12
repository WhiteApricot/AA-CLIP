import os
import cv2
import json
import random  # 必须导入
from tqdm import tqdm


def generate_g45_train_meta_subsampled(source_root, meta_root, sample_ratio=0.3):
    """
    专门为 G45data 生成训练元数据。
    1. 不复制文件。
    2. 只处理训练集。
    3. 严格检查掩码，不存在则跳过。
    4. 【新增】随机保留指定比例的图像。
    """

    # --- 1. 路径配置 ---
    REL_IMG_DIR = "images"
    REL_MASK_DIR = "ground_truth"

    ABS_IMG_DIR = os.path.join(source_root, REL_IMG_DIR)
    ABS_MASK_DIR = os.path.join(source_root, REL_MASK_DIR)

    DATASET_NAME_TRAIN = "G45_Train"
    CLASS_NAME = "G45"

    # 输出目录
    META_OUT_DIR = os.path.join(meta_root, DATASET_NAME_TRAIN)
    os.makedirs(META_OUT_DIR, exist_ok=True)
    TRAIN_JSONL = os.path.join(META_OUT_DIR, "full-shot.jsonl")

    # --- 2. 获取图像列表并采样 ---
    if not os.path.exists(ABS_IMG_DIR):
        print(f"错误: 找不到图像目录 {ABS_IMG_DIR}")
        return

    all_images = [f for f in os.listdir(ABS_IMG_DIR) if f.endswith('.png')]
    total_found = len(all_images)
    print(f"扫描到原始图像总数: {total_found}")

    # 【核心逻辑：随机采样】
    sample_size = int(total_found * sample_ratio)
    sampled_images = random.sample(all_images, sample_size)
    print(f"已随机抽取 {sample_ratio * 100}% 的数据，剩余图像数: {len(sampled_images)}")

    # --- 3. 处理并写入 ---
    valid_count = 0
    skip_count = 0

    print("正在生成训练元数据...")
    with open(TRAIN_JSONL, 'w') as f:
        for filename in tqdm(sampled_images):

            # 构造掩码的绝对路径
            abs_mask_path = os.path.join(ABS_MASK_DIR, filename)

            # 掩码检查
            if not os.path.exists(abs_mask_path):
                skip_count += 1
                continue

            mask = cv2.imread(abs_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                skip_count += 1
                continue

            # 判断 Label
            label = 1.0 if cv2.countNonZero(mask) > 0 else 0.0

            # 写入记录
            record = {
                "image_path": f"{REL_IMG_DIR}/{filename}",
                "label": label,
                "class_name": CLASS_NAME,
                "mask_path": f"{REL_MASK_DIR}/{filename}"
            }
            f.write(json.dumps(record) + '\n')
            valid_count += 1

    print(f"\n处理完成！")
    print(f"最终写入 jsonl 的有效图像: {valid_count}")
    print(f"因缺失掩码被跳过: {skip_count}")
    print(f"元数据文件: {TRAIN_JSONL}")


if __name__ == "__main__":
    SOURCE_ROOT = "G45data/train"
    AA_META_ROOT = "./dataset/metadata"

    # 在这里设置保留比例，0.3 代表 30%
    generate_g45_train_meta_subsampled(SOURCE_ROOT, AA_META_ROOT, sample_ratio=0.1)