import os
import json
import shutil
from tqdm import tqdm


def process_normal_test_set(source_img_dir, aa_data_root, aa_meta_root):
    """
    处理全正常的测试集数据
    """
    CLASS_NAME = "hubei_down"
    TEST_DS_NAME = "HubeiDown_Train"  # 新的测试集名称

    # 1. 定义并创建输出路径
    DEST_IMG_DIR = os.path.join(aa_data_root, TEST_DS_NAME, "images")
    DEST_META_DIR = os.path.join(aa_meta_root, TEST_DS_NAME)
    os.makedirs(DEST_IMG_DIR, exist_ok=True)
    os.makedirs(DEST_META_DIR, exist_ok=True)

    JSONL_PATH = os.path.join(DEST_META_DIR, "full-shot.jsonl")

    # 2. 获取所有图像文件
    all_files = [f for f in os.listdir(source_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"找到 {len(all_files)} 张正常测试图像。")

    # 3. 开始处理
    with open(JSONL_PATH, 'w') as f_json:
        for i, basename in enumerate(tqdm(all_files)):
            ext = os.path.splitext(basename)[1]
            new_name = f"test_normal_{i:04d}{ext}"

            # 复制图像到目标文件夹
            src_path = os.path.join(source_img_dir, basename)
            dest_path = os.path.join(DEST_IMG_DIR, new_name)
            shutil.copy(src_path, dest_path)

            # 写入 JSONL 记录
            # 注意：全正常数据 label 为 0.0，且不需要 mask_path
            record = {
                "image_path": f"images/{new_name}",
                "label": 0.0,
                "class_name": CLASS_NAME
            }
            f_json.write(json.dumps(record) + '\n')

    print(f"\n处理完成！")
    print(f"测试集已保存至: {os.path.join(aa_data_root, TEST_DS_NAME)}")
    print(f"元数据文件: {JSONL_PATH}")


if __name__ == "__main__":
    # 填入你存放全是正常图片的文件夹路径
    MY_NORMAL_IMAGES = "hubei/normal"

    # AA-CLIP 的标准目录
    AA_DATA_ROOT = "./data"
    AA_META_ROOT = "./dataset/metadata"

    process_normal_test_set(MY_NORMAL_IMAGES, AA_DATA_ROOT, AA_META_ROOT)