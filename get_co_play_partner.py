import os
import shutil
import pandas as pd

# ========= 配置区 =========
xlsx_path = "files.xlsx"
target_dir = r"co_play_partner_pool_counter"   # 改成你的目标路径
prefix = "[equilibrium][counter]agent1_"
dry_run = False   # 先设为 True 预览；确认无误后改成 False 真删
# =========================


def load_suffixes_from_excel(xlsx_file: str) -> set[str]:
    """
    读取 Excel，每一行只取第一个逗号前的部分，作为后缀集合。
    假设数据都在第一列。
    """
    df = pd.read_excel(xlsx_file, header=None)

    suffixes = set()
    for value in df.iloc[:, 0].dropna():
        text = str(value).strip()
        suffix = text.split(",", 1)[0].strip()
        if suffix:
            suffixes.add(suffix)

    return suffixes


def clean_folders(target_dir: str, prefix: str, valid_suffixes: set[str], dry_run: bool = True):
    """
    仅保留名字为 prefix + suffix 且 suffix 在 Excel 里的文件夹，其他文件夹删除。
    """
    valid_folder_names = {prefix + suffix for suffix in valid_suffixes}

    if not os.path.isdir(target_dir):
        raise NotADirectoryError(f"目标路径不存在或不是文件夹: {target_dir}")

    for name in os.listdir(target_dir):
        full_path = os.path.join(target_dir, name)

        # 只处理文件夹
        if not os.path.isdir(full_path):
            continue

        # 只对指定前缀的文件夹做保留/删除判断
        if name.startswith(prefix):
            if name in valid_folder_names:
                print(f"[保留] {name}")
            else:
                print(f"[删除] {name}")
                if not dry_run:
                    shutil.rmtree(full_path)
        else:
            print(f"[跳过-非目标前缀] {name}")


if __name__ == "__main__":
    suffixes = load_suffixes_from_excel(xlsx_path)
    print(f"从 Excel 中读取到 {len(suffixes)} 个有效后缀。")
    clean_folders(target_dir, prefix, suffixes, dry_run=dry_run)