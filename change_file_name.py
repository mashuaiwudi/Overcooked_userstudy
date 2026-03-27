import os

# ====== 你的目标路径 ======
BASE_DIR = "policy_pool_thinpath"   # ← 改成你的路径


def rename_folders(base_dir, dry_run=True):
    """
    dry_run=True 只打印，不真正修改
    dry_run=False 才会真的重命名
    """

    for name in os.listdir(base_dir):
        old_path = os.path.join(base_dir, name)

        # 只处理文件夹
        if not os.path.isdir(old_path):
            continue

        # 只处理包含 helping_ 的
        if "helping_" not in name:
            continue

        # 避免重复处理
        if "helping0_" in name or "helping1_" in name:
            print(f"[Skip already processed] {name}")
            continue

        if "helping_True" in name:
            new_name = name.replace(
                "helping_True",
                "helping0_True_helping1_True"
            )
        elif "helping_False" in name:
            new_name = name.replace(
                "helping_False",
                "helping0_False_helping1_False"
            )
        else:
            print(f"[Skip unknown format] {name}")
            continue

        new_path = os.path.join(base_dir, new_name)

        print(f"{name}  -->  {new_name}")

        if not dry_run:
            os.rename(old_path, new_path)


if __name__ == "__main__":
    # ====== 第一步：先预览 ======
    # rename_folders(BASE_DIR, dry_run=True)

    # ====== 确认无误后再执行 ======
    rename_folders(BASE_DIR, dry_run=False)