import os
import argparse

def should_keep(filename: str) -> bool:
    """
    Return True if the file should be kept.
    Only keep files ending with:
        _500000.zip
        _500000.png
    """
    return filename.endswith("_500000.zip") or filename.endswith("_500000.png")


def clean_directory(root_dir: str, force: bool = False):
    deleted_files = []
    kept_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            full_path = os.path.join(dirpath, file)

            if should_keep(file):
                kept_files.append(full_path)
            else:
                deleted_files.append(full_path)
                if force:
                    try:
                        os.remove(full_path)
                    except Exception as e:
                        print(f"Failed to delete {full_path}: {e}")

    print("\n========= SUMMARY =========")
    print(f"Root directory: {root_dir}")
    print(f"Files to delete: {len(deleted_files)}")
    print(f"Files kept: {len(kept_files)}")

    if not force:
        print("\n[DRY RUN MODE] No files were deleted.")
        print("Use --force to actually delete them.")

    print("\nExample files to delete (first 10):")
    for f in deleted_files[:10]:
        print("  ", f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete all files except those ending with _500000.zip and _500000.png"
    )
    parser.add_argument("root_dir", type=str, help="Root directory to clean")
    parser.add_argument("--force", action="store_true", help="Actually delete files")

    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        print("Invalid directory.")
    else:
        clean_directory(args.root_dir, force=args.force)