import argparse
import json
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

# Default folders

# Icons-50 dataset
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_WVS/source"
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_BNOT/source"
SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN/source"
TARGET_DIRS = {
    "target_WVS_1024": "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_WVS/target",
    "target_BNOT_1024": "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_BNOT/target",
    "target_GBN_1024": "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN/target",
}
OUTPUT_DIR = "experiments/outputs/z_validation_data/Icons-50_1024"


# CelebA dataset
# # SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/CelebA-5K_1024_WVS/source"
# # SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/CelebA-5K_1024_BNOT/source"
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/CelebA-5K_1024_GBN/source"
# TARGET_DIRS = {
#     "target_WVS_1024": "/groups/asharf_group/ofirgila/ControlNet/training/CelebA-5K_1024_WVS/target",
#     "target_BNOT_1024": "/groups/asharf_group/ofirgila/ControlNet/training/CelebA-5K_1024_BNOT/target",
#     "target_GBN_1024": "/groups/asharf_group/ofirgila/ControlNet/training/CelebA-5K_1024_GBN/target",
# }
# OUTPUT_DIR = "experiments/outputs/z_validation_data/CelebA-5K_1024"


# ShapeNetRender dataset
# # SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRender_Custom-3K_1600_WVS/source"
# # SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRender_Custom-3K_1600_BNOT/source"
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRender_Custom-3K_1600_GBN/source"
# TARGET_DIRS = {
#     "target_WVS_1600": "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRender_Custom-3K_1600_WVS/target",
#     "target_BNOT_1600": "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRender_Custom-3K_1600_BNOT/target",
#     "target_GBN_1600": "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRender_Custom-3K_1600_GBN/target",
# }
# OUTPUT_DIR = "experiments/outputs/z_validation_data/ShapeNetRender_Custom-3K_1600"


DEVICE = "cuda"
SPLIT_SEED = 42
VAL_SPLIT = 0.1


def parse_args():
    p = argparse.ArgumentParser(description="Export ablation predictions to .npy per epoch")
    p.add_argument("--output", default=OUTPUT_DIR, help="Base output folder for exports")
    p.add_argument("--source", default=SOURCE_DIR, help="Source images folder (validation pool)")
    p.add_argument("--targets", default=str(TARGET_DIRS), help="Target images folder (ground truth)")
    p.add_argument("--val-split", type=float, default=VAL_SPLIT, help="Fraction for validation split")
    p.add_argument("--seed", type=int, default=SPLIT_SEED, help="Deterministic seed for split")
    p.add_argument("--dry-run", action="store_true", help="Only show what would be done")
    return p.parse_args()


def list_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    p = Path(folder)
    return [str(f) for f in sorted(p.rglob("*")) if f.suffix.lower() in exts]


def select_validation_images(all_images, val_frac, seed):
    imgs = sorted(all_images)
    n_total = len(imgs)
    val_len = int(n_total * float(val_frac))
    val_len = min(max(val_len, 0), max(n_total - 1, 0))
    train_len = n_total - val_len

    all_indices = torch.randperm(
        n_total,
        generator=torch.Generator().manual_seed(int(seed)),
    ).tolist()
    val_indices = all_indices[train_len:]
    return [imgs[i] for i in val_indices]


def backup_validation_images(val_images, out_base, source_dir, target_dirs):
    out_base_p = Path(out_base)
    val_data_dir = out_base_p
    source_backup_dir = val_data_dir / "source"
    target_backup_dirs = [val_data_dir / name for name in target_dirs.keys()]
    manifest_path = out_base_p / "validation_manifest.json"

    source_backup_dir.mkdir(parents=True, exist_ok=True)
    for target_backup_dir in target_backup_dirs:
        target_backup_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    
    for img in tqdm(val_images, desc="Backing up validation images"):
        src = Path(img)
        # Backup source image
        src_dst = source_backup_dir / src.name
        if not src_dst.exists():
            shutil.copy2(src, src_dst)
        
        for target_name, target_dir in target_dirs.items():
            target_backup_dir = val_data_dir / target_name

            tgt_image = Path(target_dir) / src.relative_to(source_dir)
            
            # Backup corresponding target image
            tgt_dst_image = target_backup_dir / src.name
            shutil.copy2(tgt_image, tgt_dst_image)

            # Backup corresponding target npy file
            tgt_dst_npy = target_backup_dir / (src.stem + ".npy")
            npy_file = tgt_image.with_suffix(".npy")
            shutil.copy2(npy_file, tgt_dst_npy)
        
        manifest.append(src.name)
    
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return source_backup_dir

def main():
    args = parse_args()
    args.targets = eval(args.targets)  # Convert string representation of dict to actual dict
    out_base = Path(args.output)
    out_base.mkdir(parents=True, exist_ok=True)

    manifest_path = out_base / "validation_manifest.json"
    val_data_dir = out_base / "validation_data"
    source_backup_dir = val_data_dir / "source"
    
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        val_images = [str(source_backup_dir / name) for name in manifest]
        missing = [p for p in val_images if not Path(p).exists()]
        if missing:
            print("Validation manifest exists but some backed-up images are missing.")
            print(f"Missing count: {len(missing)}")
            return 2
        print(f"Loaded {len(val_images)} validation images from existing manifest")
    else:
        all_images = list_images(args.source)
        if len(all_images) == 0:
            print(f"No source images found in {args.source}")
            return 2

        manifest_images = select_validation_images(all_images, args.val_split, args.seed)
        source_backup_dir = backup_validation_images(manifest_images, out_base, args.source, args.targets)
        print(f"Backed up validation data to {val_data_dir}")


if __name__ == "__main__":
    main()
