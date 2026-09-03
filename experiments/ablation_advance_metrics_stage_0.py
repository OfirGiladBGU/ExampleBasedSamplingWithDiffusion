"""ablation_advance_metrics_stage_0.py -- Part 0: stage the shared ablation resources.

Copies what the ablation needs out of the shared validation folder (built once by
experiments/z_copy_validatation_data.py) into a single RESOURCES folder:

    z_validation_data/<dataset>/source/                   -> OUTPUT_DIR/resources/source/
    z_validation_data/<dataset>/target_GBN_1024/          -> OUTPUT_DIR/resources/target_GBN_1024/
    z_validation_data/<dataset>/validation_manifest.json  -> OUTPUT_DIR/resources/validation_manifest.json

and draws the per-image MinSNR timesteps once, here:

    OUTPUT_DIR/resources/minsnr_timesteps.json

Drawing them in stage 0 (rather than lazily in stage 2) makes the M6 losses reproducible:
every variant, every epoch and every re-run scores against the same timesteps.

Only the ablation's own GT target is copied (TARGET_NAME) -- the other oracles' target
folders are not used by the ablation. The target is needed by stage 2 for the M6 losses;
every other metric uses the source condition.

Layout: everything shared lives under OUTPUT_DIR/resources/ (source, target, the
manifest, the minsnr_timesteps.json stage 2 writes, and the plots stage 4 writes), so
every OTHER folder directly under OUTPUT_DIR is an ablation-variant folder.

    python experiments/ablation_advance_metrics_stage_0.py
    python experiments/ablation_advance_metrics_stage_0.py --dry-run
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

SRC_DIR = "experiments/outputs/z_validation_data/Icons-50_1024"
# OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics"
OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics_e500_b50_1024"
RESOURCES_NAME = "resources"
TARGET_NAME = "target_GBN_1024"      # the ablation's GT oracle (used by stage 2's M6)
MANIFEST_NAME = "validation_manifest.json"
TIMESTEPS_NAME = "minsnr_timesteps.json"

# MinSNR timestep draw -- consumed by stage 2's M6 losses.
SPLIT_SEED = 42            # RNG seed for the draw
EVAL_TIMESTEPS = 1000      # must match stage 2's --timesteps
SAMPLES_PER_IMAGE = 1      # timestep samples drawn per validation image


def parse_args():
    ap = argparse.ArgumentParser(
        description="Copy source/, the GT target and the manifest into OUTPUT_DIR/resources/.")
    ap.add_argument("--src", default=SRC_DIR,
                    help="Shared validation folder (built by z_copy_validatation_data.py).")
    ap.add_argument("--output", default=OUTPUT_DIR, help="Ablation output folder.")
    ap.add_argument("--target-name", default=TARGET_NAME,
                    help="Which target_<METHOD>_<points> folder is the ablation's GT.")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED,
                    help="Seed for the MinSNR timestep draw.")
    ap.add_argument("--timesteps", type=int, default=EVAL_TIMESTEPS,
                    help="Diffusion timestep count; must match stage 2's --timesteps.")
    ap.add_argument("--samples-per-image", type=int, default=SAMPLES_PER_IMAGE,
                    help="MinSNR timestep samples drawn per validation image. More samples "
                         "reduce the variance of M6 (the Component-2 term is masked to "
                         "t < 0.4 * timesteps, so single-sample images often score 0).")
    ap.add_argument("--force-timesteps", action="store_true",
                    help="Redraw minsnr_timesteps.json even if it exists. CHANGES the M6 "
                         "losses and makes them incomparable with previously scored runs.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; copy nothing.")
    return ap.parse_args()


def write_timesteps(manifest_path, out_path, args):
    """Draw the per-image MinSNR timesteps once, so every later stage is reproducible.

    Keyed by image name and drawn for EVERY name in the manifest, so any stage-2
    --num-samples sees the same values. Names are consumed in manifest order, which makes
    a larger manifest a strict superset: the first N names keep the draws they already had.

    Never silently overwritten -- M6 depends on these values, so a redraw would make new
    scores incomparable with existing ones. Pass --force-timesteps to redraw deliberately.
    """
    if out_path.exists() and not args.force_timesteps:
        existing = json.loads(out_path.read_text())
        print(f"  {TIMESTEPS_NAME}: kept existing ({len(existing)} entries) "
              f"-- pass --force-timesteps to redraw")
        return
    names = json.loads(Path(manifest_path).read_text())
    rng = np.random.RandomState(int(args.seed))
    max_t = max(1, int(args.timesteps) - 1)
    samples = max(1, int(args.samples_per_image))
    mapping = {n: [int(rng.randint(0, max_t + 1)) for _ in range(samples)] for n in names}
    out_path.write_text(json.dumps(mapping, indent=2))
    verb = "redrew" if args.force_timesteps else "wrote"
    print(f"  {TIMESTEPS_NAME}: {verb} {len(mapping)} entries "
          f"({samples} sample(s)/image, seed {args.seed}, t in [0,{max_t}])")


def main():
    args = parse_args()
    src = Path(args.src)
    res = Path(args.output) / RESOURCES_NAME

    src_source = src / "source"
    src_target = src / args.target_name
    src_manifest = src / MANIFEST_NAME
    for p, what in ((src_source, "source folder"), (src_target, "target folder"),
                    (src_manifest, "manifest")):
        if not p.exists():
            print(f"Error: {what} not found: {p}")
            return 1

    n_src = sum(1 for _ in src_source.iterdir())
    n_tgt = sum(1 for _ in src_target.iterdir())
    print(f"src: {src}")
    print(f"dst: {res}\n")
    print(f"  source/ ({n_src} files)          -> {res / 'source'}")
    print(f"  {args.target_name}/ ({n_tgt} files)  -> {res / args.target_name}")
    print(f"  {MANIFEST_NAME}             -> {res / MANIFEST_NAME}")
    print(f"  {TIMESTEPS_NAME}              -> {res / TIMESTEPS_NAME}  (drawn here)")

    if args.dry_run:
        print("\nDRY RUN: nothing copied.")
        return 0

    res.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_source, res / "source", dirs_exist_ok=True)
    shutil.copytree(src_target, res / args.target_name, dirs_exist_ok=True)
    shutil.copy2(src_manifest, res / MANIFEST_NAME)
    write_timesteps(res / MANIFEST_NAME, res / TIMESTEPS_NAME, args)

    print(f"\nCopied source/ ({n_src}) + {args.target_name}/ ({n_tgt}) + {MANIFEST_NAME} -> {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
