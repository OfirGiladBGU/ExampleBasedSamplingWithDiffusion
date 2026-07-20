#!/usr/bin/env python
"""overfit1 -- Stage A of the GT-free transfer probe.

HYBRID overfit (velocity + geometry) on the *taksim* image. This stage IS teacher-supervised
(velocity loss regresses on the GT stipple offsets) -- that is intentional: it builds the
blue-noise "spacing manifold" whose transfer overfit2 (Stage B) then tests.

It produces the checkpoint that overfit2 warm-starts from:
    <output>/<image-stem>/checkpoints/latest_controlnet.pt

Run from the repo root:  python control_gt_free/overfit1.py
Any extra CLI args are forwarded to test_overfit.py (e.g. --steps 5000).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEST_OVERFIT = os.path.join(HERE, "test_overfit.py")

# Stage-A image (the taksim dataset the overfit already used).
DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/data_taksim"
# Empirical density-warped PCF target (aggregate GBN signature). Build once with:
#   python -m control_gt_free.eval.precompute_gbn_bar --source <src> --offsets <offs> --out control_gt_free/gbn_bar.pt
GBN_BAR = os.path.join(HERE, "gbn_bar.pt")
OUTPUT_DIR = os.path.join("control_gt_free", "overfit_outputs_stageA_hybrid")

cmd = [
    sys.executable, TEST_OVERFIT,
    "--data-root", DATA_ROOT,
    "--output_dir", OUTPUT_DIR,
    "--sample-index", "0",
    "--steps", "500",
    # HYBRID: velocity anchor (spacing prior) + geometry refinement on the one-step x0 decode.
    "--gt-free-mode", "hybrid",
    "--geo-weight", "0.1", "--geo-t-max", "0.4", "--geo-warmup", "200",
    # geometry objective: PCF-dominant (the repulsion term), light capacity, no CVT.
    "--w-pcf", "1.0", "--w-cap", "0.5", "--w-cvt", "0.0",
    "--fm-coupling", "smartinit",
]
if os.path.exists(os.path.join(ROOT, GBN_BAR)) or os.path.exists(GBN_BAR):
    cmd += ["--gbn-bar", GBN_BAR]
else:
    print(f"[overfit1] WARNING: no GBN bar at {GBN_BAR} -> PCF term DISABLED (geometry = capacity only). "
          "The velocity anchor still gives spacing so Stage A converges, but build the bar for the full "
          "objective:  python -m control_gt_free.eval.precompute_gbn_bar ...")

cmd += sys.argv[1:]  # forward any ad-hoc overrides
print("[overfit1] Stage A (hybrid, taksim):\n  " + " ".join(cmd))
rc = subprocess.call(cmd, cwd=ROOT)
if rc == 0:
    # Build the PCF target (GBN bar) from THIS Stage-A output so overfit2 has everything it
    # needs (checkpoint + bar) with no manual step -- uses gt_offsets.npy + source.png.
    bar_out = os.path.join(HERE, "gbn_bar.pt")
    stage_a_abs = os.path.join(ROOT, OUTPUT_DIR)
    print("[overfit1] Stage A done -> building gbn_bar.pt from its output for overfit2 ...")
    brc = subprocess.call([sys.executable, "-m", "control_gt_free.eval.precompute_gbn_bar",
                           "--from-overfit-dir", stage_a_abs, "--out", bar_out], cwd=ROOT)
    if brc == 0 and os.path.exists(bar_out):
        print("[overfit1] gbn_bar.pt ready. Next:  python control_gt_free/overfit2.py")
    else:
        print("[overfit1] WARNING: bar build failed; overfit2 will retry building it.")
raise SystemExit(rc)
