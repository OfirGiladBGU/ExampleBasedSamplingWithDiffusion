#!/usr/bin/env python
"""overfit2 -- Stage B of the GT-free transfer probe (the actual GT-free test).

Load the Stage-A (overfit1) weights, turn velocity supervision OFF, and train the GEOMETRY
OBJECTIVE ONLY on a NEW image (monkey) that Stage A never saw.

Mechanism = ``--gt-free-mode unroll``: the model samples its OWN points by unrolling the ODE
from the monkey *smart-init* (an input-derived, density-following layout -- NOT the GT). The
loss is the composite objective (PCF-dominant) on those points. The monkey GT stipples never
enter the loss -- x_0 is used only for tensor shape -- so the training is genuinely GT-free.

Question it answers: is the blue-noise structure a LEARNED, TRANSFERABLE property of the net
(holds on a new image under the objective alone), or was it per-image memorization (collapses
once the teacher is removed on unseen input)?

Honest framing: this is "GT-free training initialized from a GT-supervised checkpoint", NOT
"pure" GT-free -- the warm-start weights are teacher-derived. See the transfer-probe spec.

Run from the repo root:  python control_gt_free/overfit2.py
Extra CLI args are forwarded to test_overfit.py.
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEST_OVERFIT = os.path.join(HERE, "test_overfit.py")

# Stage-B image: the NEW image the Stage-A model never saw.
DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/monkey"
GBN_BAR = os.path.join(HERE, "gbn_bar.pt")  # aggregate PCF signature (roughly image-independent)
STAGE_A_OUTPUT = os.path.join(ROOT, "control_gt_free", "overfit_outputs_stageA_hybrid")
OUTPUT_DIR = os.path.join("control_gt_free", "overfit_outputs_stageB_transfer")


def find_stage_a_ckpt():
    """Locate the newest latest_controlnet.pt produced by overfit1 (image stem unknown ahead of time)."""
    hits = glob.glob(os.path.join(STAGE_A_OUTPUT, "**", "checkpoints", "latest_controlnet.pt"), recursive=True)
    if not hits:
        hits = glob.glob(os.path.join(STAGE_A_OUTPUT, "**", "*.pt"), recursive=True)
    return max(hits, key=os.path.getmtime) if hits else None


init_ckpt = find_stage_a_ckpt()
if init_ckpt is None:
    sys.exit(f"[overfit2] No Stage-A checkpoint under {STAGE_A_OUTPUT}. Run overfit1.py first.")

cmd = [
    sys.executable, TEST_OVERFIT,
    "--data-root", DATA_ROOT,
    "--output_dir", OUTPUT_DIR,
    "--sample-index", "0",
    "--steps", "500",
    # GT-FREE: no velocity loss; model unrolls its own points from the monkey smart-init.
    "--gt-free-mode", "unroll",
    # objective: PCF-dominant repulsion (the crux) + light capacity for density tracking, no CVT.
    "--w-pcf", "1.0", "--w-cap", "0.3", "--w-cvt", "0.0",
    "--unroll-steps", "8", "--fm-coupling", "smartinit",
    # weights-only warm-start from Stage A (fresh optimizer/step, new image):
    "--init-ckpt", init_ckpt,
]
if not os.path.exists(GBN_BAR):
    # overfit1 normally builds this; rebuild from the Stage-A output as a fallback so the PCF
    # (repulsion) term is ON -- without it the objective is attractive-only and the test is INVALID.
    print("[overfit2] gbn_bar.pt missing -> building it from the Stage-A output ...")
    subprocess.call([sys.executable, "-m", "control_gt_free.eval.precompute_gbn_bar",
                     "--from-overfit-dir", STAGE_A_OUTPUT, "--out", GBN_BAR], cwd=ROOT)
if os.path.exists(GBN_BAR):
    cmd += ["--gbn-bar", GBN_BAR]
else:
    sys.exit("[overfit2] Could not obtain gbn_bar.pt -> PCF term would be OFF = INVALID test. "
             "Run overfit1.py first (it builds the bar).")

cmd += sys.argv[1:]
print(f"[overfit2] Stage B (GT-free unroll, monkey)\n  warm-start: {init_ckpt}\n  " + " ".join(cmd))
raise SystemExit(subprocess.call(cmd, cwd=ROOT))
