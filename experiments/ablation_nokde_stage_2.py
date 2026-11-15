#!/usr/bin/env python3
"""E8 (KDE-loss ablation), stage 2: score the stage-1 point sets with the Table 2 scorer.

Runs ablation_advance_metrics_stage_2.py on OUTPUT_DIR/<results dir>/epoch_*_npy (the scorer's
RESULTS_DIR is set here on the module; the shared script is not edited).

    python experiments/ablation_nokde_stage_2.py --results-dir sdedit_noKDE --epochs epoch_5000
    python experiments/ablation_nokde_stage_2.py --results-dir sdedit_seeded
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation_advance_metrics_stage_2 as s2  # noqa: E402
from ablation_nokde_stage_1 import OUTPUT_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--output", default=OUTPUT_DIR)
    args, rest = ap.parse_known_args()
    s2.RESULTS_DIR = args.results_dir
    sys.argv = [sys.argv[0], "--output", args.output, *rest]
    return s2.main()


if __name__ == "__main__":
    raise SystemExit(main())
