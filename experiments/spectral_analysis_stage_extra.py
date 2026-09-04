"""Stage "extra": ad-hoc cross-method comparison table for BOTH tests, printed to stdout.

Optional companion to stage 1/2. Read-only: it writes nothing. Stage 2 already emits the
same numbers as plots/spectral_summary.csv and plots/pcf_summary.csv; this exists to view
both tests side by side, and to compare target folders that are NOT in stage 2's
hardcoded METHODS list (e.g. the BNOT epsilon/iteration variants) via --methods.

Reads each <base>/<folder>_spectral/summary.json produced by spectral_analysis_stage_1.py.
Usage: spectral_analysis_stage_extra.py [--base DIR] [--methods folder:label,...]
"""
import argparse, json, os

BASE = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/spectral_analysis"
DEF = ("target_WVS_1024:WVS,target_BNOT_1024:BNOT(eps1.0),"
       "target_BNOT_1024_eps0p01:BNOT(eps0.01),target_GBN_1024:GBN,"
       "target_CN-WVS_1024:Ours-WVS,target_CN-GBN_1024:Ours-GBN")

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=BASE)
ap.add_argument("--methods", default=DEF)
a = ap.parse_args()

pairs = [p.split(":") for p in a.methods.split(",")]
data = {}
for folder, label in pairs:
    p = os.path.join(a.base, folder + "_spectral", "summary.json")
    if os.path.exists(p):
        data[label] = json.load(open(p))
    else:
        data[label] = None

keys1 = sorted({k for d in data.values() if d for k in d if k.startswith("spectrum")})
keys2 = sorted({k for d in data.values() if d for k in d if k.startswith("pcf")})

print("=" * 78)
print("TEST 1  POWER SPECTRUM   (want: low_freq -> 0, moderate peak, tail_dev -> 0)")
print("=" * 78)
print(f"{'method':<16}{'grey':>6}{'low_freq':>11}{'peak':>9}{'peak_f':>8}{'tail_dev':>10}")
for label in data:
    if data[label] is None:
        print(f"{label:<16}  (not yet computed)"); continue
    for k in keys1:
        v = data[label].get(k)
        if not v: continue
        print(f"{label:<16}{v['grey']:>6}{v['low_freq_power']:>11.5f}"
              f"{v['peak_power']:>9.3f}{v['peak_freq']:>8.3f}{v['tail_deviation']:>10.4f}")
    print()

print("=" * 78)
print("TEST 2  PAIR CORRELATION (want: exclusion_leak -> 0, one moderate peak, tail_dev -> 0)")
print("=" * 78)
print(f"{'method':<16}{'region':<16}{'leak':>8}{'peak':>9}{'peak_r':>8}{'tail_dev':>10}{'pts':>8}")
for label in data:
    if data[label] is None:
        print(f"{label:<16}  (not yet computed)"); continue
    for k in keys2:
        v = data[label].get(k)
        if not v: continue
        reg = f"{v['pattern']}/{v['region']}"
        print(f"{label:<16}{reg:<16}{v['exclusion_leak']:>8.4f}{v['peak']:>9.3f}"
              f"{v['peak_r']:>8.3f}{v['tail_deviation']:>10.4f}{v['mean_points']:>8.1f}")
    print()
