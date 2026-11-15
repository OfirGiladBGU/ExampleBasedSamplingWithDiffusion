#!/usr/bin/env python3
"""Anisotropy (dB) and low-frequency power per method, averaged over the uniform grey levels.

Reads experiments/outputs/spectral_analysis/target_<method>_spectral/spectral_g<grey>.npz written by
spectral_analysis_stage_1.py (25 realizations per grey level; anisotropy = angular variance of the
power within a radial band over its squared mean, in dB; lower = more isotropic; the floor for an
isotropic process averaged over 25 realizations is about -14 dB). Bands are in units of
radial frequency / sqrt(n); the blue-noise peak sits near 1.

Writes experiments/outputs/spectral_analysis/anisotropy_summary.md.

    python experiments/spectral_anisotropy_summary.py
"""
from pathlib import Path

import numpy as np

BASE = Path("experiments/outputs/spectral_analysis")
METHODS = ["BNOT_1024", "GBN_1024", "WVS_1024", "CN-GBN_1024", "CN-GBN-DDIM25_1024", "CN-WVS_1024",
           "CN-WVS-DDIM25_1024"]
GREYS = ["000", "064", "128", "192"]
BANDS = [("low (< 0.75)", 0.0, 0.75), ("peak (0.75-1.5)", 0.75, 1.5), ("high (1.5-3)", 1.5, 3.0)]


def main():
    lines = ["# Spectral anisotropy (uniform densities, 1024 points, 25 realizations per grey level)", "",
             "Mean anisotropy in dB over grey levels 0 / 64 / 128 / 192 (lower = more isotropic);",
             "low-freq power = mean radial power below 0.5 (lower = bluer).", "",
             "| method | " + " | ".join(b[0] for b in BANDS) + " | low-freq power |",
             "|---|" + "---|" * (len(BANDS) + 1)]
    for m in METHODS:
        d = BASE / f"target_{m}_spectral"
        if not d.is_dir():
            continue
        cells, lowp = [], []
        for _, a, b in BANDS:
            vals = []
            for g in GREYS:
                z = np.load(d / f"spectral_g{g}.npz")
                r, an = z["radial_freq"], z["anisotropy_db"]
                sel = (r >= a) & (r < b) & np.isfinite(an)
                vals.append(np.nanmean(an[sel]))
            cells.append(f"{np.mean(vals):.2f}")
        for g in GREYS:
            z = np.load(d / f"spectral_g{g}.npz")
            sel = (z["radial_freq"] < 0.5) & np.isfinite(z["radial_power"])
            lowp.append(np.mean(z["radial_power"][sel]))
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {np.mean(lowp):.4f} |")
    (BASE / "anisotropy_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
