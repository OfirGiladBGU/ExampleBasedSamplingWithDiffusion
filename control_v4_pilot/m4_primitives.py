"""m4_primitives.py -- build the primitives training set for the generalization test.

Idea: teach the anisotropy branch a LOCAL, content-independent rule on a handful
of basic shapes, then test whether it transfers to unseen real icons.

Two design requirements make this a fair test rather than memorization:
  * (theta, kappa) are sampled CONTINUOUSLY (theta in [0,180), kappa in [1,2]),
    so the branch must learn the rule, not 4 memorized control vectors.
  * several DIVERSE primitives (flat / disk / square / stripe), so anisotropy
    does not get entangled with one specific density pattern (the branch sees
    density as an input channel).

Targets use the density-aware sampler (aniso_density), so points respect each
primitive's ink density exactly as they will on real icons.

Output: a compact .npz. Only the few unique primitive images are stored (with a
per-sample index), so the file stays small.

  python m4_primitives.py --n_samples 400 --out primitives_ds.npz
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import aniso_m2 as m2
import aniso_density as ad
import m4_teacher


GRID_SIZE = 32
HIMG = 512
N_SAMPLES = 400
KAPPA_MIN = 1.0
KAPPA_MAX = 2.0
SEED = 0
OUT_NPZ = os.path.join(_HERE, "primitives_ds.npz")


def make_primitives(himg=HIMG):
    """Grayscale [0,1], dark = ink. Kept deliberately simple and diverse."""
    yy, xx = np.mgrid[0:himg, 0:himg] / float(himg)
    prims = {}

    prims["flat"] = np.zeros((himg, himg), dtype=np.float32)          # all ink

    disk = np.ones((himg, himg), dtype=np.float32)
    disk[((xx - 0.5) ** 2 + (yy - 0.5) ** 2) < 0.32 ** 2] = 0.0
    prims["disk"] = disk

    square = np.ones((himg, himg), dtype=np.float32)
    square[(np.abs(xx - 0.5) < 0.30) & (np.abs(yy - 0.5) < 0.30)] = 0.0
    prims["square"] = square

    stripe = np.ones((himg, himg), dtype=np.float32)
    stripe[np.abs(yy - 0.5) < 0.18] = 0.0
    prims["stripe"] = stripe

    return prims


def downsample_density(img, G):
    """Area-average to (G,G) -- same convention as control_v4 conditioning."""
    H, W = img.shape
    return img.reshape(G, H // G, G, W // G).mean(axis=(1, 3)).astype(np.float32)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--grid", type=int, default=GRID_SIZE)
    pa.add_argument("--himg", type=int, default=HIMG)
    pa.add_argument("--n_samples", type=int, default=N_SAMPLES)
    pa.add_argument("--kappa_min", type=float, default=KAPPA_MIN)
    pa.add_argument("--kappa_max", type=float, default=KAPPA_MAX)
    pa.add_argument("--seed", type=int, default=SEED)
    pa.add_argument("--out", default=OUT_NPZ)
    args = pa.parse_args()

    G = args.grid
    N = G * G
    rng = np.random.default_rng(args.seed)

    prims = make_primitives(args.himg)
    names = sorted(prims.keys())
    prim_imgs = np.stack([prims[n] for n in names])                    # (P,H,W)
    prim_dens = np.stack([downsample_density(prims[n], G) for n in names])  # (P,G,G)
    rhos = [ad.ink_density(prims[n]) for n in names]

    offsets, controls, prim_idx, thetas, kappas = [], [], [], [], []
    failed = 0
    # Carry the last feasible r0 forward: re-discovering it per sample wastes
    # ~2 failed full passes each time.
    r0_hint = 0.85
    t0 = time.time()
    for i in range(args.n_samples):
        p = i % len(names)
        theta = float(rng.uniform(0.0, 180.0))
        kappa = float(rng.uniform(args.kappa_min, args.kappa_max))
        th_fn, ka_fn, kmax = m2.const_field(theta, kappa)
        P, r0, att = ad.sample_exact_n_density(
            rhos[p], th_fn, ka_fn, N, rng, r0_init=r0_hint, kappa_max=kmax,
            progress=False)
        if len(P) != N:
            failed += 1
            r0_hint *= 0.9
            continue
        r0_hint = min(1.0, r0 * 1.04)
        offsets.append(m4_teacher.points_to_offset_grid(P, G))
        controls.append(m4_teacher.control_map(theta, kappa, G))
        prim_idx.append(p)
        thetas.append(theta)
        kappas.append(kappa)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (args.n_samples - i - 1)
            print("  %4d/%d  %-6s theta=%3.0f kappa=%.2f r0=%.3f | %.0fs elapsed, ETA %.0fs"
                  % (i + 1, args.n_samples, names[p], theta, kappa, r0, el, eta))

    if not offsets:
        raise SystemExit("no samples generated")

    np.savez_compressed(
        args.out,
        offsets=np.stack(offsets).astype(np.float32),
        control=np.stack(controls).astype(np.float32),
        prim_idx=np.asarray(prim_idx, dtype=np.int64),
        theta_deg=np.asarray(thetas, dtype=np.float32),
        kappa=np.asarray(kappas, dtype=np.float32),
        prim_imgs=prim_imgs.astype(np.float32),
        prim_dens=prim_dens.astype(np.float32),
        prim_names=np.asarray(names),
        grid=np.int64(G),
    )
    print("wrote %s: %d samples over %d primitives (%d failed)"
          % (args.out, len(offsets), len(names), failed))
    print("kappa range [%.2f, %.2f], theta continuous in [0,180)"
          % (min(kappas), max(kappas)))


if __name__ == "__main__":
    main()
