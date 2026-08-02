"""m4_icons_teacher.py -- build the REAL-ICONS training set for anisotropy transfer.

Replaces the 4-primitive teacher. The primitives -> icons domain shift is what
blocked M4b, and the density ablation showed the branch was keying on shape
identity. Training on many real icons removes the shift; a DISJOINT held-out icon
set makes transfer the thing we actually measure.

Same .npz schema as m4_primitives (offsets / control / prim_idx / prim_imgs /
prim_dens / prim_names), so run_m4_generalize trains on it UNCHANGED. Adds a
held-out split (test_imgs / test_dens / test_names) that run_m4_generalize
evaluates on -- a real transfer test instead of random eval icons.

Targets use the density-aware sampler (aniso_density): points respect each icon's
ink density exactly, with anisotropy reshaping (not resizing) the exclusion zone.

  python m4_icons_teacher.py --n_train_icons 60 --n_test_icons 10 \
      --samples_per_icon 8 --out icons_ds.npz
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cv2
import numpy as np

import aniso_m2 as m2
import aniso_density as ad
import m4_teacher


GRID_SIZE = 32
HIMG = 512
N_TRAIN_ICONS = 60
N_TEST_ICONS = 10
SAMPLES_PER_ICON = 8
KAPPA_MIN = 1.0
KAPPA_MAX = 2.0
SEED = 0
ICONS_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/source"
OUT_NPZ = os.path.join(_HERE, "icons_ds.npz")


def list_icons(icons_dir):
    """Deterministic sorted list of every icon file under icons_dir."""
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = []
    for root, _, fs in os.walk(icons_dir):
        for f in sorted(fs):
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(root, f))
    if not files:
        raise SystemExit("no icons found in %s" % icons_dir)
    return files


def load_icon(path, himg):
    """Grayscale [0,1], dark = ink -- same convention as the rest of the pilot."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (himg, himg), interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32) / 255.0)


def downsample_density(img, G):
    """Area-average to (G,G) -- same convention as control_v4 conditioning."""
    H, W = img.shape
    return img.reshape(G, H // G, G, W // G).mean(axis=(1, 3)).astype(np.float32)


def load_set(files, himg, G):
    """Load a list of icon paths -> (imgs, dens, names), skipping unreadable ones."""
    imgs, dens, names = [], [], []
    for p in files:
        img = load_icon(p, himg)
        if img is None:
            print("  WARN unreadable, skipped:", p)
            continue
        imgs.append(img)
        dens.append(downsample_density(img, G))
        names.append(os.path.basename(p))
    return imgs, dens, names


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--icons_dir", default=ICONS_DIR)
    pa.add_argument("--grid", type=int, default=GRID_SIZE)
    pa.add_argument("--himg", type=int, default=HIMG)
    pa.add_argument("--n_train_icons", type=int, default=N_TRAIN_ICONS)
    pa.add_argument("--n_test_icons", type=int, default=N_TEST_ICONS)
    pa.add_argument("--samples_per_icon", type=int, default=SAMPLES_PER_ICON)
    pa.add_argument("--kappa_min", type=float, default=KAPPA_MIN)
    pa.add_argument("--kappa_max", type=float, default=KAPPA_MAX)
    pa.add_argument("--seed", type=int, default=SEED)
    pa.add_argument("--out", default=OUT_NPZ)
    args = pa.parse_args()

    G = args.grid
    N = G * G
    rng = np.random.default_rng(args.seed)

    # ── deterministic disjoint train / test split ────────────────────
    files = list_icons(args.icons_dir)
    perm = rng.permutation(len(files))
    need = args.n_train_icons + args.n_test_icons
    if len(files) < need:
        raise SystemExit("only %d icons found, need %d (train+test)" % (len(files), need))
    train_files = [files[i] for i in perm[:args.n_train_icons]]
    test_files = [files[i] for i in perm[args.n_train_icons:need]]
    print("icons: %d total -> %d train, %d held-out test (disjoint, seed=%d)"
          % (len(files), len(train_files), len(test_files), args.seed))

    train_imgs, train_dens, train_names = load_set(train_files, args.himg, G)
    test_imgs, test_dens, test_names = load_set(test_files, args.himg, G)
    n_train = len(train_imgs)
    rhos = [ad.ink_density(im) for im in train_imgs]

    # ── generate anisotropic targets, round-robin over train icons ────
    offsets, controls, icon_idx, thetas, kappas = [], [], [], [], []
    r0_hint = {p: 0.85 for p in range(n_train)}   # per-icon: ink coverage varies
    failed = 0
    total = n_train * args.samples_per_icon
    t0 = time.time()
    for i in range(total):
        p = i % n_train
        theta = float(rng.uniform(0.0, 180.0))
        kappa = float(rng.uniform(args.kappa_min, args.kappa_max))
        th_fn, ka_fn, kmax = m2.const_field(theta, kappa)
        P, r0, att = ad.sample_exact_n_density(
            rhos[p], th_fn, ka_fn, N, rng, r0_init=r0_hint[p], kappa_max=kmax,
            progress=False)
        if len(P) != N:
            failed += 1
            r0_hint[p] *= 0.9
            continue
        r0_hint[p] = min(1.0, r0 * 1.04)
        offsets.append(m4_teacher.points_to_offset_grid(P, G))
        controls.append(m4_teacher.control_map(theta, kappa, G))
        icon_idx.append(p)
        thetas.append(theta)
        kappas.append(kappa)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (total - i - 1)
            print("  %4d/%d  icon=%-20s theta=%3.0f kappa=%.2f r0=%.3f | %.0fs, ETA %.0fs"
                  % (i + 1, total, train_names[p][:20], theta, kappa, r0, el, eta))

    if not offsets:
        raise SystemExit("no samples generated")

    np.savez_compressed(
        args.out,
        offsets=np.stack(offsets).astype(np.float32),
        control=np.stack(controls).astype(np.float32),
        prim_idx=np.asarray(icon_idx, dtype=np.int64),
        theta_deg=np.asarray(thetas, dtype=np.float32),
        kappa=np.asarray(kappas, dtype=np.float32),
        prim_imgs=np.stack(train_imgs).astype(np.float32),
        prim_dens=np.stack(train_dens).astype(np.float32),
        prim_names=np.asarray(train_names),
        test_imgs=np.stack(test_imgs).astype(np.float32),
        test_dens=np.stack(test_dens).astype(np.float32),
        test_names=np.asarray(test_names),
        grid=np.int64(G),
    )
    print("wrote %s: %d samples over %d train icons (%d failed), %d held-out test icons"
          % (args.out, len(offsets), n_train, failed, len(test_imgs)))
    print("kappa range [%.2f, %.2f], theta continuous in [0,180)"
          % (min(kappas), max(kappas)))


if __name__ == "__main__":
    main()
