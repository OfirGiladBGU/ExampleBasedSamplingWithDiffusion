"""m4_teacher.py -- overfit teacher generator for the anisotropy control branch.

FLAT (uniform) density canvas so the ONLY thing distinguishing conditions is the
(theta,kappa) control field -- the sharpest test of non-decomposable steering.

Each teacher sample:
  high_res       (1, Himg, Himg)  constant grayscale canvas (uniform density)
  target_density (1, G, G)        uniform density map
  offsets        (2, G, G)        GT offset grid = M2 anisotropic points OT-assigned
  control        (3, G, G)        [m*cos2theta, m*sin2theta, m], m = log(kappa)

Points -> offset grid uses exact assignment (N = G*G points to G*G cells);
offset = (point - cell_center) * G  (matches DynamicControlNet point mapping).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

import aniso_m2 as m2


def build_conditions():
    return [dict(name="k1.00_t00", theta_deg=0.0,  kappa=1.0),
            dict(name="k2.00_t00", theta_deg=0.0,  kappa=2.0),
            dict(name="k2.00_t90", theta_deg=90.0, kappa=2.0),
            dict(name="k1.60_t45", theta_deg=45.0, kappa=1.6)]


def cell_centers(G):
    W = G
    cx = (np.arange(W) + 0.5) / W
    cy = (np.arange(G) + 0.5) / G
    JJ, II = np.meshgrid(np.arange(W), np.arange(G))
    return np.stack([cx[JJ], cy[II]], axis=-1).reshape(-1, 2)


def points_to_offset_grid(points, G):
    """Assign N=G*G points to the G*G cells (exact OT) -> (2,G,G) offset grid."""
    W = G
    centers = cell_centers(G)
    assert len(points) == G * W, "need G*G points"
    d = points[:, None, :] - centers[None, :, :]
    cost = (d * d).sum(-1)
    ri, ci = linear_sum_assignment(cost)
    assigned = np.empty(G * W, dtype=int)
    assigned[ci] = ri
    off = (points[assigned] - centers) * G
    return off.reshape(G, W, 2).transpose(2, 0, 1).astype(np.float32)


def control_map(theta_deg, kappa, G):
    """(3,G,G) constant control field: [m cos2th, m sin2th, m], m = log kappa."""
    th = np.deg2rad(theta_deg)
    m = np.log(float(kappa))
    ch = np.array([m * np.cos(2 * th), m * np.sin(2 * th), m], dtype=np.float32)
    return np.broadcast_to(ch.reshape(3, 1, 1), (3, G, G)).astype(np.float32).copy()


def generate(G=32, n_per_cond=1, himg=512, img_value=0.0, seed=0):
    """Return (samples, conditions). samples: list of dicts of numpy arrays."""
    conds = build_conditions()
    N = G * G
    s0 = 0.62 / np.sqrt(N)
    high_res = np.full((1, himg, himg), float(img_value), dtype=np.float32)
    density = np.full((1, G, G), float(img_value), dtype=np.float32)
    samples = []
    for ci, cond in enumerate(conds):
        theta_fn, kappa_fn, kmax = m2.const_field(cond["theta_deg"], cond["kappa"])
        cmap = control_map(cond["theta_deg"], cond["kappa"], G)
        for si in range(n_per_cond):
            rng = np.random.default_rng(10000 * seed + 97 * ci + si + 1)
            P, s_used, _ = m2.sample_exact_n(theta_fn, kappa_fn, s0, N, rng, kmax)
            offsets = points_to_offset_grid(P, G)
            samples.append(dict(high_res=high_res.copy(), target_density=density.copy(),
                                offsets=offsets, control=cmap.copy(),
                                cond_name=cond["name"], theta_deg=cond["theta_deg"],
                                kappa=cond["kappa"]))
    return samples, conds


if __name__ == "__main__":
    s, c = generate(G=32, n_per_cond=2)
    print("generated %d teacher samples over %d conditions" % (len(s), len(c)))
