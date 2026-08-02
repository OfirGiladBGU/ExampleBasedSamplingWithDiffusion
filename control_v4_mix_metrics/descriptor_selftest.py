"""Data-free sanity check of the descriptor field math. Run this before touching real data.

Four properties are asserted, each of which has a specific way of being silently wrong:

  ORDERING      -- nn_cv and cap_cv must rank lattice < jittered < white noise. A descriptor that
                   does not recover the textbook ordering on synthetic processes cannot be trusted
                   to rank oracles.
  DENSITY       -- nn_cv must be (near-)invariant under a density gradient. This is the whole
                   reason it is normalised by local spacing; if it is not invariant it is just
                   re-reading rho and the axis would be rho-decomposable. The gradient MUST be
                   ISOTROPIC (both axes scaled): nn_cv is sensitive to local anisotropy, so an
                   x-only warp would fail this test for a reason that has nothing to do with
                   density.
  ANISOTROPY    -- aniso must be ~0 for isotropic processes and clearly positive for a line/worm
                   process. This is the descriptor expected to catch Floyd-Steinberg.
  PERIODICITY   -- pcf_peak must rank white noise (~1, no structure) < jittered < exact lattice.
                   This is descriptor 5, added because 1-4 could not tell a periodic dither from an
                   aperiodic one. If it does not separate a lattice from a jittered lattice it is
                   not measuring periodicity and does not earn its place in the stack.
  FIELD LOCALITY-- a half-and-half point set (regular left, white noise right) must show the split
                   IN THE FIELD, not just in the pooled scalar. If this fails, the descriptor is
                   effectively global and spatially-varying control is unreachable.
"""

import numpy as np

import descriptor_fields as DF

RNG = np.random.RandomState(0)
G = 32


def _flat_gray(n=512):
    return np.full((n, n), 0.5)


def lattice(n=1024, jitter=0.0, rng=RNG):
    side = int(np.sqrt(n))
    lin = (np.arange(side) + 0.5) / side
    gx, gy = np.meshgrid(lin, lin)
    p = np.stack([gx.ravel(), gy.ravel()], axis=1)
    if jitter > 0:
        p = p + (rng.rand(len(p), 2) - 0.5) * jitter / side
    return np.clip(p, 1e-6, 1 - 1e-6)


def white(n=1024, rng=RNG):
    return rng.rand(n, 2)


def lines(n=1024, n_lines=16, rng=RNG):
    """Points strung along horizontal lines -- a stand-in for FS worms.

    n_lines must be well below sqrt(n) or the fixture is not actually anisotropic: at n_lines=32
    with n=1024 the within-line and between-line spacings are both 1/32, i.e. a square lattice,
    and it would (correctly) read as isotropic. 16 lines x 64 points gives a 4:1 aspect ratio.
    """
    per = n // n_lines
    ys = (np.arange(n_lines) + 0.5) / n_lines
    p = []
    for y in ys:
        x = np.sort(rng.rand(per))
        p.append(np.stack([x, np.full(per, y) + rng.randn(per) * 0.002], axis=1))
    p = np.concatenate(p)
    if len(p) < n:
        p = np.vstack([p, rng.rand(n - len(p), 2)])
    return np.clip(p, 1e-6, 1 - 1e-6)


def isotropic_density_warp(p, strength=0.6):
    """Map the unit square onto itself so density varies, scaling BOTH axes identically.

    r -> r^(1+strength) in a radial sense about the centre keeps the map conformal-ish and, more
    importantly, isotropic: any local neighbourhood is scaled, never stretched. That is what makes
    this a pure density test rather than an anisotropy test.
    """
    q = p - 0.5
    r = np.linalg.norm(q, axis=1, keepdims=True)
    r = np.maximum(r, 1e-9)
    rn = (r / 0.7071) ** (1 + strength) * 0.7071
    out = 0.5 + q / r * rn
    return np.clip(out, 1e-6, 1 - 1e-6)


def pooled(p, gray=None):
    gray = _flat_gray() if gray is None else gray
    f, _ = DF.descriptor_fields(p, gray, G=G)
    return DF.pool_fields(f), f


def main():
    gray = _flat_gray()
    ok = True

    lat, _ = pooled(lattice(), gray)
    jit, _ = pooled(lattice(jitter=0.55), gray)
    wht, _ = pooled(white(), gray)

    print("ORDERING (expect lattice < jittered < white)")
    for key in ("nn_cv", "cap_cv"):
        print(f"  {key:10s} lattice={lat[key]:.4f}  jittered={jit[key]:.4f}  white={wht[key]:.4f}")
        if not (lat[key] < jit[key] < wht[key]):
            print(f"    FAIL: {key} does not recover the expected ordering")
            ok = False

    print("\nDENSITY INVARIANCE (isotropic warp; expect nn_cv nearly unchanged)")
    for name, gen in (("lattice", lattice()), ("white", white())):
        base, _ = pooled(gen, gray)
        warped, _ = pooled(isotropic_density_warp(gen), gray)
        rel = abs(warped["nn_cv"] - base["nn_cv"]) / max(base["nn_cv"], 1e-9)
        print(f"  {name:8s} nn_cv {base['nn_cv']:.4f} -> {warped['nn_cv']:.4f}  (rel {rel:.2%})")
        if rel > 0.35:
            print("    FAIL: nn_cv moved too much under a pure density change")
            ok = False

    print("\nANISOTROPY (expect isotropic ~ low, lines clearly higher)")
    a_lat = lat["aniso"]
    a_wht = wht["aniso"]
    a_lin, _ = pooled(lines(), gray)
    print(f"  lattice={a_lat:.4f}  white={a_wht:.4f}  lines={a_lin['aniso']:.4f}")
    if not (a_lin["aniso"] > 0.15 and max(a_lat, a_wht) < 0.10):
        print("    FAIL: line process is not read as more anisotropic than isotropic ones")
        ok = False

    # pcf_peak claims to detect LATTICE PERIODICITY, not to rank "amount of structure" -- see the
    # note in _pcf_fields. So the assertion is that an exact lattice stands clearly above EVERY
    # aperiodic process, not that the aperiodic ones are themselves ordered among each other.
    a_lin2, _ = pooled(lines(), gray)
    aperiodic = {"jittered": jit["pcf_peak"], "white": wht["pcf_peak"], "lines": a_lin2["pcf_peak"]}
    print("\nPERIODICITY (pcf_peak; expect lattice >= 2x every aperiodic process)")
    print(f"  lattice={lat['pcf_peak']:.3f}   " +
          "  ".join(f"{k}={v:.3f}" for k, v in aperiodic.items()))
    print(f"  pcf_pos   lattice={lat['pcf_pos']:.3f}  white={wht['pcf_pos']:.3f}")
    if lat["pcf_peak"] < 2.0 * max(aperiodic.values()):
        print(f"    FAIL: lattice peak {lat['pcf_peak']:.3f} not clearly above aperiodic "
              f"max {max(aperiodic.values()):.3f} -- not measuring periodicity")
        ok = False

    print("\nFIELD LOCALITY (regular left half | white noise right half)")
    left = lattice(n=1024)
    left = left[left[:, 0] < 0.5]
    right = white(n=512)
    right[:, 0] = right[:, 0] * 0.5 + 0.5
    mixed = np.vstack([left, right])
    _, fm = pooled(mixed, gray)
    fld = fm["nn_cv"]
    lo = np.nanmean(fld[:, : G // 4])
    hi = np.nanmean(fld[:, 3 * G // 4:])
    print(f"  nn_cv  left-quarter={lo:.4f}  right-quarter={hi:.4f}")
    if not (hi > lo * 1.3):
        print("    FAIL: field does not localise the style split")
        ok = False

    print("\n" + ("SELF-TEST PASSED" if ok else "SELF-TEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
