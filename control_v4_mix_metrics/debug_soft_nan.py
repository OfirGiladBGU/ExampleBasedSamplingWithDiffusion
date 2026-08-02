"""Locate the op that produces the NaN gradient in the descriptor-consistency loss.

The NaN survives three rounds of reasoning-driven clamping (11% -> 8% of steps), so this stops
guessing and asks PyTorch directly. Two halves:

  STRESS   Feed deliberately pathological coordinate sets through the loss and report which ones
           produce a non-finite gradient. These are not hypothetical -- they are what an
           undertrained model's decoded x0 actually looks like. `offsets_to_coords_gpu` adds an
           UNBOUNDED offset to the grid centres, so early predictions land far outside [0,1], pile
           into clamped edge cells, and collapse points on top of one another.

  ANOMALY  Re-run the first failing case under torch.autograd.detect_anomaly, which names the exact
           forward op whose backward produced the NaN. Slow, hence only on the failing case.

    python control_v4_mix_metrics/debug_soft_nan.py
"""

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soft_descriptors as SD  # noqa: E402

G, N, B = 32, 1024, 4


def cases(device):
    g = torch.Generator(device="cpu").manual_seed(0)
    r = lambda *s: torch.rand(*s, generator=g).to(device)
    out = {}
    out["uniform in [0,1]"] = r(B, N, 2)
    # A model that has learned nothing outputs near-zero offsets -> every point sits on its grid
    # centre, so whole windows share an identical u and var is exactly 0.
    lin = (torch.arange(G, device=device) + 0.5) / G
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)
    out["exact grid centres"] = grid.unsqueeze(0).repeat(B, 1, 1)
    out["grid + tiny jitter"] = out["exact grid centres"] + (r(B, N, 2) - 0.5) * 1e-6
    # Unbounded offsets: coords = centre + offs/G, and offs is not constrained.
    out["coords outside [0,1]"] = (r(B, N, 2) - 0.5) * 6.0
    out["all points identical"] = torch.full((B, N, 2), 0.5, device=device)
    out["half collapsed"] = torch.cat([r(B, N // 2, 2),
                                       torch.full((B, N // 2, 2), 0.3, device=device)], 1)
    out["one tight cluster"] = r(B, N, 2) * 1e-4
    out["duplicated pairs"] = r(B, N // 2, 2).repeat_interleave(2, dim=1)
    return out


def run(coords, keys, device, anomaly=False):
    coords = coords.clone().requires_grad_(True)
    grad_map = torch.rand(B, 1, 512, 512, device=device)
    rho_map = torch.rand(B, 1, 512, 512, device=device)
    req = torch.rand(B, len(keys), G, G, device=device)
    ctx = torch.autograd.detect_anomaly() if anomaly else torch.enable_grad()
    with ctx:
        loss = SD.descriptor_consistency_loss(
            coords, grad_map, req, rho_map=rho_map, keys=keys, stats=None, G=G)
        loss.backward()
    gr = coords.grad
    return float(loss.detach()), bool(torch.isfinite(gr).all()), float(gr.abs().max())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    keysets = [["nn_cv"], ["aniso"], ["edge_align"], ["nn_cv", "aniso"]]
    print(f"device={device}\n")
    print(f"{'case':24s} " + " ".join(f"{'+'.join(k):>14s}" for k in keysets))
    failures = []
    for name, coords in cases(device).items():
        cells = []
        for keys in keysets:
            try:
                lv, ok, gmax = run(coords, keys, device)
                cells.append("ok" if ok else "NaN/inf")
                if not ok:
                    failures.append((name, keys))
            except Exception as exc:
                cells.append(type(exc).__name__[:12])
                failures.append((name, keys))
        print(f"{name:24s} " + " ".join(f"{c:>14s}" for c in cells))

    if not failures:
        print()
        print("ALL CLEAN: every case x descriptor produced a finite gradient.")
        print("  These cases cover the known NaN sources -- self-selection in topk (cdist self-")
        print("  distances are not exactly 0 in fp32), coincident neighbours from duplicate-repair")
        print("  or point collapse, empty windows, and unbounded decoded coords. If NaN still")
        print("  appears in a live run it is a pattern NOT represented here, and the next step is")
        print("  to dump the offending x0_pred rather than add another guard by reasoning.")
        return 0

    name, keys = failures[0]
    print(f"\nfirst failure: {name}  keys={keys}\nre-running under detect_anomaly ...\n")
    try:
        run(cases(device)[name], keys, device, anomaly=True)
    except Exception as exc:
        msg = str(exc)
        print(type(exc).__name__ + ": " + (msg[:2000] if msg else "(no message)"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
