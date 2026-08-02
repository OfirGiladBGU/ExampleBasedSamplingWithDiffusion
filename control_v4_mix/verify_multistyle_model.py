"""Verify the MULTI-STYLE branch wiring without training data (needs only base config + ckpt).

Same three checks as verify_style_model.py, but for the K-dim one-hot style vector -- so you can
confirm the generalized model still supports the plain 2-oracle (WVS/GBN) case (run with --K 2) as
well as K=3+ (WVS/GBN/DITHER). Checks:
  1. ZERO-INIT IDENTITY : at init the style MLP outputs 0, so every one-hot vertex e_k (and None)
     gives the SAME control signal -> starts exactly at the unconditioned baseline.
  2. WIRING             : after perturbing the style MLP, distinct one-hot vertices produce
     DIFFERENT control signals -> each oracle vector is genuinely connected via temb.
  3. GRADIENT FLOW      : a loss on the control output yields non-zero gradient on the style MLP.

The control INJECTION layers are zero-init (ControlNet zero-conv) and would mask the effect, so we
activate them first (nonzero transform), exactly as in verify_style_model.py.

Run from project root.  e.g.  python control_v4_mix/verify_multistyle_model.py --K 2
"""

import argparse
import sys

import torch

sys.path.insert(0, ".")
from utils.Config import ParseSampleConfig
from control_v4_mix.DynamicControlNetMultiStyle import DynamicControlNetMultiStyle


def flat(controls):
    enc, mid = controls
    parts = [mid.reshape(-1)]
    for level in enc:
        for c in level:
            parts.append(c.reshape(-1))
    return torch.cat(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_config_path", default="config/GBN/config.json")
    p.add_argument("--base_ckpt_path", default="config/GBN/model.ckpt")
    p.add_argument("--K", type=int, default=3, help="number of oracles (2 = WVS/GBN only)")
    p.add_argument("--grid-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    K = args.K
    diffusion = ParseSampleConfig(args.base_config_path, device=device)
    diffusion.load_state_dict(torch.load(args.base_ckpt_path, map_location="cpu")["diffu"], strict=False)
    diffusion.to(device); diffusion.eval()
    denoiser = diffusion.model

    control_net = DynamicControlNetMultiStyle(
        denoiser, grid_size=args.grid_size, style_dim=K, enable_gecco=True,
        enable_adaptive_gate_injection=True,
        smart_init_features=False, sdf_features=False, batch_coords_features=False,
    ).to(device)
    control_net.eval()

    n_style = sum(pp.numel() for pp in control_net.style_mlp.parameters())
    print(f"K={K}; style_mlp params: {n_style:,} (in_dim={control_net.style_mlp[0].in_features})")

    # activate zero-init injections so temb (and thus the style vector) reaches the output.
    def activate_injections(cn):
        mods = [cn.inject_middle]
        for level in cn.injections:
            for inj in level:
                mods.append(inj)
        with torch.no_grad():
            for m in mods:
                torch.nn.init.normal_(m.transform.weight, 0.0, 0.1)
                torch.nn.init.normal_(m.transform.bias, 0.0, 0.1)
    activate_injections(control_net)

    B, G = 2, args.grid_size
    offsets_t = torch.randn(B, 2, G, G, device=device)
    t = torch.randint(0, 300, (B,), device=device)
    high_res = torch.rand(B, 1, 512, 512, device=device)
    density = torch.rand(B, 1, G, G, device=device)

    def onehot(k):
        v = torch.zeros(B, K, device=device); v[:, k] = 1.0
        return v

    def run(vec):
        return flat(control_net(offsets_t, t, high_res, density, style_vec=vec))

    with torch.no_grad():
        outs = [run(onehot(k)) for k in range(K)] + [run(None)]
    max_dev = max((outs[0] - o).abs().max().item() for o in outs[1:])
    ok1 = max_dev < 1e-6
    print(f"[1] zero-init identity (all {K} vertices + None identical): "
          f"{'PASS' if ok1 else 'FAIL'}  maxdev={max_dev:.3e}")

    with torch.no_grad():
        control_net.style_mlp[-1].weight.normal_(0, 0.1)
        control_net.style_mlp[-1].bias.normal_(0, 0.1)
        o0, o1 = run(onehot(0)), run(onehot(1))
    ok2 = not torch.allclose(o0, o1, atol=1e-6)
    print(f"[2] wiring (vertex 0 vs 1 differ after perturb): {'PASS' if ok2 else 'FAIL'}"
          f"  |o1-o0|max={(o1-o0).abs().max().item():.3e}")

    control_net.zero_grad(set_to_none=True)
    loss = run(onehot(min(1, K - 1))).pow(2).mean()
    loss.backward()
    g = control_net.style_mlp[-1].weight.grad
    ok3 = g is not None and torch.any(g != 0)
    print(f"[3] gradient reaches style_mlp: {'PASS' if ok3 else 'FAIL'}"
          f"  ||grad||={(0.0 if g is None else g.norm().item()):.3e}")

    all_ok = ok1 and ok2 and ok3
    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
