"""Verify the style branch wiring WITHOUT training data (needs only the base config + ckpt).

Checks, in order:
  1. ZERO-INIT IDENTITY: at initialization the style MLP outputs 0, so the control signal for
     s=0, s=1 and s=None are bit-identical -> the model starts exactly at the unconditioned
     baseline (nothing is perturbed before any learning).
  2. WIRING: after perturbing the style MLP's last layer, s=0 and s=1 produce DIFFERENT control
     signals -> s is genuinely connected to every control block via temb.
  3. GRADIENT FLOW: a loss on the control output produces a non-zero gradient on the style MLP
     -> training can actually learn to use s.

Run from the project root (same as train_control_style.py).
"""

import argparse
import sys

import torch

sys.path.insert(0, ".")
from utils.Config import ParseSampleConfig
from control_v4_mix.DynamicControlNetStyle import DynamicControlNetStyle


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
    p.add_argument("--grid-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    diffusion = ParseSampleConfig(args.base_config_path, device=device)
    diffusion.load_state_dict(torch.load(args.base_ckpt_path, map_location="cpu")["diffu"], strict=False)
    diffusion.to(device); diffusion.eval()
    denoiser = diffusion.model

    control_net = DynamicControlNetStyle(
        denoiser, grid_size=args.grid_size, enable_gecco=True,
        enable_adaptive_gate_injection=True,
        smart_init_features=False, sdf_features=False, batch_coords_features=False,
    ).to(device)
    control_net.eval()

    n_style = sum(p.numel() for p in control_net.style_mlp.parameters())
    print(f"style_mlp params: {n_style:,}  (requires_grad="
          f"{all(p.requires_grad for p in control_net.style_mlp.parameters())})")

    # The control INJECTION layers are zero-initialized (ControlNet zero-conv), so at init the
    # control signal is identically zero and would mask ANY upstream conditioning (temb, and hence
    # s) AND block gradient from flowing back through the zero transform. Activate them (nonzero
    # transform) to mimic a partly-trained control branch so the style effect is observable. This
    # does NOT touch the style MLP -- it stays zero-init, so check [1] remains a real test.
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

    def run(s):
        s_t = None if s is None else torch.full((B,), float(s), device=device)
        return flat(control_net(offsets_t, t, high_res, density, style_s=s_t))

    with torch.no_grad():
        o_none, o0, o1 = run(None), run(0.0), run(1.0)

    ok1 = torch.allclose(o0, o1, atol=1e-6) and torch.allclose(o0, o_none, atol=1e-6)
    print(f"[1] zero-init identity (s in {{None,0,1}} identical): {'PASS' if ok1 else 'FAIL'}"
          f"  |o1-o0|max={(o1-o0).abs().max().item():.3e}")

    with torch.no_grad():
        control_net.style_mlp[-1].weight.normal_(0, 0.1)
        control_net.style_mlp[-1].bias.normal_(0, 0.1)
        o0b, o1b = run(0.0), run(1.0)
    ok2 = not torch.allclose(o0b, o1b, atol=1e-6)
    print(f"[2] wiring (s changes control after perturb): {'PASS' if ok2 else 'FAIL'}"
          f"  |o1-o0|max={(o1b-o0b).abs().max().item():.3e}")

    control_net.zero_grad(set_to_none=True)
    s_t = torch.full((B,), 1.0, device=device)
    loss = flat(control_net(offsets_t, t, high_res, density, style_s=s_t)).pow(2).mean()
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
