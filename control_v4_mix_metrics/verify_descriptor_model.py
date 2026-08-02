"""Wiring checks for DynamicControlNetDescriptor. Run BEFORE launching any training.

Five properties, each with a specific way of being silently wrong:

  IDENTITY AT INIT   The zero-init FiLM heads must make the descriptor-conditioned net produce
                     control signals bit-identical to the plain DynamicControlNet. If they do not,
                     the descriptor branch is perturbing the model before it has learned anything,
                     and any later comparison against a baseline is confounded.
  DESCRIPTOR MATTERS Two different descriptor fields must produce IDENTICAL output at init (by the
                     above) but the gradient w.r.t. the descriptor input must be NON-ZERO -- i.e.
                     the path is connected and trainable. Zero-init is only safe if it is
                     zero-VALUE, not zero-GRADIENT; a disconnected branch also outputs zeros and
                     would pass a naive identity check forever.
  GRADIENTS FLOW     Every descriptor parameter must receive a gradient from a plain loss on the
                     control outputs.
  G-TRANSFER         Forward must work at G in {32, 48, 64} with a G-sized descriptor field, since
                     the plan requires re-verifying grid transfer after adding this branch.
  NO POSITIONAL EMB  A translated descriptor field must produce a correspondingly translated
                     modulation. Any learned absolute positional embedding would break this, and
                     the architecture constraint forbids one.

    python control_v4_mix_metrics/verify_descriptor_model.py
"""

import argparse
import copy
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from control_v4.DynamicControlNet import DynamicControlNet          # noqa: E402
from DynamicControlNetDescriptor import DynamicControlNetDescriptor  # noqa: E402

K = 5          # descriptor channels = len(descriptor_fields.CONDITIONING_KEYS)


BASE_CONFIG_PATH = "config/GBN/config.json"     # = train_control.py:83


def build_denoiser(config_path, device):
    """Exactly train_control.py:1199-1212's construction, so channel shapes match production.

    `ParseSampleConfig` RETURNS the diffusion object (it is not a config parser that feeds a
    separate constructor), and the denoiser is `diffusion.model`. No checkpoint is loaded: this
    verifies wiring, not weights, and the descriptor runs train the base from scratch anyway.
    """
    from utils.Config import ParseSampleConfig
    diffusion = ParseSampleConfig(config_path, device=device)
    diffusion.to(device)
    diffusion.eval()
    return diffusion.model


def _cond(B, G, device, ch_hi=512):
    return dict(
        offsets_t=torch.randn(B, 2, G, G, device=device),
        t=torch.randint(0, 1000, (B,), device=device),
        high_res_image=torch.rand(B, 1, ch_hi, ch_hi, device=device),
        target_density_map=torch.rand(B, 1, G, G, device=device),
    )


def flat(controls):
    enc, mid = controls
    return torch.cat([t.flatten() for lvl in enc for t in lvl] + [mid.flatten()])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_config_path", "--config", dest="config",
                    default=os.path.join(REPO, BASE_CONFIG_PATH),
                    help="base diffusion config (default: %(default)s)")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--descriptor-channels", type=int, default=K)
    ap.add_argument("--grid-size", type=int, default=32)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  config={args.config}")
    torch.manual_seed(0)
    denoiser = build_denoiser(args.config, device).eval()
    ok = True

    G = args.grid_size
    Kc = args.descriptor_channels
    torch.manual_seed(0)
    base = DynamicControlNet(denoiser, grid_size=G).to(device).eval()
    torch.manual_seed(0)
    net = DynamicControlNetDescriptor(denoiser, grid_size=G,
                                      descriptor_channels=Kc).to(device).eval()

    pristine = copy.deepcopy(net)

    B = 2
    c = _cond(B, G, device)
    d = torch.rand(B, Kc, G, G, device=device)

    # ── 1. identity at init ────────────────────────────────────────────
    with torch.no_grad():
        a = flat(base(**c))
        b = flat(net(**c, descriptor_field=d))
    delta = (a - b).abs().max().item()
    print(f"IDENTITY AT INIT      max|baseline - descriptor| = {delta:.3e}")
    if delta > 1e-5:
        print("  FAIL: zero-init FiLM is not identity -- descriptor branch perturbs the model")
        ok = False

    # ── 2/3. connectivity ONCE THE ZERO-CONVS HAVE WOKEN UP ────────────
    # At step 0 every ControlNet injection is a zero conv, so dL/dx = 0 through it and NOTHING
    # upstream gets gradient -- the baseline's own control encoder included. Measuring the
    # descriptor branch at that instant measures the injection, not the branch. So put the
    # injections into a generic non-zero state (where one optimiser step lands them) and only then
    # ask whether the descriptor path is live.
    with torch.no_grad():
        for p_ in net.injections.parameters():
            p_.normal_(0, 0.05)
        for p_ in net.inject_middle.parameters():
            p_.normal_(0, 0.05)

    net.zero_grad(set_to_none=True)
    flat(net(**c, descriptor_field=d)).pow(2).sum().backward()
    gates = [(n, p_) for n, p_ in net.named_parameters() if n.endswith(".gate")]
    live_gates = [n for n, p_ in gates if p_.grad is not None and p_.grad.abs().sum() > 0]
    print(f"FILM GATES LIVE       {len(live_gates)}/{len(gates)} zero-init gates received gradient "
          f"(these wake in one step)")
    if len(live_gates) != len(gates):
        print("  FAIL: a zero-init gate is not reachable -- that head can never activate")
        ok = False

    # With the gates open, the conv and the encoder must be live too, and the descriptor must
    # actually change the output.
    with torch.no_grad():
        for n, p_ in gates:
            p_.fill_(1.0)
    net.zero_grad(set_to_none=True)
    d_req = d.clone().requires_grad_(True)
    flat(net(**c, descriptor_field=d_req)).pow(2).sum().backward()
    gmax = d_req.grad.abs().max().item()
    with torch.no_grad():
        o1 = flat(net(**c, descriptor_field=d))
        o2 = flat(net(**c, descriptor_field=torch.rand_like(d)))
    delta_fn = (o1 - o2).abs().max().item()
    print(f"DESCRIPTOR MATTERS    with gates open: d(out)/d(descriptor) max = {gmax:.3e}, "
          f"output delta between two fields = {delta_fn:.3e}")
    if gmax <= 0:
        print("  FAIL: gradient never reaches the descriptor input -- branch is DISCONNECTED")
        ok = False
    if delta_fn <= 0:
        print("  FAIL: the descriptor does not change the output even with gates open")
        ok = False

    named = [(n, p_) for n, p_ in net.named_parameters() if n.startswith(("desc_encoder", "film"))]
    dead = [n for n, p_ in named if p_.grad is None or p_.grad.abs().sum() == 0]
    print(f"GRADIENTS FLOW        {len(named) - len(dead)}/{len(named)} descriptor tensors "
          f"received gradient")
    if dead:
        print(f"  FAIL: no gradient for {dead[:6]}{' ...' if len(dead) > 6 else ''}")
        ok = False

    # ── 4. G-transfer ──────────────────────────────────────────────────
    sizes = []
    for g in (32, 48, 64):
        try:
            with torch.no_grad():
                net(**_cond(1, g, device), descriptor_field=torch.rand(1, Kc, g, g, device=device))
            sizes.append(f"{g}:ok")
        except Exception as exc:
            sizes.append(f"{g}:FAIL({type(exc).__name__})")
            ok = False
    print(f"G-TRANSFER            {'  '.join(sizes)}")

    # ── 5. translation equivariance (no absolute positional embedding) ──
    with torch.no_grad():
        # Only the encoder matters here -- equivariance is a property of the conv trunk, and the
        # FiLM heads are 1x1 (trivially equivariant) and not on this path at all.
        for p in net.desc_encoder.parameters():
            p.normal_(0, 0.05)
        feat = net.desc_encoder(d)
        rolled = net.desc_encoder(torch.roll(d, shifts=8, dims=-1))
        # interior only: rolling wraps, and conv padding differs at the border
        err = (torch.roll(feat, shifts=8, dims=-1)[..., 4:-4, 12:-12]
               - rolled[..., 4:-4, 12:-12]).abs().max().item()
    print(f"NO POSITIONAL EMB     translation equivariance err = {err:.3e}")
    if err > 1e-4:
        print("  FAIL: descriptor encoder is position-dependent -- violates the architecture "
              "constraint and would break G-transfer")
        ok = False

    # ── report: activity is 0 at init, by construction ─────────────────
    # On a PRISTINE copy: the checks above deliberately perturb gates and encoder weights, so
    # reading film_activity off `net` afterwards would report that perturbation, not the init.
    act = pristine.film_activity(d)
    print(f"\nfilm_activity at init mean|scale| = {act.get('mean_abs_scale', float('nan')):.3e} "
          f"(0 by construction; watch this GROW during training -- if it stays ~0 the model is "
          f"ignoring the descriptor)")

    print("\n" + ("ALL CHECKS PASSED" if ok else "CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
