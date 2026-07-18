"""inspect_ckpt.py -- report the exact config baked into a control_v4 checkpoint,
so the anisotropy control branch can warm-start from it WITHOUT a silent partial
load. Run on the server (env with torch). CPU only.

    python inspect_ckpt.py /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt
"""
import sys
import torch


def find_control_sd(obj):
    if isinstance(obj, dict):
        for k in ("control_net", "model_state_dict", "state_dict"):
            v = obj.get(k)
            if isinstance(v, dict):
                return v, k
        if obj and all(hasattr(v, "shape") for v in obj.values()):
            return obj, "(raw state_dict)"
    raise SystemExit("Could not locate a control state_dict in this checkpoint.")


def main():
    path = sys.argv[1]
    ckpt = torch.load(path, map_location="cpu")
    print("=" * 70)
    print("checkpoint:", path)
    if isinstance(ckpt, dict):
        print("top-level keys:", list(ckpt.keys()))
        for meta in ("epoch", "global_step", "args"):
            if meta in ckpt:
                print("  %s: %s" % (meta, ckpt[meta]))
    sd, where = find_control_sd(ckpt)
    print("control weights found under:", where)
    print("num control params:", len(sd))

    # feature-flag fingerprints
    has_gate = any(".gate." in k for k in sd if "inject" in k)
    has_gecco = any("gecco" in k for k in sd)
    hint0 = None
    for k in sd:
        if k.endswith("input_hint_block.0.weight"):
            hint0 = tuple(sd[k].shape)
            break
    print("-" * 70)
    print("adaptive_gate_injection (has .gate. in injections):", has_gate)
    print("enable_gecco (has gecco keys)                      :", has_gecco)
    print("input_hint_block.0.weight shape                    :", hint0)
    if hint0 is not None:
        in_ch = hint0[1]
        # base = offsets(2)+density(1)=3; +sdf(1) +smart(1) +coords(2) +gecco(G)
        print("  -> hint in-channels =", in_ch,
              "(base offsets+density=3; extras = in_ch-3 =", in_ch - 3, ")")
    # does it also carry denoiser weights?
    print("contains 'locked'/denoiser keys                    :",
          any(k.startswith("locked") for k in sd))
    print("-" * 70)
    print("sample keys:")
    for k in list(sd.keys())[:12]:
        print("  ", k, tuple(sd[k].shape))


if __name__ == "__main__":
    main()
