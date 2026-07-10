"""Architecture identity: stamp it into the checkpoint, refuse to load a mismatch.

Why this exists
---------------
The conditioning architecture, the FM coupling and the Tier-2 attention setting all change
*what weights exist*. If an inference script's default drifts from the training script's,
the model is rebuilt wrong and ``load_state_dict(..., strict=False)`` silently loads almost
nothing -- you sample from a near-random network and it looks like a model failure.

This has already happened twice in this codebase (``fm_coupling`` defaulted to smartinit and
``conditioning`` to spade in sample_control.py, while training used gaussian + controlnet).
Aligning defaults by hand does not hold: they drift again. So the checkpoint carries its own
architecture and the loader asserts against it.

Deliberately EXCLUDED from the check
------------------------------------
``grid_size``   -- the model is fully convolutional with no learned positional embeddings, so
                   sampling at a different grid than training is a *feature* (G=8..112).
``ode_method`` / ``ode_steps`` / ``eta`` / ``truncation_ratio``
                -- inference-time knobs; they change results but not which weights exist.
"""

ARCH_KEYS = (
    "conditioning",
    "fm_coupling",
    "attn_middle",
    "attn_heads",
    "enable_gecco",
    "enable_adaptive_gate_injection",
    "smart_init_features",
    "sdf_features",
    "batch_coords_features",
    "concat_smart_init_grid",
)


def arch_from(source):
    """Collect the architecture-identity fields from an argparse Namespace or a dict."""
    get = source.get if isinstance(source, dict) else (lambda k, d=None: getattr(source, k, d))
    from control_fm_v2.flow_matching import get_model_config

    arch = {k: get(k) for k in ARCH_KEYS}
    arch["model_config"] = get_model_config()
    return arch


def assert_arch_matches(state, requested, strict=True, ckpt_path=""):
    """Compare a checkpoint's stamped architecture against what the caller is building.

    Returns the stamped arch (or None when the checkpoint predates the stamp).
    Raises on mismatch when ``strict`` -- otherwise prints a loud warning.
    """
    stamped = state.get("arch") if isinstance(state, dict) else None
    if not stamped:
        print(
            f"  [arch] WARNING: checkpoint {ckpt_path or ''} carries no architecture stamp "
            "(written before this guard). Cannot verify that the network being built matches "
            "the trained one."
        )
        return None

    req = arch_from(requested)
    diffs = []
    for key in ARCH_KEYS:
        want, have = req.get(key), stamped.get(key)
        if have is not None and want is not None and want != have:
            diffs.append((key, have, want))

    mc_have, mc_want = stamped.get("model_config"), req.get("model_config")
    if mc_have and mc_want and mc_have != mc_want:
        for k in sorted(set(mc_have) | set(mc_want)):
            if mc_have.get(k) != mc_want.get(k):
                diffs.append((f"model_config.{k}", mc_have.get(k), mc_want.get(k)))

    if not diffs:
        return stamped

    lines = "\n".join(f"    {k:36s} checkpoint={h!r:20s} requested={w!r}" for k, h, w in diffs)
    msg = (
        f"Architecture mismatch against checkpoint {ckpt_path or ''}:\n{lines}\n"
        "  The network would be rebuilt differently from the one that was trained, and\n"
        "  load_state_dict(strict=False) would silently drop the non-matching weights.\n"
        "  Fix the flags above (they must match training), or pass strict_arch=False to\n"
        "  override deliberately."
    )
    if strict:
        raise RuntimeError(msg)
    print("  [arch] WARNING (strict_arch=False):\n" + msg)
    return stamped

def resolve_arch(state, requested, prefer_checkpoint=True, strict=True, ckpt_path=""):
    """Return the architecture to actually BUILD.

    The checkpoint's stamp describes the weights that exist, so it is authoritative. When
    ``prefer_checkpoint`` it wins over the caller's flags, and any difference is printed
    (never applied silently). This is what keeps inference defaults from drifting away from
    whichever experiment block is currently active in train_control.py.

    With ``prefer_checkpoint=False`` the caller's flags are used and merely *validated*.
    """
    req = arch_from(requested)
    stamped = state.get("arch") if isinstance(state, dict) else None
    if not stamped:
        print(
            f"  [arch] WARNING: checkpoint {ckpt_path or ''} carries no architecture stamp; "
            "building from the supplied flags and hoping they match."
        )
        return req
    if not prefer_checkpoint:
        assert_arch_matches(state, requested, strict=strict, ckpt_path=ckpt_path)
        return req

    eff = dict(req)
    adopted = []
    for key in ARCH_KEYS + ("model_config",):
        have = stamped.get(key)
        if have is None:
            continue
        if req.get(key) != have:
            adopted.append((key, req.get(key), have))
        eff[key] = have
    if adopted:
        print(f"  [arch] adopting architecture from {ckpt_path or 'checkpoint'} "
              f"(overriding {len(adopted)} flag(s)):")
        for k, was, now in adopted:
            print(f"    {k:34s} flag={was!r:16s} -> checkpoint={now!r}")
    return eff
