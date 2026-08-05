"""Central oracle registry for the multi-style axis.

Oracle NAME -> data-root defaults live here so the command line only needs names, e.g.
    --oracles "WVS;GBN;DITHER"
Each entry may still be overridden inline as NAME:/custom/root. The list ORDER fixes the one-hot
index (WVS=0, GBN=1, DITHER=2, ...), so keep it consistent between training and eval.
Each root must contain source/, target/ (stipple ~1024 dots), and processed_offsets/.
"""

ORACLES = [
    ("WVS", "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512_WVS"),
    ("GBN", "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512_GBN"),
    ("DITHER", "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_DITHER"),
    # ("BNOT", "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_BNOT"),
]
ORACLES_MAP = dict(ORACLES)
# Default is NAMES only, so the CLI stays path-free.
ORACLES_DEFAULT = ";".join(n for n, _ in ORACLES)


def resolve_oracles(spec):
    """Parse a --oracles spec into an ordered list of (name, root).

    Each ';'-separated entry is either a bare NAME (looked up in ORACLES_MAP) or NAME:/root
    (explicit override). Raises on an unknown bare name.
    """
    out = []
    for entry in [e for e in spec.split(";") if e.strip()]:
        if ":" in entry:
            name, root = entry.split(":", 1)
            out.append((name.strip(), root.strip()))
        else:
            name = entry.strip()
            if name not in ORACLES_MAP:
                raise ValueError(
                    f"Unknown oracle '{name}'. Known names: {list(ORACLES_MAP)}. "
                    "Pass NAME:/path to use a custom root."
                )
            out.append((name, ORACLES_MAP[name]))
    return out
