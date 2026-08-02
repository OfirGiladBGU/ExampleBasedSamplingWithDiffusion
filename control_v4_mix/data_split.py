"""Canonical train/val split, keyed ONLY on the source-folder images.

This reproduces the split in control_v4/train_control.py (torch.randperm(N, seed=42); train =
first (1 - val_split), val = the rest) but bases N on the SOURCE folder image listing alone -- NOT
on which oracles happen to be loaded. That gives two guarantees the multistyle runs need:
  * the val set is IDENTICAL to control_v4's (same held-out icons -> comparable metrics), and
  * it is STABLE as oracles are added/removed (WVS/GBN -> +DITHER -> +BNOT), because the split no
    longer depends on any oracle's offset availability.

Assumes every source image has offsets (true for the icon set: 10000/10000). If some source
images lacked offsets, control_v4 would have excluded them before splitting; pass a reference
offsets dir to `offsets_filter` to reproduce that exactly.
"""

import json
import os

import torch

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def list_source_files(source_dir, offsets_filter=None):
    """Sorted source image rel-paths (with extension). If offsets_filter (a dir of .npy) is given,
    keep only stems that have a matching offset file -- exactly control_v4's intersection."""
    rels = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXT:
                rels.append(os.path.relpath(os.path.join(root, f), source_dir))
    rels = sorted(rels)
    if offsets_filter and os.path.isdir(offsets_filter):
        off = set()
        for root, _, files in os.walk(offsets_filter):
            for f in files:
                if f.endswith(".npy"):
                    off.add(os.path.splitext(os.path.relpath(os.path.join(root, f), offsets_filter))[0])
        rels = [r for r in rels if os.path.splitext(r)[0] in off]
    return rels


def source_train_val_split(source_dir, val_split=0.1, seed=42, offsets_filter=None):
    """Return (train_files, val_files) as source rel-paths. Matches control_v4's split method."""
    files = list_source_files(source_dir, offsets_filter=offsets_filter)
    n = len(files)
    val_len = min(max(int(n * val_split), 0), max(n - 1, 0))
    train_len = n - val_len
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    train = [files[i] for i in order[:train_len]]
    val = [files[i] for i in order[train_len:]]
    return train, val


def split_from_manifest(source_dir, manifest_path):
    """Val = source images whose BASENAME is listed in manifest_path (a JSON list of basenames);
    train = the rest. This reproduces an explicit reference split EXACTLY, independent of the
    source folder's structure or which oracles are loaded. Basenames are matched with extension.
    """
    with open(manifest_path) as fh:
        val_names = set(json.load(fh))
    files = list_source_files(source_dir)
    train, val = [], []
    for rel in files:
        (val if os.path.basename(rel) in val_names else train).append(rel)
    return train, val
