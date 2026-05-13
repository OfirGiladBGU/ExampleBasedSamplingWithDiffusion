"""Concatenate evaluation panels side-by-side with separator lines.

This script loads eval_panel.png from each result directory and concatenates
them horizontally, with a vertical line separator between specified groups.

Usage:
    python concat_eval_panels.py
    python concat_eval_panels.py --show-headers
    python concat_eval_panels.py --show-headers --label-map '{"sdedit":"SDEdit"}'
"""

import argparse
import json
import os
from PIL import Image, ImageDraw, ImageFont

# Configuration
RESULT_DIR_LIST = [
    "condition",
    "vanilla",
    "gecco",
    "agi",
    "full",
    "sdedit",
    "sdedit_resample",
]

# Groups: separator appears BEFORE entries in the second group
GROUPS = [
    ["condition"],
    ["vanilla", "gecco", "agi", "full"],
    ["sdedit", "sdedit_resample"],
]

BASE_DIR = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4"
OUTPUT_PATH = os.path.join(BASE_DIR, "eval_panels_combined.png")

# Separator line width and color
SEPARATOR_WIDTH = 4  # pixels
SEPARATOR_COLOR = (0, 0, 0)  # black

# Optional top header configuration
SHOW_HEADERS = True
HEADER_HEIGHT = 56
HEADER_BG_COLOR = (255, 255, 255)
HEADER_TEXT_COLOR = (0, 0, 0)
HEADER_FONT_SIZE = 52

# Optional row filtering
EXISTING_ROWS_COUNT = 10
SELECTED_ROWS_LIST = [1, 2, 4, 7]

# Default labels (can be overridden with --label-map / --label-map-file)
RESULT_LABEL_MAP = {
    "condition": "Target",
    "vanilla": "VANILLA",
    "gecco": "GECCO",
    "agi": "GATED",
    "full": "FULL",
    "sdedit": "SDEdit",
    "sdedit_resample": "SDEdit + Resample",
}


def load_panel(result_dir):
    """Load eval_panel.png for a given result directory."""
    panel_path = os.path.join(BASE_DIR, f"eval_outputs_{result_dir}", "eval_panel.png")
    if not os.path.isfile(panel_path):
        raise FileNotFoundError(f"Panel not found: {panel_path}")
    return Image.open(panel_path).convert("RGB")


def create_separator(height):
    """Create a vertical separator line."""
    return Image.new("RGB", (SEPARATOR_WIDTH, height), SEPARATOR_COLOR)


def _resolve_labels(label_map_overrides):
    labels = dict(RESULT_LABEL_MAP)
    labels.update(label_map_overrides)
    return labels


def _default_label(result_dir):
    return result_dir.replace("_", " ").title()


def _parse_label_overrides(args):
    merged = {}
    if args.label_map_file:
        with open(args.label_map_file, "r", encoding="utf-8") as f:
            from_file = json.load(f)
        if not isinstance(from_file, dict):
            raise ValueError("--label-map-file must contain a JSON object")
        merged.update({str(k): str(v) for k, v in from_file.items()})

    if args.label_map:
        from_inline = json.loads(args.label_map)
        if not isinstance(from_inline, dict):
            raise ValueError("--label-map must be a JSON object")
        merged.update({str(k): str(v) for k, v in from_inline.items()})
    return merged


def _ordered_items():
    ordered = []
    for group_idx, group in enumerate(GROUPS):
        for result_dir in group:
            ordered.append(("panel", result_dir))

        if group_idx < len(GROUPS) - 1:
            ordered.append(("separator", None))
    return ordered


def _parse_selected_rows(raw_selected_rows):
    if isinstance(raw_selected_rows, str):
        selected_rows = json.loads(raw_selected_rows)
    else:
        selected_rows = raw_selected_rows

    if not isinstance(selected_rows, list) or not selected_rows:
        raise ValueError("--selected-rows-list must be a non-empty JSON list of row indices")
    if not all(isinstance(row_idx, int) for row_idx in selected_rows):
        raise ValueError("--selected-rows-list must contain only integers")
    if any(row_idx < 0 for row_idx in selected_rows):
        raise ValueError("--selected-rows-list cannot contain negative indices")
    return selected_rows


def _select_panel_rows(img, existing_rows_count, selected_rows):
    if existing_rows_count <= 0:
        raise ValueError("--existing-rows-count must be > 0")

    if any(row_idx >= existing_rows_count for row_idx in selected_rows):
        raise ValueError(
            f"selected row index out of bounds for existing rows count {existing_rows_count}: {selected_rows}"
        )

    width, height = img.size
    bounds = [round(i * height / existing_rows_count) for i in range(existing_rows_count + 1)]

    row_crops = []
    for row_idx in selected_rows:
        top = bounds[row_idx]
        bottom = bounds[row_idx + 1]
        row_crops.append(img.crop((0, top, width, bottom)))

    out_height = sum(crop.height for crop in row_crops)
    out_img = Image.new("RGB", (width, out_height))
    y_offset = 0
    for crop in row_crops:
        out_img.paste(crop, (0, y_offset))
        y_offset += crop.height
    return out_img


def _load_header_font(font_size):
    """Load a scalable font for headers, fallback to PIL default if unavailable."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=font_size)
    except Exception:
        return ImageFont.load_default()


def _with_headers(combined, placements, labels, header_font_size):
    canvas = Image.new(
        "RGB",
        (combined.width, combined.height + HEADER_HEIGHT),
        HEADER_BG_COLOR,
    )
    canvas.paste(combined, (0, HEADER_HEIGHT))

    draw = ImageDraw.Draw(canvas)
    font = _load_header_font(header_font_size)

    for placement in placements:
        if placement["kind"] != "panel":
            continue

        result_dir = placement["result_dir"]
        label = labels.get(result_dir, _default_label(result_dir))
        left = placement["x"]
        width = placement["width"]
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = left + max((width - text_w) // 2, 0)
        y = max((HEADER_HEIGHT - text_h) // 2, 0)
        draw.text((x, y), label, fill=HEADER_TEXT_COLOR, font=font)

    return canvas


def concatenate_panels(
    show_headers=False,
    label_map_overrides=None,
    header_font_size=HEADER_FONT_SIZE,
    existing_rows_count=EXISTING_ROWS_COUNT,
    selected_rows_list=None,
):
    """Load all panels and concatenate them with separators and optional headers."""
    if label_map_overrides is None:
        label_map_overrides = {}
    if selected_rows_list is None:
        selected_rows_list = list(SELECTED_ROWS_LIST)

    labels = _resolve_labels(label_map_overrides)
    ordered = _ordered_items()

    elements = []
    placements = []
    panel_height = None

    for kind, result_dir in ordered:
        if kind == "panel":
            print(f"Loading {result_dir}...", end=" ")
            img = load_panel(result_dir)
            img = _select_panel_rows(img, existing_rows_count, selected_rows_list)
            print(f"✓ {img.size}")

            if panel_height is None:
                panel_height = img.height
            elif img.height != panel_height:
                raise ValueError(
                    f"Image height mismatch: expected {panel_height}, got {img.height}"
                )

            elements.append(img)
            placements.append(
                {
                    "kind": "panel",
                    "result_dir": result_dir,
                    "width": img.width,
                }
            )
        else:
            if panel_height is None:
                raise ValueError("Cannot add separator before loading any panel")
            sep = create_separator(panel_height)
            elements.append(sep)
            placements.append({"kind": "separator", "result_dir": None, "width": sep.width})

    if not elements:
        raise ValueError("No panels to concatenate")

    total_width = sum(img.width for img in elements)
    combined = Image.new("RGB", (total_width, panel_height))

    x_offset = 0
    for idx, img in enumerate(elements):
        combined.paste(img, (x_offset, 0))
        placements[idx]["x"] = x_offset
        x_offset += img.width

    if show_headers:
        combined = _with_headers(combined, placements, labels, header_font_size)

    return combined


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-headers",
        action=argparse.BooleanOptionalAction,
        default=SHOW_HEADERS,
        help="Render a header row with labels above each panel",
    )
    parser.add_argument(
        "--label-map",
        default="",
        help="Inline JSON mapping result dir to label, e.g. '{\"sdedit\":\"SDEdit\"}'",
    )
    parser.add_argument(
        "--label-map-file",
        default="",
        help="Path to JSON file containing {result_dir: label} mapping",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help="Output image path",
    )
    parser.add_argument(
        "--header-font-size",
        type=int,
        default=HEADER_FONT_SIZE,
        help="Header text font size in pixels",
    )
    parser.add_argument(
        "--existing-rows-count",
        type=int,
        default=EXISTING_ROWS_COUNT,
        help="How many rows each source panel currently has",
    )
    parser.add_argument(
        "--selected-rows-list",
        default=json.dumps(SELECTED_ROWS_LIST),
        help="JSON list of row indices to keep, e.g. '[1,2,4,7]' (first row is 0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    label_map_overrides = _parse_label_overrides(args)
    selected_rows_list = _parse_selected_rows(args.selected_rows_list)

    if args.header_font_size <= 0:
        raise ValueError("--header-font-size must be > 0")
    if args.existing_rows_count <= 0:
        raise ValueError("--existing-rows-count must be > 0")

    print(f"Concatenating panels from {len(RESULT_DIR_LIST)} result directories...")
    print(f"Headers: {args.show_headers}")
    print(f"Header font size: {args.header_font_size}")
    print(f"Existing rows count: {args.existing_rows_count}")
    print(f"Selected rows list: {selected_rows_list}")
    print(f"Output: {args.output}\n")

    try:
        combined = concatenate_panels(
            show_headers=args.show_headers,
            label_map_overrides=label_map_overrides,
            header_font_size=args.header_font_size,
            existing_rows_count=args.existing_rows_count,
            selected_rows_list=selected_rows_list,
        )
        combined.save(args.output)
        print(f"\n✓ Saved combined panel: {args.output}")
        print(f"  Final size: {combined.size}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
