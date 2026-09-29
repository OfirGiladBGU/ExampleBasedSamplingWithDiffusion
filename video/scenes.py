"""Manim scenes for every shot type in video/shots.json.

A shot is one ShotScene. Everything on screen is a function of the shot clock t (seconds
since the shot started): each mobject gets an updater that sets its state from t, and the
scene then simply waits for the shot duration. This keeps every shot exactly as long as its
narration (stage 3 passes the duration in) and makes shot types easy to add.

Shot types (visual.type):
    title        centered title and subtitle
    wvs          target image + Lloyd iterations of Weighted Voronoi Stippling
    figure       one or more crops of a paper figure, side by side, with labels
    pipeline     the pipeline figure, with timed highlight boxes or a camera zoom to a region
    trajectory   our sampler's reverse process, one or more panels (+ targets)
    tone         tone-correction optimization: requested density, rendered stipple, error

Point sets are rasterized with PIL (anti-aliased discs) into image mobjects, which is far
faster than thousands of Dot mobjects and looks like the paper's scatter plots.
"""

from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, ORIGIN, RIGHT, UP, Group, Line, VGroup, ImageMobject, MovingCameraScene,
                   Rectangle, RoundedRectangle, Text, VMobject, config, smooth)
from PIL import Image, ImageDraw

from common import FIG_DIR, RESULTS_DIR, TRAJ_DIR, subtitle_timeline

SUPERSAMPLE = 3                 # point discs are drawn at 3x and downsampled (anti-aliasing)
FADE_S = 0.5                    # fade-in time of the shot's content
SUBTITLE_HEIGHT = 1.05          # frame units reserved at the bottom for subtitles


# ── small helpers ────────────────────────────────────────────────────────────────────

def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def px_per_unit():
    return config.pixel_height / config.frame_height


def ramp(t, t0, t1):
    """0 before t0, 1 after t1, smooth in between."""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return float(smooth(np.clip((t - t0) / (t1 - t0), 0.0, 1.0)))


def set_alpha(mob, a):
    """Scale a mobject's visibility to a in [0,1], relative to its own opacities.

    Manim's set_opacity() forces fill and stroke to the same value, which turns an outline
    (fill 0) into a filled shape; this keeps each part's original fill/stroke opacity."""
    for m in mob.get_family():
        if isinstance(m, ImageMobject):
            m.set_opacity(a)
            continue
        if not isinstance(m, VMobject):       # plain Groups carry no opacity of their own
            continue
        if not hasattr(m, "_base_alpha"):
            m._base_alpha = (m.get_fill_opacity(), m.get_stroke_opacity())
        f, st = m._base_alpha
        m.set_fill(opacity=f * a, family=False)
        m.set_stroke(opacity=st * a, family=False)
    return mob


def gray_to_rgb(img01):
    """(H,W) float in [0,1] -> (H,W,3) uint8."""
    g = np.clip(np.asarray(img01, np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def render_points(pts, size_px, radius_px, color=(26, 26, 26), background=(255, 255, 255)):
    """(N,2) points in the unit square, y down -> (S,S,3) uint8 scatter image."""
    s = SUPERSAMPLE
    canvas = Image.new("RGB", (size_px * s, size_px * s), background)
    draw = ImageDraw.Draw(canvas)
    r = max(radius_px * s, 1.0)
    xy = np.asarray(pts, np.float64) * (size_px * s)
    for x, y in xy:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    return np.asarray(canvas.resize((size_px, size_px), Image.LANCZOS))


def render_darkness(coords, size_px, sigma_frac, gain, chunk=8192):
    """The tone experiment's renderer, at any resolution: Gaussian dots of std `sigma_frac`
    (fraction of the canvas), Beer-Lambert darkness 1 - exp(-gain * ink). Same model as
    experiments/tone_results_utils.render_darkness, evaluated on a finer pixel grid."""
    ax = (np.arange(size_px, dtype=np.float32) + 0.5) / size_px
    gx, gy = np.meshgrid(ax, ax)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    c = np.asarray(coords, np.float32)
    inv = -0.5 / (sigma_frac * sigma_frac)
    ink = np.empty(len(grid), np.float32)
    for a in range(0, len(grid), chunk):
        g = grid[a:a + chunk]
        d2 = ((g[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        ink[a:a + chunk] = np.exp(inv * d2).sum(1)
    return 1.0 - np.exp(-float(gain) * ink.reshape(size_px, size_px))


def render_ink_dots(coords, size_px, radius_frac, alpha, supersample=3):
    """Stipple drawn as sharp discs, the paper figure's depiction (tone_results_stage_3_plots.
    show_points): each disc has opacity `alpha` = 1 - exp(-ink_gain), and overlapping discs
    multiply transmittance, the same Beer-Lambert form as the renderer. Returns darkness in [0,1)."""
    s = int(supersample)
    n = size_px * s
    r = radius_frac * n
    rr = int(np.ceil(r)) + 1
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    count = np.zeros((n + 2 * rr + 2, n + 2 * rr + 2), np.float32)
    for x, y in np.asarray(coords, np.float64) * n:
        cx, cy = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - cx, y - cy
        disc = (((xx - fx + 0.5) ** 2 + (yy - fy + 0.5) ** 2) <= r * r).astype(np.float32)
        count[cy:cy + 2 * rr + 1, cx:cx + 2 * rr + 1] += disc
    count = count[rr:rr + n, rr:rr + n]
    dark = 1.0 - (1.0 - float(alpha)) ** count
    img = Image.fromarray(np.clip(dark * 255.0 + 0.5, 0, 255).astype(np.uint8))
    return np.asarray(img.resize((size_px, size_px), Image.LANCZOS), np.float32) / 255.0


def dot_radius(n_points, size_px):
    """Disc radius that reads like the paper figures at any point count."""
    spacing = size_px / max(np.sqrt(n_points), 1.0)
    return float(np.clip(0.16 * spacing, 0.9, 3.2))


def image_mobject(arr, height):
    """Image mobject shown `height` frame units tall, pre-resized to its exact on-screen pixels.

    Manim draws images with a perspective transform that does not low-pass filter, so a large
    image shrunk by the renderer aliases badly (broken thin lines and text). Resizing here with
    Lanczos first means the renderer only ever draws at about 1:1."""
    arr = np.ascontiguousarray(arr)
    h_px = max(1, int(round(height * px_per_unit())))
    w_px = max(1, int(round(h_px * arr.shape[1] / arr.shape[0])))
    if (arr.shape[0], arr.shape[1]) != (h_px, w_px):
        arr = np.asarray(Image.fromarray(arr).resize((w_px, h_px), Image.LANCZOS))
    m = ImageMobject(arr)
    m.height = height
    return m


def placeholder(height, message, color):
    """Stands in for data that stage 1 has not produced yet, so every shot still renders."""
    box = Rectangle(width=height, height=height, stroke_color=color, stroke_width=2)
    txt = Text(message, font_size=18, color=color).scale_to_fit_width(height * 0.9)
    return Group(box, txt)


def load_npz(name):
    p = TRAJ_DIR / f"{name}.npz"
    return dict(np.load(p)) if p.exists() else None


def load_result(name):
    """Real final result from stage_1_results.py (used while no trajectory file exists)."""
    p = RESULTS_DIR / f"{name}.npz"
    return dict(np.load(p)) if p.exists() else None


class Clock(Group):
    """Carries the shot time; its (dt) updater also makes Manim render every frame."""

    def __init__(self):
        super().__init__()
        self.t = 0.0
        self.add_updater(self._tick)

    def _tick(self, mob, dt):
        self.t += dt


# ── the scene ────────────────────────────────────────────────────────────────────────

class ShotScene(MovingCameraScene):
    """One shot. Stage 3 sets `shot`, `shot_len`, `speech_s` and `settings` before render()."""

    shot = None
    shot_len = 4.0
    speech_s = 0.0
    settings = None
    prev_zoom = None            # zoom box of the previous pipeline shot, for continuous camera moves
    line_spans = None           # (start, end) of each narration line in speech seconds, from stage 2

    # colors / fonts from settings
    def _style(self):
        s = self.settings
        self.fg = s["text_color"]
        self.accent = s["accent_color"]
        self.font = s["font"]

    def text(self, s, size=30, color=None, weight="NORMAL"):
        return Text(s, font=self.font, font_size=size, color=color or self.fg, weight=weight)

    def fade_in(self, mob, t0=0.0, t1=FADE_S):
        """Fade a mobject in over [t0, t1] of the shot clock."""
        set_alpha(mob, 0.0)

        def upd(m, dt):
            set_alpha(m, ramp(self.clock.t, t0, t1))
        mob.add_updater(upd)
        return mob

    def construct(self):
        self._style()
        self.clock = Clock()
        self.add(self.clock)
        self.add(self.camera.frame)
        visual = self.shot["visual"]
        getattr(self, f"shot_{visual['type']}")(visual)
        if self.settings.get("subtitles", True):
            self.add_subtitles()
        self.wait(self.shot_len)

    # content area above the subtitles
    def content_height(self):
        reserve = SUBTITLE_HEIGHT if self.settings.get("subtitles", True) else 0.0
        return config.frame_height - reserve - 0.5

    def content_center(self):
        reserve = SUBTITLE_HEIGHT if self.settings.get("subtitles", True) else 0.0
        return UP * (reserve / 2.0)

    # ── subtitles ────────────────────────────────────────────────────────────────────
    def add_subtitles(self):
        text = (self.shot.get("narration") or "").strip()
        if not text:
            return
        lead = self.settings["lead_in_s"]
        timeline, _ = subtitle_timeline(text, self._speech(), self.line_spans)
        chunks = [c for _, _, c in timeline]
        # each chunk shows from its start until the next one starts (the last one until the end)
        bounds = lead + np.array([a for a, _, _ in timeline] + [self.shot_len + 1.0 - lead])
        bottom = DOWN * (config.frame_height / 2 - SUBTITLE_HEIGHT / 2 - 0.05)
        mobs = []
        for c in chunks:
            m = self.text(c, size=26, color=self.fg)
            if m.width > config.frame_width - 1.0:
                m.scale_to_fit_width(config.frame_width - 1.0)
            m.move_to(bottom)
            set_alpha(m, 0.0)
            mobs.append(m)
            self.add(m)

        def upd(_, dt):
            t = self.clock.t
            for i, m in enumerate(mobs):
                on = bounds[i] <= t < (bounds[i + 1] if i < len(mobs) - 1 else self.shot_len + 1)
                set_alpha(m, 1.0 if on else 0.0)
        holder = Group()
        holder.add_updater(upd)
        self.add(holder)
        # subtitles stay fixed on screen while the camera zooms: follow and scale with the frame
        frame = self.camera.frame
        for m in mobs:
            base_h = m.height

            def follow(m, dt, base_h=base_h):
                s = frame.width / config.frame_width
                m.set(height=base_h * s)
                m.move_to(frame.get_center() + bottom * s)
            m.add_updater(follow)

    def _speech(self):
        return self.speech_s or max(self.shot_len - self.settings["lead_in_s"] - self.settings["tail_s"], 1.0)

    def line_span(self, k):
        """(start, end) of narration line k on the shot clock."""
        _, spans = subtitle_timeline(self.shot.get("narration", ""), self._speech(), self.line_spans)
        a, b = spans[min(k, len(spans) - 1)]
        return self.settings["lead_in_s"] + a, self.settings["lead_in_s"] + b

    def line_end(self, k):
        """Shot-clock time at which narration line k ends."""
        return self.line_span(k)[1]

    # ── title ────────────────────────────────────────────────────────────────────────
    def shot_title(self, v):
        title = self.text(v["title"], size=64, weight="BOLD")
        parts = [title]
        if v.get("subtitle"):
            parts.append(self.text(v["subtitle"], size=34, color=self.accent))
        for extra in v.get("lines", []):
            parts.append(self.text(extra, size=26))
        g = Group(*parts).arrange(DOWN, buff=0.45).move_to(self.content_center())
        if g.width > config.frame_width - 1.0:
            g.scale_to_fit_width(config.frame_width - 1.0)
        for i, m in enumerate(parts):
            self.fade_in(m, 0.15 * i, 0.15 * i + 0.8)
            self.add(m)

    # ── figure crops ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _tile_grid(tiles, cols, gap_frac=0.08):
        """Arrange equally sized RGB tiles in a grid with white gaps."""
        th, tw = tiles[0].shape[:2]
        g = int(gap_frac * th)
        rows = int(np.ceil(len(tiles) / cols))
        out = np.full((rows * th + (rows - 1) * g, cols * tw + (cols - 1) * g, 3), 255, np.uint8)
        for k, t in enumerate(tiles):
            r, c = divmod(k, cols)
            out[r * (th + g):r * (th + g) + th, c * (tw + g):c * (tw + g) + tw] = t[:th, :tw]
        return out

    def _figure_item(self, c, v, height):
        """One item of a figure shot, as an RGB array: a whole image file ("source"), real point
        samples ("points", from stage 1 results), or a crop of a paper figure. With "split": n the
        crop is cut into n equal columns; "grid_cols" arranges samples/columns in a grid."""
        if "source" in c:
            return np.asarray(Image.open(c["source"]).convert("RGB"))
        cols = int(c.get("grid_cols", 0))
        if "points" in c:
            r = load_result(c["points"])
            if r is not None:
                n = len(r["samples"])
                per_row = cols or n
                rows = int(np.ceil(n / per_row))
                px = int(round(height / (rows + 0.08 * (rows - 1)) * px_per_unit()))
                tiles = [render_points(pts, px, dot_radius(len(pts), px)) for pts in r["samples"]]
                return self._tile_grid(tiles, per_row)
        img = Image.open(FIG_DIR / c.get("image", v.get("image", ""))).convert("RGB")
        w, h = img.size
        x0, y0, x1, y1 = c.get("box", [0, 0, 1, 1])
        arr = np.asarray(img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))))
        n = int(c.get("split", 1))
        if n > 1:
            edges = np.linspace(0, arr.shape[1], n + 1).astype(int)
            tiles = [arr[:, edges[k]:edges[k + 1]] for k in range(n)]
            wmin = min(t.shape[1] for t in tiles)
            arr = self._tile_grid([t[:, :wmin] for t in tiles], cols or n)
        return arr

    def _animated_samples(self, r, height, cols):
        """Grid of point panels playing a recorded trajectory (K,S,N,2) from pure noise."""
        traj, steps = r["trajectory"], r["steps"].astype(np.float64)
        n = traj.shape[1]
        rows = int(np.ceil(n / cols))
        gap = 0.18
        size = (height - gap * (rows - 1)) / rows
        times = steps / max(steps[-1], 1.0)
        t0, t1 = FADE_S + 0.4, self.shot_len - 1.2
        state = {"k": 0}

        def on_frame(k, a):
            state["k"] = min(k + (1 if a > 0.5 else 0), len(steps) - 1)
        panels = []
        for j in range(n):
            panel, border = self._point_panel(traj[:, j], size, times, t0, t1, on_frame if j == 0 else None)
            panels.append((panel, border))
        grid = Group(*[p for p, _ in panels]).arrange_in_grid(rows=rows, cols=cols, buff=gap)
        t_last = int(r["t_start"]) if "t_start" in r else int(steps[-1])
        counter = lambda: f"reverse diffusion   t = {max(t_last - int(steps[state['k']]), 0)}"  # noqa: E731
        return grid, [b for _, b in panels], counter

    def shot_figure(self, v):
        row_h = self.content_height() - 0.9
        items_spec = v.get("crops") or [{"box": [0, 0, 1, 1], "label": ""}]
        animated = {}
        for i, c in enumerate(items_spec):
            r = load_result(c["points"]) if "points" in c else None
            if r is not None and "trajectory" in r:
                animated[i] = self._animated_samples(r, row_h, int(c.get("grid_cols", 2)))
        crops = [(None if i in animated else self._figure_item(c, v, row_h), c.get("label", ""))
                 for i, c in enumerate(items_spec)]
        items, borders, counters = [], [], []
        for i, (arr, label) in enumerate(crops):
            if i in animated:
                m, b, counter = animated[i]
                borders += b
                counters.append(counter)
            else:
                m = image_mobject(arr, row_h)
            cap = self.text(label, size=26) if label else None
            items.append(Group(m, cap).arrange(UP, buff=0.25) if cap else m)
        g = Group(*items).arrange(RIGHT, buff=0.5)
        if g.width > config.frame_width - 0.8:
            g.scale_to_fit_width(config.frame_width - 0.8)
        g.move_to(self.content_center())
        for i, it in enumerate(items):
            self.fade_in(it, 0.3 * i, 0.3 * i + FADE_S)
            self.add(it)
        for b in borders:
            self.fade_in(b)
            self.add(b)
        for counter in counters:
            self.add(self._counter("{}", g.get_bottom() + DOWN * 0.35, counter))

    # ── pipeline figure: highlights or zoom ──────────────────────────────────────────
    def _pipeline_source(self, v):
        path = FIG_DIR / v["image"]
        if not path.exists():
            self.add(placeholder(4, f"run stage_1_figures.py ({v['image']})", self.fg))
            return None
        return Image.open(path).convert("RGB")

    def _box_to_frame(self, fig, box):
        """Normalized image box (x0,y0,x1,y1; y down) -> (center, width, height) in frame units."""
        x0, y0, x1, y1 = box
        left, top = fig.get_left()[0], fig.get_top()[1]
        cx = left + (x0 + x1) / 2 * fig.width
        cy = top - (y0 + y1) / 2 * fig.height
        return np.array([cx, cy, 0.0]), (x1 - x0) * fig.width, (y1 - y0) * fig.height

    def shot_pipeline(self, v):
        src = self._pipeline_source(v)
        if src is None:
            return
        if "zoom" in v:
            self._pipeline_zoom(src, v)
            return
        sw, sh = src.size
        scale = min((config.frame_width - 0.4) / sw, self.content_height() / sh)
        fig = image_mobject(np.asarray(src), sh * scale).move_to(self.content_center())
        self.add(fig)
        self.fade_in(fig)
        for hl in v.get("highlights", []):
            self._pipeline_highlight(fig, hl)

    def _pipeline_zoom(self, src, v):
        """Camera move over the figure, rendered by cropping the full-resolution source every
        frame (PIL Lanczos), so zoomed-in views use the original pixels instead of an enlarged,
        aliased copy. The view slides from the previous shot's region to this one's."""
        reserve = SUBTITLE_HEIGHT if self.settings.get("subtitles", True) else 0.0
        w_u, h_u = config.frame_width, config.frame_height - reserve
        w_px, h_px = int(round(w_u * px_per_unit())), int(round(h_u * px_per_unit()))
        aspect = w_px / h_px
        sw, sh = src.size
        # white margin around the source, so views centred near an edge never read outside it
        pad_x = int(max(sh * aspect - sw, 0) / 2 + 0.25 * sw)
        pad_y = int(max(sw / aspect - sh, 0) / 2 + 0.25 * sh)
        canvas = Image.new("RGB", (sw + 2 * pad_x, sh + 2 * pad_y), self.settings["background"])
        canvas.paste(src, (pad_x, pad_y))

        def view(box):
            """(center x, center y, width) in source pixels for a normalized box (None = whole figure)."""
            if box is None:
                return sw / 2, sh / 2, max(sw, sh * aspect) * 1.02
            x0, y0, x1, y1 = box[0] * sw, box[1] * sh, box[2] * sw, box[3] * sh
            w = max(x1 - x0, (y1 - y0) * aspect) * 1.04
            # centred on the region even past the figure's edge: the canvas is padded with white
            return (x0 + x1) / 2, (y0 + y1) / 2, w

        a0, a1 = view(self.prev_zoom), view(v["zoom"])
        move = float(v.get("move_s", 1.4))

        def view_at(t):
            """(center x, center y, width, height) of the view in source pixels."""
            a = ramp(t, 0.0, move)
            cx = a0[0] + (a1[0] - a0[0]) * a
            cy = a0[1] + (a1[1] - a0[1]) * a
            w = float(np.exp(np.log(a0[2]) + (np.log(a1[2]) - np.log(a0[2])) * a))   # even zoom speed
            w *= 1.0 - 0.03 * ramp(t, move, self.shot_len)                          # slow push-in
            return cx, cy, w, w / aspect

        def frame_at(t):
            cx, cy, w, h = view_at(t)
            box = (cx - w / 2 + pad_x, cy - h / 2 + pad_y, cx + w / 2 + pad_x, cy + h / 2 + pad_y)
            return np.asarray(canvas.resize((w_px, h_px), Image.LANCZOS, box=box))

        m = ImageMobject(frame_at(0.0))
        m.height = h_u
        m.move_to(UP * (reserve / 2.0))
        cache = {"t": None}

        def upd(mob, dt):
            t = round(self.clock.t, 4)
            if t == cache["t"]:
                return
            cache["t"] = t
            mob.pixel_array = np.dstack([frame_at(t), np.full((h_px, w_px), 255, np.uint8)])
        m.add_updater(upd)
        self.add(m)

        # boxes marking the part the narration is talking about, one narration line each
        center_y = reserve / 2.0
        for hl in v.get("highlights", []):
            self._zoom_highlight(hl, view_at, sw, sh, w_u, h_u, center_y)

    def _zoom_highlight(self, hl, view_at, sw, sh, w_u, h_u, center_y):
        """A rounded box around a normalized figure region, following the camera, visible while
        narration line hl["on_line"] is spoken."""
        start, end = self.line_span(int(hl.get("on_line", 0)))
        color = hl.get("color", self.accent)
        x0, y0, x1, y1 = hl["box"]
        pad = float(hl.get("pad", 0.006))
        x0, y0, x1, y1 = x0 - pad, y0 - pad * 3.1, x1 + pad, y1 + pad * 3.1   # figure is ~3.1:1

        def geometry(t):
            cx, cy, w, h = view_at(t)
            sx = lambda x: (x * sw - (cx - w / 2)) / w * w_u - w_u / 2           # noqa: E731
            sy = lambda y: center_y + h_u / 2 - (y * sh - (cy - h / 2)) / h * h_u  # noqa: E731
            return (sx(x0) + sx(x1)) / 2, (sy(y0) + sy(y1)) / 2, sx(x1) - sx(x0), sy(y0) - sy(y1)

        cxs, cys, ws, hs = geometry(0.0)
        rect = RoundedRectangle(corner_radius=0.1, width=ws, height=hs, stroke_color=color, stroke_width=7)
        rect.move_to([cxs, cys, 0])
        state = {"g": (cxs, cys, ws, hs)}

        def upd(r, dt):
            t = self.clock.t
            g = geometry(t)
            if any(abs(a - b) > 1e-4 for a, b in zip(g, state["g"])):
                r.become(RoundedRectangle(corner_radius=0.1, width=g[2], height=g[3],
                                          stroke_color=color, stroke_width=7).move_to([g[0], g[1], 0]))
                r.__dict__.pop("_base_alpha", None)
                for sm in r.get_family():
                    sm.__dict__.pop("_base_alpha", None)
                state["g"] = g
            set_alpha(r, ramp(t, start - 0.15, start + 0.25) * (1.0 - ramp(t, end, end + 0.35)))
        rect.add_updater(upd)
        set_alpha(rect, 0.0)
        self.add(rect)

    def _pipeline_highlight(self, fig, hl):
        c, w, h = self._box_to_frame(fig, hl["box"])
        color = hl.get("color", self.accent)
        if "on_line" in hl:
            # shown exactly while narration line k is spoken
            t0, t_off = self.line_span(int(hl["on_line"]))
        else:
            t0 = float(hl["at"]) * self.shot_len
            # "off_after_line": k -> fade out when narration line k has been spoken
            t_off = self.line_end(int(hl["off_after_line"])) if "off_after_line" in hl else None
        pad = float(hl.get("pad_units", 0.1))
        rect = RoundedRectangle(corner_radius=0.12, width=w + pad, height=h + pad,
                                stroke_color=color, stroke_width=6).move_to(c)
        parts = [rect]
        if hl.get("label"):
            label = self.text(hl["label"], size=26, color=color, weight="BOLD")
            # labels go outside the figure: above it for boxes in its upper half, below otherwise
            upper = c[1] > fig.get_center()[1]
            edge = fig.get_top() if upper else fig.get_bottom()
            label.move_to([c[0], edge[1], 0]).shift((UP if upper else DOWN) * (label.height / 2 + 0.15))
            label.add_background_rectangle(color=self.settings["background"], opacity=0.85, buff=0.08)
            parts.append(label)

        def upd(m, dt):
            t = self.clock.t
            on = ramp(t, t0, t0 + 0.4)
            nxt = self._next_highlight_time(hl)
            dim = 1.0 - 0.65 * ramp(t, nxt, nxt + 0.4) if nxt is not None else 1.0
            off = 1.0 - ramp(t, t_off, t_off + 0.4) if t_off is not None else 1.0
            set_alpha(m, on * dim * off)
        for m in parts:
            m.add_updater(upd)
            self.add(m)

    def _next_highlight_time(self, hl):
        """When the next "at"-timed highlight starts (earlier ones dim then); None if none."""
        if "at" not in hl:
            return None
        times = sorted(float(h["at"]) for h in self.shot["visual"].get("highlights", []) if "at" in h)
        later = [x for x in times if x > float(hl["at"])]
        return later[0] * self.shot_len if later else None

    # ── point-set animations (WVS and our sampler) ───────────────────────────────────
    def _point_panel(self, frames, size_units, times, t0, t1, label_fn=None):
        """An image mobject showing `frames` (K,N,2) interpolated over clock [t0, t1].

        `times` (K,) are the positions of the frames on a 0..1 progress axis."""
        size_px = int(round(size_units * px_per_unit()))
        radius = dot_radius(frames.shape[1], size_px)
        first = render_points(frames[0], size_px, radius)
        m = image_mobject(first, size_units)
        cache = {"key": None}

        def upd(mob, dt):
            p = ramp(self.clock.t, t0, t1) if t1 > t0 else 1.0
            k = int(np.clip(np.searchsorted(times, p, side="right") - 1, 0, len(times) - 2))
            span = max(times[k + 1] - times[k], 1e-9)
            a = float(np.clip((p - times[k]) / span, 0.0, 1.0))
            key = (k, round(a, 3))
            if key == cache["key"]:
                return
            cache["key"] = key
            pts = frames[k] * (1 - a) + frames[k + 1] * a
            mob.pixel_array = np.dstack([render_points(pts, size_px, radius),
                                         np.full((size_px, size_px), 255, np.uint8)])
            if label_fn is not None:
                label_fn(k, a)
        m.add_updater(upd)
        border = Rectangle(width=size_units, height=size_units, stroke_color="#BBBBBB", stroke_width=1.5)
        border.add_updater(lambda b, dt: b.move_to(m.get_center()))
        return m, border

    def _counter(self, fmt, position, values_fn):
        """Text that is rebuilt only when its value changes."""
        holder = Group()
        state = {"val": None, "mob": None}

        def upd(h, dt):
            val = values_fn()
            if val == state["val"]:
                if state["mob"] is not None and self.clock.t <= FADE_S + 0.1:   # finish the fade-in
                    set_alpha(state["mob"], ramp(self.clock.t, 0.0, FADE_S))
                return
            state["val"] = val
            new = self.text(fmt.format(val), size=26).move_to(position)
            if state["mob"] is None:
                state["mob"] = new
                h.add(new)
            else:
                # change the existing text in place: Manim fixes the list of drawn objects when
                # the shot starts, so a removed text would keep being drawn under the new one
                state["mob"].become(new)
            if hasattr(state["mob"], "_base_alpha"):
                for m in state["mob"].get_family():
                    m.__dict__.pop("_base_alpha", None)
            set_alpha(state["mob"], ramp(self.clock.t, 0.0, FADE_S))
        holder.add_updater(upd)
        return holder

    def shot_wvs(self, v):
        d = load_npz(v["data"])
        size = min(self.content_height() - 1.2, 5.6)
        if d is None:
            self.add(placeholder(size, f"run stage_1_wvs.py ({v['data']})", self.fg))
            return
        frames = d["points"]
        K = len(frames)
        t0, t1 = FADE_S + 0.2, self.shot_len - 1.2
        # Lloyd moves are large at first and tiny later: spend time evenly in log-iterations
        times = np.log1p(np.arange(K)) / np.log1p(K - 1)
        state = {"k": 0}

        def on_frame(k, a):
            state["k"] = k + (1 if a > 0.5 else 0)
        target = image_mobject(gray_to_rgb(d["image"]), size)
        panel, border = self._point_panel(frames, size, times, t0, t1, on_frame)
        row = Group(target, panel).arrange(RIGHT, buff=0.9).move_to(self.content_center() + DOWN * 0.2)
        border.move_to(panel.get_center())
        cap_t = self.text("Target", size=28).next_to(target, UP, buff=0.25)
        cap_p = self.text(v.get("label", "Weighted Voronoi Stippling"), size=28).next_to(panel, UP, buff=0.25)
        counter = self._counter(v.get("counter", "iteration") + " {:d} / " + f"{K - 1}",
                                panel.get_bottom() + DOWN * 0.35, lambda: state["k"])
        for m in (target, panel, border, cap_t, cap_p):
            self.fade_in(m)
            self.add(m)
        self.add(counter)

    def _crossfade_panel(self, first, last, size_units, t0, t1):
        """Two real point sets (prior, final result), crossfaded over clock [t0, t1]."""
        size_px = int(round(size_units * px_per_unit()))
        radius = dot_radius(last.shape[0], size_px)
        a_img = render_points(first, size_px, radius).astype(np.float32)
        b_img = render_points(last, size_px, radius).astype(np.float32)
        m = image_mobject(a_img.astype(np.uint8), size_units)
        cache = {"a": None}

        def upd(mob, dt):
            a = round(ramp(self.clock.t, t0, t1), 3)
            if a == cache["a"]:
                return
            cache["a"] = a
            rgb = (a_img * (1 - a) + b_img * a + 0.5).astype(np.uint8)
            mob.pixel_array = np.dstack([rgb, np.full((size_px, size_px), 255, np.uint8)])
        m.add_updater(upd)
        border = Rectangle(width=size_units, height=size_units, stroke_color="#BBBBBB", stroke_width=1.5)
        border.add_updater(lambda b, dt: b.move_to(m.get_center()))
        return m, border

    def shot_trajectory(self, v):
        panels = v["panels"]
        each_target = any(p.get("show_target") for p in panels)
        n = len(panels)
        cols = n + (0 if each_target or not v.get("target") else 1)
        avail_w = config.frame_width - 0.8 - 0.45 * (cols - 1)
        rows_h = self.content_height() - (1.4 if each_target else 1.0)
        size = min(avail_w / cols, rows_h / (2 if each_target else 1), 4.2)

        # timeline: prior (clean rejection sample, held), forward noising, reverse process, hold
        t_prior, t_noise = FADE_S + 1.3, FADE_S + 2.1
        t_end = max(t_noise + 1.0, self.shot_len - 1.1)
        state = {"step": None, "result_only": True}
        animate = bool(v.get("animate", True))
        if animate:
            t_fade0, t_fade1 = FADE_S + 0.6 + 0.25 * self.shot_len, FADE_S + 1.8 + 0.25 * self.shot_len
        else:                     # show the final result from the start
            t_fade0, t_fade1 = -2.0, -1.0
        cells = []
        for p in panels:
            # "trajectories": false -> crossfade prior -> final even if a trajectory file exists
            d = load_npz(p["traj"]) if animate and v.get("trajectories", True) else None
            if d is None:
                r = load_result(p["traj"])
                if r is None:
                    cells.append((placeholder(size, f"run stage_1_trajectories.py ({p['traj']})", self.fg), None, p))
                    continue
                panel, border = self._crossfade_panel(r["prior"], r["final"], size, t_fade0, t_fade1)
                cells.append((panel, border, p, r))
                continue
            state["result_only"] = False
            frames = np.concatenate([d["prior"][None], d["points"]], axis=0)
            steps = d["steps"].astype(np.float64)
            t_start = float(d["t_start"])
            state["t_start"] = int(t_start)
            # progress axis: prior at 0, noised start at a, then linear in denoising steps
            a0 = (t_noise - t_prior) / (t_end - t_prior)
            times = np.concatenate([[0.0], a0 + (1 - a0) * steps / max(steps[-1], 1.0)])

            def on_frame(k, a, steps=steps, t_start=t_start):
                if k == 0:
                    state["step"] = None
                else:
                    s = steps[min(k - 1 + (1 if a > 0.5 else 0), len(steps) - 1)]
                    state["step"] = int(round(t_start - s))
            panel, border = self._point_panel(frames, size, times, t_prior, t_end, on_frame)
            cells.append((panel, border, p, d))

        items = []
        for c in cells:
            panel, border, p = c[0], c[1], c[2]
            cap = self.text(p.get("label", ""), size=28)
            if each_target and len(c) > 3:
                tgt = image_mobject(gray_to_rgb(c[3]["image"]), size)
                col = Group(cap, tgt, panel).arrange(DOWN, buff=0.22)
            else:
                col = Group(cap, panel).arrange(DOWN, buff=0.22)
            items.append((col, panel, border))
        cols_g = [it[0] for it in items]
        if v.get("target") and not each_target:
            td = load_npz(v["target"]) or load_result(v["target"])
            tgt = (image_mobject(gray_to_rgb(td["image"]), size) if td is not None
                   else placeholder(size, f"missing {v['target']}", self.fg))
            cols_g.insert(0, Group(self.text("Target", size=28, weight="BOLD"), tgt).arrange(DOWN, buff=0.22))
        row = Group(*cols_g).arrange(RIGHT, buff=0.45, aligned_edge=UP).move_to(self.content_center() + UP * 0.15)
        for m in cols_g:
            self.fade_in(m)
            self.add(m)
        for _, panel, border in items:
            if border is not None:
                border.move_to(panel.get_center())
                self.fade_in(border)
                self.add(border)

        def label():
            t = self.clock.t
            if state["result_only"]:
                if not animate:
                    return "our results"
                return "rejection sampling" if t < (t_fade0 + t_fade1) / 2 else "our result"
            if t < t_prior:
                return "rejection sampling"
            if t < t_noise:
                return f"adding noise   (t = {state.get('t_start', 500)})"
            step = state["step"] if state["step"] is not None else state.get("t_start", 500)
            return f"reverse diffusion   t = {step}"
        self.add(self._counter("{}", row.get_bottom() + DOWN * 0.4, label))

    # ── tone correction ──────────────────────────────────────────────────────────────
    def shot_tone(self, v):
        d = load_npz(v["data"])
        size = min((config.frame_width - 2.0) / 3, self.content_height() - 1.6)
        if d is None:
            r = load_result(v["data"])
            if r is None:
                self.add(placeholder(size, f"run stage_1_tone.py ({v['data']})", self.fg))
            else:
                self._tone_result(r)
            return
        import matplotlib
        cmap = matplotlib.colormaps["RdBu_r"]
        rho, dark, target = d["rho"], d["dark"], d["target"]
        err = dark - target[None]
        lim = float(np.percentile(np.abs(err[0]), 99)) or 1.0
        K = len(rho)
        t0, t1 = FADE_S + 0.4, self.shot_len - 1.2

        def rgb_rho(x):
            return gray_to_rgb(1.0 - x)                     # show density as ink (dark = dense)

        def rgb_dark(x):
            return gray_to_rgb(1.0 - x)

        def rgb_err(x):
            return (cmap(np.clip(x / lim * 0.5 + 0.5, 0, 1))[..., :3] * 255).astype(np.uint8)

        def resize(arr, px):
            return np.asarray(Image.fromarray(arr).resize((px, px), Image.BILINEAR))

        px = int(round(size * px_per_unit()))
        makers = [(rho, rgb_rho, "Requested density"), (dark, rgb_dark, "Rendered stipple"),
                  (err, rgb_err, "Error (red: too dark, blue: too light)")]
        mobs = []
        state = {"k": 0}
        for series, to_rgb, title in makers:
            m = image_mobject(resize(to_rgb(series[0]), px), size)

            def upd(mob, dt, series=series, to_rgb=to_rgb, cache={"key": None}):
                p = ramp(self.clock.t, t0, t1)
                f = p * (K - 1)
                k = int(min(f, K - 2))
                a = f - k
                key = (k, round(a, 2))
                if key == cache["key"]:
                    return
                cache["key"] = key
                state["k"] = k + (1 if a > 0.5 else 0)
                x = series[k] * (1 - a) + series[k + 1] * a
                mob.pixel_array = np.dstack([resize(to_rgb(x), px), np.full((px, px), 255, np.uint8)])
            m.add_updater(upd)
            cap = self.text(title, size=24)
            if cap.width > size + 0.3:
                cap.scale_to_fit_width(size + 0.3)
            mobs.append(Group(cap, m).arrange(DOWN, buff=0.2))
        row = Group(*mobs).arrange(RIGHT, buff=0.5).move_to(self.content_center() + UP * 0.2)
        for m in mobs:
            self.fade_in(m)
            self.add(m)
        steps, psnr = d["steps"], d["psnr"]
        counter = self._counter("optimization step {}", row.get_bottom() + DOWN * 0.35 + LEFT * 2.2,
                                lambda: int(steps[min(state["k"], K - 1)]))
        score = self._counter("PSNR {:.2f} dB", row.get_bottom() + DOWN * 0.35 + RIGHT * 2.2,
                              lambda: float(psnr[min(state["k"], K - 1)]))
        self.add(counter, score)


    def _tone_result(self, r):
        """Uncorrected vs optimized (real end states), crossfaded while the real loss curve of
        the 300 optimizer steps is drawn; the crossfade follows the loss, not a fake state."""
        import matplotlib
        cmap = matplotlib.colormaps["RdBu_r"]
        target = r["target"]
        loss = r["field_loss"].astype(np.float64)
        K = len(loss)
        t0, t1 = FADE_S + 0.4, self.shot_len - 1.4
        gap = 0.35
        size = min((config.frame_width - 1.2 - 3 * gap) / 4, self.content_height() - 1.8)
        px = int(round(size * px_per_unit()))
        err0 = r["none_dark"] - target
        lim = float(np.percentile(np.abs(err0), 99)) or 1.0

        def resize(arr):
            return np.asarray(Image.fromarray(arr).resize((px, px), Image.LANCZOS)).astype(np.float32)

        def ink(x):
            return resize(gray_to_rgb(1.0 - x))

        def err_rgb(x):
            rgb = (cmap(np.clip(x / lim * 0.5 + 0.5, 0, 1))[..., :3] * 255).astype(np.uint8)
            return resize(rgb) if rgb.shape[0] != px else rgb.astype(np.float32)

        if "none_coords" in r:
            # Re-render the stipples from their points at the panel's own resolution (the stored
            # renders are 256 px and would be enlarged); the dot size is kept relative to the canvas.
            sig = float(r["dot_sigma_px"]) / float(r["render_res"])

            def dark_at(coords):                        # 2x supersampled, then filtered down
                big = render_darkness(coords, 2 * px, sig, r["ink_gain"])
                g = Image.fromarray(np.clip(big * 255.0 + 0.5, 0, 255).astype(np.uint8))
                return np.asarray(g.resize((px, px), Image.LANCZOS), np.float32) / 255.0
            dark0, dark1 = dark_at(r["none_coords"]), dark_at(r["field_coords"])
            tgt = 1.0 - np.asarray(Image.fromarray((r["image"] * 255).astype(np.uint8)).resize((px, px), Image.LANCZOS),
                                   np.float32) / 255.0
            e0, e1 = dark0 - tgt, dark1 - tgt           # errors: from the renderer's raster
            # stipple panel: sharp discs, as in the paper figure (radius 1.8 sigma, Beer-Lambert alpha)
            radius = 1.8 * sig
            alpha = 1.0 - float(np.exp(-float(r["ink_gain"])))
            dark0 = render_ink_dots(r["none_coords"], px, radius, alpha)
            dark1 = render_ink_dots(r["field_coords"], px, radius, alpha)
        else:
            dark0, dark1, e0, e1 = r["none_dark"], r["field_dark"], err0, r["field_dark"] - target
        pairs = [(ink(r["none_rho"]), ink(r["field_rho"]), "Requested density"),
                 (ink(dark0), ink(dark1), "Rendered stipple"),
                 (err_rgb(e0), err_rgb(e1), "Error (red: too dark, blue: too light)")]
        drop = (loss[0] - loss) / max(loss[0] - loss[-1], 1e-12)       # 0 -> 1 as the loss falls
        state = {"k": 0}

        def progress():
            return ramp(self.clock.t, t0, t1)

        cols = []
        for a_img, b_img, title in pairs:
            m = image_mobject(a_img.astype(np.uint8), size)

            def upd(mob, dt, a_img=a_img, b_img=b_img, cache={"a": None}):
                k = int(round(progress() * (K - 1)))
                state["k"] = k
                a = round(float(np.clip(drop[k], 0, 1)), 3)
                if a == cache["a"]:
                    return
                cache["a"] = a
                rgb = (a_img * (1 - a) + b_img * a + 0.5).astype(np.uint8)
                mob.pixel_array = np.dstack([rgb, np.full((px, px), 255, np.uint8)])
            m.add_updater(upd)
            cap = self.text(title, size=22)
            if cap.width > size + 0.2:
                cap.scale_to_fit_width(size + 0.2)
            cols.append(Group(cap, m).arrange(DOWN, buff=0.2))

        # loss curve panel: axes + a polyline revealed step by step
        w_ax, h_ax = size, size * 0.8
        origin = np.array([-w_ax / 2, -h_ax / 2, 0.0])
        axes = VGroup(Line(origin, origin + RIGHT * w_ax, color=self.fg, stroke_width=2),
                      Line(origin, origin + UP * h_ax, color=self.fg, stroke_width=2))
        lo, hi = float(loss.min()), float(loss.max())
        xs = np.linspace(0, w_ax, K)
        ys = (loss - lo) / max(hi - lo, 1e-12) * h_ax * 0.92
        curve_pts = np.stack([origin[0] + xs, origin[1] + ys, np.zeros(K)], axis=1)
        curve = VMobject(stroke_color=self.accent, stroke_width=3)
        curve.set_points_as_corners(curve_pts[:2])

        def draw(c, dt):
            k = max(2, int(round(progress() * (K - 1))) + 1)
            c.set_points_as_corners(curve_pts[:k])
        curve.add_updater(draw)
        cap_l = self.text("Loss over 300 optimizer steps", size=22)
        if cap_l.width > size + 0.2:
            cap_l.scale_to_fit_width(size + 0.2)
        plot = VGroup(axes, curve)
        cols.append(Group(cap_l, plot).arrange(DOWN, buff=0.2))

        row = Group(*cols).arrange(RIGHT, buff=gap, aligned_edge=UP).move_to(self.content_center() + UP * 0.25)
        # the curve's points were built around the old origin; move them with the axes
        curve_pts += axes[0].get_start() - origin
        for m in cols:
            self.fade_in(m)
            self.add(m)

        def label():
            k = state["k"]
            if k >= K - 1:
                return f"PSNR {float(r['none_psnr']):.2f} dB  →  {float(r['field_psnr']):.2f} dB"
            return f"optimization step {k + 1} / {K}"
        self.add(self._counter("{}", row.get_bottom() + DOWN * 0.4, label))


__all__ = ["ShotScene", "ORIGIN"]
