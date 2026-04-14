"""
HUD overlays and debug visualizer for the exercise classifier.

Public functions (called from main.py):
    - load_feedback_emojis(project_root)  — load emoji PNGs once at startup.
    - draw_landmarks_on_image(...)        — skeleton with exercise-aware coloring.
    - draw_active_hud(...)                — Concept C gamified AR overlay.
    - draw_no_pose_hud(...)               — "NO POSE DETECTED" banner + rep summary.
    - draw_feedback_visualizer(...)       — dual sparkline graph (active + predicted).
"""

import os

import cv2
import numpy as np

from runtime_logic.rep_counter import (
    EXERCISE_CONFIGS,
    SWITCH_PROGRESS_THRESHOLD,
    get_exercise_metric_value,
)


# ---------------------------------------------------------------------------
# Colors (BGR)
# ---------------------------------------------------------------------------

C_ORANGE = (30, 165, 255)
C_PURPLE = (255, 68, 136)
C_WHITE = (255, 255, 255)
C_MUTED = (170, 180, 195)
C_GREEN = (60, 210, 80)
C_RED = (60, 60, 220)
C_YELLOW = (30, 220, 255)
C_DARK = (8, 10, 14)

ALPHA_PILL = 0.72
ALPHA_BANNER = 0.68

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

RADIUS = 10


# ---------------------------------------------------------------------------
# Exercise-aware skeleton config
# ---------------------------------------------------------------------------

POSE_CONNECTIONS = frozenset(
    [
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (25, 27),
        (24, 26),
        (26, 28),
    ]
)

# Connections highlighted in orange for each exercise (the "working" limbs).
_ACTIVE_CONNECTIONS = {
    "curl": frozenset([(13, 15), (14, 16)]),
    "shoulder_press": frozenset([(11, 13), (12, 14), (13, 15), (14, 16)]),
    "squat": frozenset([(23, 25), (25, 27), (24, 26), (26, 28)]),
}

# Landmark indices highlighted in orange for each exercise.
_ACTIVE_JOINTS = {
    "curl": frozenset([13, 14, 15, 16]),
    "shoulder_press": frozenset([11, 12, 13, 14]),
    "squat": frozenset([23, 24, 25, 26]),
}


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def _rounded_fill(canvas, x1, y1, x2, y2, r, color):
    """Fill a rounded rectangle on canvas."""
    if x2 <= x1 or y2 <= y1:
        return
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(canvas, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(canvas, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [
        (x1 + r, y1 + r),
        (x2 - r, y1 + r),
        (x1 + r, y2 - r),
        (x2 - r, y2 - r),
    ]:
        cv2.circle(canvas, (cx, cy), r, color, -1, cv2.LINE_AA)


def _pill(frame, x, y, w, h, alpha=ALPHA_PILL, color=C_DARK):
    overlay = frame.copy()
    _rounded_fill(overlay, x, y, x + w, y + h, RADIUS, color)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _txt(frame, text, x, y, color=C_WHITE, scale=0.55, thickness=1, font=FONT):
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _txt_size(text, scale=0.55, thickness=1):
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h


# ---------------------------------------------------------------------------
# Emoji overlay
# ---------------------------------------------------------------------------


def load_feedback_emojis(project_root):
    """Load happy/angry emoji PNGs from the project's image directory."""
    for sub in ("graphics/Images", "graphics/images", "excercise_classifcation/Images"):
        d = os.path.join(project_root, sub)
        if os.path.isdir(d):
            break
    happy = cv2.imread(os.path.join(d, "happyemoji.png"), cv2.IMREAD_UNCHANGED)
    angry = cv2.imread(os.path.join(d, "angryemoji.png"), cv2.IMREAD_UNCHANGED)
    return {"happy": happy, "angry": angry}


def _overlay_emoji(frame, icon, x, y, size):
    """Alpha-blend an emoji icon at (x, y) with given size."""
    if icon is None:
        return
    ih, iw = icon.shape[:2]
    if ih == 0 or iw == 0:
        return
    scale = size / float(max(ih, iw))
    icon = cv2.resize(icon, (max(1, int(iw * scale)), max(1, int(ih * scale))))
    fh, fw = frame.shape[:2]
    ih, iw = icon.shape[:2]
    # Clamp to frame bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + iw), min(fh, y + ih)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    crop = icon[: y2 - y1, : x2 - x1]
    if crop.shape[2] == 4:
        a = crop[:, :, 3:4].astype("float32") / 255.0
        frame[y1:y2, x1:x2] = (
            a * crop[:, :, :3] + (1 - a) * roi.astype("float32")
        ).astype("uint8")
    else:
        frame[y1:y2, x1:x2] = crop[:, :, :3]


# ---------------------------------------------------------------------------
# Concept C — component drawers
# ---------------------------------------------------------------------------


def _draw_scan_lines(frame):
    """Subtle CRT-style horizontal lines — darken every 4th row by ~12%."""
    frame[::4, :] = (frame[::4, :].astype(np.float32) * 0.88).astype(np.uint8)


def _draw_corner_brackets(frame):
    """Orange L-shaped brackets at all four corners."""
    fh, fw = frame.shape[:2]
    m, s, t = 10, 22, 2
    c = C_ORANGE
    # top-left
    cv2.line(frame, (m, m), (m + s, m), c, t, cv2.LINE_AA)
    cv2.line(frame, (m, m), (m, m + s), c, t, cv2.LINE_AA)
    # top-right
    cv2.line(frame, (fw - m, m), (fw - m - s, m), c, t, cv2.LINE_AA)
    cv2.line(frame, (fw - m, m), (fw - m, m + s), c, t, cv2.LINE_AA)
    # bottom-left
    cv2.line(frame, (m, fh - m), (m + s, fh - m), c, t, cv2.LINE_AA)
    cv2.line(frame, (m, fh - m), (m, fh - m - s), c, t, cv2.LINE_AA)
    # bottom-right
    cv2.line(frame, (fw - m, fh - m), (fw - m - s, fh - m), c, t, cv2.LINE_AA)
    cv2.line(frame, (fw - m, fh - m), (fw - m, fh - m - s), c, t, cv2.LINE_AA)


def _draw_exercise_tag(frame, exercise_name):
    """Top-center orange pill with exercise name."""
    fw = frame.shape[1]
    text = exercise_name.replace("_", " ").upper()
    scale, thick = 0.52, 2
    (tw, th), _ = cv2.getTextSize(text, FONT_BOLD, scale, thick)
    pad_w, pad_h = 18, 7
    pill_w = tw + pad_w * 2
    pill_h = th + pad_h * 2
    x = (fw - pill_w) // 2
    y = 10

    overlay = frame.copy()
    _rounded_fill(overlay, x, y, x + pill_w, y + pill_h, 12, C_ORANGE)
    cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

    cv2.putText(
        frame,
        text,
        (x + pad_w, y + pad_h + th),
        FONT_BOLD,
        scale,
        C_WHITE,
        thick,
        cv2.LINE_AA,
    )


def _draw_rep_boxes(frame, rep_count, direction):
    """Top-left stat boxes: rep count + direction, anchored to the left edge."""
    y0 = 10
    x0 = 0
    border = 3

    # — Box 1: rep count ——————————————
    rep_str = str(rep_count)
    num_scale, num_thick = 1.6, 3
    (nw, nh), _ = cv2.getTextSize(rep_str, FONT_BOLD, num_scale, num_thick)
    sub_text = "REPS"
    sub_scale = 0.38
    (sw, _), _ = cv2.getTextSize(sub_text, FONT, sub_scale, 1)

    b1_w = border + 12 + nw + 8 + sw + 10
    b1_h = nh + 18

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + b1_w, y0 + b1_h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + border, y0 + b1_h), C_ORANGE, -1)

    cv2.putText(
        frame,
        rep_str,
        (x0 + border + 10, y0 + b1_h - 8),
        FONT_BOLD,
        num_scale,
        C_WHITE,
        num_thick,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        sub_text,
        (x0 + border + 10 + nw + 6, y0 + b1_h - 10),
        FONT,
        sub_scale,
        C_ORANGE,
        1,
        cv2.LINE_AA,
    )

    if not direction:
        return

    # — Box 2: direction ——————————————
    dir_str = str(direction).upper()
    d_scale, d_thick = 0.60, 2
    (dw, dh), _ = cv2.getTextSize(dir_str, FONT_BOLD, d_scale, d_thick)

    b2_w = border + 12 + dw + 12
    b2_h = dh + 14
    y1 = y0 + b1_h + 5

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (x0, y1), (x0 + b2_w, y1 + b2_h), C_DARK, -1)
    cv2.addWeighted(overlay2, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (x0, y1), (x0 + border, y1 + b2_h), C_PURPLE, -1)

    cv2.putText(
        frame,
        dir_str,
        (x0 + border + 10, y1 + b2_h - 7),
        FONT_BOLD,
        d_scale,
        C_ORANGE,
        d_thick,
        cv2.LINE_AA,
    )


def _draw_form_ring(frame, feedback_message):
    """Bottom-right circular progress ring showing form score."""
    fh, fw = frame.shape[:2]

    if feedback_message is None:
        pct = 0
    elif feedback_message.lower().startswith("good"):
        pct = 85
    else:
        pct = 40

    ring_r = 26
    margin = 14
    # Position above the feedback bar (which is ~44px tall).
    cx = fw - margin - ring_r
    cy = fh - 44 - margin - ring_r

    # Dark background circle.
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), ring_r + 5, C_DARK, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Track.
    cv2.circle(frame, (cx, cy), ring_r, (50, 50, 55), 3, cv2.LINE_AA)

    if pct > 0:
        sweep = int(360 * pct / 100)
        arc_color = C_GREEN if pct >= 70 else C_YELLOW
        cv2.ellipse(
            frame,
            (cx, cy),
            (ring_r, ring_r),
            -90,
            0,
            sweep,
            arc_color,
            3,
            cv2.LINE_AA,
        )

    # Center text.
    pct_text = f"{pct}%" if pct > 0 else "--"
    tw, th = _txt_size(pct_text, scale=0.35)
    _txt(frame, pct_text, cx - tw // 2, cy + th // 2, C_WHITE, scale=0.35)

    # "FORM" label above ring.
    lw, lh = _txt_size("FORM", scale=0.28)
    _txt(frame, "FORM", cx - lw // 2, cy - ring_r - 5, C_MUTED, scale=0.28)


def _draw_bottom_bar(frame, feedback_message, is_good, emoji_images):
    """Bottom gradient fade + form feedback chip + emoji."""
    fh, fw = frame.shape[:2]
    bar_h = 44

    # Gradient fade: transparent → dark at bottom.
    region = frame[fh - bar_h : fh].astype(np.float32)
    weights = np.linspace(0.0, 0.88, bar_h).reshape(-1, 1, 1)
    dark = np.array([8, 8, 12], dtype=np.float32)
    frame[fh - bar_h : fh] = (region * (1 - weights) + dark * weights).astype(
        np.uint8
    )

    # Feedback chip.
    fb_color = C_GREEN if is_good else C_YELLOW
    chip_bg = (15, 50, 15) if is_good else (15, 40, 50)
    chip_text = feedback_message
    scale, thick = 0.44, 1
    tw, th = _txt_size(chip_text, scale=scale, thickness=thick)
    chip_x = 10
    chip_h = th + 12
    chip_w = tw + 22
    chip_y = fh - (bar_h - chip_h) // 2 - chip_h

    overlay = frame.copy()
    _rounded_fill(overlay, chip_x, chip_y, chip_x + chip_w, chip_y + chip_h, 10, chip_bg)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    _txt(frame, chip_text, chip_x + 11, chip_y + chip_h - 6, fb_color, scale=scale)

    # Emoji to the right of the chip.
    emoji_key = "happy" if is_good else "angry"
    icon = emoji_images.get(emoji_key)
    emoji_size = 36
    emoji_x = chip_x + chip_w + 8
    emoji_y = fh - bar_h + (bar_h - emoji_size) // 2
    _overlay_emoji(frame, icon, emoji_x, emoji_y, emoji_size)


# ---------------------------------------------------------------------------
# Skeleton overlay
# ---------------------------------------------------------------------------


def draw_landmarks_on_image(
    rgb_image, detection_result, landmark_indices, landmark_names, exercise_name=None
):
    """Draw skeleton with purple lines; active joints/connections highlighted orange."""
    annotated = np.copy(rgb_image)
    active_conns = _ACTIVE_CONNECTIONS.get(exercise_name, frozenset())
    active_joints = _ACTIVE_JOINTS.get(exercise_name, frozenset())

    for pose_landmarks in detection_result.pose_landmarks:
        h, w = annotated.shape[:2]

        for conn in POSE_CONNECTIONS:
            start = pose_landmarks[conn[0]]
            end = pose_landmarks[conn[1]]
            is_active = conn in active_conns or tuple(reversed(conn)) in active_conns
            color = C_ORANGE if is_active else C_PURPLE
            line_w = 2 if is_active else 1
            cv2.line(
                annotated,
                (int(start.x * w), int(start.y * h)),
                (int(end.x * w), int(end.y * h)),
                color,
                line_w,
                cv2.LINE_AA,
            )

        for name in landmark_names:
            idx = landmark_indices[name]
            landmark = pose_landmarks[idx]
            px, py = int(landmark.x * w), int(landmark.y * h)
            is_active = idx in active_joints
            if is_active:
                # Outer ring + filled dot.
                cv2.circle(annotated, (px, py), 7, C_ORANGE, 1, cv2.LINE_AA)
                cv2.circle(annotated, (px, py), 4, C_ORANGE, -1, cv2.LINE_AA)
            else:
                cv2.circle(annotated, (px, py), 4, C_PURPLE, -1, cv2.LINE_AA)

    return annotated


# ---------------------------------------------------------------------------
# Active HUD — Concept C
# ---------------------------------------------------------------------------


def draw_active_hud(
    frame,
    exercise_name,
    rep_counts,
    feedback_message,
    emoji_images,
    rep_direction_label=None,
    # legacy params kept for API compatibility, not used in new design
    feedback_bottom_margin=35,
    emoji_margin=28,
    emoji_size=56,
):
    """Draw Concept C gamified AR overlay."""
    if exercise_name is None:
        return

    _draw_scan_lines(frame)
    _draw_corner_brackets(frame)
    _draw_exercise_tag(frame, exercise_name)
    _draw_rep_boxes(frame, rep_counts.get(exercise_name, 0), rep_direction_label)

    if feedback_message is not None:
        is_good = feedback_message.lower().startswith("good")
        _draw_form_ring(frame, feedback_message)
        _draw_bottom_bar(frame, feedback_message, is_good, emoji_images)
    else:
        _draw_form_ring(frame, None)


# ---------------------------------------------------------------------------
# No-pose HUD
# ---------------------------------------------------------------------------


def draw_no_pose_hud(frame, exercise_name, rep_counts):
    """Show 'NO POSE DETECTED' banner and rep summary."""
    _draw_scan_lines(frame)
    _draw_corner_brackets(frame)

    msg = "NO POSE DETECTED"
    tw, th = _txt_size(msg, scale=0.58)
    pw, ph = tw + 20, th + 14
    _pill(frame, 12, 12, pw, ph, alpha=0.75, color=(18, 8, 8))
    _txt(frame, msg, 22, 12 + ph - 6, C_RED, scale=0.58)
    if exercise_name is not None:
        _txt(
            frame,
            f"REPS  {rep_counts[exercise_name]}",
            16,
            12 + ph + 28,
            C_ORANGE,
            scale=1.0,
            thickness=2,
            font=FONT_BOLD,
        )


# ---------------------------------------------------------------------------
# Signal graph — pretty dual sparkline
# ---------------------------------------------------------------------------


def _get_metric_series(exercise_name, stream):
    if not stream or exercise_name is None:
        return None
    cfg = EXERCISE_CONFIGS.get(exercise_name, {})
    try:
        top_t = float(cfg["top_threshold"])
        bot_t = float(cfg["bottom_threshold"])
    except (KeyError, TypeError, ValueError):
        return None
    series = [
        v
        for v in (get_exercise_metric_value(f, exercise_name) for f in stream)
        if v is not None
    ]
    if not series:
        return None
    pad = abs(top_t - bot_t) * 0.15
    lo = min(top_t, bot_t) - pad
    hi = max(top_t, bot_t) + pad
    return {"series": series, "y_bounds": (lo, hi), "top_t": top_t, "bot_t": bot_t}


def _draw_pretty_sparkline(frame, metric, sx, sy, sw, sh, accent, switch_thresh_y=None):
    """Render a single polished sparkline with area fill and glowing tip."""
    series = metric["series"]
    y_min, y_max = metric["y_bounds"]
    top_t, bot_t = metric["top_t"], metric["bot_t"]

    if y_max - y_min < 1e-6:
        y_max = y_min + 1.0

    def _py(val):
        r = np.clip((val - y_min) / (y_max - y_min), 0, 1)
        return int(sy + sh - r * sh)

    n = len(series)

    # — Area fill under the curve ——————————————————
    if n >= 2:
        pts = [
            (sx + int(i / max(n - 1, 1) * (sw - 1)), _py(series[i]))
            for i in range(n)
        ]
        poly = np.array(
            pts + [(sx + sw - 1, sy + sh), (sx, sy + sh)], dtype=np.int32
        )
        area_overlay = frame.copy()
        cv2.fillPoly(area_overlay, [poly], accent)
        cv2.addWeighted(area_overlay, 0.10, frame, 0.90, 0, frame)

    # — Threshold lines (dashed) ———————————————————
    for thresh_val, t_color in [(top_t, C_GREEN), (bot_t, C_ORANGE)]:
        ty = _py(thresh_val)
        if sy <= ty <= sy + sh:
            dash, gap = 4, 4
            for dx in range(sx, sx + sw, dash + gap):
                cv2.line(
                    frame,
                    (dx, ty),
                    (min(dx + dash, sx + sw), ty),
                    t_color,
                    1,
                    cv2.LINE_AA,
                )

    # — Switch threshold dashed line (red, predicted only) ———
    if switch_thresh_y is not None and sy <= switch_thresh_y <= sy + sh:
        dash, gap = 3, 5
        for dx in range(sx, sx + sw, dash + gap):
            cv2.line(
                frame,
                (dx, switch_thresh_y),
                (min(dx + dash, sx + sw), switch_thresh_y),
                C_RED,
                1,
                cv2.LINE_AA,
            )

    # — Series line ————————————————————————————————
    if n >= 2:
        for i in range(1, n):
            x1p = sx + int((i - 1) / max(n - 1, 1) * (sw - 1))
            x2p = sx + int(i / max(n - 1, 1) * (sw - 1))
            cv2.line(
                frame,
                (x1p, _py(series[i - 1])),
                (x2p, _py(series[i])),
                C_WHITE,
                2,
                cv2.LINE_AA,
            )

    # — Glowing tip dot ————————————————————————————
    if series:
        tip_x = sx + sw - 1
        tip_y = _py(series[-1])
        for glow_r, glow_a in ((9, 0.08), (6, 0.16), (4, 0.28)):
            ov = frame.copy()
            cv2.circle(ov, (tip_x, tip_y), glow_r, C_ORANGE, -1)
            cv2.addWeighted(ov, glow_a, frame, 1.0 - glow_a, 0, frame)
        cv2.circle(frame, (tip_x, tip_y), 3, C_ORANGE, -1, cv2.LINE_AA)

        # Current value label
        val_text = f"{series[-1]:.2f}"
        vw, vh = _txt_size(val_text, scale=0.30)
        label_x = max(sx, tip_x - vw - 2)
        _txt(frame, val_text, label_x, sy + sh + 10, C_MUTED, scale=0.30)


def draw_feedback_visualizer(
    frame,
    exercise_name,
    feedback_state,
    rep_count,
    switch_state=None,
):
    """Render pretty dual-sparkline signal card at top-right."""
    if exercise_name is None or feedback_state is None:
        return

    active_stream = list(feedback_state.rolling_buffer)
    predicted_ex = switch_state.get("predicted_exercise") if switch_state else None
    predicted_stream = (
        list(switch_state.get("predicted_stream", [])) if switch_state else []
    )

    fh, fw = frame.shape[:2]

    # Card layout constants.
    PAD = 10
    HEADER_H = 18
    LABEL_H = 16
    SPARK_H = 44
    VALUE_H = 14
    SECTION_H = LABEL_H + SPARK_H + VALUE_H
    GAP = 10
    CARD_W = min(300, max(180, int(fw * 0.25)))
    CARD_H = PAD + HEADER_H + GAP + SECTION_H + GAP + SECTION_H + PAD
    CARD_X = fw - CARD_W - 8
    CARD_Y = 8

    # — Card background ——————————————————————————
    overlay = frame.copy()
    _rounded_fill(overlay, CARD_X, CARD_Y, CARD_X + CARD_W, CARD_Y + CARD_H, 12, C_DARK)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Orange left accent strip.
    cv2.rectangle(
        frame,
        (CARD_X, CARD_Y + 4),
        (CARD_X + 3, CARD_Y + CARD_H - 4),
        C_ORANGE,
        -1,
    )

    # Subtle card border.
    cv2.rectangle(
        frame,
        (CARD_X, CARD_Y),
        (CARD_X + CARD_W, CARD_Y + CARD_H),
        (40, 42, 48),
        1,
        cv2.LINE_AA,
    )

    # — Header ——————————————————————————————————
    header_y = CARD_Y + PAD
    _txt(frame, "SIGNALS", CARD_X + PAD + 4, header_y + 11, C_MUTED, scale=0.32)

    spark_w = CARD_W - PAD * 2 - 4

    # — Draw each section ——————————————————————
    sections = [
        (exercise_name, active_stream, "ACTIVE", C_WHITE, None),
        (predicted_ex, predicted_stream, "PREDICTED", C_MUTED, switch_state),
    ]

    for idx, (ex, stream, label, accent, sw_state) in enumerate(sections):
        sec_y = CARD_Y + PAD + HEADER_H + GAP + idx * (SECTION_H + GAP)
        spark_x = CARD_X + PAD + 4
        spark_y = sec_y + LABEL_H

        # Section label row.
        _txt(frame, label, spark_x, sec_y + 11, C_MUTED, scale=0.28)
        lbl_w, _ = _txt_size(label, scale=0.28)
        dot_x = spark_x + lbl_w + 5
        # Status dot.
        dot_color = C_ORANGE if ex else (60, 62, 68)
        cv2.circle(frame, (dot_x + 3, sec_y + 7), 3, dot_color, -1, cv2.LINE_AA)
        # Exercise name.
        ex_text = ex.replace("_", " ").upper() if ex else "---"
        _txt(frame, ex_text, dot_x + 10, sec_y + 11, C_ORANGE if ex else C_MUTED, scale=0.30)

        metric = _get_metric_series(ex, stream)

        # Spark background.
        ov = frame.copy()
        cv2.rectangle(ov, (spark_x, spark_y), (spark_x + spark_w, spark_y + SPARK_H), (14, 15, 18), -1)
        cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)

        if metric:
            # Compute switch-threshold Y for predicted panel.
            thresh_y_px = None
            if idx == 1 and sw_state:
                top_t, bot_t = metric["top_t"], metric["bot_t"]
                start_ext = sw_state.get("predicted_start_extremity")
                if start_ext == "top":
                    sl, tl = top_t, bot_t
                elif start_ext == "bottom":
                    sl, tl = bot_t, top_t
                else:
                    sl, tl = min(top_t, bot_t), max(top_t, bot_t)
                tv = sl + SWITCH_PROGRESS_THRESHOLD * (tl - sl)
                y_min, y_max = metric["y_bounds"]
                if y_max - y_min > 1e-6:
                    r = np.clip((tv - y_min) / (y_max - y_min), 0, 1)
                    thresh_y_px = int(spark_y + SPARK_H - r * SPARK_H)

            _draw_pretty_sparkline(
                frame, metric, spark_x, spark_y, spark_w, SPARK_H,
                accent=C_ORANGE, switch_thresh_y=thresh_y_px,
            )
        else:
            _txt(
                frame, "no data",
                spark_x + spark_w // 2 - 16,
                spark_y + SPARK_H // 2 + 4,
                C_MUTED, scale=0.26,
            )
