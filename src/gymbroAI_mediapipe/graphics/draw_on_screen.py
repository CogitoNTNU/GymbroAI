"""
HUD overlays and debug visualizer for the exercise classifier.

Public functions (called from main.py):
    - load_feedback_emojis(project_root)  — load emoji PNGs once at startup.
    - draw_landmarks_on_image(...)        — skeleton connections and joint dots.
    - draw_active_hud(...)                — exercise name, rep count, direction, feedback.
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
# Hyperparameters
# ---------------------------------------------------------------------------

# Colors (BGR).
C_ORANGE = (30, 165, 255)
C_WHITE = (255, 255, 255)
C_MUTED = (170, 180, 195)
C_GREEN = (60, 210, 80)
C_RED = (60, 60, 220)
C_YELLOW = (30, 220, 255)
C_DARK = (8, 10, 14)

# Transparency for pill backgrounds and the feedback banner.
ALPHA_PILL = 0.72
ALPHA_BANNER = 0.68

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# Corner radius for rounded pills.
RADIUS = 10


# ---------------------------------------------------------------------------
# Skeleton overlay
# ---------------------------------------------------------------------------

# Pairs of MediaPipe landmark indices that form the body skeleton.
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


def draw_landmarks_on_image(
    rgb_image, detection_result, landmark_indices, landmark_names
):
    """Draw skeleton connections and joint dots on the frame."""
    annotated = np.copy(rgb_image)
    for pose_landmarks in detection_result.pose_landmarks:
        h, w = annotated.shape[:2]

        for connection in POSE_CONNECTIONS:
            start = pose_landmarks[connection[0]]
            end = pose_landmarks[connection[1]]
            cv2.line(
                annotated,
                (int(start.x * w), int(start.y * h)),
                (int(end.x * w), int(end.y * h)),
                (0, 255, 0),
                2,
            )

        for name in landmark_names:
            idx = landmark_indices[name]
            landmark = pose_landmarks[idx]
            cv2.circle(
                annotated,
                (int(landmark.x * w), int(landmark.y * h)),
                5,
                (0, 0, 255),
                -1,
            )

    return annotated


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
    """Draw a semi-transparent rounded pill background."""
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


def _overlay_bottom_right(frame, icon, margin, size):
    """Alpha-blend an icon into the bottom-right corner of frame."""
    if icon is None:
        return
    ih, iw = icon.shape[:2]
    if ih == 0 or iw == 0:
        return
    scale = size / float(max(ih, iw))
    icon = cv2.resize(icon, (max(1, int(iw * scale)), max(1, int(ih * scale))))
    fh, fw = frame.shape[:2]
    ih, iw = icon.shape[:2]
    x1, y1 = fw - margin - iw, fh - margin - ih
    roi = frame[y1 : y1 + ih, x1 : x1 + iw]
    crop = icon[:ih, :iw]
    if crop.shape[2] == 4:
        a = crop[:, :, 3:4].astype("float32") / 255.0
        frame[y1 : y1 + ih, x1 : x1 + iw] = (
            a * crop[:, :, :3] + (1 - a) * roi.astype("float32")
        ).astype("uint8")
    else:
        frame[y1 : y1 + ih, x1 : x1 + iw] = crop[:, :, :3]


# ---------------------------------------------------------------------------
# Active HUD
# ---------------------------------------------------------------------------


def draw_active_hud(
    frame,
    exercise_name,
    rep_counts,
    feedback_message,
    emoji_images,
    rep_direction_label=None,
    feedback_bottom_margin=35,
    emoji_margin=28,
    emoji_size=56,
):
    """Draw the main overlay: exercise name, rep count, direction, and form feedback."""
    if exercise_name is None:
        return

    fh, fw = frame.shape[:2]
    x0, y0 = 12, 12

    # Pill 1: exercise name.
    ex_label = exercise_name.replace("_", " ").upper()
    tw, th = _txt_size(ex_label, scale=0.70, thickness=2)
    p1w = tw + 20
    p1h = th + 14
    _pill(frame, x0, y0, p1w, p1h)
    _txt(
        frame,
        ex_label,
        x0 + 10,
        y0 + p1h - 6,
        C_ORANGE,
        scale=0.70,
        thickness=2,
        font=FONT_BOLD,
    )

    # Pill 2: rep count + direction arrow.
    y1 = y0 + p1h + 4
    rep_val = rep_counts.get(exercise_name, 0)
    rep_str = str(rep_val)
    direction = str(rep_direction_label).upper() if rep_direction_label else ""
    nw, nh = _txt_size(rep_str, scale=1.6, thickness=3)
    dw, _ = _txt_size(direction, scale=0.72, thickness=2)
    p2h = nh + 16
    p2w = nw + 24 + (dw + 14 if direction else 0)
    _pill(frame, x0, y1, p2w, p2h)
    _txt(frame, rep_str, x0 + 10, y1 + p2h - 6, C_WHITE, scale=1.6, thickness=3)
    if direction:
        _txt(
            frame,
            direction,
            x0 + 10 + nw + 14,
            y1 + p2h - 8,
            C_ORANGE,
            scale=0.72,
            thickness=2,
            font=FONT_BOLD,
        )

    # Feedback banner + emoji.
    if feedback_message is not None:
        is_good = feedback_message.lower().startswith("good")
        fb_color = C_GREEN if is_good else C_YELLOW
        fb_y = fh - feedback_bottom_margin - 40
        _txt(frame, "FORM", x0, fb_y, C_MUTED, scale=0.38)
        fw_t, fh_t = _txt_size(feedback_message, scale=0.50)
        fbw = fw_t + 20
        fbh = fh_t + 14
        _pill(frame, x0, fb_y + 6, fbw, fbh, alpha=ALPHA_BANNER)
        _txt(frame, feedback_message, x0 + 10, fb_y + 6 + fbh - 5, fb_color, scale=0.50)
        _overlay_bottom_right(
            frame,
            emoji_images.get("happy" if is_good else "angry"),
            emoji_margin,
            emoji_size,
        )


# ---------------------------------------------------------------------------
# No-pose HUD
# ---------------------------------------------------------------------------


def draw_no_pose_hud(frame, exercise_name, rep_counts):
    """Show 'NO POSE DETECTED' banner and rep summary."""
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
# Debug visualizer — dual sparkline (active + predicted)
# ---------------------------------------------------------------------------


def _get_metric_series(exercise_name, stream):
    """Build the metric series and Y bounds for a sparkline from a frame stream."""
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

    return {
        "series": series,
        "y_bounds": (lo, hi),
        "top_t": top_t,
        "bot_t": bot_t,
    }


def _draw_sparkline(frame, series, y_min, y_max, top_t, bot_t, x, y, w, h):
    """Render a single sparkline graph with threshold lines."""
    if y_max - y_min < 1e-6:
        y_max = y_min + 1.0

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    def _val_to_py(val):
        r = np.clip((val - y_min) / (y_max - y_min), 0, 1)
        return int(y + h - r * h)

    # Green = top threshold, orange = bottom threshold.
    cv2.line(
        frame,
        (x, _val_to_py(top_t)),
        (x + w, _val_to_py(top_t)),
        C_GREEN,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (x, _val_to_py(bot_t)),
        (x + w, _val_to_py(bot_t)),
        C_ORANGE,
        1,
        cv2.LINE_AA,
    )

    if not series:
        return

    # Draw metric line — stretch all frames to fill full width.
    n = len(series)
    for i in range(1, n):
        x1p = x + int((i - 1) / max(n - 1, 1) * (w - 1))
        x2p = x + int(i / max(n - 1, 1) * (w - 1))
        p1 = (x1p, _val_to_py(series[i - 1]))
        p2 = (x2p, _val_to_py(series[i]))
        cv2.line(frame, p1, p2, C_WHITE, 2, cv2.LINE_AA)

    # Tip dot.
    cv2.circle(frame, (x + w - 1, _val_to_py(series[-1])), 3, C_ORANGE, -1, cv2.LINE_AA)


def draw_feedback_visualizer(
    frame,
    exercise_name,
    feedback_state,
    rep_count,
    switch_state=None,
):
    """Render dual sparkline: ACTIVE on top, PREDICTED below.
    Red dashed line on predicted = switch progress threshold."""
    if exercise_name is None or feedback_state is None:
        return

    active_stream = list(feedback_state.rolling_buffer)
    predicted_ex = switch_state.get("predicted_exercise") if switch_state else None
    predicted_stream = (
        list(switch_state.get("predicted_stream", [])) if switch_state else []
    )

    fh, fw = frame.shape[:2]

    # Layout constants.
    PAD = 8
    LABEL_H = 22
    SPARK_H = 52
    CARD_H = LABEL_H + SPARK_H + 12
    GAP = 6
    P_W = min(310, max(200, int(fw * 0.26)))
    P_H = PAD + CARD_H + GAP + CARD_H + PAD
    P_X = fw - P_W
    P_Y = 0

    _pill(frame, P_X, P_Y, P_W, P_H, alpha=0.76)
    spark_w = P_W - PAD * 2

    for idx, (ex, stream, lbl) in enumerate(
        [
            (exercise_name, active_stream, "ACTIVE"),
            (predicted_ex, predicted_stream, "PREDICTED"),
        ]
    ):
        card_y = P_Y + PAD + idx * (CARD_H + GAP)
        spark_x = P_X + PAD
        spark_y = card_y + LABEL_H

        # Label + exercise name.
        _txt(frame, lbl, spark_x, card_y + 10, C_MUTED, scale=0.30)
        lbl_w, _ = _txt_size(lbl, scale=0.30)
        ex_text = ex.upper() if ex else "\u2014"
        ex_col = C_ORANGE if ex else C_MUTED
        _txt(frame, ex_text, spark_x + lbl_w + 6, card_y + 10, ex_col, scale=0.34)

        metric = _get_metric_series(ex, stream)

        if metric:
            _draw_sparkline(
                frame,
                metric["series"],
                metric["y_bounds"][0],
                metric["y_bounds"][1],
                metric["top_t"],
                metric["bot_t"],
                spark_x,
                spark_y,
                spark_w,
                SPARK_H,
            )
            cur = metric["series"][-1]
            cur_label = f"{cur:.2f}"
            _txt(frame, cur_label, spark_x, spark_y + SPARK_H + 10, C_WHITE, scale=0.28)
        else:
            ov = frame.copy()
            cv2.rectangle(
                ov,
                (spark_x, spark_y),
                (spark_x + spark_w, spark_y + SPARK_H),
                C_DARK,
                -1,
            )
            cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
            _txt(
                frame,
                "no data",
                spark_x + 3,
                spark_y + SPARK_H // 2 + 4,
                C_MUTED,
                scale=0.26,
            )

        # Red dashed line on predicted card = switch progress threshold.
        if idx == 1 and metric:
            top_t = metric["top_t"]
            bot_t = metric["bot_t"]
            thresh = SWITCH_PROGRESS_THRESHOLD
            start_ext = (
                switch_state.get("predicted_start_extremity") if switch_state else None
            )
            if start_ext == "top":
                start_level, target_level = top_t, bot_t
            elif start_ext == "bottom":
                start_level, target_level = bot_t, top_t
            else:
                start_level = min(top_t, bot_t)
                target_level = max(top_t, bot_t)
            thresh_val = start_level + thresh * (target_level - start_level)
            y_min, y_max = metric["y_bounds"]
            r = np.clip((thresh_val - y_min) / (y_max - y_min), 0, 1)
            thresh_y = int(spark_y + SPARK_H - r * SPARK_H)
            seg = 5
            for sx in range(spark_x, spark_x + spark_w, seg * 2):
                cv2.line(
                    frame,
                    (sx, thresh_y),
                    (min(sx + seg, spark_x + spark_w), thresh_y),
                    C_RED,
                    1,
                    cv2.LINE_AA,
                )
