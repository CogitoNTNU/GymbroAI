import cv2

from form_feedback.excercise_configs.form_feedback_utils import calculate_angle
from rep_counter.rep_counter import (
    CURL_BOTTOM_ANGLE,
    CURL_TOP_ANGLE,
    SHOULDER_PRESS_BOTTOM_ELBOW_ANGLE,
    SHOULDER_PRESS_BOTTOM_WRIST_OFFSET,
    SHOULDER_PRESS_TOP_ELBOW_ANGLE,
    SHOULDER_PRESS_TOP_WRIST_OFFSET,
    SQUAT_BOTTOM_ANGLE,
    SQUAT_TOP_ANGLE,
)


def _joint_angle(frame, a_name, b_name, c_name):
    p1 = (frame[f"{a_name}_x"], frame[f"{a_name}_y"])
    p2 = (frame[f"{b_name}_x"], frame[f"{b_name}_y"])
    p3 = (frame[f"{c_name}_x"], frame[f"{c_name}_y"])
    return calculate_angle(p1, p2, p3)


def _metric_series(exercise_name, stream):
    if exercise_name == "curl":
        series = [
            (
                _joint_angle(f, "left_shoulder", "left_elbow", "left_wrist")
                + _joint_angle(f, "right_shoulder", "right_elbow", "right_wrist")
            )
            / 2.0
            for f in stream
        ]
        return {
            "label": "Elbow angle",
            "series": series,
            "y_bounds": (30.0, 175.0),
            "thresholds": [
                (CURL_TOP_ANGLE, (80, 255, 80), "top"),
                (CURL_BOTTOM_ANGLE, (80, 200, 255), "bottom"),
            ],
        }

    if exercise_name == "squat":
        series = [
            (
                _joint_angle(f, "left_hip", "left_knee", "left_ankle")
                + _joint_angle(f, "right_hip", "right_knee", "right_ankle")
            )
            / 2.0
            for f in stream
        ]
        return {
            "label": "Knee angle",
            "series": series,
            "y_bounds": (70.0, 185.0),
            "thresholds": [
                (SQUAT_BOTTOM_ANGLE, (80, 255, 80), "bottom"),
                (SQUAT_TOP_ANGLE, (80, 200, 255), "top"),
            ],
        }

    if exercise_name == "shoulder_press":
        shoulder_y_series = [
            (f["left_shoulder_y"] + f["right_shoulder_y"]) / 2.0 for f in stream
        ]
        wrist_y_series = [
            (f["left_wrist_y"] + f["right_wrist_y"]) / 2.0 for f in stream
        ]
        wrist_offset_series = [
            shoulder_y - wrist_y
            for shoulder_y, wrist_y in zip(shoulder_y_series, wrist_y_series)
        ]

        series = [
            (
                _joint_angle(f, "left_shoulder", "left_elbow", "left_wrist")
                + _joint_angle(f, "right_shoulder", "right_elbow", "right_wrist")
            )
            / 2.0
            for f in stream
        ]
        return {
            "label": "Elbow angle",
            "series": series,
            "y_bounds": (70.0, 175.0),
            "thresholds": [
                (SHOULDER_PRESS_TOP_ELBOW_ANGLE, (80, 255, 80), "top"),
                (SHOULDER_PRESS_BOTTOM_ELBOW_ANGLE, (80, 200, 255), "bottom"),
            ],
            "wrist_offset_series": wrist_offset_series,
        }

    return None


def _draw_series_graph(
    frame,
    series,
    thresholds,
    y_min,
    y_max,
    panel_x,
    panel_y,
    graph_w,
    graph_h,
    threshold_scale=0.38,
):
    if y_max - y_min < 1e-6:
        y_max = y_min + 1.0

    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + graph_w, panel_y + graph_h),
        (70, 70, 70),
        1,
    )

    for value, color, name in thresholds:
        ratio = (value - y_min) / max(y_max - y_min, 1e-6)
        y = int(panel_y + graph_h - ratio * graph_h)
        if panel_y <= y <= panel_y + graph_h:
            cv2.line(frame, (panel_x, y), (panel_x + graph_w, y), color, 1)
            cv2.putText(
                frame,
                name,
                (panel_x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                threshold_scale,
                color,
                1,
            )

    max_points = graph_w
    visible = series[-max_points:]
    if len(visible) == 1:
        ratio = (visible[0] - y_min) / max(y_max - y_min, 1e-6)
        y = int(panel_y + graph_h - ratio * graph_h)
        cv2.circle(frame, (panel_x + graph_w - 1, y), 3, (0, 255, 255), -1)
        return

    for i in range(1, len(visible)):
        x1 = panel_x + i - 1
        x2 = panel_x + i
        r1 = (visible[i - 1] - y_min) / max(y_max - y_min, 1e-6)
        r2 = (visible[i] - y_min) / max(y_max - y_min, 1e-6)
        y1 = int(panel_y + graph_h - r1 * graph_h)
        y2 = int(panel_y + graph_h - r2 * graph_h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)


def _get_live_values(exercise_name, frame):
    if frame is None:
        return []

    if exercise_name == "curl":
        elbow_angle = (
            _joint_angle(frame, "left_shoulder", "left_elbow", "left_wrist")
            + _joint_angle(frame, "right_shoulder", "right_elbow", "right_wrist")
        ) / 2.0
        return [f"live elbow: {elbow_angle:.1f} deg"]

    if exercise_name == "squat":
        knee_angle = (
            _joint_angle(frame, "left_hip", "left_knee", "left_ankle")
            + _joint_angle(frame, "right_hip", "right_knee", "right_ankle")
        ) / 2.0
        return [f"live knee: {knee_angle:.1f} deg"]

    if exercise_name == "shoulder_press":
        elbow_angle = (
            _joint_angle(frame, "left_shoulder", "left_elbow", "left_wrist")
            + _joint_angle(frame, "right_shoulder", "right_elbow", "right_wrist")
        ) / 2.0
        shoulder_y = (frame["left_shoulder_y"] + frame["right_shoulder_y"]) / 2.0
        wrist_y = (frame["left_wrist_y"] + frame["right_wrist_y"]) / 2.0
        wrist_offset = shoulder_y - wrist_y
        return [
            f"live elbow: {elbow_angle:.1f} deg",
            f"live wrist offset: {wrist_offset:.3f}",
            f"top uses: off>{SHOULDER_PRESS_TOP_WRIST_OFFSET:.2f} + ang>{SHOULDER_PRESS_TOP_ELBOW_ANGLE}",
            f"bot uses: off<={SHOULDER_PRESS_BOTTOM_WRIST_OFFSET:.2f} + ang<{SHOULDER_PRESS_BOTTOM_ELBOW_ANGLE}",
        ]

    return []


def _draw_stream_card(
    frame,
    title,
    exercise_name,
    stream,
    card_x,
    card_y,
    card_w,
    card_h,
    ui_scale,
):
    cv2.rectangle(
        frame,
        (card_x, card_y),
        (card_x + card_w, card_y + card_h),
        (65, 65, 65),
        1,
    )

    cv2.putText(
        frame,
        title,
        (card_x + 8, card_y + max(14, int(18 * ui_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45 * ui_scale,
        (220, 220, 220),
        1,
    )

    if exercise_name is None:
        cv2.putText(
            frame,
            "exercise: none",
            (card_x + 8, card_y + max(30, int(36 * ui_scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40 * ui_scale,
            (160, 160, 160),
            1,
        )
        return

    cv2.putText(
        frame,
        f"exercise: {exercise_name}",
        (card_x + 8, card_y + max(30, int(36 * ui_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40 * ui_scale,
        (180, 220, 255),
        1,
    )

    if not stream:
        cv2.putText(
            frame,
            "stream empty",
            (card_x + 8, card_y + max(48, int(56 * ui_scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40 * ui_scale,
            (160, 160, 160),
            1,
        )
        return

    metric_data = _metric_series(exercise_name, stream)
    if metric_data is None:
        cv2.putText(
            frame,
            "no metric",
            (card_x + 8, card_y + max(48, int(56 * ui_scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40 * ui_scale,
            (160, 160, 160),
            1,
        )
        return

    graph_x = card_x + 8
    graph_y = card_y + max(44, int(52 * ui_scale))
    graph_w = max(40, card_w - 16)
    graph_h = max(50, int(card_h * 0.42))
    y_min, y_max = metric_data["y_bounds"]

    _draw_series_graph(
        frame,
        metric_data["series"],
        metric_data["thresholds"],
        y_min,
        y_max,
        graph_x,
        graph_y,
        graph_w,
        graph_h,
        threshold_scale=max(0.26, 0.34 * ui_scale),
    )

    current_metric = metric_data["series"][-1] if metric_data["series"] else 0.0
    info_y = graph_y + graph_h + max(14, int(16 * ui_scale))
    cv2.putText(
        frame,
        f"{metric_data['label']}: {current_metric:.1f}",
        (card_x + 8, info_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40 * ui_scale,
        (200, 220, 255),
        1,
    )
    cv2.putText(
        frame,
        f"frames: {len(stream)}",
        (card_x + 8, info_y + max(14, int(16 * ui_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40 * ui_scale,
        (200, 200, 200),
        1,
    )


def draw_feedback_visualizer(
    frame,
    exercise_name,
    feedback_state,
    rep_count,
    switch_state=None,
):
    """Draw a compact panel showing the data stream and feedback decision path."""
    if exercise_name is None or feedback_state is None:
        return

    active_stream = list(feedback_state.rolling_buffer)
    predicted_exercise = None
    predicted_stream = []
    predicted_progress = 0.0
    predicted_start_extremity = None
    if switch_state is not None:
        predicted_exercise = switch_state.get("predicted_exercise")
        predicted_stream = list(switch_state.get("predicted_stream", []))
        predicted_progress = switch_state.get("predicted_progress", 0.0)
        predicted_start_extremity = switch_state.get("predicted_start_extremity")

    frame_h, frame_w = frame.shape[:2]
    panel_margin = max(10, int(min(frame_w, frame_h) * 0.02))

    # Wider panel to show active/predicted streams side-by-side.
    panel_w = int(frame_w * 0.52)
    panel_w = max(520, min(panel_w, 820))

    panel_h = int(frame_h * 0.48)
    panel_h = max(240, min(panel_h, 420))

    panel_x = max(frame_w - panel_w - panel_margin, 10)
    panel_y = panel_margin

    # Scale typography and spacing with panel width.
    ui_scale = max(0.75, min(1.0, panel_w / 460.0))
    title_scale = 0.56 * ui_scale
    body_scale = 0.40 * ui_scale
    line_step = max(12, int(18 * ui_scale))

    # Translucent panel background.
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (15, 15, 15),
        -1,
    )
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0.0, frame)
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (90, 90, 90),
        1,
    )

    cv2.putText(
        frame,
        "Feedback Visualizer [active + predicted streams]",
        (panel_x + 10, panel_y + max(16, int(22 * ui_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (230, 230, 230),
        1,
    )

    cards_top = panel_y + max(30, int(40 * ui_scale))
    cards_gap = max(10, int(12 * ui_scale))
    card_w = (panel_w - 20 - cards_gap) // 2
    card_h = max(120, int(panel_h * 0.55))

    _draw_stream_card(
        frame,
        "ACTIVE STREAM",
        exercise_name,
        active_stream,
        panel_x + 10,
        cards_top,
        card_w,
        card_h,
        ui_scale,
    )

    _draw_stream_card(
        frame,
        "PREDICTED STREAM",
        predicted_exercise,
        predicted_stream,
        panel_x + 10 + card_w + cards_gap,
        cards_top,
        card_w,
        card_h,
        ui_scale,
    )

    text_bottom = cards_top + card_h + max(16, int(20 * ui_scale))

    decision_lines = [
        f"rep count: {rep_count} (last feedback rep: {feedback_state.last_rep_count})",
        f"cycle state: start={feedback_state.cycle_start_extremity}, opposite_seen={feedback_state.seen_opposite_extremity}",
        f"cycle frames: {feedback_state.frames_in_cycle}, active buffer: {len(active_stream)} frames",
        "feedback updates only when rep count increases",
        "final message = highest-priority failed check",
        f"predicted: {predicted_exercise}, start={predicted_start_extremity}, progress={predicted_progress * 100:.1f}%",
        "switch gate: 15% predicted movement",
    ]

    live_values = _get_live_values(
        exercise_name, active_stream[-1] if active_stream else None
    )
    lines = decision_lines + live_values

    y = text_bottom
    panel_bottom = panel_y + panel_h
    max_lines = max(1, (panel_bottom - y - 6) // line_step)
    for line in lines[:max_lines]:
        cv2.putText(
            frame,
            line,
            (panel_x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            body_scale,
            (215, 215, 215),
            1,
        )
        y += line_step
