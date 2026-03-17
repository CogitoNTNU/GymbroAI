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
    frame, series, thresholds, y_min, y_max, panel_x, panel_y, graph_w, graph_h
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
                0.38,
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


def draw_feedback_visualizer(frame, exercise_name, feedback_state, rep_count):
    """Draw a compact panel showing the data stream and feedback decision path."""
    if exercise_name is None or feedback_state is None:
        return

    stream = list(feedback_state.rolling_buffer)
    metric_data = _metric_series(exercise_name, stream)
    if metric_data is None:
        return

    panel_w = 470
    panel_h = 300 if exercise_name == "shoulder_press" else 260
    panel_x = max(frame.shape[1] - panel_w - 18, 10)
    panel_y = 15

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
        f"Feedback Visualizer [{exercise_name}]",
        (panel_x + 10, panel_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (230, 230, 230),
        1,
    )

    graph_x = panel_x + 10
    graph_y = panel_y + 36
    graph_w = panel_w - 20
    graph_h = 110
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
    )

    current_metric = metric_data["series"][-1] if metric_data["series"] else 0.0
    cv2.putText(
        frame,
        f"stream metric: {metric_data['label']} ({current_metric:.1f})",
        (panel_x + 10, panel_y + 164),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (180, 220, 255),
        1,
    )

    text_bottom = panel_y + 184
    if exercise_name == "shoulder_press":
        wrist_offset_series = metric_data.get("wrist_offset_series", [])
        wrist_offset = wrist_offset_series[-1] if wrist_offset_series else 0.0
        top_angle_ok = current_metric > SHOULDER_PRESS_TOP_ELBOW_ANGLE
        top_offset_ok = wrist_offset > SHOULDER_PRESS_TOP_WRIST_OFFSET
        bottom_angle_ok = current_metric < SHOULDER_PRESS_BOTTOM_ELBOW_ANGLE
        bottom_offset_ok = wrist_offset <= SHOULDER_PRESS_BOTTOM_WRIST_OFFSET

        cv2.putText(
            frame,
            f"wrist offset: {wrist_offset:.3f} (top>{SHOULDER_PRESS_TOP_WRIST_OFFSET:.2f}, bottom<={SHOULDER_PRESS_BOTTOM_WRIST_OFFSET:.2f})",
            (panel_x + 10, panel_y + 184),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (180, 220, 255),
            1,
        )
        cv2.putText(
            frame,
            f"TOP condition: angle={top_angle_ok} and offset={top_offset_ok}",
            (panel_x + 10, panel_y + 202),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (215, 215, 215),
            1,
        )
        cv2.putText(
            frame,
            f"BOTTOM condition: angle={bottom_angle_ok} and offset={bottom_offset_ok}",
            (panel_x + 10, panel_y + 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (215, 215, 215),
            1,
        )
        text_bottom = panel_y + 240

    decision_lines = [
        f"rep count: {rep_count} (last feedback rep: {feedback_state.last_rep_count})",
        f"cycle state: start={feedback_state.cycle_start_extremity}, opposite_seen={feedback_state.seen_opposite_extremity}",
        f"cycle frames: {feedback_state.frames_in_cycle}, buffer: {len(stream)} frames",
        "feedback updates only when rep count increases",
        "final message = highest-priority failed check",
    ]

    live_values = _get_live_values(exercise_name, stream[-1] if stream else None)
    lines = decision_lines + live_values

    y = text_bottom
    for line in lines[:6]:
        cv2.putText(
            frame,
            line,
            (panel_x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (215, 215, 215),
            1,
        )
        y += 18
