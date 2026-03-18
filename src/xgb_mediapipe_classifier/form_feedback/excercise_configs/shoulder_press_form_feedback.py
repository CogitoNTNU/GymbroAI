from .form_feedback_utils import (
    choose_feedback_message,
    get_mean_axis_series,
    joint_angle,
    motion_metrics,
)

WRIST_STACK_MAX = 0.07
WRIST_LEVEL_MAX = 0.06
ELBOW_BALANCE_MAX = 15.0
MIN_DESCENT_SECONDS = 2.0
MAX_DESCENT_VELOCITY = 75.0
MIN_USEFUL_ANGLE_SPAN = 25.0


def _build_metrics(rep_stream, dt):
    left_elbow_angles = [
        joint_angle(frame, "left_shoulder", "left_elbow", "left_wrist")
        for frame in rep_stream
    ]
    right_elbow_angles = [
        joint_angle(frame, "right_shoulder", "right_elbow", "right_wrist")
        for frame in rep_stream
    ]
    elbow_angles = [
        (left_angle + right_angle) / 2.0
        for left_angle, right_angle in zip(left_elbow_angles, right_elbow_angles)
    ]
    shoulder_y = get_mean_axis_series(
        rep_stream, "left_shoulder", "right_shoulder", "y"
    )
    wrist_y = get_mean_axis_series(rep_stream, "left_wrist", "right_wrist", "y")
    wrist_level = [
        abs(frame["left_wrist_y"] - frame["right_wrist_y"]) for frame in rep_stream
    ]
    wrist_stack_offset = [
        max(
            abs(frame["left_wrist_x"] - frame["left_elbow_x"]),
            abs(frame["right_wrist_x"] - frame["right_elbow_x"]),
        )
        for frame in rep_stream
    ]
    elbow_balance = [
        abs(left_angle - right_angle)
        for left_angle, right_angle in zip(left_elbow_angles, right_elbow_angles)
    ]

    return {
        "left_elbow_angles": left_elbow_angles,
        "right_elbow_angles": right_elbow_angles,
        "elbow_angles": elbow_angles,
        "shoulder_y": shoulder_y,
        "wrist_y": wrist_y,
        "wrist_level": wrist_level,
        "wrist_stack_offset": wrist_stack_offset,
        "elbow_balance": elbow_balance,
        "motion": motion_metrics(elbow_angles, dt),
    }


def _evaluate_joint_tracking(metrics):
    return [
        {
            "passed": max(metrics["wrist_stack_offset"]) <= WRIST_STACK_MAX,
            "message": "Stack wrists directly above elbows",
            "priority": 1,
        },
        {
            "passed": max(metrics["wrist_level"]) <= WRIST_LEVEL_MAX,
            "message": "Keep wrists level",
            "priority": 2,
        },
        {
            "passed": max(metrics["elbow_balance"]) <= ELBOW_BALANCE_MAX,
            "message": "Keep both arms moving evenly",
            "priority": 3,
        },
    ]


def _evaluate_tempo(metrics):
    # Gate tempo checks so they do not dominate when movement range is too small.
    has_useful_motion = metrics["motion"]["angle_span"] >= MIN_USEFUL_ANGLE_SPAN
    return [
        {
            "passed": (
                not has_useful_motion
                or metrics["motion"]["duration"] > MIN_DESCENT_SECONDS
            ),
            "message": "Control the lowering phase",
            "priority": 6,
        },
        {
            "passed": (
                not has_useful_motion
                or metrics["motion"]["velocity"] <= MAX_DESCENT_VELOCITY
            ),
            "message": "Slow the descent slightly",
            "priority": 7,
        },
    ]


def analyze_rep(rep_stream, dt):
    # Evaluate movement quality in stages to keep the logic maintainable.
    metrics = _build_metrics(rep_stream, dt)
    rules = []
    rules.extend(_evaluate_joint_tracking(metrics))
    rules.extend(_evaluate_tempo(metrics))
    return choose_feedback_message(rules)
