import numpy as np

from .form_feedback_utils import (
    calculate_angle,
    choose_feedback_message,
    joint_angle,
    motion_metrics,
)

# ── Elbow position ─────────────────────────────────────────────────────────────
# How far each elbow is allowed to drift sideways from directly below its own shoulder,
# expressed as a fraction of shoulder width.
#   0.0  = elbow must be exactly under the shoulder (impossible to achieve perfectly)
#   0.25 = elbow may be up to 25 % of shoulder width away — a reasonably strict but fair limit
# Raise this value if "Keep elbows pinned" fires when your form actually looks fine.
ELBOW_DRIFT_MAX = 0.45

# Elbow should stay close to a right angle (90 degrees) relative to the shoulder line.
# This is measured for both sides through the rep.
SHOULDER_ELBOW_TARGET_ANGLE = 90.0
SHOULDER_ELBOW_ANGLE_TOLERANCE = 20.0

# ── Tempo ──────────────────────────────────────────────────────────────────────
# Minimum time (seconds) the descent phase (lowering the weight) should take.
# Raise this to demand a slower, more controlled descent.
MIN_DESCENT_SECONDS = 2.0

# Maximum average angular velocity (degrees / second) during the descent.
# Raise this to allow faster lowering before triggering feedback.
MAX_DESCENT_VELOCITY = 85.0

# Ignore tempo entirely if the elbow moved through fewer than this many degrees.
# Prevents false feedback on very short or partial reps.
MIN_USEFUL_ANGLE_SPAN = 28.0


def _build_metrics(rep_stream, dt):
    left_elbow_angles = [
        joint_angle(frame, "left_shoulder", "left_elbow", "left_wrist")
        for frame in rep_stream
    ]
    right_elbow_angles = [
        joint_angle(frame, "right_shoulder", "right_elbow", "right_wrist")
        for frame in rep_stream
    ]

    # Average of left and right elbow angles per frame (shoulder → elbow → wrist).
    elbow_angles = [
        (left_angle + right_angle) / 2.0
        for left_angle, right_angle in zip(left_elbow_angles, right_elbow_angles)
    ]

    # Per-frame lateral drift of each elbow from its own shoulder,
    # normalised to shoulder width so that body size doesn't matter.
    shoulder_width = [
        max(abs(frame["right_shoulder_x"] - frame["left_shoulder_x"]), 1e-6)
        for frame in rep_stream
    ]
    left_elbow_drift = [
        abs(frame["left_elbow_x"] - frame["left_shoulder_x"]) / sw
        for frame, sw in zip(rep_stream, shoulder_width)
    ]
    right_elbow_drift = [
        abs(frame["right_elbow_x"] - frame["right_shoulder_x"]) / sw
        for frame, sw in zip(rep_stream, shoulder_width)
    ]

    # Angle between shoulder line and each shoulder->elbow segment.
    # Left side angle: right_shoulder -> left_shoulder -> left_elbow
    # Right side angle: left_shoulder -> right_shoulder -> right_elbow
    shoulder_elbow_line_angles = []
    for frame in rep_stream:
        left_angle = calculate_angle(
            (frame["right_shoulder_x"], frame["right_shoulder_y"]),
            (frame["left_shoulder_x"], frame["left_shoulder_y"]),
            (frame["left_elbow_x"], frame["left_elbow_y"]),
        )
        right_angle = calculate_angle(
            (frame["left_shoulder_x"], frame["left_shoulder_y"]),
            (frame["right_shoulder_x"], frame["right_shoulder_y"]),
            (frame["right_elbow_x"], frame["right_elbow_y"]),
        )
        shoulder_elbow_line_angles.append((left_angle + right_angle) / 2.0)

    return {
        "elbow_angles": elbow_angles,
        "shoulder_elbow_line_angles": shoulder_elbow_line_angles,
        "left_elbow_drift": left_elbow_drift,
        "right_elbow_drift": right_elbow_drift,
        "motion": motion_metrics(elbow_angles, dt),
    }


def _evaluate_elbow_position(metrics):
    shoulder_elbow_errors = [
        abs(angle - SHOULDER_ELBOW_TARGET_ANGLE)
        for angle in metrics["shoulder_elbow_line_angles"]
    ]

    return [
        {
            # Use percentile to avoid failing from 1-2 noisy frames.
            "passed": np.percentile(shoulder_elbow_errors, 90)
            <= SHOULDER_ELBOW_ANGLE_TOLERANCE,
            "message": "pin elbows to sides",
            "priority": 1,
        },
    ]


def _evaluate_tempo(metrics):
    # Skip tempo checks when there wasn't enough movement to judge meaningfully.
    has_useful_motion = metrics["motion"]["angle_span"] >= MIN_USEFUL_ANGLE_SPAN
    return [
        {
            "passed": (
                not has_useful_motion
                or metrics["motion"]["duration"] > MIN_DESCENT_SECONDS
            ),
            "message": "Control the lowering speed",
            "priority": 5,
        },
        {
            "passed": (
                not has_useful_motion
                or metrics["motion"]["velocity"] <= MAX_DESCENT_VELOCITY
            ),
            "message": "Lower with smoother control",
            "priority": 6,
        },
    ]


def analyze_rep(rep_stream, dt):
    metrics = _build_metrics(rep_stream, dt)
    rules = []
    rules.extend(_evaluate_elbow_position(metrics))
    rules.extend(_evaluate_tempo(metrics))
    return choose_feedback_message(rules)
