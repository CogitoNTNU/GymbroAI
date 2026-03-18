from .form_feedback_utils import (
    choose_feedback_message,
    joint_angle,
    motion_metrics,
)

# Judge knees-out near the bottom of the squat, where valgus matters most.
KNEE_OUT_RATIO_MIN = 0.74
KNEE_SYMMETRY_MAX = 18.0
HIP_SHIFT_MAX = 0.18
MIN_DESCENT_SECONDS = 2.0
MAX_DESCENT_VELOCITY = 70.0
MIN_USEFUL_ANGLE_SPAN = 20.0
BOTTOM_PHASE_MARGIN = 18.0


def _build_metrics(rep_stream, dt):
    knee_width = [
        abs(frame["right_knee_x"] - frame["left_knee_x"]) for frame in rep_stream
    ]
    ankle_width = [
        max(abs(frame["right_ankle_x"] - frame["left_ankle_x"]), 1e-6)
        for frame in rep_stream
    ]
    knee_out_ratio = [
        knee_span / ankle_span for knee_span, ankle_span in zip(knee_width, ankle_width)
    ]

    left_knee_angles = [
        joint_angle(frame, "left_hip", "left_knee", "left_ankle")
        for frame in rep_stream
    ]
    right_knee_angles = [
        joint_angle(frame, "right_hip", "right_knee", "right_ankle")
        for frame in rep_stream
    ]
    lr_knee_balance = [
        abs(left_angle - right_angle)
        for left_angle, right_angle in zip(left_knee_angles, right_knee_angles)
    ]
    knee_angles = [
        (left_angle + right_angle) / 2.0
        for left_angle, right_angle in zip(left_knee_angles, right_knee_angles)
    ]
    ankle_center_x = [
        (frame["left_ankle_x"] + frame["right_ankle_x"]) / 2.0 for frame in rep_stream
    ]
    hip_center_x = [
        (frame["left_hip_x"] + frame["right_hip_x"]) / 2.0 for frame in rep_stream
    ]
    hip_shift = [
        abs(hip_mid - ankle_mid) / ankle_span
        for hip_mid, ankle_mid, ankle_span in zip(
            hip_center_x, ankle_center_x, ankle_width
        )
    ]

    return {
        "knee_width": knee_width,
        "ankle_width": ankle_width,
        "knee_out_ratio": knee_out_ratio,
        "lr_knee_balance": lr_knee_balance,
        "hip_shift": hip_shift,
        "knee_angles": knee_angles,
        "motion": motion_metrics(knee_angles, dt),
    }


def _get_bottom_phase_values(values, knee_angles):
    # Only evaluate some squat cues near the deepest part of the rep.
    # Using the whole rep is too noisy because knees naturally re-stack more at the top.
    deepest_knee_angle = min(knee_angles)
    cutoff_angle = deepest_knee_angle + BOTTOM_PHASE_MARGIN
    bottom_phase_values = [
        value
        for value, knee_angle in zip(values, knee_angles)
        if knee_angle <= cutoff_angle
    ]
    return bottom_phase_values or values


def _evaluate_joint_tracking(metrics):
    bottom_phase_knee_out = _get_bottom_phase_values(
        metrics["knee_out_ratio"], metrics["knee_angles"]
    )

    return [
        {
            # Judge knee position at the bottom, not at the top or during transitions.
            "passed": min(bottom_phase_knee_out) >= KNEE_OUT_RATIO_MIN,
            "message": "Push knees out, not inward",
            "priority": 1,
        },
        {
            "passed": max(metrics["lr_knee_balance"]) <= KNEE_SYMMETRY_MAX,
            "message": "Keep knee bend more balanced",
            "priority": 2,
        },
        {
            "passed": max(metrics["hip_shift"]) <= HIP_SHIFT_MAX,
            "message": "Keep your weight centered over your feet",
            "priority": 3,
        },
    ]


def _evaluate_tempo(metrics):
    has_useful_motion = metrics["motion"]["angle_span"] >= MIN_USEFUL_ANGLE_SPAN
    return [
        {
            "passed": (
                not has_useful_motion
                or metrics["motion"]["duration"] > MIN_DESCENT_SECONDS
            ),
            "message": "Slow down on the way down",
            "priority": 5,
        },
        {
            "passed": (
                not has_useful_motion
                or metrics["motion"]["velocity"] <= MAX_DESCENT_VELOCITY
            ),
            "message": "Control squat descent a bit more",
            "priority": 6,
        },
    ]


def analyze_rep(rep_stream, dt):
    # Pipeline: derive metrics, evaluate rules, then emit a single actionable cue.
    metrics = _build_metrics(rep_stream, dt)
    rules = []
    rules.extend(_evaluate_joint_tracking(metrics))
    rules.extend(_evaluate_tempo(metrics))
    return choose_feedback_message(rules)
