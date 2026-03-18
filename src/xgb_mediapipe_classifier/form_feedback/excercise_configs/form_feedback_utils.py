import numpy as np


def get_axis_series(rep_stream, landmark_name, axis):
    return [frame[f"{landmark_name}_{axis}"] for frame in rep_stream]


def get_mean_axis_series(rep_stream, left_landmark, right_landmark, axis):
    return [
        (frame[f"{left_landmark}_{axis}"] + frame[f"{right_landmark}_{axis}"]) / 2.0
        for frame in rep_stream
    ]


def choose_feedback_message(rules, default_message="Good form"):
    """Return the highest-priority failed rule, or default if all rules pass."""
    failed_rules = [rule for rule in rules if not rule["passed"]]
    if not failed_rules:
        return default_message
    failed_rules.sort(key=lambda rule: rule.get("priority", 999))
    return failed_rules[0]["message"]


def calculate_angle(p1, p2, p3):
    a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos_angle = np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def joint_angle(lm, a_name, b_name, c_name):
    p1 = (lm[f"{a_name}_x"], lm[f"{a_name}_y"])
    p2 = (lm[f"{b_name}_x"], lm[f"{b_name}_y"])
    p3 = (lm[f"{c_name}_x"], lm[f"{c_name}_y"])
    return calculate_angle(p1, p2, p3)


def descent_metrics(angle_series, dt):
    duration = max((len(angle_series) - 1) * dt, dt)
    velocity = abs(angle_series[-1] - angle_series[0]) / duration
    return duration, velocity


def motion_metrics(angle_series, dt):
    """Compute coarse motion properties for the captured movement phase."""
    if not angle_series:
        return {
            "duration": dt,
            "velocity": 0.0,
            "angle_span": 0.0,
            "start_angle": 0.0,
            "end_angle": 0.0,
        }

    duration = max((len(angle_series) - 1) * dt, dt)
    start_angle = angle_series[0]
    end_angle = angle_series[-1]
    angle_span = float(max(angle_series) - min(angle_series))
    velocity = abs(end_angle - start_angle) / duration
    return {
        "duration": duration,
        "velocity": velocity,
        "angle_span": angle_span,
        "start_angle": start_angle,
        "end_angle": end_angle,
    }
