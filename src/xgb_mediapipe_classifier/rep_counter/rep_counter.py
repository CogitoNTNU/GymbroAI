from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from form_feedback.excercise_configs.form_feedback_utils import calculate_angle

# Lowered thresholds to make extremity confirmation and rep detection less strict.
REP_EXTREMITY_STREAK = 2

SQUAT_BOTTOM_ANGLE = 120
SQUAT_TOP_ANGLE = 150

CURL_TOP_ANGLE = 50
CURL_BOTTOM_ANGLE = 135

SHOULDER_PRESS_TOP_WRIST_OFFSET = 0.06
SHOULDER_PRESS_TOP_ELBOW_ANGLE = 140
SHOULDER_PRESS_BOTTOM_WRIST_OFFSET = 0.04
SHOULDER_PRESS_BOTTOM_ELBOW_ANGLE = 100

REP_COUNT_AT = {
    "squat": "bottom",
    "curl": "top",
    "shoulder_press": "top",
}


@dataclass
class RepState:
    last_extremity: Optional[str] = None
    pending_extremity: Optional[str] = None
    pending_count: int = 0


def create_rep_counter_state(encoder):
    rep_counts = {exercise_name: 0 for exercise_name in encoder.classes_}
    rep_states = defaultdict(RepState)
    return rep_counts, rep_states


def detect_rep_extremity(curr_lm, exercise_name):
    angle_triplets = {
        "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
        "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
        "left_knee": ("left_hip", "left_knee", "left_ankle"),
        "right_knee": ("right_hip", "right_knee", "right_ankle"),
    }

    angles = {}
    for name, (a_name, b_name, c_name) in angle_triplets.items():
        p1 = (curr_lm[f"{a_name}_x"], curr_lm[f"{a_name}_y"])
        p2 = (curr_lm[f"{b_name}_x"], curr_lm[f"{b_name}_y"])
        p3 = (curr_lm[f"{c_name}_x"], curr_lm[f"{c_name}_y"])
        angles[name] = calculate_angle(p1, p2, p3)

    elbow_angle = (angles["left_elbow"] + angles["right_elbow"]) / 2.0
    knee_angle = (angles["left_knee"] + angles["right_knee"]) / 2.0
    shoulder_y = (curr_lm["left_shoulder_y"] + curr_lm["right_shoulder_y"]) / 2.0
    wrist_y = (curr_lm["left_wrist_y"] + curr_lm["right_wrist_y"]) / 2.0

    if exercise_name == "squat":
        if knee_angle < SQUAT_BOTTOM_ANGLE:
            return "bottom"
        if knee_angle > SQUAT_TOP_ANGLE:
            return "top"
        return None

    if exercise_name == "curl":
        if elbow_angle < CURL_TOP_ANGLE:
            return "top"
        if elbow_angle > CURL_BOTTOM_ANGLE:
            return "bottom"
        return None

    if exercise_name == "shoulder_press":
        if (
            wrist_y < shoulder_y - SHOULDER_PRESS_TOP_WRIST_OFFSET
            and elbow_angle > SHOULDER_PRESS_TOP_ELBOW_ANGLE
        ):
            return "top"
        if (
            wrist_y >= shoulder_y - SHOULDER_PRESS_BOTTOM_WRIST_OFFSET
            and elbow_angle < SHOULDER_PRESS_BOTTOM_ELBOW_ANGLE
        ):
            return "bottom"
        return None

    return None


def _confirm_extremity_transition(state, extremity):
    if state.last_extremity == extremity:
        state.pending_extremity = None
        state.pending_count = 0
        return False

    if state.pending_extremity == extremity:
        state.pending_count += 1
    else:
        state.pending_extremity = extremity
        state.pending_count = 1

    if state.pending_count < REP_EXTREMITY_STREAK:
        return False

    state.pending_extremity = None
    state.pending_count = 0
    return True


def update_rep_counts(exercise_name, curr_lm, rep_counts, rep_states):
    if exercise_name is None:
        return

    extremity = detect_rep_extremity(curr_lm, exercise_name)
    if extremity is None:
        return

    state = rep_states[exercise_name]
    if not _confirm_extremity_transition(state, extremity):
        return

    if state.last_extremity is None:
        state.last_extremity = extremity
        return

    count_at = REP_COUNT_AT.get(exercise_name, "top")
    if extremity == count_at:
        rep_counts[exercise_name] += 1
    state.last_extremity = extremity


def reset_rep_counter_tracking(rep_states):
    for state in rep_states.values():
        state.pending_extremity = None
        state.pending_count = 0
