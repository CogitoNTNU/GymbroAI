"""
Form feedback analyser for exercise reps.

Pipeline:
    1. Each frame, get_form_feedback() is called with the current landmarks.
    2. detect_rep_extremity() determines if we're at "top", "bottom", or in between.
    3. Frames are collected into a buffer while the user is mid-rep.
    4. A full rep is bottom -> top -> bottom. When returning to "bottom",
       the buffer is passed to an exercise-specific analyser.
    5. Each analyser runs a chain of checks (if/elif). First failure wins.
       If all checks pass, returns "Good form".
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from runtime_logic.rep_counter import detect_rep_extremity


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Curl: shoulder-shoulder-elbow angle should be ~90° (upper arm vertical).
CURL_TARGET_ANGLE = 90.0
CURL_ANGLE_TOLERANCE = 8.0

# Shoulder press: max allowed average Y difference between wrists.
SHOULDER_PRESS_WRIST_Y_TOLERANCE = 0.03

# Squat: tolerance before a knee is considered caving inward (X coords).
SQUAT_KNEE_INWARD_TOLERANCE = 0.01
# Fraction of frames where knees cave before flagging it.
SQUAT_KNEE_INWARD_RATIO = 0.3

# All exercises: minimum frames of downward motion for controlled negative.
# ~2 seconds at 30fps.
MIN_DOWN_FRAMES = 60


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class FeedbackState:
    message: str | None = None
    last_extremity: str | None = None
    collecting: bool = False
    rolling_buffer: list = field(default_factory=list)

    def reset(self):
        self.last_extremity = None
        self.collecting = False
        self.rolling_buffer.clear()


def create_feedback_state():
    return defaultdict(FeedbackState)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def calculate_angle(p1, p2, p3):
    """Angle in degrees at p2 formed by p1-p2-p3."""
    a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos_angle = np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _shoulder_shoulder_elbow_angle(frame, side):
    """Angle at the working shoulder between the opposite shoulder and elbow.
    ~90° when the upper arm hangs straight down."""
    opposite = "right" if side == "left" else "left"
    try:
        opp_shoulder = (
            frame[f"{opposite}_shoulder_x"],
            frame[f"{opposite}_shoulder_y"],
        )
        shoulder = (frame[f"{side}_shoulder_x"], frame[f"{side}_shoulder_y"])
        elbow = (frame[f"{side}_elbow_x"], frame[f"{side}_elbow_y"])
    except KeyError:
        return None
    return calculate_angle(opp_shoulder, shoulder, elbow)


def _count_down_frames(rep_stream, y_key):
    """Count frames where Y is increasing (moving down in mediapipe coords)."""
    prev = None
    count = 0
    for frame in rep_stream:
        val = frame.get(y_key)
        if val is None:
            continue
        if prev is not None and val > prev:
            count += 1
        prev = val
    return count


# ---------------------------------------------------------------------------
# Exercise analysers — each is a chain of checks, first failure wins.
# ---------------------------------------------------------------------------


def _analyze_curl_rep(rep_stream):
    if not rep_stream or len(rep_stream) < 4:
        return None

    # Check 1: elbows pinned to torso
    angles = []
    for frame in rep_stream:
        for side in ("left", "right"):
            a = _shoulder_shoulder_elbow_angle(frame, side)
            if a is not None:
                angles.append(a)

    if angles and abs(np.mean(angles) - CURL_TARGET_ANGLE) > CURL_ANGLE_TOLERANCE:
        return "Pin elbows to torso"

    # Check 2: controlled negative
    if _count_down_frames(rep_stream, "left_wrist_y") < MIN_DOWN_FRAMES:
        return "Slow down the negative"

    return "Good form"


def _analyze_shoulder_press_rep(rep_stream):
    if not rep_stream or len(rep_stream) < 4:
        return None

    # Check 1: bar level
    diffs = []
    for frame in rep_stream:
        ly = frame.get("left_wrist_y")
        ry = frame.get("right_wrist_y")
        if ly is not None and ry is not None:
            diffs.append(abs(ly - ry))

    if diffs and np.mean(diffs) > SHOULDER_PRESS_WRIST_Y_TOLERANCE:
        return "Keep the bar level"

    # Check 2: controlled negative
    if _count_down_frames(rep_stream, "left_wrist_y") < MIN_DOWN_FRAMES:
        return "Slow down the negative"

    return "Good form"


def _analyze_squat_rep(rep_stream):
    if not rep_stream or len(rep_stream) < 4:
        return None

    # Check 1: knees pointing outward
    inward_count = 0
    total = 0
    for frame in rep_stream:
        lk_x = frame.get("left_knee_x")
        lh_x = frame.get("left_hip_x")
        rk_x = frame.get("right_knee_x")
        rh_x = frame.get("right_hip_x")
        if None in (lk_x, lh_x, rk_x, rh_x):
            continue
        total += 1
        if (
            lk_x < lh_x - SQUAT_KNEE_INWARD_TOLERANCE
            or rk_x > rh_x + SQUAT_KNEE_INWARD_TOLERANCE
        ):
            inward_count += 1

    if total > 0 and inward_count / total > SQUAT_KNEE_INWARD_RATIO:
        return "Push knees outward"

    # Check 2: controlled negative
    if _count_down_frames(rep_stream, "left_hip_y") < MIN_DOWN_FRAMES:
        return "Slow down the negative"

    return "Good form"


# Map exercise name to its analyser.
_ANALYSERS = {
    "curl": _analyze_curl_rep,
    "shoulder_press": _analyze_shoulder_press_rep,
    "squat": _analyze_squat_rep,
}


# ---------------------------------------------------------------------------
# Main feedback loop — called every frame from main.py
# ---------------------------------------------------------------------------


def get_form_feedback(exercise_name, curr_lm, feedback_states, rep_counts):
    """Collect frames during a rep, analyse when rep completes (returns to bottom)."""
    if exercise_name is None:
        return None

    state = feedback_states[exercise_name]
    extremity = detect_rep_extremity(curr_lm, exercise_name)

    # Mid-movement: keep collecting.
    if extremity is None:
        if state.collecting and curr_lm is not None:
            state.rolling_buffer.append(dict(curr_lm))
        return state.message

    # At an extremity: record the frame.
    if state.collecting and curr_lm is not None:
        state.rolling_buffer.append(dict(curr_lm))

    # Returned to bottom -> full rep complete, analyse.
    if extremity == "bottom" and state.last_extremity != "bottom":
        analyser = _ANALYSERS.get(exercise_name)
        if analyser and state.collecting and state.rolling_buffer:
            state.message = analyser(state.rolling_buffer)
        state.rolling_buffer.clear()
        state.collecting = True

    state.last_extremity = extremity
    return state.message


def get_feedback_message(exercise_name, feedback_states):
    if exercise_name is None:
        return None
    return feedback_states[exercise_name].message


def reset_form_feedback_tracking(feedback_states):
    for state in feedback_states.values():
        state.reset()
