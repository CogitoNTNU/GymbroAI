from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from rep_counter import detect_rep_extremity
from rep_counter.rep_counter import REP_COUNT_AT

from .excercise_configs.curl_form_feedback import analyze_rep as analyze_curl_rep
from .excercise_configs.shoulder_press_form_feedback import (
    analyze_rep as analyze_shoulder_press_rep,
)
from .excercise_configs.squat_form_feedback import analyze_rep as analyze_squat_rep

FRAME_DT = 0.033
BUFFER_SECONDS = 8
BUFFER_MAX_FRAMES = int(BUFFER_SECONDS / FRAME_DT)  # ~242 frames at 30 fps
REP_EXTREMITY_STREAK = 2

FORM_FEEDBACK_ANALYZERS = {
    "squat": analyze_squat_rep,
    "curl": analyze_curl_rep,
    "shoulder_press": analyze_shoulder_press_rep,
}

FORM_FEEDBACK_LANDMARKS = {
    "squat": [
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    "curl": [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ],
    "shoulder_press": [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ],
}


@dataclass
class FeedbackState:
    message: Optional[str] = None
    last_extremity: Optional[str] = None
    pending_extremity: Optional[str] = None
    pending_count: int = 0
    collecting: bool = False
    rep_completed_pending: bool = False
    cycle_start_extremity: Optional[str] = None
    seen_opposite_extremity: bool = False
    frames_in_cycle: int = 0
    # Track the rep count at the time of the last feedback update.
    # Feedback is only produced when this has been exceeded by the rep counter,
    # ensuring the system never fires on partial or phantom movements.
    last_rep_count: int = 0
    # Rolling buffer: always keeps the last 8 seconds of landmark data.
    rolling_buffer: deque = field(
        default_factory=lambda: deque(maxlen=BUFFER_MAX_FRAMES)
    )

    def reset(self):
        # On pose loss: clear cycle tracking and pending rep state.
        self.last_extremity = None
        self.pending_extremity = None
        self.pending_count = 0
        self.collecting = False
        self.rep_completed_pending = False
        self.cycle_start_extremity = None
        self.seen_opposite_extremity = False
        self.frames_in_cycle = 0
        self.rolling_buffer.clear()


def create_feedback_state():
    return defaultdict(FeedbackState)


def extract_feedback_landmarks(curr_lm, exercise_name):
    subset = {}
    for name in FORM_FEEDBACK_LANDMARKS.get(exercise_name, []):
        subset[f"{name}_x"] = curr_lm[f"{name}_x"]
        subset[f"{name}_y"] = curr_lm[f"{name}_y"]
        subset[f"{name}_z"] = curr_lm[f"{name}_z"]
    return subset


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


def _append_cycle_frame(state, curr_lm, exercise_name):
    """Append a frame only while a cycle stream is active."""
    if not state.collecting:
        return
    state.rolling_buffer.append(extract_feedback_landmarks(curr_lm, exercise_name))
    state.frames_in_cycle += 1


def _start_cycle_stream(state, curr_lm, exercise_name, rest_extremity):
    """Start a new stream at resting position."""
    state.collecting = True
    state.cycle_start_extremity = rest_extremity
    state.seen_opposite_extremity = False
    state.rolling_buffer.clear()
    state.rolling_buffer.append(extract_feedback_landmarks(curr_lm, exercise_name))
    state.frames_in_cycle = 1


def _extract_rep_stream(state):
    if not state.rolling_buffer:
        return []
    return list(state.rolling_buffer)


def _finalize_feedback_at_rest(state, exercise_name, current_rep_count):
    """Emit feedback after a counted rep returns to rest, then reset stream."""
    rep_stream = _extract_rep_stream(state)
    analyzer = FORM_FEEDBACK_ANALYZERS.get(exercise_name)
    if analyzer is not None and rep_stream:
        state.message = analyzer(rep_stream, FRAME_DT)
        state.last_rep_count = current_rep_count

    # Reset collection after feedback; a new stream starts when rest is reached again.
    state.collecting = False
    state.rep_completed_pending = False
    state.cycle_start_extremity = None
    state.seen_opposite_extremity = False
    state.frames_in_cycle = 0
    state.rolling_buffer.clear()


def _rest_extremity_for_exercise(exercise_name):
    count_at = REP_COUNT_AT.get(exercise_name, "top")
    return "bottom" if count_at == "top" else "top"


def _handle_confirmed_extremity_transition(
    state,
    exercise_name,
    curr_lm,
    extremity,
    current_rep_count,
):
    """Feedback flow: rest -> movement -> counted rep -> rest => emit feedback."""
    rest_extremity = _rest_extremity_for_exercise(exercise_name)

    # Start stream only when a stable resting extremity is confirmed.
    if not state.collecting:
        if extremity == rest_extremity:
            _start_cycle_stream(state, curr_lm, exercise_name, rest_extremity)
        return

    if extremity != rest_extremity:
        state.seen_opposite_extremity = True
        return

    if state.rep_completed_pending and state.seen_opposite_extremity:
        _finalize_feedback_at_rest(state, exercise_name, current_rep_count)
        # Immediately start next stream from the same confirmed rest point.
        _start_cycle_stream(state, curr_lm, exercise_name, rest_extremity)


def get_form_feedback(exercise_name, curr_lm, feedback_states, rep_counts):
    if exercise_name is None:
        return None

    state = feedback_states[exercise_name]
    current_rep_count = rep_counts.get(exercise_name, 0)
    if current_rep_count > state.last_rep_count:
        state.rep_completed_pending = True

    # Feed cycle stream while actively collecting from rest to next rest.
    _append_cycle_frame(state, curr_lm, exercise_name)

    extremity = detect_rep_extremity(curr_lm, exercise_name)
    if extremity is None:
        return state.message

    if not _confirm_extremity_transition(state, extremity):
        return state.message

    _handle_confirmed_extremity_transition(
        state,
        exercise_name,
        curr_lm,
        extremity,
        current_rep_count,
    )

    state.last_extremity = extremity
    return state.message


def get_feedback_message(exercise_name, feedback_states):
    if exercise_name is None:
        return None
    return feedback_states[exercise_name].message


def reset_form_feedback_tracking(feedback_states):
    for state in feedback_states.values():
        state.reset()
