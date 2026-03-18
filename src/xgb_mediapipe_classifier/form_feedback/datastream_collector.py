from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from rep_counter import get_rep_count_at_map

FRAME_DT = 0.033
BUFFER_SECONDS = 8
BUFFER_MAX_FRAMES = int(BUFFER_SECONDS / FRAME_DT)  # ~242 frames at 30 fps
REP_EXTREMITY_STREAK = 2


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
    # Feedback is emitted only after a new rep count appears.
    last_rep_count: int = 0
    rolling_buffer: deque = field(
        default_factory=lambda: deque(maxlen=BUFFER_MAX_FRAMES)
    )

    def reset(self):
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


def snapshot_landmarks(curr_lm):
    return dict(curr_lm)


def confirm_extremity_transition(state, extremity):
    """Return True only after the same new extremity is seen in a short streak."""
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


def append_cycle_frame(state, curr_lm):
    """Append one frame to the active cycle stream, if collection is active."""
    if not state.collecting:
        return
    state.rolling_buffer.append(snapshot_landmarks(curr_lm))
    state.frames_in_cycle += 1


def start_cycle_stream(state, curr_lm, rest_extremity):
    """Start a fresh cycle stream from a confirmed rest extremity."""
    state.collecting = True
    state.cycle_start_extremity = rest_extremity
    state.seen_opposite_extremity = False
    state.rolling_buffer.clear()
    state.rolling_buffer.append(snapshot_landmarks(curr_lm))
    state.frames_in_cycle = 1


def extract_rep_stream(state):
    if not state.rolling_buffer:
        return []
    return list(state.rolling_buffer)


def finalize_feedback_at_rest(state, exercise_name, current_rep_count, analyze_rep_fn):
    """Analyze the captured stream and reset state at the rest extremity."""
    rep_stream = extract_rep_stream(state)
    if rep_stream:
        state.message = analyze_rep_fn(exercise_name, rep_stream)
        state.last_rep_count = current_rep_count

    state.collecting = False
    state.rep_completed_pending = False
    state.cycle_start_extremity = None
    state.seen_opposite_extremity = False
    state.frames_in_cycle = 0
    state.rolling_buffer.clear()


def rest_extremity_for_exercise(exercise_name):
    """Infer the resting extremity from profile-driven count-at metadata."""
    count_at = get_rep_count_at_map().get(exercise_name, "top")
    return "bottom" if count_at == "top" else "top"


def handle_confirmed_extremity_transition(
    state,
    exercise_name,
    curr_lm,
    extremity,
    current_rep_count,
    analyze_rep_fn,
):
    """Drive stream collection lifecycle across stable extremity transitions."""
    rest_extremity = rest_extremity_for_exercise(exercise_name)

    if not state.collecting:
        if extremity == rest_extremity:
            start_cycle_stream(state, curr_lm, rest_extremity)
        return

    if extremity != rest_extremity:
        state.seen_opposite_extremity = True
        return

    if state.rep_completed_pending and state.seen_opposite_extremity:
        finalize_feedback_at_rest(
            state,
            exercise_name,
            current_rep_count,
            analyze_rep_fn,
        )
        start_cycle_stream(state, curr_lm, rest_extremity)
