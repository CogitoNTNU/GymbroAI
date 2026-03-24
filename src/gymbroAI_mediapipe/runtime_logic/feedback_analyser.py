from collections import defaultdict, deque
from dataclasses import dataclass, field

from runtime_logic.rep_counter import (
    detect_rep_extremity,
    get_feature_value,
    get_motion_profiles,
)

EXTREMITY_STREAK = 2
FEEDBACK_BUFFER_MAX_FRAMES = 240


@dataclass
class FeedbackState:
    message: str | None = None
    last_rep_count: int = 0
    rep_completed_pending: bool = False
    last_extremity: str | None = None
    pending_extremity: str | None = None
    pending_count: int = 0
    cycle_start_extremity: str | None = None
    seen_opposite_extremity: bool = False
    frames_in_cycle: int = 0
    rolling_buffer: deque = field(
        default_factory=lambda: deque(maxlen=FEEDBACK_BUFFER_MAX_FRAMES)
    )

    def reset(self):
        self.rep_completed_pending = False
        self.last_extremity = None
        self.pending_extremity = None
        self.pending_count = 0
        self.cycle_start_extremity = None
        self.seen_opposite_extremity = False
        self.frames_in_cycle = 0
        self.rolling_buffer.clear()


def create_feedback_state():
    return defaultdict(FeedbackState)


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

    if state.pending_count < EXTREMITY_STREAK:
        return False

    state.pending_extremity = None
    state.pending_count = 0
    return True


def _append_cycle_frame(state, curr_lm):
    if curr_lm is None:
        return
    state.rolling_buffer.append(dict(curr_lm))
    state.frames_in_cycle = len(state.rolling_buffer)


def analyze_generic_rep(exercise_name, rep_stream):
    profile = get_motion_profiles().get(exercise_name)
    if not rep_stream:
        return None

    if profile is None:
        # Fallback when dynamic motion profiles are unavailable.
        return "Good form"

    selected = (
        profile.selected_features
        if profile.selected_features
        else [profile.feature_name]
    )
    if not selected:
        return "Need more stable motion"

    per_feature_values = {}
    for feature_name in selected:
        values = [
            value
            for value in (
                get_feature_value(frame, feature_name) for frame in rep_stream
            )
            if value is not None
        ]
        if len(values) >= 4:
            per_feature_values[feature_name] = values

    if not per_feature_values:
        return "Need more stable motion"

    span_ok = 0
    full_range_hits = 0
    crossing_scores = []
    used_count = 0

    for feature_name, values in per_feature_values.items():
        spec = profile.feature_profiles.get(feature_name)
        if spec is None:
            continue

        used_count += 1
        span = max(values) - min(values)
        if span >= spec.min_significant_motion * 0.8:
            span_ok += 1

        upper_hits = sum(1 for value in values if value >= spec.top_threshold)
        lower_hits = sum(1 for value in values if value <= spec.bottom_threshold)
        if upper_hits > 0 and lower_hits > 0:
            full_range_hits += 1

        midpoint_crossings = sum(
            1
            for idx in range(1, len(values))
            if (values[idx - 1] - spec.midpoint) * (values[idx] - spec.midpoint) < 0
        )
        crossing_scores.append(midpoint_crossings)

    if used_count == 0:
        return "Need more stable motion"

    if span_ok < max(1, used_count // 2 + (used_count % 2)):
        return "Increase range of motion"

    if full_range_hits < max(1, used_count // 2 + (used_count % 2)):
        return "Hit both ends of the movement"

    avg_crossings = sum(crossing_scores) / max(len(crossing_scores), 1)
    if avg_crossings < 2:
        return "Move with fuller controlled cycles"

    return "Good form"


def analyze_rep(exercise_name, rep_stream):
    return analyze_generic_rep(exercise_name, rep_stream)


def _handle_confirmed_extremity_transition(
    state,
    exercise_name,
    curr_lm,
    extremity,
    current_rep_count,
):
    if state.cycle_start_extremity is None:
        state.cycle_start_extremity = extremity
        state.seen_opposite_extremity = False
        return

    if extremity != state.cycle_start_extremity:
        state.seen_opposite_extremity = True
        return

    if not state.seen_opposite_extremity:
        return

    # Completed one full cycle by returning to starting extremity.
    if state.rep_completed_pending and state.rolling_buffer:
        state.message = analyze_rep(exercise_name, list(state.rolling_buffer))
        state.last_rep_count = current_rep_count
        state.rep_completed_pending = False

    state.cycle_start_extremity = extremity
    state.seen_opposite_extremity = False
    state.rolling_buffer.clear()
    if curr_lm is not None:
        state.rolling_buffer.append(dict(curr_lm))
    state.frames_in_cycle = len(state.rolling_buffer)


def get_form_feedback(exercise_name, curr_lm, feedback_states, rep_counts):
    if exercise_name is None:
        return None

    state = feedback_states[exercise_name]
    current_rep_count = rep_counts.get(exercise_name, 0)
    if current_rep_count > state.last_rep_count:
        state.rep_completed_pending = True

    _append_cycle_frame(state, curr_lm)

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


__all__ = [
    "create_feedback_state",
    "analyze_rep",
    "analyze_generic_rep",
    "get_form_feedback",
    "get_feedback_message",
    "reset_form_feedback_tracking",
]
