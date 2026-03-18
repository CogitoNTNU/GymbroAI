from rep_counter import (
    detect_rep_extremity,
    get_feature_value,
    get_motion_profiles,
)

from .datastream_collector import (
    append_cycle_frame,
    confirm_extremity_transition,
    create_feedback_state,
    handle_confirmed_extremity_transition,
)
from .excercise_configs import get_exercise_analyzer


def analyze_generic_rep(exercise_name, rep_stream):
    """Analyze one completed rep stream and return a short feedback message.

    This function is feedback-only logic. It does not manage buffers,
    stream collection, or extremity state transitions.
    """
    profile = get_motion_profiles().get(exercise_name)
    if profile is None or not rep_stream:
        return None

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
    """Run hardcoded exercise checks, while reusing dynamic profile values.

    Exercise recognition and rep counting stay dynamic.
    Form checks can be exercise-specific and hardcoded.
    """
    profile = get_motion_profiles().get(exercise_name)
    analyzer = get_exercise_analyzer(exercise_name)
    if analyzer is None:
        return analyze_generic_rep(exercise_name, rep_stream)
    return analyzer(rep_stream, 0.033, profile)


def get_form_feedback(exercise_name, curr_lm, feedback_states, rep_counts):
    """Main feedback entry-point used by the runtime loop.

    Stream/cycle collection details are delegated to datastream_collector.
    This function coordinates rep-count gating and message updates.
    """
    if exercise_name is None:
        return None

    state = feedback_states[exercise_name]
    current_rep_count = rep_counts.get(exercise_name, 0)
    if current_rep_count > state.last_rep_count:
        state.rep_completed_pending = True

    # Keep collecting frames only while a cycle stream is active.
    append_cycle_frame(state, curr_lm)

    extremity = detect_rep_extremity(curr_lm, exercise_name)
    if extremity is None:
        return state.message

    # Ignore noisy one-frame extremity flips.
    if not confirm_extremity_transition(state, extremity):
        return state.message

    # Progress or finalize the cycle after a stable extremity transition.
    handle_confirmed_extremity_transition(
        state,
        exercise_name,
        curr_lm,
        extremity,
        current_rep_count,
        analyze_rep,
    )

    state.last_extremity = extremity
    return state.message


def get_feedback_message(exercise_name, feedback_states):
    """Read the latest feedback message for the current displayed exercise."""
    if exercise_name is None:
        return None
    return feedback_states[exercise_name].message


def reset_form_feedback_tracking(feedback_states):
    """Reset collector state for all exercises when pose tracking is lost."""
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
