"""
Exercise switching with dual-stream smoothing.

Pipeline:
    1. Raw model predictions are filtered through a streak filter — the model
       must predict the same exercise N times in a row before it's accepted.
    2. If the accepted prediction differs from the active exercise, a candidate
       stream is started and movement progress is tracked.
    3. Once the user moves past the switch threshold (e.g. 70% of the range),
       the candidate is promoted to active and the rep counter is seeded.
"""

from collections import deque

from runtime_logic.rep_counter import (
    EXERCISE_CONFIGS,
    SWITCH_PROGRESS_THRESHOLD,
    get_switch_progress,
)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Number of consecutive identical predictions required before accepting.
PREDICTION_STREAK_REQUIRED = 5


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def initialize_dual_stream_state(stream_max_frames):
    return {
        "active_stream": deque(maxlen=stream_max_frames),
        "predicted_stream": deque(maxlen=stream_max_frames),
        "predicted_exercise": None,
        "predicted_start_extremity": None,
        "predicted_progress": 0.0,
        "streak_label": None,
        "streak_count": 0,
    }


def _reset_candidate(state):
    """Clear all candidate/predicted fields."""
    state["predicted_exercise"] = None
    state["predicted_start_extremity"] = None
    state["predicted_progress"] = 0.0
    state["predicted_stream"].clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_streak_filter(state, prediction):
    """Return the prediction only if it has been stable for N frames."""
    if prediction == state["streak_label"]:
        state["streak_count"] += 1
    else:
        state["streak_label"] = prediction
        state["streak_count"] = 1

    if state["streak_count"] < PREDICTION_STREAK_REQUIRED:
        return None
    return prediction


def _determine_start_extremity(exercise_name, curr_landmarks):
    """Decide whether the user is in the upper or lower half of the
    exercise range. Used as the starting point for progress tracking."""
    cfg = EXERCISE_CONFIGS.get(exercise_name, {})
    top_t = cfg.get("top_threshold", 1)
    bot_t = cfg.get("bottom_threshold", 0)
    midpoint = (top_t + bot_t) / 2.0
    try:
        metric = cfg["metric"](curr_landmarks)
    except (KeyError, TypeError):
        metric = midpoint
    return "top" if metric >= midpoint else "bottom"


def _seed_rep_state(rep_states, exercise_name, start_extremity):
    """Set the rep counter's starting extremity so the first rep counts correctly."""
    if rep_states is None or start_extremity is None:
        return
    rep_states[exercise_name].last_extremity = start_extremity
    rep_states[exercise_name].pending_extremity = None
    rep_states[exercise_name].pending_count = 0


# ---------------------------------------------------------------------------
# Main update — called every frame from main.py
# ---------------------------------------------------------------------------


def update_active_exercise_with_dual_stream(
    curr_landmarks,
    predicted_exercise_name,
    active_exercise_name,
    state,
    rep_states=None,
):
    # Step 1: streak filter.
    predicted_exercise_name = _apply_streak_filter(state, predicted_exercise_name)

    # Always feed the active stream.
    if active_exercise_name is not None:
        state["active_stream"].append(curr_landmarks)

    # No stable prediction yet.
    if predicted_exercise_name is None:
        return active_exercise_name

    # First prediction ever — activate immediately.
    if active_exercise_name is None:
        state["active_stream"].clear()
        state["active_stream"].append(curr_landmarks)
        return predicted_exercise_name

    # Prediction matches active — nothing to switch.
    if predicted_exercise_name == active_exercise_name:
        _reset_candidate(state)
        return active_exercise_name

    # New candidate — start tracking.
    if state["predicted_exercise"] != predicted_exercise_name:
        state["predicted_exercise"] = predicted_exercise_name
        state["predicted_start_extremity"] = _determine_start_extremity(
            predicted_exercise_name, curr_landmarks
        )
        state["predicted_progress"] = 0.0
        state["predicted_stream"].clear()

    # Collect candidate frames and measure progress.
    state["predicted_stream"].append(curr_landmarks)
    state["predicted_progress"] = get_switch_progress(
        curr_landmarks,
        predicted_exercise_name,
        state["predicted_start_extremity"],
    )

    # Promote candidate once progress passes threshold.
    if state["predicted_progress"] >= SWITCH_PROGRESS_THRESHOLD:
        active_exercise_name = predicted_exercise_name
        state["active_stream"] = deque(
            state["predicted_stream"],
            maxlen=state["active_stream"].maxlen,
        )
        _seed_rep_state(
            rep_states, predicted_exercise_name, state["predicted_start_extremity"]
        )
        _reset_candidate(state)

    return active_exercise_name
