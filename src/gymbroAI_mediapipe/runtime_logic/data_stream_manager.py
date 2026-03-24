# Queue type used to store fixed-length frame streams.
from collections import deque

# Rep-counter helpers used to detect phase and compute switch progress.
from runtime_logic.rep_counter import (
    detect_rep_extremity,
    get_switch_progress,
    get_switch_progress_threshold,
)


# Create the initial dual-stream state used by exercise-switch smoothing.
def initialize_dual_stream_state(stream_max_frames):
    return {
        # Frames currently assigned to the active displayed exercise.
        "active_stream": deque(maxlen=stream_max_frames),
        # Frames currently assigned to a candidate predicted exercise.
        "predicted_stream": deque(maxlen=stream_max_frames),
        # Candidate exercise label being evaluated for a possible switch.
        "predicted_exercise": None,
        # Extremity (top/bottom) where candidate switch movement started.
        "predicted_start_extremity": None,
        # Normalized candidate movement progress from start extremity.
        "predicted_progress": 0.0,
    }


# Update active label using active/predicted streams and progress gating.
def update_active_exercise_with_dual_stream(
    curr_landmarks,
    predicted_exercise_name,
    active_exercise_name,
    dual_stream_state,
    rep_states=None,
):
    # Always keep active stream updated while an active label exists.
    if active_exercise_name is not None:
        dual_stream_state["active_stream"].append(curr_landmarks)

    # If there is no prediction, keep the current active label unchanged.
    if predicted_exercise_name is None:
        return active_exercise_name

    # First valid prediction becomes active immediately when nothing is active yet.
    if active_exercise_name is None:
        dual_stream_state["active_stream"].clear()
        dual_stream_state["active_stream"].append(curr_landmarks)
        return predicted_exercise_name

    # If prediction matches active label, clear candidate state and continue.
    if predicted_exercise_name == active_exercise_name:
        reset_predicted_stream(dual_stream_state)
        return active_exercise_name

    # When candidate label changes, restart candidate tracking from scratch.
    if dual_stream_state["predicted_exercise"] != predicted_exercise_name:
        dual_stream_state["predicted_exercise"] = predicted_exercise_name
        # Capture current movement phase as anchor for progress computation.
        dual_stream_state["predicted_start_extremity"] = detect_rep_extremity(
            curr_landmarks,
            predicted_exercise_name,
        )
        # Reset progress for this new candidate.
        dual_stream_state["predicted_progress"] = 0.0
        # Clear candidate stream so it contains only frames for this candidate.
        dual_stream_state["predicted_stream"].clear()

    # Append current frame into candidate stream history.
    dual_stream_state["predicted_stream"].append(curr_landmarks)

    # If start extremity was unknown, attempt to detect it from current frame.
    if dual_stream_state["predicted_start_extremity"] is None:
        dual_stream_state["predicted_start_extremity"] = detect_rep_extremity(
            curr_landmarks,
            predicted_exercise_name,
        )

    # Compute candidate movement progress only when a start extremity exists.
    if dual_stream_state["predicted_start_extremity"] is not None:
        dual_stream_state["predicted_progress"] = get_switch_progress(
            curr_landmarks,
            predicted_exercise_name,
            dual_stream_state["predicted_start_extremity"],
        )

    # Read configured minimum progress required to accept a label switch.
    switch_threshold = get_switch_progress_threshold(predicted_exercise_name)

    # Promote candidate to active once progress passes threshold.
    if dual_stream_state["predicted_progress"] >= switch_threshold:
        active_exercise_name = predicted_exercise_name
        # Replace active stream with candidate stream at switch time.
        dual_stream_state["active_stream"] = deque(
            dual_stream_state["predicted_stream"],
            maxlen=dual_stream_state["active_stream"].maxlen,
        )
        # Seed rep state with the start extremity so the rep counter knows
        # where the movement began and counts the first rep correctly.
        if (
            rep_states is not None
            and dual_stream_state["predicted_start_extremity"] is not None
        ):
            rep_states[predicted_exercise_name].last_extremity = dual_stream_state[
                "predicted_start_extremity"
            ]
            rep_states[predicted_exercise_name].pending_extremity = None
            rep_states[predicted_exercise_name].pending_count = 0
        # Clear candidate state now that switch is complete.
        reset_predicted_stream(dual_stream_state)

    # Return whichever label is active after processing this frame.
    return active_exercise_name


# Reset all candidate/predicted stream fields back to neutral state.
def reset_predicted_stream(dual_stream_state):
    # Remove candidate exercise label.
    dual_stream_state["predicted_exercise"] = None
    # Remove candidate start phase anchor.
    dual_stream_state["predicted_start_extremity"] = None
    # Reset candidate progress value.
    dual_stream_state["predicted_progress"] = 0.0
    # Clear candidate frame history.
    dual_stream_state["predicted_stream"].clear()


# Backward-compatible alias for callers using the old function name.
def update_active_excercise(
    curr_landmarks,
    predicted_excercise_name,
    active_exercise_name,
    dual_stream_state,
    rep_states=None,
):
    return update_active_exercise_with_dual_stream(
        curr_landmarks,
        predicted_excercise_name,
        active_exercise_name,
        dual_stream_state,
        rep_states=rep_states,
    )
