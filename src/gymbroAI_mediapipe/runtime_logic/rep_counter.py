"""
Rep counter for exercise tracking.

Each exercise has a metric function that converts raw landmarks into a single
number, plus top/bottom thresholds that define the extremities of the movement.
A rep is counted each time the metric crosses the count_at extremity after
visiting the opposite one.
"""

from collections import defaultdict
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Movement progress (0-1) required to accept an exercise switch.
SWITCH_PROGRESS_THRESHOLD = 0.5

# Consecutive frames at the same extremity before confirming a transition.
REP_EXTREMITY_STREAK = 2


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _shoulder_width(lm):
    """Euclidean distance between shoulders. Scales with camera distance,
    so dividing by it makes other metrics distance-independent."""
    dx = lm["left_shoulder_x"] - lm["right_shoulder_x"]
    dy = lm["left_shoulder_y"] - lm["right_shoulder_y"]
    return max((dx**2 + dy**2) ** 0.5, 1e-6)


def _body_center_y(lm):
    """Y midpoint between shoulder center and hip center."""
    hip_cy = (lm["left_hip_y"] + lm["right_hip_y"]) / 2.0
    sho_cy = (lm["left_shoulder_y"] + lm["right_shoulder_y"]) / 2.0
    return (hip_cy + sho_cy) / 2.0


# ---------------------------------------------------------------------------
# Metric functions — each returns a single number from landmarks
# ---------------------------------------------------------------------------


def _avg_wrist_norm_y(lm):
    """Wrist Y above body centre, normalized by shoulder width."""
    cy = _body_center_y(lm)
    sw = _shoulder_width(lm)
    left = (cy - lm["left_wrist_y"]) / sw
    right = (cy - lm["right_wrist_y"]) / sw
    return (left + right) / 2.0


def _avg_ankle_norm_y(lm):
    """Euclidean distance between hip center and ankle center,
    normalized by shoulder width."""
    sw = _shoulder_width(lm)
    hip_x = (lm["left_hip_x"] + lm["right_hip_x"]) / 2.0
    hip_y = (lm["left_hip_y"] + lm["right_hip_y"]) / 2.0
    ankle_x = (lm["left_ankle_x"] + lm["right_ankle_x"]) / 2.0
    ankle_y = (lm["left_ankle_y"] + lm["right_ankle_y"]) / 2.0
    dist = ((hip_x - ankle_x) ** 2 + (hip_y - ankle_y) ** 2) ** 0.5
    return dist / sw


# ---------------------------------------------------------------------------
# Exercise configs
# ---------------------------------------------------------------------------

EXERCISE_CONFIGS = {
    "squat": {
        "top_threshold": 2.8,
        "bottom_threshold": 2.0,
        "count_at": "bottom",
        "metric": _avg_ankle_norm_y,
    },
    "curl": {
        "top_threshold": 0.35,
        "bottom_threshold": -0.3,
        "count_at": "top",
        "metric": _avg_wrist_norm_y,
    },
    "shoulder_press": {
        "top_threshold": 3.4,
        "bottom_threshold": 1.6,
        "count_at": "top",
        "metric": _avg_wrist_norm_y,
    },
}


# ---------------------------------------------------------------------------
# Extremity detection
# ---------------------------------------------------------------------------


def _get_metric(lm, exercise_name):
    """Return (metric, top_threshold, bottom_threshold) or None."""
    config = EXERCISE_CONFIGS.get(exercise_name)
    if config is None:
        return None
    try:
        metric = config["metric"](lm)
    except KeyError:
        return None
    return metric, config["top_threshold"], config["bottom_threshold"]


def detect_rep_extremity(curr_lm, exercise_name):
    """Return "top", "bottom", or None based on current metric vs thresholds."""
    result = _get_metric(curr_lm, exercise_name)
    if result is None:
        return None

    metric, top_t, bot_t = result
    if top_t < bot_t:
        if metric <= top_t:
            return "top"
        if metric >= bot_t:
            return "bottom"
    else:
        if metric >= top_t:
            return "top"
        if metric <= bot_t:
            return "bottom"
    return None


# ---------------------------------------------------------------------------
# Rep counting
# ---------------------------------------------------------------------------


@dataclass
class RepState:
    last_extremity: str | None = None
    pending_extremity: str | None = None
    pending_count: int = 0


def create_rep_counter_state(encoder):
    rep_counts = {name: 0 for name in encoder.classes_}
    rep_states = defaultdict(RepState)
    return rep_counts, rep_states


def _confirm_extremity_transition(state, extremity):
    """Debounce: require REP_EXTREMITY_STREAK consecutive frames at a new
    extremity before accepting the transition."""
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
    """Called every frame. Counts a rep when the metric reaches count_at."""
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

    if extremity == EXERCISE_CONFIGS[exercise_name]["count_at"]:
        rep_counts[exercise_name] += 1
    state.last_extremity = extremity


def reset_rep_counter_tracking(rep_states):
    for state in rep_states.values():
        state.pending_extremity = None
        state.pending_count = 0


# ---------------------------------------------------------------------------
# Switch progress — used by data_stream_manager
# ---------------------------------------------------------------------------


def get_switch_progress(curr_lm, exercise_name, start_extremity):
    """Return normalized movement progress [0,1] from the starting extremity."""
    result = _get_metric(curr_lm, exercise_name)
    if result is None:
        return 0.0

    metric, top_t, bot_t = result
    if start_extremity == "top":
        start_level, target_level = top_t, bot_t
    elif start_extremity == "bottom":
        start_level, target_level = bot_t, top_t
    else:
        return 0.0

    delta = target_level - start_level
    if abs(delta) <= 1e-6:
        return 0.0
    return max(0.0, min(1.0, (metric - start_level) / delta))


# ---------------------------------------------------------------------------
# Public getters
# ---------------------------------------------------------------------------


def get_exercise_metric_value(curr_lm, exercise_name):
    """Return the raw metric value for a single frame, or None."""
    result = _get_metric(curr_lm, exercise_name)
    if result is None:
        return None
    return result[0]


def get_rep_direction_label(exercise_name, rep_states):
    """Return 'UP' or 'DOWN' to indicate current movement direction."""
    if exercise_name is None:
        return None

    state = rep_states.get(exercise_name)
    if state is None:
        return None

    # Pending extremity indicates current movement direction before full confirmation.
    if state.pending_extremity == "top":
        return "UP"
    if state.pending_extremity == "bottom":
        return "DOWN"

    # Last confirmed extremity: show where to go next.
    if state.last_extremity == "bottom":
        return "UP"
    if state.last_extremity == "top":
        return "DOWN"

    return None
