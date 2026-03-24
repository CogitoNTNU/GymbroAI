import os
import re
import ast
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np


def calculate_angle(p1, p2, p3):
    a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos_angle = np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


REP_EXTREMITY_STREAK = 2

_DEFAULT_EXERCISE_CONFIG = {
    # Squat: hip height above ankle, normalised by torso length.
    # Positive = hips above ankles (physically up).  Line goes UP when standing, DOWN when squatting.
    #   metric >= top_threshold (0.9)  → "top"    (standing)
    #   metric <= bottom_threshold (0.6) → "bottom" (squatted)
    # count_at = "bottom" → count each time the squat depth is reached.
    "squat_config": {
        "tracked_value": "(left_hip_above_ankle + right_hip_above_ankle) / 2",
        "top_threshold": 0.9,
        "bottom_threshold": 0.6,
        "count_at": "bottom",
    },
    # Curl: normalised wrist Y, positive = wrists above body centre.
    # Line goes UP when curling, DOWN when extending.
    #   metric >= top_threshold (0.3)  → "top"    (wrists near shoulder/chest — curled)
    #   metric <= bottom_threshold (-0.5) → "bottom" (wrists below body centre — extended)
    "curl_config": {
        "tracked_value": "(left_wrist_norm_y_up + right_wrist_norm_y_up) / 2",
        "top_threshold": 0.3,
        "bottom_threshold": -0.5,
        "count_at": "top",
    },
    # Shoulder press: normalised wrist Y, positive = wrists above body centre.
    # Line goes UP when pressing overhead, DOWN when lowering.
    #   metric >= top_threshold (0.9)  → "top"    (wrists above head — pressed)
    #   metric <= bottom_threshold (0.45) → "bottom" (wrists near shoulder — start)
    "shoulder_press_config": {
        "tracked_value": "(left_wrist_norm_y_up + right_wrist_norm_y_up) / 2",
        "top_threshold": 0.9,
        "bottom_threshold": 0.45,
        "count_at": "top",
    },
}


def _default_rep_config():
    return {
        "global_config": {
            "switch_progress_threshold": 0.25,
        },
        "exercise_configs": {
            name: dict(config) for name, config in _DEFAULT_EXERCISE_CONFIG.items()
        },
    }


def _models_dir():
    return os.path.join(
        os.path.dirname(__file__),
        "excercise_classifcation",
        "models",
    )


def _rep_config_path():
    models_dir = _models_dir()
    preferred_path = os.path.join(models_dir, "exercise_configs.json")
    legacy_path = os.path.join(models_dir, "rep_counting_config.json")

    if os.path.exists(preferred_path):
        return preferred_path
    return legacy_path


def _normalize_loaded_config(loaded):
    # Preferred schema.
    if "global_config" in loaded and "exercise_configs" in loaded:
        return loaded

    # Backward compatibility: convert previous schema on read.
    global_config = {
        "switch_progress_threshold": loaded.get("switch_progress_threshold", 0.25)
    }

    raw_exercises = loaded.get("exercise_configs") or loaded.get("exercises") or {}
    normalized_exercises = {}
    if isinstance(raw_exercises, dict):
        for key, config in raw_exercises.items():
            if not isinstance(config, dict):
                continue

            exercise_name = (
                key[:-7] if isinstance(key, str) and key.endswith("_config") else key
            )
            normalized_key = f"{exercise_name}_config"
            merged = {
                "tracked_value": config.get("tracked_value"),
                "top_threshold": config.get("top_threshold"),
                "bottom_threshold": config.get("bottom_threshold"),
                "count_at": config.get("count_at", "top"),
            }

            tracked_angles = config.get("tracked_angles")
            if (
                (
                    not isinstance(merged.get("tracked_value"), str)
                    or not merged.get("tracked_value")
                )
                and isinstance(tracked_angles, list)
                and tracked_angles
            ):
                expression = " + ".join(str(name) for name in tracked_angles)
                merged["tracked_value"] = f"({expression}) / {len(tracked_angles)}"

            normalized_exercises[normalized_key] = merged

    return {
        "global_config": global_config,
        "exercise_configs": normalized_exercises,
    }


def _load_rep_count_config():
    # Always use hardcoded defaults — JSON files are ignored.
    return _default_rep_config()


_REP_CONFIG = _load_rep_count_config()
REP_COUNT_AT = {
    exercise_key[:-7] if exercise_key.endswith("_config") else exercise_key: str(
        exercise_config.get("count_at", "top")
    )
    for exercise_key, exercise_config in _REP_CONFIG.get("exercise_configs", {}).items()
}


def _get_exercise_config(exercise_name):
    exercise_configs = _REP_CONFIG.get("exercise_configs", {})
    if exercise_name in exercise_configs:
        return exercise_configs[exercise_name]

    config_key = f"{exercise_name}_config"
    if config_key in exercise_configs:
        return exercise_configs[config_key]

    return {}


def get_count_at(exercise_name):
    exercise_config = _get_exercise_config(exercise_name)
    return str(exercise_config.get("count_at", "top"))


def get_switch_progress_threshold(exercise_name=None):
    try:
        threshold = float(
            _REP_CONFIG.get("global_config", {}).get("switch_progress_threshold", 0.25)
        )
        # Accept both ratio-style values (0.25) and percentage-style values (50).
        if threshold > 1.0:
            threshold = threshold / 100.0
        return max(0.0, min(1.0, threshold))
    except (TypeError, ValueError):
        return 0.25


def _angle_triplet_from_name(angle_name):
    if not isinstance(angle_name, str):
        return None

    clean_name = angle_name[:-6] if angle_name.endswith("_angle") else angle_name
    parts = clean_name.split("_")
    if len(parts) < 4:
        return None

    side = parts[0]
    if side not in ("left", "right"):
        return None

    return (
        f"{side}_{parts[1]}",
        f"{side}_{parts[2]}",
        f"{side}_{parts[3]}",
    )


def _body_center_and_torso_y(curr_lm):
    """Return (body_center_y, torso_y_scale) using mid-shoulder and mid-hip.

    body_center_y is the midpoint between shoulder centre and hip centre.
    torso_y_scale is the vertical distance between them, used as a normalisation
    factor so the metric is body-size independent.
    """
    hip_cy = (curr_lm["left_hip_y"] + curr_lm["right_hip_y"]) / 2.0
    sho_cy = (curr_lm["left_shoulder_y"] + curr_lm["right_shoulder_y"]) / 2.0
    body_cy = (hip_cy + sho_cy) / 2.0
    torso = max(abs(sho_cy - hip_cy), 1e-6)
    return body_cy, torso


def _calculate_named_metrics(curr_lm, tracked_names):
    """Resolve each name in tracked_names to a numeric value.

    Supported suffixes (all normalised by torso Y length, positive = physically up):
      _norm_y_up    {side}_{landmark}_norm_y_up
                    → (body_center_y - landmark_y) / torso_y
                    Positive when the landmark is above body centre.

      _above_ankle  {side}_{landmark}_above_ankle
                    → (avg_ankle_y - landmark_y) / torso_y
                    Positive when the landmark is above the ankle level.
                    Useful for squats where the whole body moves vertically.

      Angle triplet {side}_{j1}_{j2}_{j3}  → angle in degrees (legacy).
    """
    metrics = {}
    body_cy, torso = None, None

    for name in tracked_names:
        if name.endswith("_norm_y_up"):
            if body_cy is None:
                body_cy, torso = _body_center_and_torso_y(curr_lm)
            landmark = name[: -len("_norm_y_up")]  # e.g. "left_wrist"
            y_key = f"{landmark}_y"
            if y_key not in curr_lm:
                continue
            metrics[name] = (body_cy - curr_lm[y_key]) / torso
            continue

        if name.endswith("_above_ankle"):
            if body_cy is None:
                body_cy, torso = _body_center_and_torso_y(curr_lm)
            landmark = name[: -len("_above_ankle")]  # e.g. "left_hip"
            y_key = f"{landmark}_y"
            if y_key not in curr_lm:
                continue
            avg_ankle_y = (
                curr_lm.get("left_ankle_y", 0) + curr_lm.get("right_ankle_y", 0)
            ) / 2.0
            metrics[name] = (avg_ankle_y - curr_lm[y_key]) / torso
            continue

        triplet = _angle_triplet_from_name(name)
        if triplet is None:
            continue
        a_name, b_name, c_name = triplet
        p1 = (curr_lm[f"{a_name}_x"], curr_lm[f"{a_name}_y"])
        p2 = (curr_lm[f"{b_name}_x"], curr_lm[f"{b_name}_y"])
        p3 = (curr_lm[f"{c_name}_x"], curr_lm[f"{c_name}_y"])
        metrics[name] = calculate_angle(p1, p2, p3)

    return metrics


def _extract_angle_names_from_expression(expression):
    if not isinstance(expression, str):
        return []
    names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)
    return names


def _safe_eval_tracked_expression(expression, value_map):
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            raise ValueError("Unsupported operator")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        if isinstance(node, ast.Name):
            if node.id not in value_map:
                raise ValueError("Unknown variable")
            return value_map[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Unsupported expression")

    parsed = ast.parse(expression, mode="eval")
    return float(_eval(parsed))


def _exercise_metric(curr_lm, exercise_name):
    exercise_config = _get_exercise_config(exercise_name)
    if not exercise_config:
        return None

    tracked_value = exercise_config.get("tracked_value")
    if not isinstance(tracked_value, str) or not tracked_value.strip():
        return None

    tracked_names = _extract_angle_names_from_expression(tracked_value)
    if not tracked_names:
        return None

    named_angles = _calculate_named_metrics(curr_lm, tracked_names)
    if not named_angles:
        return None

    try:
        metric = _safe_eval_tracked_expression(tracked_value, named_angles)
    except (ValueError, ZeroDivisionError, SyntaxError):
        return None

    try:
        top_threshold = float(exercise_config["top_threshold"])
        bottom_threshold = float(exercise_config["bottom_threshold"])
    except (KeyError, TypeError, ValueError):
        return None

    top_is_lower = top_threshold < bottom_threshold
    return metric, top_threshold, bottom_threshold, top_is_lower


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
    metric_data = _exercise_metric(curr_lm, exercise_name)
    if metric_data is None:
        return None

    metric, top_threshold, bottom_threshold, top_is_lower = metric_data
    if top_is_lower:
        if metric <= top_threshold:
            return "top"
        if metric >= bottom_threshold:
            return "bottom"
        return None

    if metric >= top_threshold:
        return "top"
    if metric <= bottom_threshold:
        return "bottom"
    return None


def _exercise_motion_metric(curr_lm, exercise_name):
    metric_data = _exercise_metric(curr_lm, exercise_name)
    if metric_data is None:
        return None

    metric, top_threshold, bottom_threshold, _ = metric_data
    return metric, top_threshold, bottom_threshold


def get_rep_count_at_map():
    return dict(REP_COUNT_AT)


def get_motion_profiles():
    # Compatibility shim for legacy imports. Dynamic profile learning is not wired here.
    return {}


def get_feature_value(curr_lm, feature_name):
    # Compatibility shim for legacy imports.
    return curr_lm.get(feature_name)


def get_rep_config():
    return _REP_CONFIG


def get_rep_config_source_path():
    return _rep_config_path()


def _progress_from_extremity(metric, top_threshold, bottom_threshold, start_extremity):
    if start_extremity not in ("top", "bottom"):
        return 0.0

    if start_extremity == "top":
        start_level = top_threshold
        target_level = bottom_threshold
    else:
        start_level = bottom_threshold
        target_level = top_threshold

    delta = target_level - start_level
    if abs(delta) <= 1e-6:
        return 0.0

    # Directional progress: only movement toward target increases progress.
    progress = (metric - start_level) / delta
    return max(0.0, min(1.0, progress))


def get_switch_progress(curr_lm, exercise_name, start_extremity):
    """Return normalized movement progress [0,1] from the starting extremity."""
    metric_data = _exercise_motion_metric(curr_lm, exercise_name)
    if metric_data is None:
        return 0.0

    metric, top_threshold, bottom_threshold = metric_data
    return _progress_from_extremity(
        metric,
        top_threshold,
        bottom_threshold,
        start_extremity,
    )


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

    count_at = get_count_at(exercise_name)
    if extremity == count_at:
        rep_counts[exercise_name] += 1
    state.last_extremity = extremity


def reset_rep_counter_tracking(rep_states):
    for state in rep_states.values():
        state.pending_extremity = None
        state.pending_count = 0


def get_exercise_metric_value(curr_lm, exercise_name):
    """Return the raw tracked metric value for a single frame, or None.

    Single source of truth for the angle used by switching and rep-counting —
    the visualiser uses this so the graph always matches what the system sees.
    """
    result = _exercise_metric(curr_lm, exercise_name)
    if result is None:
        return None
    return result[0]  # (metric, top_threshold, bottom_threshold, top_is_lower)


def get_exercise_metric_unit(exercise_name):
    """Return the display unit string for the exercise metric.

    Returns '' for normalised position metrics (_norm_y_up, _above_ankle),
    and 'deg' for angle-based metrics.
    """
    config = _get_exercise_config(exercise_name)
    tracked = config.get("tracked_value", "")
    if "_norm_y_up" in tracked or "_above_ankle" in tracked:
        return ""
    return "deg"
