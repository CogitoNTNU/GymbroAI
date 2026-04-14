"""
Train the XGBoost exercise classifier and generate rep-counting thresholds.

Outputs (written to models/):
    - model_updated.pkl          — trained XGBClassifier
    - encoder_updated.pkl        — LabelEncoder (exercise name <-> int)
    - feature_config_updated.pkl — landmark list used for feature extraction
    - exercise_configs.json      — auto-tuned rep counting thresholds (reference only,
                                   runtime uses hardcoded values in rep_counter.py)
"""

import json
import os
import re
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# XGBoost training.
N_ESTIMATORS = 300
MAX_DEPTH = 6
LEARNING_RATE = 0.1
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
TEST_SIZE = 0.2

# Threshold tuning — minimum margin between top and bottom thresholds.
MIN_THRESHOLD_MARGIN_DEGREES = 8.0
REP_THRESHOLD_MARGIN_RATIO = 0.20

# Written to exercise_configs.json for reference.
GLOBAL_SWITCH_PROGRESS_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# Landmark and exercise definitions
# ---------------------------------------------------------------------------

BODY_LANDMARKS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
]

ANGLE_TRIPLETS = {
    "left_shoulder_elbow_wrist": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_shoulder_elbow_wrist": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_hip_knee_ankle": ("left_hip", "left_knee", "left_ankle"),
    "right_hip_knee_ankle": ("right_hip", "right_knee", "right_ankle"),
}

# Per-exercise settings for threshold auto-tuning.
EXERCISE_ANGLE_SELECTION = {
    "squat": {"top_is_lower": False, "count_at": "bottom"},
    "curl": {"top_is_lower": True, "count_at": "top"},
    "shoulder_press": {"top_is_lower": False, "count_at": "top"},
}

EXERCISE_TRACKED_VALUE = {
    "squat": ["left_hip_knee_ankle", "right_hip_knee_ankle"],
    "curl": ["left_shoulder_elbow_wrist", "right_shoulder_elbow_wrist"],
    "shoulder_press": ["left_shoulder_elbow_wrist", "right_shoulder_elbow_wrist"],
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_relative_positions(row):
    """Compute body-center-relative, torso-normalized 3D positions for each landmark."""
    hip_cx = (row["left_hip_x"] + row["right_hip_x"]) / 2.0
    hip_cy = (row["left_hip_y"] + row["right_hip_y"]) / 2.0
    hip_cz = (row["left_hip_z"] + row["right_hip_z"]) / 2.0

    sho_cx = (row["left_shoulder_x"] + row["right_shoulder_x"]) / 2.0
    sho_cy = (row["left_shoulder_y"] + row["right_shoulder_y"]) / 2.0
    sho_cz = (row["left_shoulder_z"] + row["right_shoulder_z"]) / 2.0

    body_cx = (hip_cx + sho_cx) / 2.0
    body_cy = (hip_cy + sho_cy) / 2.0
    body_cz = (hip_cz + sho_cz) / 2.0

    torso = max(
        np.sqrt(
            (sho_cx - hip_cx) ** 2 + (sho_cy - hip_cy) ** 2 + (sho_cz - hip_cz) ** 2
        ),
        1e-6,
    )

    feats = []
    for name in BODY_LANDMARKS:
        feats.extend(
            [
                (row[f"{name}_x"] - body_cx) / torso,
                (row[f"{name}_y"] - body_cy) / torso,
                (row[f"{name}_z"] - body_cz) / torso,
            ]
        )
    return feats


# ---------------------------------------------------------------------------
# Rep counting config (angle-based threshold auto-tuning)
# ---------------------------------------------------------------------------


def _calculate_angle(p1, p2, p3):
    """Angle in degrees at p2 formed by p1-p2-p3."""
    a = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float32)
    b = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a <= 1e-6 or norm_b <= 1e-6:
        return None
    cos_angle = np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _angle_from_row(row, angle_name):
    """Compute a named angle from a dataframe row."""
    triplet = ANGLE_TRIPLETS.get(angle_name)
    if triplet is None:
        return None
    a_name, b_name, c_name = triplet
    required = [
        f"{a_name}_x",
        f"{a_name}_y",
        f"{b_name}_x",
        f"{b_name}_y",
        f"{c_name}_x",
        f"{c_name}_y",
    ]
    if any(col not in row for col in required):
        return None
    p1 = (row[f"{a_name}_x"], row[f"{a_name}_y"])
    p2 = (row[f"{b_name}_x"], row[f"{b_name}_y"])
    p3 = (row[f"{c_name}_x"], row[f"{c_name}_y"])
    return _calculate_angle(p1, p2, p3)


def _compute_metric_series(exercise_df, tracked_angles):
    """Average the tracked angles across all frames for one exercise."""
    series = []
    for _, row in exercise_df.iterrows():
        values = [
            a
            for name in tracked_angles
            if (a := _angle_from_row(row, name)) is not None
        ]
        if values:
            series.append(float(sum(values) / len(values)))
    return series


def _thresholds_from_series(series, top_is_lower):
    """Derive top/bottom thresholds from percentiles of a metric series."""
    if len(series) < 10:
        return None

    lower_q = float(np.percentile(series, 20))
    upper_q = float(np.percentile(series, 80))
    if upper_q - lower_q <= 1e-6:
        return None

    span = upper_q - lower_q
    margin = max(MIN_THRESHOLD_MARGIN_DEGREES, span * REP_THRESHOLD_MARGIN_RATIO)

    if top_is_lower:
        top_t = min(lower_q + margin, upper_q - 1e-3)
        bot_t = max(upper_q - margin, lower_q + 1e-3)
    else:
        top_t = max(upper_q - margin, lower_q + 1e-3)
        bot_t = min(lower_q + margin, upper_q - 1e-3)

    if abs(top_t - bot_t) <= 1e-6:
        return (lower_q, upper_q) if top_is_lower else (upper_q, lower_q)

    return top_t, bot_t


def _build_rep_counting_config(df):
    """Auto-tune rep counting thresholds from training data."""
    exercise_configs = {}
    for exercise_name, settings in EXERCISE_ANGLE_SELECTION.items():
        exercise_df = df[df["exercise"] == exercise_name]
        if exercise_df.empty:
            continue

        tracked_angles = list(EXERCISE_TRACKED_VALUE[exercise_name])
        series = _compute_metric_series(exercise_df, tracked_angles)
        thresholds = _thresholds_from_series(series, bool(settings["top_is_lower"]))
        if thresholds is None:
            continue

        top_t, bot_t = thresholds
        exercise_configs[f"{exercise_name}_config"] = {
            "tracked_value": f"({' + '.join(tracked_angles)}) / {len(tracked_angles)}",
            "top_threshold": float(top_t),
            "bottom_threshold": float(bot_t),
            "count_at": str(settings["count_at"]),
        }

    return {
        "global_config": {
            "switch_progress_threshold": GLOBAL_SWITCH_PROGRESS_THRESHOLD
        },
        "exercise_configs": exercise_configs,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(data_dir):
    """Load CSVs from data_dir. Prefers merged <exercise>.csv, falls back to numbered files."""
    merged_by_label = {}
    numbered_by_label = defaultdict(list)

    for filename in os.listdir(data_dir):
        if not filename.endswith(".csv"):
            continue
        numbered_match = re.match(r"^(.+?)(\d+)\.csv$", filename)
        if numbered_match:
            label = numbered_match.group(1).rstrip("_-")
            numbered_by_label[label].append(os.path.join(data_dir, filename))
            continue
        merged_match = re.match(r"^(.+?)\.csv$", filename)
        if merged_match:
            label = merged_match.group(1).rstrip("_-")
            merged_by_label[label] = os.path.join(data_dir, filename)

    exercises = {}
    for label in set(merged_by_label) | set(numbered_by_label):
        if label in merged_by_label:
            exercises[label] = [merged_by_label[label]]
        else:
            exercises[label] = sorted(numbered_by_label[label])

    all_data = []
    for label, files in exercises.items():
        for file_path in files:
            df = pd.read_csv(file_path)
            df["exercise"] = label
            all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    print(
        f"Loaded {len(combined)} frames from {sum(len(v) for v in exercises.values())} files"
    )
    return combined


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    # Load and extract features.
    print("Loading data...")
    df = load_data(data_dir)

    print("Extracting features...")
    X = np.array(
        [extract_relative_positions(row) for _, row in df.iterrows()], dtype=np.float32
    )
    y = np.array([row["exercise"] for _, row in df.iterrows()])
    print(f"Samples: {len(X)} | Features: {X.shape[1]}")

    # Encode labels and split.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=42, stratify=y_encoded
    )

    # Train.
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)}...")
    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

    # Save model artifacts.
    joblib.dump(model, os.path.join(models_dir, "model_updated.pkl"))
    joblib.dump(label_encoder, os.path.join(models_dir, "encoder_updated.pkl"))
    joblib.dump(
        {"body_landmarks": BODY_LANDMARKS},
        os.path.join(models_dir, "feature_config_updated.pkl"),
    )

    # Save auto-tuned rep counting config (reference only).
    exercise_config_path = os.path.join(models_dir, "exercise_configs.json")
    with open(exercise_config_path, "w", encoding="utf-8") as f:
        json.dump(_build_rep_counting_config(df), f, indent=2)

    print(
        "Saved: model_updated.pkl, encoder_updated.pkl, feature_config_updated.pkl, exercise_configs.json"
    )


if __name__ == "__main__":
    main()
