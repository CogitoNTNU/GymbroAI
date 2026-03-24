import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import re
from collections import defaultdict

# Default threshold tuning values kept local to avoid external config dependency.
MIN_THRESHOLD_MARGIN_DEGREES = 8.0
REP_THRESHOLD_MARGIN_RATIO = 0.20

# ── Definitions ──────────────────────────────────────────────────────────────

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
    "right_shoulder_elbow_wrist": (
        "right_shoulder",
        "right_elbow",
        "right_wrist",
    ),
    "left_hip_knee_ankle": ("left_hip", "left_knee", "left_ankle"),
    "right_hip_knee_ankle": ("right_hip", "right_knee", "right_ankle"),
}

GLOBAL_SWITCH_PROGRESS_THRESHOLD = 0.15

EXERCISE_ANGLE_SELECTION = {
    "squat": {
        "top_is_lower": False,
        "count_at": "bottom",
    },
    "curl": {
        "top_is_lower": True,
        "count_at": "top",
    },
    "shoulder_press": {
        "top_is_lower": False,
        "count_at": "top",
    },
}

EXERCISE_TRACKED_VALUE = {
    "squat": ["left_hip_knee_ankle", "right_hip_knee_ankle"],
    "curl": ["left_shoulder_elbow_wrist", "right_shoulder_elbow_wrist"],
    "shoulder_press": ["left_shoulder_elbow_wrist", "right_shoulder_elbow_wrist"],
}

# ── Features: Relative positions in 3D space ─────────────────────────────────


def extract_relative_positions(row):
    # Get hip and shoulder landmarks
    l_hip_x, l_hip_y, l_hip_z = row["left_hip_x"], row["left_hip_y"], row["left_hip_z"]
    r_hip_x, r_hip_y, r_hip_z = (
        row["right_hip_x"],
        row["right_hip_y"],
        row["right_hip_z"],
    )
    l_sho_x, l_sho_y, l_sho_z = (
        row["left_shoulder_x"],
        row["left_shoulder_y"],
        row["left_shoulder_z"],
    )
    r_sho_x, r_sho_y, r_sho_z = (
        row["right_shoulder_x"],
        row["right_shoulder_y"],
        row["right_shoulder_z"],
    )

    # Calculate hip center
    hip_cx = (l_hip_x + r_hip_x) / 2.0
    hip_cy = (l_hip_y + r_hip_y) / 2.0
    hip_cz = (l_hip_z + r_hip_z) / 2.0

    # Calculate shoulder center
    sho_cx = (l_sho_x + r_sho_x) / 2.0
    sho_cy = (l_sho_y + r_sho_y) / 2.0
    sho_cz = (l_sho_z + r_sho_z) / 2.0

    # Body center = midpoint between hip center and shoulder center
    body_cx = (hip_cx + sho_cx) / 2.0
    body_cy = (hip_cy + sho_cy) / 2.0
    body_cz = (hip_cz + sho_cz) / 2.0

    # Torso height for scaling (3D distance)
    torso_h = np.sqrt(
        (sho_cx - hip_cx) ** 2 + (sho_cy - hip_cy) ** 2 + (sho_cz - hip_cz) ** 2
    )
    torso_h = max(torso_h, 1e-6)  # avoid div-by-zero

    feats = []
    for name in BODY_LANDMARKS:
        # Normalize all three dimensions relative to body center
        nx = (row[f"{name}_x"] - body_cx) / torso_h
        ny = (row[f"{name}_y"] - body_cy) / torso_h
        nz = (row[f"{name}_z"] - body_cz) / torso_h
        feats.extend([nx, ny, nz])

    return feats


def calculate_angle(p1, p2, p3):
    a = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float32)
    b = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a <= 1e-6 or norm_b <= 1e-6:
        return None
    cos_angle = np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _angle_from_row(row, angle_name):
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
    if any(column not in row for column in required):
        return None

    p1 = (row[f"{a_name}_x"], row[f"{a_name}_y"])
    p2 = (row[f"{b_name}_x"], row[f"{b_name}_y"])
    p3 = (row[f"{c_name}_x"], row[f"{c_name}_y"])
    return calculate_angle(p1, p2, p3)


def _compute_exercise_metric_series(exercise_df, tracked_angles):
    series = []
    for _, row in exercise_df.iterrows():
        values = []
        for angle_name in tracked_angles:
            angle = _angle_from_row(row, angle_name)
            if angle is not None:
                values.append(angle)
        if values:
            series.append(float(sum(values) / len(values)))
    return series


def _thresholds_from_metric_series(series, top_is_lower):
    if len(series) < 10:
        return None

    lower_q = float(np.percentile(series, 20))
    upper_q = float(np.percentile(series, 80))
    if upper_q - lower_q <= 1e-6:
        return None

    span = upper_q - lower_q
    margin = max(MIN_THRESHOLD_MARGIN_DEGREES, span * REP_THRESHOLD_MARGIN_RATIO)

    if top_is_lower:
        top_threshold = min(lower_q + margin, upper_q - 1e-3)
        bottom_threshold = max(upper_q - margin, lower_q + 1e-3)
    else:
        top_threshold = max(upper_q - margin, lower_q + 1e-3)
        bottom_threshold = min(lower_q + margin, upper_q - 1e-3)

    if abs(top_threshold - bottom_threshold) <= 1e-6:
        if top_is_lower:
            return lower_q, upper_q
        return upper_q, lower_q

    return top_threshold, bottom_threshold


def build_rep_counting_config(df):
    exercise_configs = {}
    for exercise_name, settings in EXERCISE_ANGLE_SELECTION.items():
        exercise_df = df[df["exercise"] == exercise_name]
        if exercise_df.empty:
            continue

        tracked_angles = list(EXERCISE_TRACKED_VALUE[exercise_name])
        top_is_lower = bool(settings["top_is_lower"])
        metric_series = _compute_exercise_metric_series(exercise_df, tracked_angles)
        thresholds = _thresholds_from_metric_series(metric_series, top_is_lower)
        if thresholds is None:
            continue

        tracked_expression = f"({' + '.join(tracked_angles)}) / {len(tracked_angles)}"
        top_threshold, bottom_threshold = thresholds
        exercise_configs[f"{exercise_name}_config"] = {
            "tracked_value": tracked_expression,
            "top_threshold": float(top_threshold),
            "bottom_threshold": float(bottom_threshold),
            "count_at": str(settings["count_at"]),
        }

    return {
        "global_config": {
            "switch_progress_threshold": GLOBAL_SWITCH_PROGRESS_THRESHOLD,
        },
        "exercise_configs": exercise_configs,
    }


# ── Data loading ──────────────────────────────────────────────────────────────


def load_data(data_dir):
    """
    Scans data_dir for CSVs and loads one set per exercise label.

    Preferred input is merged files: <exercise>.csv.
    Backward compatibility: if merged file is missing, numbered files
    (<exercise>1.csv, <exercise>2.csv, ...) are used.
    """
    merged_by_label = {}
    numbered_by_label = defaultdict(list)

    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
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
    labels = set(merged_by_label) | set(numbered_by_label)
    for label in labels:
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


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    print("Loading data...")
    df = load_data(data_dir)

    print("Extracting features...")
    X, y = [], []

    for _, row in df.iterrows():
        X.append(extract_relative_positions(row))
        y.append(row["exercise"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    n_angles = 0
    n_positions = len(BODY_LANDMARKS) * 3
    print(
        f"Samples: {len(X)} | Features: {X.shape[1]} (angles={n_angles}, positions={n_positions})"
    )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)}...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")

    joblib.dump(model, os.path.join(models_dir, "model_updated.pkl"))
    joblib.dump(label_encoder, os.path.join(models_dir, "encoder_updated.pkl"))

    # Save feature config so the inference script uses identical extraction
    feature_config = {
        "angle_triplets": [],
        "body_landmarks": BODY_LANDMARKS,
        "n_angles": n_angles,
        "n_positions": n_positions,
        "total_features": X.shape[1],
    }
    joblib.dump(feature_config, os.path.join(models_dir, "feature_config_updated.pkl"))

    exercise_configs = build_rep_counting_config(df)

    exercise_config_path = os.path.join(models_dir, "exercise_configs.json")
    with open(exercise_config_path, "w", encoding="utf-8") as handle:
        json.dump(exercise_configs, handle, indent=2)

    # Legacy output retained so older code paths continue to work.
    rep_config_path = os.path.join(models_dir, "rep_counting_config.json")
    with open(rep_config_path, "w", encoding="utf-8") as handle:
        json.dump(exercise_configs, handle, indent=2)

    print(
        "Saved: model_updated.pkl, encoder_updated.pkl, "
        "feature_config_updated.pkl, exercise_configs.json, rep_counting_config.json"
    )


if __name__ == "__main__":
    main()
