import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import re
from collections import defaultdict

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


# ── Data loading ──────────────────────────────────────────────────────────────


def load_data(data_dir):
    """
    Scans data_dir for CSVs, strips trailing digits from filename to get label.
    e.g. squat1.csv → 'squat',  shoulder_press2.csv → 'shoulder_press'
    """
    exercises = defaultdict(list)
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            match = re.match(r"^(.+?)\d*\.csv$", filename)
            if match:
                label = match.group(1).rstrip("_-")
                exercises[label].append(os.path.join(data_dir, filename))

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

    print("Saved: model_updated.pkl, encoder_updated.pkl, feature_config_updated.pkl")


if __name__ == "__main__":
    main()
