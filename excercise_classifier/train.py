import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import re
from collections import defaultdict


# ── Landmark definitions ──────────────────────────────────────────────────────

# Body landmarks used for relative position features (excludes face/head details)
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

# Joint angle triplets: (point_a, VERTEX, point_c) → angle at vertex
ANGLE_TRIPLETS = [
    ("left_shoulder", "left_hip", "left_knee"),
    ("right_shoulder", "right_hip", "right_knee"),
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_shoulder", "left_elbow"),
    ("right_hip", "right_shoulder", "right_elbow"),
    # Extra angles that help distinguish exercises
    ("left_elbow", "left_shoulder", "right_shoulder"),  # L shoulder abduction
    ("right_elbow", "right_shoulder", "left_shoulder"),  # R shoulder abduction
]


# ── Feature 1: Angles ─────────────────────────────────────────────────────────


def calculate_angle(p1, p2, p3):
    """Angle at vertex p2 formed by p1→p2 and p3→p2 (degrees)."""
    a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos_angle = np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def extract_angles(row):
    angles = []
    for a_name, vertex_name, c_name in ANGLE_TRIPLETS:
        p1 = (row[f"{a_name}_x"], row[f"{a_name}_y"])
        p2 = (row[f"{vertex_name}_x"], row[f"{vertex_name}_y"])
        p3 = (row[f"{c_name}_x"], row[f"{c_name}_y"])
        angles.append(calculate_angle(p1, p2, p3))
    return angles  # length = len(ANGLE_TRIPLETS)


# ── Feature 2: Relative positions in 3D space ────────────────────────────────
#
# Raw MediaPipe coordinates change with where you stand in the frame, so we
# normalise them in 3D space (x, y, z):
#   1. Calculate body center as the midpoint between hip center and shoulder center
#   2. Translate all coordinates so body center is at origin (0, 0, 0)
#   3. Divide by torso height for scale invariance
# This gives scale-invariant and position-invariant 3D coordinates.


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

    return feats  # length = len(BODY_LANDMARKS) * 3


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
        f"[DATA] Loaded {len(combined)} frames from {sum(len(v) for v in exercises.values())} files"
    )
    for label, files in sorted(exercises.items()):
        n = sum(len(pd.read_csv(f)) for f in files)
        print(f"         {label:20s}  {n} frames  ({len(files)} file(s))")
    return combined


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 80)
    print("EXERCISE CLASSIFIER TRAINING - BODY LANDMARKS ONLY")
    print("=" * 80)
    print(
        f"Using {len(BODY_LANDMARKS)} body landmarks for features (ankles/feet removed)"
    )
    print("Extracting 2 feature types in 3D space:")
    print(f"  - Angles: {len(ANGLE_TRIPLETS)} joint angles")
    print(
        f"  - Positions: {len(BODY_LANDMARKS)} × 3 = {len(BODY_LANDMARKS) * 3} normalized 3D coordinates (x,y,z)"
    )
    print("\nNormalization: Body center (torso midpoint) at origin (0,0,0)")
    print("               Scaled by 3D torso height for position/scale invariance")
    print("=" * 80)
    print()

    print("Loading data...")
    df = load_data(data_dir)

    print("\nExtracting features  (angles + 3D positions)...")
    X, y = [], []

    df_grouped = df.groupby("exercise")

    for exercise_label, group in df_grouped:
        rows = group.reset_index(drop=True)
        for _, row in rows.iterrows():
            angles = extract_angles(row)
            positions = extract_relative_positions(row)

            X.append(angles + positions)
            y.append(exercise_label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    n_angles = len(ANGLE_TRIPLETS)
    n_positions = len(BODY_LANDMARKS) * 3
    print(
        f"\n[FEATURES]  angles={n_angles}  positions={n_positions}  total={X.shape[1]}"
    )
    print(f"[SAMPLES]   {len(X)} frames  ×  {X.shape[1]} features")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)}...")
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
    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("\nSaving model...")
    joblib.dump(model, os.path.join(models_dir, "model_updated.pkl"))
    joblib.dump(label_encoder, os.path.join(models_dir, "encoder_updated.pkl"))

    # Save feature config so the inference script uses identical extraction
    feature_config = {
        "angle_triplets": ANGLE_TRIPLETS,
        "body_landmarks": BODY_LANDMARKS,
        "n_angles": n_angles,
        "n_positions": n_positions,
        "total_features": X.shape[1],
    }
    joblib.dump(feature_config, os.path.join(models_dir, "feature_config_updated.pkl"))

    print("\nSaved to models/:")
    print("  - model_updated.pkl")
    print("  - encoder_updated.pkl")
    print("  - feature_config_updated.pkl")
    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
