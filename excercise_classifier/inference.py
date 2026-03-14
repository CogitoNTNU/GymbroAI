import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from collections import defaultdict


LABEL_SWITCH_STREAK = 20
REP_EXTREMITY_STREAK = 3

# ── Body landmarks only (matches training data) ──────────────────────────────

# MediaPipe indices for body landmarks (excludes face/head details)
BODY_LANDMARK_INDICES = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# Ordered list of tracked landmarks.
# Ankles and feet stay here for squat rep counting and full skeleton drawing,
# but the classifier feature vector comes from feature_config["body_landmarks"].
LANDMARK_NAMES = [
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
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

# Pose connections for visualization (using body landmarks only)
POSE_CONNECTIONS = frozenset(
    [
        # Shoulders
        (11, 12),  # left_shoulder to right_shoulder
        # Left arm
        (11, 13),  # left_shoulder to left_elbow
        (13, 15),  # left_elbow to left_wrist
        # Right arm
        (12, 14),  # right_shoulder to right_elbow
        (14, 16),  # right_elbow to right_wrist
        # Torso
        (11, 23),  # left_shoulder to left_hip
        (12, 24),  # right_shoulder to right_hip
        (23, 24),  # left_hip to right_hip
        # Left leg
        (23, 25),  # left_hip to left_knee
        (25, 27),  # left_knee to left_ankle
        (27, 29),  # left_ankle to left_heel
        (27, 31),  # left_ankle to left_foot_index
        (29, 31),  # left_heel to left_foot_index
        # Right leg
        (24, 26),  # right_hip to right_knee
        (26, 28),  # right_knee to right_ankle
        (28, 30),  # right_ankle to right_heel
        (28, 32),  # right_ankle to right_foot_index
        (30, 32),  # right_heel to right_foot_index
    ]
)


# ── Feature extraction (must exactly mirror train_updated.py) ────────────────


def calculate_angle(p1, p2, p3):
    a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos_angle = np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def extract_angles(lm, angle_triplets):
    angles = []
    for a_name, vertex_name, c_name in angle_triplets:
        p1 = (lm[f"{a_name}_x"], lm[f"{a_name}_y"])
        p2 = (lm[f"{vertex_name}_x"], lm[f"{vertex_name}_y"])
        p3 = (lm[f"{c_name}_x"], lm[f"{c_name}_y"])
        angles.append(calculate_angle(p1, p2, p3))
    return angles


def _normalise(lm, body_landmarks):
    """
    Translate to body center origin, scale by torso height in 3D space.
    Body center = midpoint between hip center and shoulder center.
    """
    # Hip center
    hip_cx = (lm["left_hip_x"] + lm["right_hip_x"]) / 2.0
    hip_cy = (lm["left_hip_y"] + lm["right_hip_y"]) / 2.0
    hip_cz = (lm["left_hip_z"] + lm["right_hip_z"]) / 2.0

    # Shoulder center
    sho_cx = (lm["left_shoulder_x"] + lm["right_shoulder_x"]) / 2.0
    sho_cy = (lm["left_shoulder_y"] + lm["right_shoulder_y"]) / 2.0
    sho_cz = (lm["left_shoulder_z"] + lm["right_shoulder_z"]) / 2.0

    # Body center (midpoint between hip and shoulder centers)
    body_cx = (hip_cx + sho_cx) / 2.0
    body_cy = (hip_cy + sho_cy) / 2.0
    body_cz = (hip_cz + sho_cz) / 2.0

    # 3D torso height for scaling
    torso = max(
        np.sqrt(
            (sho_cx - hip_cx) ** 2 + (sho_cy - hip_cy) ** 2 + (sho_cz - hip_cz) ** 2
        ),
        1e-6,
    )

    return {
        name: (
            (lm[f"{name}_x"] - body_cx) / torso,
            (lm[f"{name}_y"] - body_cy) / torso,
            (lm[f"{name}_z"] - body_cz) / torso,
        )
        for name in body_landmarks
    }


def extract_relative_positions(lm, body_landmarks):
    norm = _normalise(lm, body_landmarks)
    feats = []
    for name in body_landmarks:
        feats.extend(norm[name])  # Add x, y, z
    return feats


def build_feature_vector(curr_lm, feature_config):
    angles = extract_angles(curr_lm, feature_config["angle_triplets"])
    positions = extract_relative_positions(curr_lm, feature_config["body_landmarks"])
    return np.array(angles + positions, dtype=np.float32)


def detect_rep_extremity(curr_lm, exercise_name):
    left_elbow = calculate_angle(
        (curr_lm["left_shoulder_x"], curr_lm["left_shoulder_y"]),
        (curr_lm["left_elbow_x"], curr_lm["left_elbow_y"]),
        (curr_lm["left_wrist_x"], curr_lm["left_wrist_y"]),
    )
    right_elbow = calculate_angle(
        (curr_lm["right_shoulder_x"], curr_lm["right_shoulder_y"]),
        (curr_lm["right_elbow_x"], curr_lm["right_elbow_y"]),
        (curr_lm["right_wrist_x"], curr_lm["right_wrist_y"]),
    )
    left_knee = calculate_angle(
        (curr_lm["left_hip_x"], curr_lm["left_hip_y"]),
        (curr_lm["left_knee_x"], curr_lm["left_knee_y"]),
        (curr_lm["left_ankle_x"], curr_lm["left_ankle_y"]),
    )
    right_knee = calculate_angle(
        (curr_lm["right_hip_x"], curr_lm["right_hip_y"]),
        (curr_lm["right_knee_x"], curr_lm["right_knee_y"]),
        (curr_lm["right_ankle_x"], curr_lm["right_ankle_y"]),
    )

    elbow_angle = (left_elbow + right_elbow) / 2.0
    knee_angle = (left_knee + right_knee) / 2.0
    shoulder_y = (curr_lm["left_shoulder_y"] + curr_lm["right_shoulder_y"]) / 2.0
    wrist_y = (curr_lm["left_wrist_y"] + curr_lm["right_wrist_y"]) / 2.0

    if exercise_name == "squat":
        if knee_angle < 105:
            return "bottom"
        if knee_angle > 155:
            return "top"
        return None

    if exercise_name == "curl":
        if elbow_angle < 60:
            return "top"
        if elbow_angle > 145:
            return "bottom"
        return None

    if exercise_name == "shoulder_press":
        if wrist_y < shoulder_y - 0.08 and elbow_angle > 145:
            return "top"
        if wrist_y >= shoulder_y - 0.02 and elbow_angle < 110:
            return "bottom"
        return None

    return None


def update_rep_counts(displayed_exercise, curr_lm, rep_counts, rep_states):
    if displayed_exercise is None:
        return

    extremity = detect_rep_extremity(curr_lm, displayed_exercise)
    if extremity is None:
        return

    state = rep_states[displayed_exercise]

    if state["last_extremity"] == extremity:
        state["pending_extremity"] = None
        state["pending_count"] = 0
        return

    if state["pending_extremity"] == extremity:
        state["pending_count"] += 1
    else:
        state["pending_extremity"] = extremity
        state["pending_count"] = 1

    if state["pending_count"] < REP_EXTREMITY_STREAK:
        return

    state["pending_extremity"] = None
    state["pending_count"] = 0

    if state["last_extremity"] is None:
        state["last_extremity"] = extremity
        return

    if state["last_extremity"] != extremity:
        rep_counts[displayed_exercise] += 0.5
        state["last_extremity"] = extremity


def draw_rep_summary(frame, displayed_exercise, rep_counts):
    y = 120
    if displayed_exercise is not None:
        cv2.putText(
            frame,
            f"Reps: {rep_counts[displayed_exercise]:.1f}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 200, 0),
            2,
        )
        y += 35

    for exercise_name, count in sorted(rep_counts.items()):
        cv2.putText(
            frame,
            f"{exercise_name}: {count:.1f}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            2,
        )
        y += 28


# ── Helpers ───────────────────────────────────────────────────────────────────


def landmarks_to_dict(pose_landmarks):
    """Convert MediaPipe landmarks to dict (body landmarks only) with x, y, z."""
    lm = {}
    for name in LANDMARK_NAMES:
        idx = BODY_LANDMARK_INDICES[name]
        landmark = pose_landmarks[idx]
        lm[f"{name}_x"] = landmark.x
        lm[f"{name}_y"] = landmark.y
        lm[f"{name}_z"] = landmark.z
    return lm


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw skeleton and landmarks on image (body landmarks only)."""
    annotated = np.copy(rgb_image)
    for pose_landmarks in detection_result.pose_landmarks:
        h, w = annotated.shape[:2]

        # Draw connections
        for connection in POSE_CONNECTIONS:
            s = pose_landmarks[connection[0]]
            e = pose_landmarks[connection[1]]
            cv2.line(
                annotated,
                (int(s.x * w), int(s.y * h)),
                (int(e.x * w), int(e.y * h)),
                (0, 255, 0),
                2,
            )

        # Draw body landmarks only
        for name in LANDMARK_NAMES:
            idx = BODY_LANDMARK_INDICES[name]
            lm = pose_landmarks[idx]
            cv2.circle(annotated, (int(lm.x * w), int(lm.y * h)), 5, (0, 0, 255), -1)

    return annotated


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    model = joblib.load(os.path.join(models_dir, "model_updated.pkl"))
    encoder = joblib.load(os.path.join(models_dir, "encoder_updated.pkl"))
    feature_config = joblib.load(os.path.join(models_dir, "feature_config_updated.pkl"))

    print("=" * 80)
    print("EXERCISE CLASSIFIER INFERENCE - BODY LANDMARKS ONLY (Angles + Positions)")
    print("=" * 80)
    print(f"Model loaded. Classes: {encoder.classes_}")
    print(
        f"Feature vector: {feature_config['total_features']} dims  "
        f"(angles={feature_config['n_angles']}  "
        f"positions={feature_config['n_positions']})"
    )
    print(
        f"Using {len(feature_config['body_landmarks'])} body landmarks for classification "
        f"(ankles/feet removed from features)"
    )
    print("Normalization: 3D body center at origin (0,0,0), scaled by torso height")
    print("Rep counting: +0.5 at each confirmed extremity, tracked per exercise")
    print("=" * 80)
    print()

    model_path = os.path.join(
        os.path.dirname(__file__), "data", "pose_landmarker_full.task"
    )
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = 0
    displayed_exercise = None
    pending_exercise = None
    pending_count = 0
    rep_counts = {exercise_name: 0.0 for exercise_name in encoder.classes_}
    rep_states = defaultdict(
        lambda: {
            "last_extremity": None,
            "pending_extremity": None,
            "pending_count": 0,
        }
    )

    print("Camera opened. Press Q to quit.")

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if result.pose_landmarks:
                frame = draw_landmarks_on_image(frame, result)

                curr_landmarks = landmarks_to_dict(result.pose_landmarks[0])

                features = build_feature_vector(curr_landmarks, feature_config)
                prediction = model.predict(features.reshape(1, -1))[0]
                exercise = encoder.inverse_transform([prediction])[0]

                if displayed_exercise is None:
                    displayed_exercise = exercise
                    pending_exercise = None
                    pending_count = 0
                elif exercise == displayed_exercise:
                    pending_exercise = None
                    pending_count = 0
                elif exercise == pending_exercise:
                    pending_count += 1
                    if pending_count >= LABEL_SWITCH_STREAK:
                        displayed_exercise = exercise
                        pending_exercise = None
                        pending_count = 0
                else:
                    pending_exercise = exercise
                    pending_count = 1

                update_rep_counts(
                    displayed_exercise, curr_landmarks, rep_counts, rep_states
                )

                cv2.putText(
                    frame,
                    f"Exercise: {displayed_exercise.upper()}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    3,
                )
                if pending_exercise is not None:
                    cv2.putText(
                        frame,
                        f"Switching: {pending_exercise.upper()} {pending_count}/{LABEL_SWITCH_STREAK}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                draw_rep_summary(frame, displayed_exercise, rep_counts)
            else:
                # No pose detected — reset so stale data doesn't carry over
                pending_exercise = None
                pending_count = 0
                for state in rep_states.values():
                    state["pending_extremity"] = None
                    state["pending_count"] = 0
                cv2.putText(
                    frame,
                    "No pose detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )
                draw_rep_summary(frame, displayed_exercise, rep_counts)

            cv2.imshow("Exercise Classifier - Body Only (3D)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_timestamp_ms += 33

    cap.release()
    cv2.destroyAllWindows()
    print("\nExiting...")


if __name__ == "__main__":
    main()
