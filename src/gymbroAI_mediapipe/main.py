# ---------------------------------------------------------------------
# Imports.
# ---------------------------------------------------------------------
import os

import cv2
import joblib
import mediapipe as mp
import numpy as np

from runtime_logic.data_stream_manager import (
    initialize_dual_stream_state,
    update_active_exercise_with_dual_stream,
)
from runtime_logic.feedback_analyser import (
    create_feedback_state,
    get_feedback_message,
    get_form_feedback,
    reset_form_feedback_tracking,
)
from runtime_logic.rep_counter import (
    create_rep_counter_state,
    get_rep_config_source_path,
    reset_rep_counter_tracking,
    update_rep_counts,
)

from graphics.draw_on_screen import (
    draw_active_hud,
    draw_feedback_visualizer,
    draw_no_pose_hud,
    load_feedback_emojis,
)

WINDOW_NAME = "Exercise Classifier - Body Only (3D)"

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
}

LANDMARK_NAMES = list(BODY_LANDMARK_INDICES)

POSE_CONNECTIONS = frozenset(
    [
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (25, 27),
        (24, 26),
        (26, 28),
    ]
)


def landmarks_to_dict(pose_landmarks):
    lm = {}
    for name in LANDMARK_NAMES:
        idx = BODY_LANDMARK_INDICES[name]
        landmark = pose_landmarks[idx]
        lm[f"{name}_x"] = landmark.x
        lm[f"{name}_y"] = landmark.y
        lm[f"{name}_z"] = landmark.z
    return lm


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated = np.copy(rgb_image)
    for pose_landmarks in detection_result.pose_landmarks:
        h, w = annotated.shape[:2]

        for connection in POSE_CONNECTIONS:
            start = pose_landmarks[connection[0]]
            end = pose_landmarks[connection[1]]
            cv2.line(
                annotated,
                (int(start.x * w), int(start.y * h)),
                (int(end.x * w), int(end.y * h)),
                (0, 255, 0),
                2,
            )

        for name in LANDMARK_NAMES:
            idx = BODY_LANDMARK_INDICES[name]
            landmark = pose_landmarks[idx]
            cv2.circle(
                annotated,
                (int(landmark.x * w), int(landmark.y * h)),
                5,
                (0, 0, 255),
                -1,
            )

    return annotated


def load_classifier(models_dir):
    model = joblib.load(os.path.join(models_dir, "model_updated.pkl"))
    encoder = joblib.load(os.path.join(models_dir, "encoder_updated.pkl"))
    feature_config = joblib.load(os.path.join(models_dir, "feature_config_updated.pkl"))
    return model, encoder, feature_config


def predict_exercise(curr_lm, model, encoder, feature_config):
    features = build_feature_vector(curr_lm, feature_config)
    prediction = model.predict(features.reshape(1, -1))[0]

    try:
        return encoder.inverse_transform([prediction])[0]
    except Exception:
        return str(prediction)


def build_feature_vector(curr_lm, feature_config):
    positions = extract_relative_positions(curr_lm, feature_config["body_landmarks"])
    return np.array(positions, dtype=np.float32)


def extract_relative_positions(lm, body_landmarks):
    hip_cx = (lm["left_hip_x"] + lm["right_hip_x"]) / 2.0
    hip_cy = (lm["left_hip_y"] + lm["right_hip_y"]) / 2.0
    hip_cz = (lm["left_hip_z"] + lm["right_hip_z"]) / 2.0

    sho_cx = (lm["left_shoulder_x"] + lm["right_shoulder_x"]) / 2.0
    sho_cy = (lm["left_shoulder_y"] + lm["right_shoulder_y"]) / 2.0
    sho_cz = (lm["left_shoulder_z"] + lm["right_shoulder_z"]) / 2.0

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
    for name in body_landmarks:
        feats.extend(
            [
                (lm[f"{name}_x"] - body_cx) / torso,
                (lm[f"{name}_y"] - body_cy) / torso,
                (lm[f"{name}_z"] - body_cz) / torso,
            ]
        )
    return feats


# ---------------------------------------------------------------------
# Main configuration (adjust these values here).
# ---------------------------------------------------------------------
DEBUG_MODE = True
FRAME_STEP_MS = 33
QUIT_KEY = "q"
TOGGLE_VISUALIZER_KEY = "v"
SHOW_FULLSCREEN_WINDOW = False

STREAM_MAX_FRAMES = 240

FEEDBACK_BOTTOM_MARGIN = 35
EMOJI_MARGIN = 28
EMOJI_SIZE = 64


def get_rep_direction_label(exercise_name, rep_states):
    if exercise_name is None:
        return None

    state = rep_states.get(exercise_name)
    if state is None:
        return None

    # Pending extremity indicates current movement direction before full confirmation.
    if state.pending_extremity == "top":
        return "DOWN"
    if state.pending_extremity == "bottom":
        return "UP"

    # Fallback to last confirmed extremity.
    if state.last_extremity == "bottom":
        return "DOWN"
    if state.last_extremity == "top":
        return "UP"

    return None


def main():
    # ---------------------------------------------------------------------
    # Initialization.
    # ---------------------------------------------------------------------

    # Load classifier model, label encoder, and feature config.
    model, encoder, feature_config = load_classifier(
        os.path.join(
            os.path.dirname(__file__),
            "runtime_logic",
            "excercise_classifcation",
            "models",
        )
    )
    print(f"Rep config source: {get_rep_config_source_path()}")

    # Configure MediaPipe pose landmarker.
    model_path = os.path.join(
        os.path.dirname(__file__),
        "mediapipe_models",
        "pose_landmarker_full.task",
    )
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )

    # Initialize camera capture and display window.
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Enable fullscreen mode when configured.
    if SHOW_FULLSCREEN_WINDOW:
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )

    # MediaPipe VIDEO mode requires a monotonically increasing timestamp.
    frame_timestamp_ms = 0

    # Currently displayed exercise label after stream smoothing.
    displayed_exercise = None

    # Toggle that controls whether the debug visualizer is drawn.
    show_feedback_visualizer = DEBUG_MODE

    # Load feedback emojis once and reuse them across all frames.
    emoji_images = load_feedback_emojis(os.path.dirname(__file__))

    # Rep counter state:
    # rep_counts stores total counted reps per exercise.
    # rep_states stores per-exercise transition memory used for extremity debouncing.
    rep_counts, rep_states = create_rep_counter_state(encoder)

    # Form feedback state:
    # Keeps per-exercise rolling buffers and cycle flags so feedback is evaluated
    # after a completed rep cycle instead of every single frame.
    feedback_states = create_feedback_state()

    # Dual-stream switching state:
    # - active_stream: frames for the currently displayed exercise label.
    # - predicted_stream: frames for a candidate new label while validating a switch.
    # - predicted_progress/start_extremity: blocks early switches and reduces flicker.
    dual_stream_state = initialize_dual_stream_state(STREAM_MAX_FRAMES)

    # ---------------------------------------------------------------------
    # Main realtime loop.
    # ---------------------------------------------------------------------
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            # Read one frame from camera; exit if capture fails.
            success, frame = cap.read()
            if not success:
                break

            # flip frame and run pose detection.
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if result.pose_landmarks:
                # Draw pose skeleton overlay.
                frame = draw_landmarks_on_image(frame, result)

                # Convert landmarks to model input and predict exercise class.
                curr_landmarks = landmarks_to_dict(result.pose_landmarks[0])
                raw_exercise = predict_exercise(
                    curr_landmarks,
                    model,
                    encoder,
                    feature_config,
                )

                # Smooth exercise switching using active/predicted streams.
                # Only AFTER threshold gating should we update rep counts/feedback.
                displayed_exercise = update_active_exercise_with_dual_stream(
                    curr_landmarks,
                    raw_exercise,
                    displayed_exercise,
                    dual_stream_state,
                    rep_states=rep_states,
                )

                # Update rep counting and form-feedback pipelines using the SMOOTHED exercise.
                # This ensures reps are only counted after passing the 50% progress gate.
                update_rep_counts(
                    displayed_exercise, curr_landmarks, rep_counts, rep_states
                )
                get_form_feedback(
                    displayed_exercise, curr_landmarks, feedback_states, rep_counts
                )

                # Draw HUD for active exercise, reps, and feedback.
                feedback = get_feedback_message(displayed_exercise, feedback_states)
                rep_direction = get_rep_direction_label(displayed_exercise, rep_states)
                draw_active_hud(
                    frame,
                    displayed_exercise,
                    rep_counts,
                    feedback,
                    emoji_images,
                    rep_direction_label=rep_direction,
                    feedback_bottom_margin=FEEDBACK_BOTTOM_MARGIN,
                    emoji_margin=EMOJI_MARGIN,
                    emoji_size=EMOJI_SIZE,
                )

                # Draw optional debug visualizer.
                if show_feedback_visualizer and displayed_exercise is not None:
                    draw_feedback_visualizer(
                        frame,
                        displayed_exercise,
                        feedback_states[displayed_exercise],
                        rep_counts[displayed_exercise],
                        dual_stream_state,
                    )
            else:
                # Reset transition-sensitive state when no pose is detected.
                reset_form_feedback_tracking(feedback_states)
                reset_rep_counter_tracking(rep_states)
                draw_no_pose_hud(frame, displayed_exercise, rep_counts)
                dual_stream_state = initialize_dual_stream_state(STREAM_MAX_FRAMES)

            # Present frame and process keyboard controls.
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(TOGGLE_VISUALIZER_KEY):
                show_feedback_visualizer = not show_feedback_visualizer
            if key == ord(QUIT_KEY):
                break

            # Advance synthetic timestamp for MediaPipe VIDEO mode.
            frame_timestamp_ms += FRAME_STEP_MS

    # ---------------------------------------------------------------------
    # Cleanup.
    # ---------------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()


# Run main when this script is executed directly.
if __name__ == "__main__":
    main()
