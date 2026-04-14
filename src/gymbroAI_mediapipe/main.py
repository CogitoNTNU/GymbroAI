"""
Real-time exercise classifier with rep counting and form feedback.

Pipeline:
    1. Capture frames from the webcam and run MediaPipe pose detection.
    2. Convert detected landmarks into a feature vector and classify
       the exercise with an XGBoost model.
    3. Smooth exercise switching via dual-stream progress gating
       (data_stream_manager).
    4. Count reps using metric thresholds (rep_counter).
    5. Analyse form after each completed rep (feedback_analyser).
    6. Draw HUD overlays and an optional debug visualizer (draw_on_screen).
"""

import os

import cv2
import mediapipe as mp

from runtime_logic.excercise_classifcation.classifier_runtime import (
    load_classifier,
    predict_exercise,
)

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
    get_rep_direction_label,
    reset_rep_counter_tracking,
    update_rep_counts,
)

from graphics.draw_on_screen import (
    draw_active_hud,
    draw_feedback_visualizer,
    draw_landmarks_on_image,
    draw_no_pose_hud,
    load_feedback_emojis,
)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

WINDOW_NAME = "Exercise Classifier"

# Set to True to show the signal graph on startup (toggle with V key).
DEBUG_MODE = True

# Approximate frame interval for MediaPipe VIDEO mode timestamp (~30 fps).
FRAME_STEP_MS = 33

QUIT_KEY = "q"
TOGGLE_VISUALIZER_KEY = "v"
SHOW_FULLSCREEN_WINDOW = False

# Maximum frames stored in the active/predicted data streams.
STREAM_MAX_FRAMES = 240

# HUD layout.
FEEDBACK_BOTTOM_MARGIN = 35
EMOJI_MARGIN = 28
EMOJI_SIZE = 100


# ---------------------------------------------------------------------------
# MediaPipe landmark configuration
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Landmark helpers
# ---------------------------------------------------------------------------


def landmarks_to_dict(pose_landmarks):
    """Convert MediaPipe pose landmarks to a flat {name_x, name_y, name_z} dict."""
    lm = {}
    for name in LANDMARK_NAMES:
        idx = BODY_LANDMARK_INDICES[name]
        landmark = pose_landmarks[idx]
        lm[f"{name}_x"] = landmark.x
        lm[f"{name}_y"] = landmark.y
        lm[f"{name}_z"] = landmark.z
    return lm


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    # -- Load classifier model, label encoder, and feature config. --
    model, encoder, feature_config = load_classifier(
        os.path.join(
            os.path.dirname(__file__),
            "runtime_logic",
            "excercise_classifcation",
            "models",
        )
    )

    # -- Configure MediaPipe pose landmarker. --
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

    # -- Camera and window. --
    cap = cv2.VideoCapture(1)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if SHOW_FULLSCREEN_WINDOW:
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )

    # -- Runtime state. --
    frame_timestamp_ms = 0
    displayed_exercise = None
    show_feedback_visualizer = DEBUG_MODE

    emoji_images = load_feedback_emojis(os.path.dirname(__file__))
    rep_counts, rep_states = create_rep_counter_state(encoder)
    feedback_states = create_feedback_state()
    dual_stream_state = initialize_dual_stream_state(STREAM_MAX_FRAMES)

    # -- Frame loop. --
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
                frame = draw_landmarks_on_image(
                    frame, result, BODY_LANDMARK_INDICES, LANDMARK_NAMES, displayed_exercise
                )

                curr_landmarks = landmarks_to_dict(result.pose_landmarks[0])
                raw_exercise = predict_exercise(
                    curr_landmarks,
                    model,
                    encoder,
                    feature_config,
                )

                # Smooth exercise switching via dual-stream progress gating.
                displayed_exercise = update_active_exercise_with_dual_stream(
                    curr_landmarks,
                    raw_exercise,
                    displayed_exercise,
                    dual_stream_state,
                    rep_states=rep_states,
                )

                # Update rep counting and form feedback using the smoothed exercise.
                update_rep_counts(
                    displayed_exercise, curr_landmarks, rep_counts, rep_states
                )
                get_form_feedback(
                    displayed_exercise, curr_landmarks, feedback_states, rep_counts
                )

                # Draw HUD.
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
                # No pose detected — reset transition-sensitive state.
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

            frame_timestamp_ms += FRAME_STEP_MS

    # -- Cleanup. --
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
