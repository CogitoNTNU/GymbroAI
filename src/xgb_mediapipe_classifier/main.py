import os
import cv2
import mediapipe as mp
from classifier.inference import (
    draw_hud_text,
    draw_landmarks_on_image,
    landmarks_to_dict,
    load_classifier,
    predict_exercise,
)
from form_feedback.runtime import (
    create_feedback_state,
    get_feedback_message,
    get_form_feedback,
    reset_form_feedback_tracking,
)
from form_feedback.visualizer import draw_feedback_visualizer
from rep_counter import (
    create_rep_counter_state,
    reset_rep_counter_tracking,
    update_rep_counts,
)

LABEL_SWITCH_STREAK = 20
FEEDBACK_BOTTOM_MARGIN = 35
WINDOW_NAME = "Gymbro AI Classifier"


def smooth_displayed_exercise(
    raw_exercise, displayed_exercise, pending_exercise, pending_count
):
    """Stabilize label changes to avoid rapid flicker between classes."""
    if displayed_exercise is None:
        return raw_exercise, None, 0

    if raw_exercise == displayed_exercise:
        return displayed_exercise, None, 0

    if raw_exercise == pending_exercise:
        pending_count += 1
        if pending_count >= LABEL_SWITCH_STREAK:
            return raw_exercise, None, 0
        return displayed_exercise, pending_exercise, pending_count

    return displayed_exercise, raw_exercise, 1


def draw_rep_summary(frame, exercise_name, rep_counts, y):
    if exercise_name is None:
        return
    draw_hud_text(
        frame,
        f"Reps: {rep_counts[exercise_name]}",
        (20, y),
        (255, 200, 0),
        1.2,
        3,
    )


def get_feedback_position(frame):
    """Anchor feedback close to the bottom edge regardless of camera resolution."""
    return 20, frame.shape[0] - FEEDBACK_BOTTOM_MARGIN


def draw_active_hud(frame, exercise_name, rep_counts, feedback):
    if exercise_name is None:
        return
    draw_hud_text(
        frame,
        f"Exercise: {exercise_name.upper()}",
        (20, 60),
        (0, 255, 0),
        1.4,
        4,
    )
    draw_rep_summary(frame, exercise_name, rep_counts, 115)
    if feedback is not None:
        feedback_color = (
            (0, 255, 0) if feedback.lower().startswith("good") else (0, 165, 255)
        )
        draw_hud_text(
            frame,
            f"Form: {feedback}",
            get_feedback_position(frame),
            feedback_color,
            1.0,
            3,
        )


def process_stable_pose(
    displayed_exercise,
    curr_landmarks,
    rep_counts,
    feedback_states,
    rep_states,
):
    """Update tracking completely based on the smoothed and stabilized exercise class."""
    if displayed_exercise is not None:
        update_rep_counts(displayed_exercise, curr_landmarks, rep_counts, rep_states)
        get_form_feedback(
            displayed_exercise, curr_landmarks, feedback_states, rep_counts
        )


def handle_no_pose(frame, exercise_name, rep_counts, feedback_states, rep_states):
    reset_form_feedback_tracking(feedback_states)
    reset_rep_counter_tracking(rep_states)
    draw_hud_text(frame, "No pose detected", (20, 60), (0, 0, 255), 1.4, 4)
    draw_rep_summary(frame, exercise_name, rep_counts, 115)


def main():
    # 1) Load classifier model, label encoder, and feature config.
    model, encoder, feature_config = load_classifier(
        os.path.join(os.path.dirname(__file__), "classifier", "models")
    )

    # 2) Configure pose landmarker.
    model_path = os.path.join(
        os.path.dirname(__file__),
        "classifier",
        "mediapipe",
        "pose_landmarker_full.task",
    )
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )

    # 3) Open camera and initialize runtime state.
    # Try finding a working camera, prioritizing index 1 and 2 to bypass OBS at 0
    cap = None
    for cam_idx in [1, 2, 0, 3, 4]:
        temp_cap = cv2.VideoCapture(cam_idx)
        if temp_cap.isOpened():
            success, frame = temp_cap.read()
            # Often OBS virtual camera has a specific default resolution,
            # but even just trying index 1 first usually finds the built-in webcam.
            if success:
                cap = temp_cap
                break
            else:
                temp_cap.release()
        else:
            temp_cap.release()

    if cap is None:
        raise RuntimeError("Could not find a valid camera feed.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    frame_timestamp_ms = 0
    displayed_exercise = None
    pending_exercise = None
    pending_count = 0
    show_feedback_visualizer = True
    rep_counts, rep_states = create_rep_counter_state(encoder)
    feedback_states = create_feedback_state()

    # 4) Main processing loop.
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            # 4a) Capture frame.
            success, frame = cap.read()
            if not success:
                break

            # 4b) Run pose detection.
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # 4c) Update classification/rep/feedback state and draw HUD.
            if result.pose_landmarks:
                frame = draw_landmarks_on_image(frame, result)
                curr_landmarks = landmarks_to_dict(result.pose_landmarks[0])
                raw_exercise = predict_exercise(
                    curr_landmarks, model, encoder, feature_config
                )

                displayed_exercise, pending_exercise, pending_count = (
                    smooth_displayed_exercise(
                        raw_exercise,
                        displayed_exercise,
                        pending_exercise,
                        pending_count,
                    )
                )

                process_stable_pose(
                    displayed_exercise,
                    curr_landmarks,
                    rep_counts,
                    feedback_states,
                    rep_states,
                )

                feedback = get_feedback_message(displayed_exercise, feedback_states)
                draw_active_hud(frame, displayed_exercise, rep_counts, feedback)

                # Visualizer now follows the smoothed active exercise
                if show_feedback_visualizer and displayed_exercise is not None:
                    draw_feedback_visualizer(
                        frame,
                        displayed_exercise,
                        feedback_states[displayed_exercise],
                        rep_counts[displayed_exercise],
                    )
            else:
                handle_no_pose(
                    frame, displayed_exercise, rep_counts, feedback_states, rep_states
                )

            # 4d) Show frame and handle exit.
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("v"):
                show_feedback_visualizer = not show_feedback_visualizer
            if key == ord("q"):
                break

            frame_timestamp_ms += 33

    # 5) Cleanup camera resources.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
