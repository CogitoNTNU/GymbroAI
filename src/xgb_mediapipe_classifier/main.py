import os
from collections import deque

import cv2
import mediapipe as mp
from classifier.inference import (
    WINDOW_NAME,
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
    detect_rep_extremity,
    get_switch_progress,
    reset_rep_counter_tracking,
    update_rep_counts,
)

FEEDBACK_BOTTOM_MARGIN = 35
EMOJI_MARGIN = 28
EMOJI_SIZE = 64
SWITCH_PROGRESS_THRESHOLD = 0.15
STREAM_MAX_FRAMES = 240


def _empty_switch_state():
    return {
        "active_stream": deque(maxlen=STREAM_MAX_FRAMES),
        "predicted_stream": deque(maxlen=STREAM_MAX_FRAMES),
        "predicted_exercise": None,
        "predicted_start_extremity": None,
        "predicted_progress": 0.0,
    }


def update_active_exercise_with_dual_stream(
    curr_landmarks,
    raw_exercise,
    active_exercise,
    switch_state,
):
    # Always keep a stream for the active exercise when available.
    if active_exercise is not None:
        switch_state["active_stream"].append(curr_landmarks)

    if raw_exercise is None:
        return active_exercise

    # First valid prediction initializes active label and active stream.
    if active_exercise is None:
        switch_state["active_stream"].clear()
        switch_state["active_stream"].append(curr_landmarks)
        return raw_exercise

    # If prediction matches active label, predicted stream must be inactive.
    if raw_exercise == active_exercise:
        switch_state["predicted_exercise"] = None
        switch_state["predicted_start_extremity"] = None
        switch_state["predicted_progress"] = 0.0
        switch_state["predicted_stream"].clear()
        return active_exercise

    # If predicted class changes, start a fresh predicted stream.
    if switch_state["predicted_exercise"] != raw_exercise:
        switch_state["predicted_exercise"] = raw_exercise
        switch_state["predicted_start_extremity"] = detect_rep_extremity(
            curr_landmarks,
            raw_exercise,
        )
        switch_state["predicted_progress"] = 0.0
        switch_state["predicted_stream"].clear()

    switch_state["predicted_stream"].append(curr_landmarks)

    if switch_state["predicted_start_extremity"] is None:
        switch_state["predicted_start_extremity"] = detect_rep_extremity(
            curr_landmarks,
            raw_exercise,
        )

    if switch_state["predicted_start_extremity"] is not None:
        switch_state["predicted_progress"] = get_switch_progress(
            curr_landmarks,
            raw_exercise,
            switch_state["predicted_start_extremity"],
        )

    if switch_state["predicted_progress"] >= SWITCH_PROGRESS_THRESHOLD:
        # Promote predicted stream to active stream and switch label.
        active_exercise = raw_exercise
        switch_state["active_stream"] = deque(
            switch_state["predicted_stream"],
            maxlen=STREAM_MAX_FRAMES,
        )
        switch_state["predicted_exercise"] = None
        switch_state["predicted_start_extremity"] = None
        switch_state["predicted_progress"] = 0.0
        switch_state["predicted_stream"].clear()

    return active_exercise


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


def load_feedback_emojis(project_root):
    image_dir = os.path.join(project_root, "classifier", "images")
    happy = cv2.imread(os.path.join(image_dir, "happyemoji.png"), cv2.IMREAD_UNCHANGED)
    angry = cv2.imread(os.path.join(image_dir, "angryemoji.png"), cv2.IMREAD_UNCHANGED)
    return {
        "happy": happy,
        "angry": angry,
    }


def _overlay_bottom_right(frame, icon):
    if icon is None:
        return

    icon_h, icon_w = icon.shape[:2]
    if icon_h == 0 or icon_w == 0:
        return

    scale = EMOJI_SIZE / float(max(icon_h, icon_w))
    resized = cv2.resize(
        icon, (max(1, int(icon_w * scale)), max(1, int(icon_h * scale)))
    )

    h, w = frame.shape[:2]
    rh, rw = resized.shape[:2]
    x1 = max(0, w - EMOJI_MARGIN - rw)
    y1 = max(0, h - EMOJI_MARGIN - rh)
    x2 = min(w, x1 + rw)
    y2 = min(h, y1 + rh)

    roi = frame[y1:y2, x1:x2]
    icon_crop = resized[: y2 - y1, : x2 - x1]

    if icon_crop.shape[2] == 4:
        alpha = icon_crop[:, :, 3:4].astype("float32") / 255.0
        rgb = icon_crop[:, :, :3].astype("float32")
        roi_float = roi.astype("float32")
        blended = alpha * rgb + (1.0 - alpha) * roi_float
        frame[y1:y2, x1:x2] = blended.astype("uint8")
    else:
        frame[y1:y2, x1:x2] = icon_crop[:, :, :3]


def draw_feedback_emoji(frame, is_good_form, emoji_images):
    icon_key = "happy" if is_good_form else "angry"
    _overlay_bottom_right(frame, emoji_images.get(icon_key))


def draw_active_hud(frame, exercise_name, rep_counts, feedback, emoji_images):
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
        is_good_form = feedback.lower().startswith("good")
        feedback_color = (0, 255, 0) if is_good_form else (0, 165, 255)
        draw_hud_text(
            frame,
            f"Form: {feedback}",
            get_feedback_position(frame),
            feedback_color,
            1.0,
            3,
        )
        draw_feedback_emoji(frame, is_good_form, emoji_images)


def process_pose_frame(
    pose_landmarks,
    model,
    encoder,
    feature_config,
    rep_counts,
    feedback_states,
    rep_states,
):
    curr_landmarks = landmarks_to_dict(pose_landmarks)
    exercise_name = predict_exercise(curr_landmarks, model, encoder, feature_config)
    update_rep_counts(exercise_name, curr_landmarks, rep_counts, rep_states)
    # rep_counts is passed so feedback is only generated when a real rep was counted.
    get_form_feedback(exercise_name, curr_landmarks, feedback_states, rep_counts)
    return exercise_name, curr_landmarks


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
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    frame_timestamp_ms = 0
    displayed_exercise = None
    show_feedback_visualizer = True
    emoji_images = load_feedback_emojis(os.path.dirname(__file__))
    rep_counts, rep_states = create_rep_counter_state(encoder)
    feedback_states = create_feedback_state()
    switch_state = _empty_switch_state()

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
                raw_exercise, curr_landmarks = process_pose_frame(
                    pose_landmarks=result.pose_landmarks[0],
                    model=model,
                    encoder=encoder,
                    feature_config=feature_config,
                    rep_counts=rep_counts,
                    feedback_states=feedback_states,
                    rep_states=rep_states,
                )
                displayed_exercise = update_active_exercise_with_dual_stream(
                    curr_landmarks,
                    raw_exercise,
                    displayed_exercise,
                    switch_state,
                )
                feedback = get_feedback_message(displayed_exercise, feedback_states)
                draw_active_hud(
                    frame,
                    displayed_exercise,
                    rep_counts,
                    feedback,
                    emoji_images,
                )
                if show_feedback_visualizer and displayed_exercise is not None:
                    draw_feedback_visualizer(
                        frame,
                        displayed_exercise,
                        feedback_states[displayed_exercise],
                        rep_counts[displayed_exercise],
                        switch_state,
                    )
            else:
                handle_no_pose(
                    frame, displayed_exercise, rep_counts, feedback_states, rep_states
                )
                switch_state = _empty_switch_state()

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
