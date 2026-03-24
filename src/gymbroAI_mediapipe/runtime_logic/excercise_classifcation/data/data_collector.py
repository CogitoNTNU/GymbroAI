import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# ============================================================================
# EXERCISE CONFIGURATION - CHANGE THIS TO RECORD A NEW EXERCISE
# ============================================================================
# Set this to the name of the exercise you want to record
# Examples: "pushups", "ssquats", "bicep_curl", "jumping_jacks", "plank"
# Use lowercase and underscores for multi-word exercises
EXERCISE_LABEL = "shoulder_press"
# ============================================================================

# Hvor mange millisekunder mellom hvert datasample som blir lagret
SAMPLE_RATE = 100

# Maks antall personer å detektere
NUM_POSES = 1

# Sett til True for å inkludere visibility-score for hvert landmark
INCLUDE_VISIBILITY = True

# Filsti for output-CSV. Mappen lages automatisk.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__))

# Path til MediaPipe-modellen
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")

# Timing constants
COUNTDOWN_SECONDS = 10
RECORDING_SECONDS = 30

# ============================================================================
# BODY LANDMARKS ONLY - Excludes face/eye/ear/mouth/hand details
# ============================================================================
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

# Ordered list of landmark names for CSV output
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
# These connections are based on the full MediaPipe skeleton
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


def get_next_output_file():
    """Returnerer neste ledige filnavn, f.eks. squat1.csv."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    i = 1
    while True:
        path = os.path.join(OUTPUT_DIR, f"{EXERCISE_LABEL}{i}.csv")
        if not os.path.exists(path):
            return path
        i += 1


def build_csv_header():
    """Bygger header-raden for CSV basert på innstillingene."""
    header = []
    for name in LANDMARK_NAMES:
        header += [f"{name}_x", f"{name}_y", f"{name}_z"]
        if INCLUDE_VISIBILITY:
            header.append(f"{name}_vis")
    return header


def landmarks_to_row(label, timestamp_ms, pose_landmarks):
    """
    Konverterer kun body landmarks til en CSV-rad med RAW MediaPipe data.
    Lagrer x, y, z koordinater direkte fra MediaPipe uten normalisering.
    """
    row = []
    for name in LANDMARK_NAMES:
        idx = BODY_LANDMARK_INDICES[name]
        lm = pose_landmarks[idx]
        row += [round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)]
        if INCLUDE_VISIBILITY:
            row.append(round(lm.visibility, 6))
    return row


def draw_landmarks_on_image(rgb_image, detection_result):
    """Tegner skjelett og landmarks på bildet (kun body landmarks)."""
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for pose_landmarks in pose_landmarks_list:
        # Draw connections
        for connection in POSE_CONNECTIONS:
            start_point = pose_landmarks[connection[0]]
            end_point = pose_landmarks[connection[1]]
            h, w = annotated_image.shape[:2]
            cv2.line(
                annotated_image,
                (int(start_point.x * w), int(start_point.y * h)),
                (int(end_point.x * w), int(end_point.y * h)),
                (0, 255, 0),
                2,
            )

        # Draw landmarks (only body landmarks)
        h, w = annotated_image.shape[:2]
        for name in LANDMARK_NAMES:
            idx = BODY_LANDMARK_INDICES[name]
            lm = pose_landmarks[idx]
            cv2.circle(
                annotated_image, (int(lm.x * w), int(lm.y * h)), 5, (0, 0, 255), -1
            )

    return annotated_image


def draw_ui(frame, state, time_remaining, sample_count):
    """Tegner HUD med status-info på skjermen."""
    h, w = frame.shape[:2]

    if state == "countdown":
        # Large countdown number in center
        countdown_text = str(int(time_remaining) + 1)
        font_scale = 5
        thickness = 10
        text_size = cv2.getTextSize(
            countdown_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        cv2.putText(
            frame,
            countdown_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            thickness,
        )
        cv2.putText(
            frame,
            "Get ready...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2,
        )
    elif state == "recording":
        rec_text = "● REC"
        cv2.putText(
            frame, rec_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
        )
        cv2.putText(
            frame,
            f"Time left: {int(time_remaining)}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Samples: {sample_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    cv2.putText(
        frame,
        f"Label: {EXERCISE_LABEL}",
        (10, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        "Q: avslutt tidlig",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
    )
    return frame


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = get_next_output_file()

    # Opprett ny CSV-fil med header
    csv_file = open(output_file, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(build_csv_header())
    csv_file.flush()

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=NUM_POSES,
    )

    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = 0
    last_sample_ms = -SAMPLE_RATE
    sample_count = 0

    # Timing state
    state = "countdown"  # States: "countdown", "recording", "done"
    start_time = time.time()
    countdown_start = start_time
    recording_start = None

    print("=" * 60)
    print(f"TIMED RECORDING: {EXERCISE_LABEL.upper()}")
    print("=" * 60)
    print(f"Output file: {output_file}")
    print(f"\nCountdown: {COUNTDOWN_SECONDS} seconds")
    print(f"Recording duration: {RECORDING_SECONDS} seconds")
    print("\nRecording BODY LANDMARKS ONLY (16 landmarks)")
    print("Excluding: nose, face, eyes, ears, mouth, hand details")
    print("Saving RAW MediaPipe data (x, y, z coordinates)")
    print("\nGet ready to perform the exercise when countdown reaches 0!")
    print("Press Q to quit early")
    print("=" * 60)

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            isCameraOpen, frame = cap.read()
            if not isCameraOpen:
                break

            current_time = time.time()

            # State machine for timing
            if state == "countdown":
                elapsed = current_time - countdown_start
                time_remaining = COUNTDOWN_SECONDS - elapsed
                if elapsed >= COUNTDOWN_SECONDS:
                    state = "recording"
                    recording_start = current_time
                    print("[INFO] Opptak STARTET!")

            elif state == "recording":
                elapsed = current_time - recording_start
                time_remaining = RECORDING_SECONDS - elapsed
                if elapsed >= RECORDING_SECONDS:
                    state = "done"
                    print("[INFO] Opptak FULLFØRT!")
                    break
            else:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # Datainnsamling - only during recording state
            if state == "recording" and result.pose_landmarks:
                elapsed_since_last = frame_timestamp_ms - last_sample_ms
                if elapsed_since_last >= SAMPLE_RATE:
                    for pose_landmarks in result.pose_landmarks:
                        row = landmarks_to_row(
                            EXERCISE_LABEL, frame_timestamp_ms, pose_landmarks
                        )
                        writer.writerow(row)
                    csv_file.flush()
                    last_sample_ms = frame_timestamp_ms
                    sample_count += 1

            # Tegning
            annotated = draw_landmarks_on_image(rgb_frame, result)
            display = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            display = draw_ui(display, state, time_remaining, sample_count)

            cv2.imshow("MediaPipe Data Collector - Body Only", display)

            frame_timestamp_ms += 33

            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                print("[INFO] Avbrutt av bruker")
                break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

    if sample_count == 0:
        os.remove(output_file)
        print("[WARNING] Ingen data ble samlet inn. Filen ble slettet.")
    else:
        print(f"\n[SUCCESS] Samlet inn {sample_count} samples til {output_file}")
        print(f"[INFO] Data inneholder kun {len(LANDMARK_NAMES)} body landmarks")


if __name__ == "__main__":
    main()
