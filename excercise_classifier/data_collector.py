import cv2
import mediapipe as mp
import numpy as np
import csv
import os


# Hvor mange millisekunder mellom hvert datasample som blir lagret
SAMPLE_RATE = 100

# Maks antall personer å detektere
NUM_POSES = 1

# Sett til True for å inkludere visibility-score for hvert landmark
INCLUDE_VISIBILITY = True

# Navnet filen blir lagret som
EXERCISE_LABEL = "squat"  

# Filsti for output-CSV. Mappen lages automatisk.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# Path til MediaPipe-modellen
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")

# Koblinger mellom landmarks, "strekene"
POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31),
    (29, 31), (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
])

LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def get_next_output_file():
    """Returnerer neste ledige filnavn, f.eks. squat.csv."""
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
    """Konverterer ett sett med pose-landmarks til en CSV-rad."""
    row = []
    for lm in pose_landmarks:
        row += [round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)]
        if INCLUDE_VISIBILITY:
            row.append(round(lm.visibility, 6))
    return row


def draw_landmarks_on_image(rgb_image, detection_result):
    """Tegner skjelett og landmarks på bildet."""
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for pose_landmarks in pose_landmarks_list:
        for connection in POSE_CONNECTIONS:
            start_point = pose_landmarks[connection[0]]
            end_point   = pose_landmarks[connection[1]]
            h, w = annotated_image.shape[:2]
            cv2.line(
                annotated_image,
                (int(start_point.x * w), int(start_point.y * h)),
                (int(end_point.x   * w), int(end_point.y   * h)),
                (0, 255, 0), 2
            )
        for lm in pose_landmarks:
            h, w = annotated_image.shape[:2]
            cv2.circle(annotated_image, (int(lm.x * w), int(lm.y * h)), 5, (0, 0, 255), -1)

    return annotated_image


def draw_ui(frame, recording, sample_count):
    """Tegner HUD med status-info på skjermen."""
    h, w = frame.shape[:2]
    rec_color = (0, 0, 255) if recording else (150, 150, 150)
    rec_text  = "● REC" if recording else "○ PAUSED"
    cv2.putText(frame, rec_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rec_color, 2)
    cv2.putText(frame, f"Label: {EXERCISE_LABEL}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Samples: {sample_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "SPACE: start/stopp  |  Q: avslutt", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
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

    cap = cv2.VideoCapture(0)           # åpner kameraet
    frame_timestamp_ms = 0              # MediaPipe krever timestamp for å prossesere videoer i rekkefølge
    last_sample_ms     = -SAMPLE_RATE   # sørger for sample ved første frame
    sample_count       = 0
    recording          = False

    print(f"[INFO] Skriver til: {output_file}")
    print("[INFO] Trykk MELLOMROM for å starte/stoppe innsamling. Trykk Q for å avslutte.")

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            isCameraOpen, frame = cap.read()
            if not isCameraOpen:
                break

            frame = cv2.flip(frame, 1) # flipper kameraet
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # Datainnsamling 
            if recording and result.pose_landmarks:
                elapsed_since_last = frame_timestamp_ms - last_sample_ms
                if elapsed_since_last >= SAMPLE_RATE:
                    for pose_landmarks in result.pose_landmarks:
                        row = landmarks_to_row(EXERCISE_LABEL, frame_timestamp_ms, pose_landmarks)
                        writer.writerow(row)
                    csv_file.flush()
                    last_sample_ms = frame_timestamp_ms
                    sample_count  += 1


            # Tegning 
            annotated = draw_landmarks_on_image(rgb_frame, result)
            display   = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            display   = draw_ui(display, recording, sample_count)

            cv2.imshow("MediaPipe Data Collector", display)

            frame_timestamp_ms += 33

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                recording = not recording
                state = "STARTET" if recording else "STOPPET"
                print(f"[INFO] Innsamling {state}. Samples så langt: {sample_count}")

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

    if sample_count == 0:
        os.remove(output_file)
        print("[FERDIG] Ingen data ble logget – ingen fil ble lagret.")
    else:
        print(f"[FERDIG] {sample_count} samples lagret i {output_file}")


if __name__ == "__main__":
    main()
