import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
COLUMNS = [
    "timestamp",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "label",
    "person",
]

EXERCISE_NAMES = {
    0: "Unknown",
    1: "Squat",
    2: "Bicep Curl",
    3: "Shoulder Press",
    4: "Rows",
    5: "Tricep Extension",
}


def load_all_data(data_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for path in csv_files:
        df = pd.read_csv(path, header=None, names=COLUMNS)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = pd.to_numeric(data["timestamp"])
    data["label"] = pd.to_numeric(data["label"], errors="coerce").fillna(0).astype(int)
    return data


def plot_session(df: pd.DataFrame, title: str):
    """Plot accel and gyro for a single recording session."""
    t = df["timestamp"] - df["timestamp"].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(title, fontsize=13)

    axes[0].plot(t, df["accel_x"], label="X")
    axes[0].plot(t, df["accel_y"], label="Y")
    axes[0].plot(t, df["accel_z"], label="Z")
    axes[0].set_ylabel("Acceleration (g)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, df["gyro_x"], label="X")
    axes[1].plot(t, df["gyro_y"], label="Y")
    axes[1].plot(t, df["gyro_z"], label="Z")
    axes[1].set_ylabel("Rotation (deg/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()


def plot_all_exercises(data: pd.DataFrame):
    """One subplot per exercise, overlaying all persons, showing accel magnitude."""
    exercise_ids = sorted(data["label"].unique())
    n = len(exercise_ids)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle("Acceleration Magnitude by Exercise", fontsize=14)

    for ax, ex_id in zip(axes, exercise_ids):
        ex_name = EXERCISE_NAMES.get(ex_id, f"Label {ex_id}")
        subset = data[data["label"] == ex_id]

        for person, grp in subset.groupby("person"):
            grp = grp.sort_values("timestamp")
            t = grp["timestamp"] - grp["timestamp"].iloc[0]
            mag = (
                grp["accel_x"] ** 2 + grp["accel_y"] ** 2 + grp["accel_z"] ** 2
            ) ** 0.5
            ax.plot(t.values, mag.values, label=person, alpha=0.8)

        ax.set_title(ex_name)
        ax.set_ylabel("|accel| (g)")
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()


def plot_per_file(data_dir: str):
    """Plot each CSV file individually."""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    for path in csv_files:
        df = pd.read_csv(path, header=None, names=COLUMNS)
        df["timestamp"] = pd.to_numeric(df["timestamp"])
        title = os.path.relpath(path, data_dir)
        plot_session(df, title)


def plot_accel_magnitude(csv_path: str):
    """Plot acceleration magnitude (sqrt(ax^2 + ay^2 + az^2)) for a single CSV file."""
    df = pd.read_csv(csv_path, header=None, names=COLUMNS)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    t = df["timestamp"] - df["timestamp"].iloc[0]
    mag = (df["accel_x"] ** 2 + df["accel_y"] ** 2 + df["accel_z"] ** 2) ** 0.5

    _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, mag, color="tab:blue")
    ax.set_title(f"Acceleration Magnitude — {os.path.basename(csv_path)}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|accel| (g)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


if __name__ == "__main__":
    data = load_all_data(DATA_DIR)

    print(
        f"Loaded {len(data)} samples from {data['person'].nunique()} persons "
        f"across {data['label'].nunique()} exercise(s)."
    )
    print(
        "Exercises:",
        {k: EXERCISE_NAMES.get(k, k) for k in sorted(data["label"].unique())},
    )

    # Overview: accel magnitude per exercise
    plot_all_exercises(data)

    # Individual file plots
    plot_per_file(DATA_DIR)

    # Acceleration magnitude for each CSV file
    for csv_path in sorted(
        glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    ):
        plot_accel_magnitude(csv_path)

    plt.show()
