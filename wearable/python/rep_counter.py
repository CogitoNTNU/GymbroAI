"""
rep_counter.py — counts exercise repetitions from raw IMU CSV data.

Supported files (placed in data/):
  - bicep_curl_dennis_1.csv
  - rows_dennis_1.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# ---------------------------------------------------------------------------
# Configuration — tweak these without touching the algorithm
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

FILES = [
    os.path.join("shoulder_press", "shoulder_press_gustav_1.csv"),
]

# Expected column names (order matters when the CSV has no header)
COLUMN_NAMES = ["timestamp", "accel_x", "accel_y", "accel_z",
                "gyro_x", "gyro_y", "gyro_z", "label", "person"]

# Candidate acceleration column name fragments (case-insensitive substring match)
ACCEL_CANDIDATES = ["accel_x", "accel_y", "accel_z", "ax", "ay", "az"]


#sample
SAMPLE_RATE = 50 #ms

# Low-pass filter
FILTER_CUTOFF_HZ = 3.0      # cut-off frequency in Hz
FILTER_ORDER     = 4        # Butterworth filter order

# Peak detection
MIN_PEAK_DISTANCE_SAMPLES = 20  # minimum samples between counted peaks

# Adaptive threshold
ADAPTIVE_THRESHOLD_FACTOR = 0.8  # T = factor * mean(last two valid peaks)
SEED_THRESHOLD            = 0.3  # used until two valid peaks have been found


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(filepath):
    """Load CSV; assign COLUMN_NAMES if the file has no header or mismatched columns."""
    df = pd.read_csv(filepath)
    # If columns look like integers the file had no header — reassign
    if df.columns[0].isdigit() or list(df.columns) == list(range(len(df.columns))):
        df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES[:len(df.columns)])
    return df


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def find_accel_columns(df):
    """Return the three acceleration column names by matching ACCEL_CANDIDATES."""
    matched = []
    for col in df.columns:
        col_lower = col.lower()
        if any(cand in col_lower for cand in ["accel_x", "ax"]):
            matched.append((col, "x"))
        elif any(cand in col_lower for cand in ["accel_y", "ay"]):
            matched.append((col, "y"))
        elif any(cand in col_lower for cand in ["accel_z", "az"]):
            matched.append((col, "z"))
    if len(matched) < 3:
        raise ValueError(f"Could not find 3 acceleration columns. Found: {matched}")
    # Return in x, y, z order
    matched.sort(key=lambda t: t[1])
    return [col for col, _ in matched]


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def select_best_axis(df, accel_cols):
    """Return (column_name, signal_array) for the axis with the highest variance."""
    variances = {col: df[col].var() for col in accel_cols}
    best_col = max(variances, key=variances.get)
    print(f"  Variances: { {c: f'{v:.4f}' for c, v in variances.items()} }")
    print(f"  Selected axis: {best_col}")
    return best_col, df[best_col].to_numpy(dtype=float)


def lowpass_filter(signal, cutoff_hz, sample_rate, order=FILTER_ORDER):
    """Apply a zero-phase Butterworth low-pass filter."""
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# Repetition counting with adaptive threshold
# ---------------------------------------------------------------------------

def count_reps(filtered_signal):
    """
    Detect peaks and count reps using an adaptive threshold.

    Returns
    -------
    rep_indices : list[int]   — sample indices of counted reps
    rep_count   : int
    """
    # Find all candidate peaks (no height filter yet, only distance)
    candidate_idx, _ = find_peaks(filtered_signal,
                                  distance=MIN_PEAK_DISTANCE_SAMPLES)

    rep_indices = []
    last_two_amplitudes = []  # rolling buffer of the last two counted peak values
    threshold = SEED_THRESHOLD  # bootstrap threshold

    for idx in candidate_idx:
        amplitude = filtered_signal[idx]

        if amplitude > threshold:
            rep_indices.append(idx)

            # Update rolling buffer
            last_two_amplitudes.append(amplitude)
            if len(last_two_amplitudes) > 2:
                last_two_amplitudes.pop(0)

            # Recompute threshold once we have two valid peaks
            if len(last_two_amplitudes) == 2:
                threshold = ADAPTIVE_THRESHOLD_FACTOR * np.mean(last_two_amplitudes)

    return rep_indices, len(rep_indices)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(df, accel_cols, best_col, raw_signal, filtered_signal,
                 rep_indices, rep_count, title):
    """Two-panel figure: all raw axes | selected axis with counted reps."""
    samples = np.arange(len(raw_signal))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(title, fontsize=13)

    # --- Plot 1: all raw acceleration axes ---
    for col in accel_cols:
        ax1.plot(samples, df[col].to_numpy(dtype=float), label=col)
    ax1.set_ylabel("Acceleration")
    ax1.set_title("Raw acceleration — all axes")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: selected axis + counted reps ---
    ax2.plot(samples, raw_signal, color="steelblue", label=f"{best_col} (raw)")
    ax2.plot(samples, filtered_signal, color="orange", linewidth=1.5,
             label="filtered")
    if rep_indices:
        ax2.scatter(rep_indices, filtered_signal[rep_indices],
                    color="red", zorder=5, label="counted rep")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Acceleration")
    ax2.set_title(f"Selected axis: {best_col}  |  Reps counted: {rep_count}")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------

def process_file(filepath):
    filename = os.path.basename(filepath)
    full_path = os.path.join(DATA_DIR, filepath)

    if not os.path.exists(full_path):
        print(f"Skipping '{filename}' — file not found in {DATA_DIR}")
        return

    print(f"\nProcessing: {filename}")

    df = load_csv(full_path)
    accel_cols = find_accel_columns(df)
    print(f"  Acceleration columns: {accel_cols}")

    best_col, raw_signal = select_best_axis(df, accel_cols)

    filtered_signal = lowpass_filter(raw_signal, FILTER_CUTOFF_HZ, SAMPLE_RATE)

    rep_indices, rep_count = count_reps(filtered_signal)
    print(f"  Repetitions counted: {rep_count}")

    title = f"{filename}  —  {rep_count} reps"
    plot_results(df, accel_cols, best_col, raw_signal, filtered_signal,
                 rep_indices, rep_count, title)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    for fpath in FILES:
        process_file(fpath)


if __name__ == "__main__":
    main()
