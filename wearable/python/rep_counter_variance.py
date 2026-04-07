"""
rep_counter_variance.py — counts exercise reps using a dual-threshold approach.

Threshold 1 (rolling stats):
    mean + 1 standard deviation of the last 100 signal samples at the time of
    each candidate peak.  If fewer than 100 samples have been seen, all samples
    so far are used.

Threshold 2 (rep history):
    mean amplitude of the two most recent valid reps.
    A seed value (SEED_THRESHOLD_FACTOR × max candidate peak) is used until
    two valid reps have been accumulated.

A peak is counted as a rep only when it exceeds BOTH thresholds.

The accel axis with the highest variance is selected automatically.
An interactive menu lets you pick any CSV file found under data/.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

COLUMN_NAMES = ["time", "ax", "ay", "az", "gx", "gy", "gz", "label", "person"]

ACCEL_COLS = ["ax", "ay", "az"]

SAMPLE_RATE = 50          # Hz
FILTER_CUTOFF_HZ = 20.0
FILTER_ORDER = 2

# Minimum distance between candidate peaks (samples)
MIN_PEAK_DISTANCE_SAMPLES = 20

# Rolling-stats window (Threshold 1)
ROLLING_WINDOW = 100      # samples

# Rep-history threshold factor (Threshold 2)
#   T2 = REP_THRESHOLD_FACTOR × mean(last two valid peak amplitudes)
REP_THRESHOLD_FACTOR = 0.8

# Seed for T2 before two valid reps have been found
#   seed = SEED_THRESHOLD_FACTOR × max(candidate peak amplitudes)
SEED_THRESHOLD_FACTOR = 0.3


# ---------------------------------------------------------------------------
# File discovery & interactive selection
# ---------------------------------------------------------------------------

def discover_files():
    """Recursively find all CSV files under DATA_DIR, sorted by path."""
    paths = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    return sorted(set(paths))


def select_file():
    """Print a numbered menu and return the path chosen by the user."""
    files = discover_files()
    if not files:
        raise FileNotFoundError(f"No CSV files found under {DATA_DIR}")

    print("\nAvailable data files:")
    print("-" * 55)
    for i, path in enumerate(files):
        rel = os.path.relpath(path, DATA_DIR)
        size = os.path.getsize(path)
        size_str = f"{size // 1024} KB" if size >= 1024 else f"{size} B"
        print(f"  [{i + 1:2d}]  {rel:<40}  {size_str}")
    print("-" * 55)

    while True:
        try:
            choice = int(input(f"Select file [1–{len(files)}]: "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
            print(f"  Please enter a number between 1 and {len(files)}.")
        except ValueError:
            print("  Invalid input — please enter a number.")


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(filepath):
    """Load CSV; fall back to COLUMN_NAMES if the file has no header row."""
    df = pd.read_csv(filepath)
    if str(df.columns[0]).lstrip("-").replace(".", "", 1).isdigit():
        df = pd.read_csv(filepath, header=None,
                         names=COLUMN_NAMES[: len(df.columns)])
    missing = [c for c in ACCEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns {missing}. "
            f"File has: {list(df.columns)}"
        )
    return df


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def best_accel_axis(df):
    """Return (column_name, raw_numpy_array) for the highest-variance accel axis."""
    variances = {c: df[c].var() for c in ACCEL_COLS}
    var_str = ", ".join(f"{c}: {v:.3f}" for c, v in variances.items())
    print(f"  Accel variances — {var_str}")
    best = max(variances, key=variances.get)
    print(f"  Selected axis   — {best}")
    return best, df[best].to_numpy(dtype=float)


def lowpass(signal, cutoff_hz=FILTER_CUTOFF_HZ, fs=SAMPLE_RATE, order=FILTER_ORDER):
    """Zero-phase Butterworth low-pass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype="low", analog=False)
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# Dual-threshold repetition counting
# ---------------------------------------------------------------------------

def count_reps_dual_threshold(filtered_signal):
    """
    Detect peaks and count reps using two simultaneous thresholds.

    Threshold 1 — rolling stats:
        T1(i) = mean + std of filtered_signal[max(0, i-ROLLING_WINDOW+1) : i+1]
        Adapts to the recent signal level and noise floor.

    Threshold 2 — rep history:
        T2 = REP_THRESHOLD_FACTOR × mean(amplitudes of last two valid reps).
        Seed value used until two valid reps have been found.

    A peak counts as a rep only when amplitude > T1  AND  amplitude > T2.

    Returns
    -------
    rep_indices   : list[int]   sample indices of counted reps
    t1_at_peaks   : list[float] T1 value at each candidate peak (for plotting)
    t2_at_peaks   : list[float] T2 value at each candidate peak (for plotting)
    candidate_idx : ndarray     indices of all distance-filtered candidate peaks
    """
    candidate_idx, _ = find_peaks(filtered_signal,
                                  distance=MIN_PEAK_DISTANCE_SAMPLES)
    if len(candidate_idx) == 0:
        return [], [], [], candidate_idx

    # Seed T2 relative to the tallest candidate peak so it scales with magnitude
    seed_t2 = SEED_THRESHOLD_FACTOR * filtered_signal[candidate_idx].max()

    rep_indices = []
    last_two_amps = []   # rolling buffer for T2
    t2 = seed_t2
    t1_at_peaks = []
    t2_at_peaks = []

    for idx in candidate_idx:
        # --- Threshold 1: rolling mean + 1 std ---
        window_start = max(0, idx - ROLLING_WINDOW + 1)
        window = filtered_signal[window_start: idx + 1]
        t1 = window.mean() + window.std()

        t1_at_peaks.append(t1)
        t2_at_peaks.append(t2)

        amp = filtered_signal[idx]
        if amp > t1 and amp > t2:
            rep_indices.append(idx)

            # Update T2 buffer
            last_two_amps.append(amp)
            if len(last_two_amps) > 2:
                last_two_amps.pop(0)
            if len(last_two_amps) == 2:
                t2 = REP_THRESHOLD_FACTOR * np.mean(last_two_amps)

    return rep_indices, t1_at_peaks, t2_at_peaks, candidate_idx


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(raw_signal, filtered_signal, accel_col,
                 candidate_idx, rep_indices,
                 t1_at_peaks, t2_at_peaks,
                 title):
    """
    Two-panel figure:
      Top    — all raw accel axes (context)
      Bottom — selected axis: raw, filtered, thresholds, and counted reps
    """
    samples = np.arange(len(raw_signal))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{title}  —  {len(rep_indices)} reps counted", fontsize=13)

    # ── Top panel: raw selected axis ────────────────────────────────────────
    ax1.plot(samples, raw_signal, color="steelblue", alpha=0.5,
             linewidth=1, label=f"{accel_col} raw")
    ax1.plot(samples, filtered_signal, color="steelblue", linewidth=1.8,
             label=f"{accel_col} filtered")
    ax1.set_ylabel("Acceleration")
    ax1.set_title(f"Selected axis: {accel_col} (highest variance)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Bottom panel: thresholds + rep markers ──────────────────────────────
    ax2.plot(samples, filtered_signal, color="steelblue", linewidth=1.5,
             label="filtered signal")

    # Scatter thresholds at candidate peak positions
    if len(candidate_idx):
        ax2.scatter(candidate_idx, t1_at_peaks, color="orange", s=20,
                    zorder=4, marker="^", label="T1: rolling mean+std")
        ax2.scatter(candidate_idx, t2_at_peaks, color="purple", s=20,
                    zorder=4, marker="v", label="T2: rep-history mean")
        # Candidate peaks (not counted)
        rejected = [idx for idx in candidate_idx if idx not in rep_indices]
        if rejected:
            ax2.scatter(rejected, filtered_signal[rejected],
                        color="grey", s=30, zorder=4, label="rejected candidate")

    # Counted reps
    if rep_indices:
        ax2.scatter(rep_indices, filtered_signal[rep_indices],
                    color="red", s=120, zorder=5, marker="*",
                    label=f"counted rep ({len(rep_indices)})")

    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Acceleration")
    ax2.set_title("Dual-threshold rep detection")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------

def process(filepath):
    print(f"\nProcessing: {os.path.relpath(filepath, DATA_DIR)}")

    df = load_csv(filepath)
    accel_col, raw_signal = best_accel_axis(df)

    filtered_signal = lowpass(raw_signal)

    rep_indices, t1_at_peaks, t2_at_peaks, candidate_idx = \
        count_reps_dual_threshold(filtered_signal)

    print(f"  Candidate peaks : {len(candidate_idx)}")
    print(f"  Reps counted    : {len(rep_indices)}")

    title = os.path.relpath(filepath, DATA_DIR)
    plot_results(raw_signal, filtered_signal, accel_col,
                 candidate_idx, rep_indices,
                 t1_at_peaks, t2_at_peaks,
                 title)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    while True:
        filepath = select_file()
        process(filepath)
        again = input("\nProcess another file? [y/N]: ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()
