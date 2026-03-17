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
    os.path.join("bicep_curl", "bicep_curl_gustav_1.csv"),
    os.path.join("rows", "rows_dennis_1.csv"),
]

# Expected column names (order matters when the CSV has no header)
COLUMN_NAMES = ["timestamp", "accel_x", "accel_y", "accel_z",
                "gyro_x", "gyro_y", "gyro_z", "label", "person"]

# Candidate acceleration column name fragments (case-insensitive substring match)
ACCEL_CANDIDATES = ["accel_x", "accel_y", "accel_z", "ax", "ay", "az"]

# Low-pass filter
FILTER_CUTOFF_HZ = 3.0      # cut-off frequency in Hz
FILTER_ORDER     = 4        # Butterworth filter order

# Peak detection
MIN_PEAK_DISTANCE_SAMPLES = 20   # minimum samples between counted peaks
MIN_PEAK_PROMINENCE       = 0.05 # minimum prominence to even be a candidate

# Adaptive threshold
ADAPTIVE_THRESHOLD_FACTOR = 0.6  # T = factor * mean(last two valid peaks)
SEED_THRESHOLD            = 0.3  # used until two valid peaks have been found

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV, assigning column names when no header is present."""
    # Peek at the first line to decide whether a header exists
    with open(filepath, "r") as fh:
        first = fh.readline()

    first_col = first.split(",")[0].strip().lower()
    # Accept any non-numeric first cell as a header row
    try:
        float(first_col)
        has_header = False
    except ValueError:
        has_header = True

    if has_header:
        df = pd.read_csv(filepath)
        # Normalise timestamp column: rename 'time' -> 'timestamp' if needed
        if "timestamp" not in df.columns:
            time_cols = [c for c in df.columns if c.lower() in ("time", "timestamp", "t")]
            if time_cols:
                df = df.rename(columns={time_cols[0]: "timestamp"})
    else:
        df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)

    # Ensure timestamp is numeric
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    return df


def find_accel_columns(df: pd.DataFrame) -> list[str]:
    """Return the acceleration columns present in df, matched by substring."""
    matched = []
    for col in df.columns:
        for cand in ACCEL_CANDIDATES:
            if cand.lower() in col.lower() and col not in matched:
                matched.append(col)
                break
    if len(matched) < 3:
        raise ValueError(
            f"Expected at least 3 acceleration columns, found: {matched}\n"
            f"Available columns: {list(df.columns)}"
        )
    return matched[:3]  # keep exactly three


def estimate_sample_rate(timestamps: pd.Series) -> float:
    """Estimate sample rate in Hz from a timestamp column (seconds)."""
    diffs = np.diff(timestamps.values)
    median_dt = np.median(diffs[diffs > 0])
    return float(1.0 / median_dt)

# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def select_best_axis(df: pd.DataFrame, accel_cols: list[str]) -> tuple[str, np.ndarray]:
    """Return the column name and values of the axis with highest variance."""
    variances = {col: df[col].var() for col in accel_cols}
    best = max(variances, key=variances.__getitem__)
    return best, df[best].values


def lowpass_filter(signal: np.ndarray, cutoff_hz: float,
                   sample_rate: float, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter."""
    nyq = 0.5 * sample_rate
    norm_cutoff = min(cutoff_hz / nyq, 0.99)  # clamp below Nyquist
    b, a = butter(order, norm_cutoff, btype="low")
    return filtfilt(b, a, signal)


def detect_candidate_peaks(filtered: np.ndarray,
                            min_distance: int,
                            min_prominence: float) -> np.ndarray:
    """Find all candidate peaks in the filtered signal."""
    peaks, _ = find_peaks(
        filtered,
        distance=min_distance,
        prominence=min_prominence,
    )
    return peaks

# ---------------------------------------------------------------------------
# Adaptive threshold repetition counter
# ---------------------------------------------------------------------------

def count_reps(filtered: np.ndarray,
               candidate_peaks: np.ndarray,
               threshold_factor: float = ADAPTIVE_THRESHOLD_FACTOR,
               seed_threshold: float = SEED_THRESHOLD,
               min_distance: int = MIN_PEAK_DISTANCE_SAMPLES) -> tuple[list[int], list[float]]:
    """
    Walk through candidate peaks and count a rep whenever a peak exceeds
    the adaptive threshold T = factor * mean(last two valid peak amplitudes).

    Returns
    -------
    counted_indices : list of sample indices that were counted as reps
    thresholds_at_count : adaptive threshold value at each counted peak
    """
    last_two: list[float] = []      # amplitudes of the two most-recent counted peaks
    counted: list[int] = []
    thresholds: list[float] = []
    last_counted_idx = -min_distance  # allow the very first peak to be counted

    for idx in candidate_peaks:
        amp = float(filtered[idx])

        # Compute current threshold
        if len(last_two) < 2:
            threshold = seed_threshold
        else:
            threshold = threshold_factor * np.mean(last_two)

        # Enforce minimum distance between counted peaks
        if (idx - last_counted_idx) < min_distance:
            continue

        if amp >= threshold:
            counted.append(idx)
            thresholds.append(threshold)
            last_two = (last_two + [amp])[-2:]  # keep only the last two
            last_counted_idx = idx

    return counted, thresholds

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(raw: np.ndarray,
                 filtered: np.ndarray,
                 counted_peaks: list[int],
                 axis_name: str,
                 filename: str) -> None:
    """Plot raw signal, filtered signal, and counted-rep markers."""
    rep_count = len(counted_peaks)
    samples = np.arange(len(raw))

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(samples, raw, color="lightsteelblue", linewidth=0.8,
            alpha=0.7, label=f"Raw {axis_name}")
    ax.plot(samples, filtered, color="steelblue", linewidth=1.5,
            label="Filtered")

    if counted_peaks:
        ax.plot(counted_peaks, filtered[counted_peaks],
                "rv", markersize=10, zorder=5, label=f"Counted rep")

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Acceleration (g)")
    ax.set_title(f"{filename}  —  Reps counted: {rep_count}  (axis: {axis_name})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_file(filepath: str) -> None:
    filename = os.path.basename(filepath)
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")

    # 1. Load data
    df = load_csv(filepath)

    # 2. Identify acceleration columns
    accel_cols = find_accel_columns(df)
    print(f"  Acceleration columns: {accel_cols}")

    # 3. Estimate sample rate
    if "timestamp" in df.columns:
        sample_rate = estimate_sample_rate(df["timestamp"])
    else:
        sample_rate = 25.0  # fallback default
        print(f"  Warning: no timestamp column — assuming {sample_rate} Hz")
    print(f"  Estimated sample rate: {sample_rate:.1f} Hz")

    # 4. Select axis with highest variance
    best_axis, raw_signal = select_best_axis(df, accel_cols)
    print(f"  Best axis (highest variance): {best_axis}")

    # 5. Low-pass filter
    filtered = lowpass_filter(raw_signal, FILTER_CUTOFF_HZ, sample_rate, FILTER_ORDER)

    # 6. Candidate peaks
    candidates = detect_candidate_peaks(
        filtered, MIN_PEAK_DISTANCE_SAMPLES, MIN_PEAK_PROMINENCE
    )
    print(f"  Candidate peaks found: {len(candidates)}")

    # 7 & 8. Adaptive threshold → count reps
    counted, _ = count_reps(filtered, candidates)
    print(f"  Repetitions counted: {len(counted)}")

    # 9. Plot
    plot_results(raw_signal, filtered, counted, best_axis, filename)


def main() -> None:
    for fname in FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            print(f"Skipping '{fname}' — file not found in {DATA_DIR}")
            continue
        process_file(fpath)


if __name__ == "__main__":
    main()
