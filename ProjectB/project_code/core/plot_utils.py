''' This file contains code to define plot functions and other related utils.
    Important to note: many of the functions were used during the experimentation process for data analysis,
    and therefore are not currently in use, and may be incompatible with the current data structure. '''
import time
from pathlib import Path

import librosa
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import shap

# List of drone movement names
movement_names = ["ascent", "away", "descent", "hover", "towards", "turn"]

# List of features
features_list = ["mfcc", "harmonic", "dwt"]#

# List of classifiers
classifiers_list = ["random forest", "svm rbf", "mlp"]

# Get the correct path, regardless of working dir
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent


# Define all plots functions


def plot_mfcc_stats(data, label, feature_type, flag=False):
    """
        Plot mfcc statistics for class label

    Args:
        data: statistics vector to plot
        label: label for this statistics vector
        feature_type: feature type name (mfcc)
        flag: False= The data contains statistics of one sample of a class.
              True= The data contains the statistics average over all samples of a class.

    """
    if flag:
        stats_path = Path(project_root / f"plots/final/{feature_type}/stats/average")
    else:
        stats_path = Path(project_root / f"plots/final/{feature_type}/stats/sample")
    stats_path.mkdir(parents=True, exist_ok=True)

    quarter_sizes = [222, 222, 222, 237]
    start_indices = np.cumsum([0] + quarter_sizes)
    bar_width = 0.2
    spacing = 0.3
    features_time = ["Max Over Time", "Min Over Time", "Std Over Time", "Median Over Time",
                     "IQR Over Time", "Mean Delta Over Time", "Mean Delta-Delta Over Time",
                     "Std Delta Over Time", "Std Delta-Delta Over Time"]

    features_coeff = ["Max Over Coefficients", "Min Over Coefficients", "Std Over Coefficients",
                      "Median Over Coefficients", "IQR Over Coefficients"]
    COEFF = 13
    coeffs = [f"Coeff{i + 1}" for i in range(COEFF)]  # List of Coefficients

    time_frames = [21, 21, 21, 24, 87]  # Time frame per quarter

    stats_time = {
        f"{feature}": np.array([
            data[start + i * COEFF:start + (i + 1) * COEFF] for start in start_indices
        ])
        for i, feature in enumerate(features_time)
    }

    full_seg_coeff_start = 117 + start_indices[-1]

    stats_coeff = {
        f"{feature}": data[full_seg_coeff_start + i * time_frames[-1]:
                           full_seg_coeff_start + (i + 1) * time_frames[-1]]
        for i, feature in enumerate(features_coeff)
    }

    # Over Time
    plt.figure(figsize=(40, 20))
    for i, stat in enumerate(stats_time, start=1):
        plt.subplot(3, 3, i)
        feature_stats = stats_time[stat]
        br = np.arange(1, len(coeffs)+1) * (bar_width * 5 + spacing)

        plt.bar(br, feature_stats[0, :], color="r", width=bar_width,
                edgecolor="grey", label="Q1")
        plt.bar(br + bar_width, feature_stats[1, :], color="g", width=bar_width,
                edgecolor="grey", label="Q2")
        plt.bar(br + 2 * bar_width, feature_stats[2, :], color="b", width=bar_width,
                edgecolor="grey", label="Q3")
        plt.bar(br + 3 * bar_width, feature_stats[3, :], color="c", width=bar_width,
                edgecolor="grey", label="Q4")
        plt.bar(br + 4 * bar_width, feature_stats[4, :], color="m",
                width=bar_width, edgecolor="grey", label="Full Segment")

        plt.xticks(br + (bar_width * 2), coeffs)
        plt.title(f"{stat}", fontweight="bold", fontsize=15)
        plt.xlabel("Coefficients")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    if flag:
        plt.suptitle("Average Statistics Over Time", fontsize=30, fontweight="bold")
    else:
        plt.suptitle("Statistics Over Time", fontsize=30, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(stats_path / f"{feature_type}_{label}_stats_over_time.png")
    plt.close()

    # Over Coefficients - Full Segment
    plt.figure(figsize=(30, 20))
    for i, stat in enumerate(stats_coeff, start=1):
        plt.subplot(2, 3, i)
        br = np.arange(1, time_frames[-1]+1)
        plt.plot(br, stats_coeff[stat])
        plt.title(f"{stat}", fontweight="bold", fontsize=15)
        plt.xlabel("Time Frames")
        plt.ylabel("Amplitude")

    if flag:
        plt.suptitle("Full Segment - Average Statistics Over Coefficients", fontsize=30, fontweight="bold")
    else:
        plt.suptitle("Full Segment - Statistics Over Coefficients", fontsize=30, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(stats_path / f"{feature_type}_{label}_stats_over_coeffs.png")
    plt.close()


def plot_mfcc_features(data, label, feature_type):
    """
        Plot mfcc raw features for class label

    Args:
        data: raw features vector to plot
        label: label for this raw features vector
        feature_type: feature type name (mfcc)

    """
    feature_path = Path(project_root / f"plots/final/{feature_type}/features")
    feature_path.mkdir(parents=True, exist_ok=True)

    x_tick_interval = 20
    # Convert time frame index to time units (seconds)
    time_bins = librosa.times_like(data, sr=44100)
    plt.figure(figsize=(10, 10))
    sns.heatmap(data, cmap="viridis", xticklabels=False)
    # Show x-axis ticks in seconds (with precision .2f)
    plt.xticks(ticks=np.arange(0, data.shape[1], x_tick_interval),
               labels=[f"{time_bins[i]:.2f}" for i in range(0, data.shape[1], x_tick_interval)])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Coefficient Index")
    plt.title(f"{feature_type} {label} Feature Vector Heatmap")
    plt.savefig(feature_path / f"{feature_type}_{label}_features.png")
    plt.close()


def plot_mfcc_features_all(data, feature_type):
    """
        Plot mfcc raw features for all classes

    Args:
        data: a dictionary with 6 raw feature vectors
        feature_type: feature type name

    """
    feature_path = Path(project_root / f"plots/final/{feature_type}/features")
    feature_path.mkdir(parents=True, exist_ok=True)

    x_tick_interval = 20  # Show every 10th time frame
    all_values = np.concatenate([v.flatten() for v in data.values()])
    vmin = np.min(all_values)
    vmax = np.max(all_values)
    # Convert time frame index to time units (seconds)
    plt.figure(figsize=(20, 10))
    for idx, (label,values) in enumerate(data.items(), start=1):
        time_bins = librosa.times_like(values, sr=44100)
        plt.subplot(2,3,idx)
        sns.heatmap(values, cmap="viridis", xticklabels=False, vmin=vmin, vmax=vmax)
        # Show x-axis ticks in seconds (with precision .2f)
        plt.xticks(ticks=np.arange(0, values.shape[1], x_tick_interval),
                   labels=[f"{time_bins[i]:.2f}" for i in range(0, values.shape[1], x_tick_interval)])
        plt.xlabel("Time (seconds)")
        plt.ylabel("Coefficient Index")
        plt.title(f"{feature_type} {label} Feature Vector Heatmap", fontweight="bold", fontsize=15)
    plt.tight_layout()
    plt.savefig(feature_path / f"{feature_type}_features_all_sample.png")
    plt.close()


def plot_mfcc_mistaken(move1, move2, label1, label2, feature_type, flag=False):
    """
        Plot mfcc statistics for two classes that are similar to one another

    Args:
        move1: first statistics vector to plot
        move2: second statistics vector to plot
        label1: first label
        label2: second label
        feature_type: feature type name (mfcc)
        flag: False= The data contains statistics samples of two classes that were mistaken for one another.
              True= The data contains the statistics average of two classes that are similar to one another.
    """
    start_fs = 222 * 3 + 237
    features_time = ["Max Over Time", "Min Over Time", "Std Over Time", "Median Over Time",
                     "IQR Over Time", "Mean Delta Over Time", "Mean Delta-Delta Over Time",
                     "Std Delta Over Time", "Std Delta-Delta Over Time"]

    features_coeff = ["Max Over Coefficients", "Min Over Coefficients", "Std Over Coefficients",
                      "Median Over Coefficients", "IQR Over Coefficients"]
    COEFF = 13

    stats_time = {
        f"{feature}": np.array([
            move1[start_fs + i * COEFF: start_fs + (i + 1) * COEFF],
            move2[start_fs + i * COEFF: start_fs + (i + 1) * COEFF]
        ])
        for i, feature in enumerate(features_time)
    }

    start_fs_coeffs = 117 + start_fs
    FULL_FRAME = 87

    stats_coeff = {
        f"{feature}": np.array([
            move1[start_fs_coeffs + i * FULL_FRAME: start_fs_coeffs + (i + 1) * FULL_FRAME],
            move2[start_fs_coeffs + i * FULL_FRAME: start_fs_coeffs + (i + 1) * FULL_FRAME]
        ])
        for i, feature in enumerate(features_coeff)
    }

    # Over Time
    plt.figure(figsize=(40, 20))
    for i, stat in enumerate(stats_time, start=1):
        plt.subplot(3, 3, i)
        feature_stats = stats_time[stat]
        br = np.arange(1, COEFF+1)

        plt.plot(br, feature_stats[0, :], color="b", linewidth=2, label=label1)
        plt.plot(br, feature_stats[1, :], color="r", linestyle="dashed", linewidth=2, label=label2)

        plt.title(f"{stat}", fontweight="bold", fontsize=15)
        plt.xlabel("Coefficients")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if flag:
        plt.suptitle(f"Full Segment Statistics Over Time- Average {label1} vs {label2}", fontsize=30)
        plt.savefig(f"{feature_type}_average_{label1}_{label2}_stats_over_time.png")
    else:
        plt.suptitle(f"Full Segment Statistics Over Time- {label1} vs {label2}", fontsize=30)
        plt.savefig(f"{feature_type}_misclassified_{label1}_{label2}_stats_over_time.png")
    plt.close()

    # Over Coefficients - Full Segment
    plt.figure(figsize=(30, 20))
    for i, stat in enumerate(stats_coeff, start=1):
        plt.subplot(2, 3, i)
        feature_stats = stats_coeff[stat]
        br = np.arange(1, FULL_FRAME+1)

        plt.plot(br, feature_stats[0, :], color="b", linewidth=2, label=label1)
        plt.plot(br, feature_stats[1, :], color="r", linestyle="dashed", linewidth=2, label=label2)
        plt.title(f"{stat}", fontweight="bold", fontsize=15)
        plt.xlabel("Time Frames")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if flag:
        plt.suptitle(f"Full Segment Statistics Over Coefficients- Average {label1} vs {label2}", fontsize=30)
        plt.savefig(f"{feature_type}_average_{label1}_{label2}_stats_over_coeffs.png")
    else:
        plt.suptitle(f"Full Segment Statistics Over Coefficients- {label1} vs {label2}", fontsize=30)
        plt.savefig(f"{feature_type}_misclassified_{label1}_{label2}_stats_over_coeffs.png")
    plt.close()


def plot_harmonic_stats(data, label, feature_type, flag=False):
    """
        Plot harmonic statistics for class label

    Args:
        data: statistics vector to plot
        label: label for this statistics vector
        feature_type: feature type name (harmonic)
        flag: False= The data contains statistics of one sample of a class.
              True= The data contains the statistics average over all samples of a class.

    """
    if flag:
        stats_path = Path(project_root / f"plots/final/{feature_type}/stats/average")
    else:
        stats_path = Path(project_root / f"plots/final/{feature_type}/stats/sample")
    stats_path.mkdir(parents=True, exist_ok=True)

    bar_width = 0.2
    features = ["Harmonic Centroid", "Harmonic Bandwidth", "Harmonic Rolloff", "Harmonic Magnitude Delta",
                "Harmonic Magnitude Delta-Delta", "Percussive Onset", "Percussive Contrast- Full Segment"]
    features_mag = ["Std Harmonic Magnitude Over Time", "Median Harmonic Magnitude Over Time",
                    "Std Harmonic Magnitude Over Frequency", "Median Harmonic Magnitude Over Frequency",
                    "Std Percussive Magnitude Over Time", "Median Percussive Magnitude Over Time",
                    "Std Percussive Magnitude Over Frequency", "Median Percussive Magnitude Over Frequency"]
    quarters = ["Q1", "Q2", "Q3", "Q4", "FullSegment"]  # List of quarters
    bands = [f"Band{i + 1}" for i in range(7)]  # List of spectral bands (for Percussive Contrast)
    per_quarter = 48
    offsets = [0, 4, 8, 12, 14, 16]
    stats = {
        f"{feature}": np.array([data[start + j * per_quarter: offsets[i + 1] + j * per_quarter]
                                for j in range(5)])  # 5 quarters
        for i, (feature, start) in enumerate(zip(features, offsets[:-1]))
    }
    stats["Percussive Onset"] = np.vstack([
                                              data[16 + i * per_quarter: 20 + i * per_quarter] for i in
                                              range(4)] + [data[1068:1072]])
    stats["Percussive Contrast- Full Segment"] = data[1072:1100].reshape(4, 7)

    freq_bin_size = 257
    time_frame_size = 173
    quarter_sizes = [freq_bin_size, freq_bin_size, time_frame_size, time_frame_size]

    start_indices_harmonic = np.cumsum([208] + quarter_sizes)
    start_indices_percussive = np.cumsum([1100] + quarter_sizes)

    stats_mag = {}
    for i, feature in enumerate(features_mag):
        if i < 4:  # Harmonic
            start = start_indices_harmonic[i]
        else:
            start = start_indices_percussive[i - 4]
        end = start + quarter_sizes[i % 4]
        stats_mag[feature] = data[start:end]

    # Regular stats
    plt.figure(figsize=(20, 12))
    for i, feature in enumerate(features, start=1):
        plt.subplot(3, 3, i)
        feature_stats = stats[feature]
        if "Contrast" in feature:
            br = np.arange(1, len(bands)+1)
            plt.bar(br, feature_stats[0, :], color="r", width=bar_width,
                    edgecolor="grey", label="max")
            plt.bar(br + bar_width, feature_stats[1, :], color="g", width=bar_width,
                    edgecolor="grey", label="min")
            plt.bar(br + 2 * bar_width, feature_stats[2, :], color="b", width=bar_width,
                    edgecolor="grey", label="std")
            plt.bar(br + 3 * bar_width, feature_stats[3, :], color="c", width=bar_width,
                    edgecolor="grey", label="median")

            plt.xticks(br + bar_width, bands)
            plt.xlabel("Frequency Bands")
        else:
            br = np.arange(1, len(quarters)+1)
            if "Delta" in feature:
                plt.bar(br, feature_stats[:, 0], color="r", width=bar_width,
                        edgecolor="grey", label="mean")
                plt.bar(br + bar_width, feature_stats[:, 1], color="g", width=bar_width,
                        edgecolor="grey", label="std")
            else:
                plt.bar(br, feature_stats[:, 0], color="r", width=bar_width,
                        edgecolor="grey", label="max")
                plt.bar(br + bar_width, feature_stats[:, 1], color="g", width=bar_width,
                        edgecolor="grey", label="min")
                plt.bar(br + 2 * bar_width, feature_stats[:, 2], color="b", width=bar_width,
                        edgecolor="grey", label="std")
                plt.bar(br + 3 * bar_width, feature_stats[:, 3], color="c", width=bar_width,
                        edgecolor="grey", label="median")

            plt.xticks(br + bar_width, quarters)
            plt.xlabel("Quarters")
        plt.ylabel("Amplitude")
        plt.title(f"{feature}", fontweight="bold", fontsize=15)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    if flag:
        plt.suptitle("Average Regular Statistics", fontsize=30, fontweight="bold")
    else:
        plt.suptitle("Regular Statistics", fontsize=30, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(stats_path / f"{feature_type}_{label}_stats.png")
    plt.close()

    # Magnitude stats
    plt.figure(figsize=(20, 12))
    for i, feature in enumerate(features_mag):
        plt.subplot(3, 3, i + 1)
        if i % 4 < 2:  # Over Time
            br = np.arange(1, freq_bin_size+1)
            plt.plot(br, stats_mag[feature])
            plt.title(f"{feature}", fontweight="bold", fontsize=15)
            plt.xlabel("Frequency Bins")
            plt.ylabel("Amplitude")
        else:  # Over Frequency
            br = np.arange(1, time_frame_size+1)
            plt.plot(br, stats_mag[feature])
            plt.title(f"{feature}", fontweight="bold", fontsize=15)
            plt.xlabel("Time Frames")
            plt.ylabel("Amplitude")

    if flag:
        plt.suptitle("Average Full Segment Magnitude Statistics", fontsize=30, fontweight="bold")
    else:
        plt.suptitle("Full Segment Magnitude Statistics", fontsize=30, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(stats_path / f"{feature_type}_{label}_stats_magnitude.png")
    plt.close()


def plot_heatmap(data, time_bins, freq_bins, title, subplot_idx):
    """
        Helper function to plot harmonic heatmaps

    Args:
        data: features vector to plot
        time_bins: the time bins vector
        freq_bins: the frequency bins vector
        title:  title of the plot
        subplot_idx: index of this subplot

    """
    x_tick_interval = 20  # Show every 10th time frame
    y_tick_interval = 5  # Show every 5th frequency bin
    plt.subplot(2, 2, subplot_idx)
    sns.heatmap(data, cmap="viridis", cbar_kws={"label": "Amplitude"}, xticklabels=False, yticklabels=False)
    plt.xticks(ticks=np.arange(0, data.shape[1], x_tick_interval), rotation=45,
               labels=[f"{time_bins[i]:.2f}" for i in range(0, data.shape[1], x_tick_interval)])
    plt.yticks(ticks=np.arange(0, data.shape[0], y_tick_interval),
               labels=[f"{freq_bins[i]:.0f}" for i in range(0, data.shape[0], y_tick_interval)])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)


def plot_harmonic_features(data, label, feature_type):
    """
       Plot harmonic raw features for class label

   Args:
       data: raw features vector to plot
       label: label for this raw features vector
       feature_type: feature type name (harmonic)

   """
    feature_path = Path(project_root / f"plots/final/{feature_type}/features")
    feature_path.mkdir(parents=True, exist_ok=True)

    harmonic_mag, percussive_mag = data
    midpoint = harmonic_mag.shape[0] // 2

    segments = [
        (harmonic_mag[:midpoint, :], "Harmonic Feature Vector Heatmap- Lower Frequencies"),
        (harmonic_mag[midpoint:, :], "Harmonic Feature Vector Heatmap- Higher Frequencies"),
        (percussive_mag[:midpoint, :], "Percussive Feature Vector Heatmap- Lower Frequencies"),
        (percussive_mag[midpoint:, :], "Percussive Feature Vector Heatmap- Higher Frequencies")
    ]

    # Convert frequency bin index to frequency units (Hz)
    freq_bins = librosa.fft_frequencies(sr=44100, n_fft=512)
    midpoint_freq_idx = len(freq_bins) // 2

    # Convert time frame index to time units (seconds)
    time_bins = librosa.frames_to_time(np.arange(harmonic_mag.shape[1]), sr=44100, n_fft=512, hop_length=256)

    plt.figure(figsize=(20, 20))
    for i, (segment, title) in enumerate(segments, start=1):
        plot_heatmap(segment, time_bins, freq_bins[midpoint_freq_idx * (i // 3):], title, i)

    plt.tight_layout()
    plt.savefig(feature_path / f"{feature_type}_{label}_features.png")
    plt.close()


def plot_harmonic_mistaken(move1, move2, label1, label2, feature_type, flag=False):
    """
        Plot harmonic statistics for two classes that are similar to one another

    Args:
        move1: first statistics vector to plot
        move2: second statistics vector to plot
        label1: first label
        label2: second label
        feature_type: feature type name (harmonic)
        flag: False= The data contains statistics samples of two classes that were mistaken for one another.
              True= The data contains the statistics average of two classes that are similar to one another.
    """
    bar_width = 0.2
    features = ["Harmonic Centroid", "Harmonic Bandwidth", "Harmonic Rolloff", "Harmonic Magnitude Delta",
                "Harmonic Magnitude Delta-Delta", "Percussive Onset", "Max Percussive Contrast",
                "Min Percussive Contrast", "Std Percussive Contrast", "Median Percussive Contrast"]
    features_mag = ["Std Harmonic Magnitude Over Time", "Median Harmonic Magnitude Over Time",
                    "Std Harmonic Magnitude Over Frequency", "Median Harmonic Magnitude Over Frequency",
                    "Std Percussive Magnitude Over Time", "Median Percussive Magnitude Over Time",
                    "Std Percussive Magnitude Over Frequency", "Median Percussive Magnitude Over Frequency"]

    stat_names = ["Max", "Min", "Std", "Median"]
    stat_names_delta = ["Mean", "Std"]

    bands = [f"Band{i + 1}" for i in range(7)]  # List of spectral bands (for Percussive Contrast)

    offsets_harmonic = [0, 4, 8, 12, 14, 16]
    offsets_percussive = [0, 4, 11, 18, 25, 32]
    per_quarter = 48
    start_fs = 4 * per_quarter

    stats = {
        f"{feature}": np.array([
            move1[start_fs + start: start_fs + offsets_harmonic[i + 1]],
            move2[start_fs + start: start_fs + offsets_harmonic[i + 1]]
        ])
        for i, (feature, start) in enumerate(zip(features, offsets_harmonic[:-1]))
    }

    freq_bin_size = 257
    time_frame_size = 173

    start_fs_percussive = start_fs + 16 + 2 * freq_bin_size + 2 * time_frame_size

    percussive_stats = {
        f"{feature}": np.array([
            move1[start_fs_percussive + start: start_fs_percussive + offsets_percussive[i + 1]],
            move2[start_fs_percussive + start: start_fs_percussive + offsets_percussive[i + 1]]
        ])
        for i, (feature, start) in enumerate(zip(features[len(stats):], offsets_percussive[:-1]))
    }

    stats.update(percussive_stats)

    quarter_sizes = [freq_bin_size, freq_bin_size, time_frame_size, time_frame_size]
    start_indices_harmonic = np.cumsum([208] + quarter_sizes)
    start_indices_percussive = np.cumsum([1100] + quarter_sizes)

    stats_mag = {}
    for i, feature in enumerate(features_mag):
        if i < 4:  # Harmonic
            start = start_indices_harmonic[i]
        else:
            start = start_indices_percussive[i - 4]
        end = start + quarter_sizes[i % 4]
        stats_mag[feature] = np.array([
            move1[start:end], move2[start:end]
        ])

    # Regular stats
    plt.figure(figsize=(30, 20))
    for i, feature in enumerate(features, start=1):
        plt.subplot(3, 4, i)
        feature_stats = stats[feature]
        br = np.arange(1, len(feature_stats[0, :])+1)
        plt.bar(br, feature_stats[0, :], color="r", width=bar_width,
                edgecolor="grey", label=label1)
        plt.bar(br + bar_width, feature_stats[1, :], color="b", width=bar_width,
                edgecolor="grey", label=label2)

        if "Contrast" in feature:
            plt.xticks(br + bar_width, bands)
            plt.xlabel("Frequency Bands")
        else:
            plt.xlabel("Stats")
            if "Delta" in feature:
                plt.xticks(br + bar_width, stat_names_delta)
            else:
                plt.xticks(br + bar_width, stat_names)

        plt.ylabel("Amplitude")
        plt.title(f"{feature}", fontweight="bold", fontsize=15)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if flag:
        plt.suptitle(f"Full Segment Statistics- Average {label1} vs {label2}", fontsize=30)
        plt.savefig(f"{feature_type}_average_{label1}_{label2}_stats.png")
    else:
        plt.suptitle(f"Full Segment Statistics- {label1} vs {label2}", fontsize=30)
        plt.savefig(f"{feature_type}_misclassified_{label1}_{label2}_stats.png")
    plt.close()

    # Magnitude stats
    plt.figure(figsize=(30, 30))
    for i, feature in enumerate(features_mag):
        plt.subplot(3, 3, i + 1)
        feature_stats = stats_mag[feature]

        if i % 4 < 2:  # Over Time
            br = np.arange(1, freq_bin_size+1)
            plt.plot(br, feature_stats[0, :], color="b", linewidth=2, label=label1)
            plt.plot(br, feature_stats[1, :], color="r", linestyle="dashed", linewidth=2, label=label2)
            plt.xlabel("Frequency Bins")
        else:  # Over Frequency
            br = np.arange(1, time_frame_size+1)
            plt.plot(br, feature_stats[0, :], color="b", linewidth=2, label=label1)
            plt.plot(br, feature_stats[1, :], color="r", linestyle="dashed", linewidth=2, label=label2)
            plt.xlabel("Time Frames")

        plt.ylabel("Amplitude")
        plt.title(f"{feature}", fontweight="bold", fontsize=15)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if flag:
        plt.suptitle(f"Full Segment Magnitude Statistics- Average {label1} vs {label2}", fontsize=30)
        plt.savefig(f"{feature_type}_average_{label1}_{label2}_stats_magnitude.png")
    else:
        plt.suptitle(f"Full Segment Magnitude Statistics- {label1} vs {label2}", fontsize=30)
        plt.savefig(f"{feature_type}_misclassified_{label1}_{label2}_stats_magnitude.png")
    plt.close()


def plot_dwt_stats(data, label, feature_type, flag=False):
    """
        Plot dwt statistics for class label

    Args:
        data: statistics vector to plot
        label: label for this statistics vector
        feature_type: feature type name (dwt)
        flag: False= The data contains statistics of one sample of a class.
              True= The data contains the statistics average over all samples of a class.

    """
    if flag:
        stats_path = Path(project_root / f"plots/final/{feature_type}/stats/average")
    else:
        stats_path = Path(project_root / f"plots/final/{feature_type}/stats/sample")
    stats_path.mkdir(parents=True, exist_ok=True)

    bar_width = 0.2
    spacing = 0.3
    features = ["Max Coeff", "Min Coeff", "Std Coeff", "Median Coeff",
                "IQR Coeff", "Mean Delta", "Mean Delta-Delta",
                "Std Delta", "Std Delta-Delta", "ZCR"]
    features_ac = ["Max Autocorrelation", "Min Autocorrelation", "Mean Autocorrelation",
                   "Std Autocorrelation"]
    coeffs = ["Approximation", "Detail level 1", "Detail level 2", "Detail level 3"]

    quarters = ["Q1", "Q2", "Q3", "Q4", "FullSegment"]  # List of quarters
    per_quarter = 256
    per_coeff = 64

    stats = {
        f"{feature}": np.array([
            [data[i + j * per_quarter + k * per_coeff] for k in range(len(coeffs))]
            # 4 coeffs
            for j in range(len(quarters))  # 5 quarters
        ])
        for i, feature in enumerate(features)
    }

    stats_num = 10

    stats_ac = {
        f"{feature}": np.array([
            [data[stats_num + i + j * per_quarter + k * per_coeff] for k in range(len(coeffs))]
            # 4 coeffs
            for j in range(len(quarters))  # 5 quarters
        ])
        for i, feature in enumerate(features_ac)
    }

    lag_size = 50
    start_lags = 4 * per_quarter + 14
    lags = {
        f"{coeff}": data[start_lags + i * per_coeff: lag_size + start_lags + i * per_coeff]
        for i, coeff in enumerate(coeffs)
    }

    # Stats
    plt.figure(figsize=(30, 20))
    for i, stat in enumerate(stats, start=1):
        plt.subplot(3, 4, i)
        feature_stats = stats[stat]
        br = np.arange(1, len(coeffs)+1) * (bar_width * 5 + spacing)

        plt.bar(br, feature_stats[0, :], color="r", width=bar_width,
                edgecolor="grey", label="Q1")
        plt.bar(br + bar_width, feature_stats[1, :], color="g", width=bar_width,
                edgecolor="grey", label="Q2")
        plt.bar(br + 2 * bar_width, feature_stats[2, :], color="b", width=bar_width,
                edgecolor="grey", label="Q3")
        plt.bar(br + 3 * bar_width, feature_stats[3, :], color="c", width=bar_width,
                edgecolor="grey", label="Q4")
        plt.bar(br + 4 * bar_width, feature_stats[4, :], color="m",
                width=bar_width, edgecolor="grey", label="Full Segment")

        plt.xticks(br + (bar_width * 2), coeffs)
        plt.title(f"{stat}", fontweight="bold", fontsize=15)
        plt.xlabel("Features")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    if flag:
        plt.suptitle("Average Regular Statistics", fontsize=30, fontweight="bold")
    else:
        plt.suptitle("Full Regular Statistics", fontsize=30, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(stats_path / f"{feature_type}_{label}_stats.png")
    plt.close()

    # Autocorrelation stats
    plt.figure(figsize=(30, 20))
    for i, stat in enumerate(stats_ac, start=1):
        plt.subplot(3, 3, i)
        feature_stats = stats_ac[stat]
        br = np.arange(1, len(coeffs)+1) * (bar_width * 5 + spacing)

        plt.bar(br, feature_stats[0, :], color="r", width=bar_width,
                edgecolor="grey", label="Q1")
        plt.bar(br + bar_width, feature_stats[1, :], color="g", width=bar_width,
                edgecolor="grey", label="Q2")
        plt.bar(br + 2 * bar_width, feature_stats[2, :], color="b", width=bar_width,
                edgecolor="grey", label="Q3")
        plt.bar(br + 3 * bar_width, feature_stats[3, :], color="c", width=bar_width,
                edgecolor="grey", label="Q4")
        plt.bar(br + 4 * bar_width, feature_stats[4, :], color="m",
                width=bar_width, edgecolor="grey", label="Full Segment")

        plt.xticks(br + (bar_width * 2), coeffs)
        plt.title(f"{stat}", fontweight="bold", fontsize=15)
        plt.xlabel("Features")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Lags stats
    for i, stat in enumerate(lags, start=len(stats_ac) + 1):
        plt.subplot(3, 3, i)
        br = np.arange(1, lag_size+1)

        plt.plot(br, lags[stat])
        plt.title(f"{stat}- Full Segment {lag_size} Lags", fontweight="bold", fontsize=15)
        plt.xlabel("Lag Number")
        plt.ylabel("Amplitude")

    if flag:
        plt.suptitle("Average Autocorrelation Statistics", fontsize=30, fontweight="bold")
    else:
        plt.suptitle("Autocorrelation Statistics", fontsize=30, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(stats_path / f"{feature_type}_{label}_stats_autocorrelation.png")
    plt.close()


def plot_dwt_features(data, label, feature_type):
    """
       Plot dwt raw features for class label

   Args:
       data: raw features vector to plot
       label: label for this raw features vector
       feature_type: feature type name (dwt)

   """
    feature_path = Path(project_root / f"plots/final/{feature_type}/features")
    feature_path.mkdir(parents=True, exist_ok=True)

    # Plot Approximation and Detail per level
    for i in range(len(data)):
        plt.subplot(2, 2, i + 1)
        coeff = data[i]

        if i == 0:  # Approx
            plt.plot(coeff, color="b")
            plt.title(f"Approximation")
        else:
            plt.plot(coeff, color="r")
            plt.title(f"Level {i} Detail")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(feature_path / f"{feature_type}_{label}_features.png")
    plt.close()


def plot_dwt_mistaken(move1, move2, label1, label2, feature_type, flag=False):
    """
        Plot dwt statistics for two classes that are similar to one another

    Args:
        move1: first statistics vector to plot
        move2: second statistics vector to plot
        label1: first label
        label2: second label
        feature_type: feature type name (dwt)
        flag: False= The data contains statistics samples of two classes that were mistaken for one another.
              True= The data contains the statistics average of two classes that are similar to one another.
    """
    bar_width = 0.2
    features = ["Max Coeff", "Min Coeff", "Std Coeff", "Median Coeff",
                "IQR Coeff", "Mean Delta", "Mean Delta-Delta",
                "Std Delta", "Std Delta-Delta", "ZCR"]
    features_ac = ["Max Autocorrelation", "Min Autocorrelation", "Mean Autocorrelation",
                   "Std Autocorrelation"]
    coeffs = ["Approximation", "Detail level 1", "Detail level 2", "Detail level 3"]

    quarters = ["Q1", "Q2", "Q3", "Q4", "FullSegment"]  # List of quarters
    per_quarter = 256
    per_coeff = 64

    start_fs = 4 * per_quarter

    stats = {
        f"{feature}": np.array([
            [move1[start_fs + i + k * per_coeff] for k in range(len(coeffs))],
            [move2[start_fs + i + k * per_coeff] for k in range(len(coeffs))]
            # 4 coeffs
        ])
        for i, feature in enumerate(features)
    }

    stats_num = 10

    stats_ac = {
        f"{feature}": np.array([
            [move1[start_fs + stats_num + i + k * per_coeff] for k in range(len(coeffs))],
            [move2[start_fs + stats_num + i + k * per_coeff] for k in range(len(coeffs))]
        ])
        for i, feature in enumerate(features_ac)
    }

    lag_size = 50
    start_lags = start_fs + stats_num + len(stats_ac)
    lags = {
        f"{coeff}": np.array([
            move1[start_lags + i * per_coeff: lag_size + start_lags + i * per_coeff],
            move2[start_lags + i * per_coeff: lag_size + start_lags + i * per_coeff]
        ])
        for i, coeff in enumerate(coeffs)
    }

    # Stats
    plt.figure(figsize=(30, 20))
    for i, stat in enumerate(stats, start=1):
        plt.subplot(3, 4, i)
        feature_stats = stats[stat]
        br = np.arange(1, len(coeffs)+1)

        plt.bar(br, feature_stats[0, :], color="r", width=bar_width,
                edgecolor="grey", label=label1)
        plt.bar(br + bar_width, feature_stats[1, :], color="b", width=bar_width,
                edgecolor="grey", label=label2)

        plt.xticks(br + bar_width, coeffs)
        plt.title(f"{stat}", fontweight="bold", fontsize=15)
        plt.xlabel("Coefficients (Approx/Detail)")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if flag:
        plt.suptitle(f"Full Segment Statistics- Average {label1} vs {label2}", fontsize=30)
        plt.savefig(f"{feature_type}_average_{label1}_{label2}_stats.png")
    else:
        plt.suptitle(f"Full Segment Statistics- {label1} vs {label2}", fontsize=30)
        plt.savefig(f"{feature_type}_misclassified_{label1}_{label2}_stats.png")
    plt.close()

    # Autocorrelation stats
    plt.figure(figsize=(30, 20))
    for i, stat in enumerate(stats_ac, start=1):
        plt.subplot(3, 3, i)
        feature_stats = stats_ac[stat]
        br = np.arange(1, len(coeffs)+1)

        plt.bar(br, feature_stats[0, :], color="r", width=bar_width,
                edgecolor="grey", label=label1)
        plt.bar(br + bar_width, feature_stats[1, :], color="b", width=bar_width,
                edgecolor="grey", label=label2)

        plt.xticks(br + bar_width, coeffs)
        plt.title(f"{stat}", fontweight="bold", fontsize=15)
        plt.xlabel("Coefficients (Approx/Detail)")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Lags stats
    for i, stat in enumerate(lags, start=len(stats_ac) + 1):
        plt.subplot(3, 3, i)
        feature_stats = lags[stat]
        br = np.arange(1, lag_size+1)

        plt.plot(br, feature_stats[0, :], color="b", linewidth=2, label=label1)
        plt.plot(br, feature_stats[1, :], color="r", linestyle="dashed", linewidth=2, label=label2)
        plt.title(f"{stat}- Full Segment {lag_size} Lags", fontweight="bold", fontsize=15)
        plt.xlabel("Lag Number")
        plt.ylabel("Amplitude")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if flag:
        plt.suptitle(f"Full Segment Autocorrelation Statistics- Average {label1} vs {label2}", fontsize=30)
        plt.savefig(f"{feature_type}_average_{label1}_{label2}_stats_autocorrelation.png")
    else:
        plt.suptitle(f"Full Segment Autocorrelation Statistics- {label1} vs {label2}", fontsize=30)
        plt.savefig(f"{feature_type}_misclassified_{label1}_{label2}_stats_autocorrelation.png")
    plt.close()


# Mapping feature type to plot function (according to selector)
# sel=0: plot raw features (single), sel=1: plot statistics, sel=2: plot raw features (all classes),
plot_functions = {
    "mfcc": [plot_mfcc_features, plot_mfcc_stats, plot_mfcc_mistaken],
    "harmonic": [plot_harmonic_features, plot_harmonic_stats, plot_harmonic_mistaken],
    "dwt": [plot_dwt_features, plot_dwt_stats, plot_dwt_mistaken]
}


def plot_sample_features(raw_features, labels, feature_type):
    # Get first sample of each label
    first_features = {
        label: raw_features[labels == label][0]
        for label in np.unique(labels)
    }
    # 6 plots
    for label, vec in first_features.items():
        plot_functions[feature_type][0](vec, label, feature_type)

    # A single plot with 6 subplots
    # Impossible for other feature types, as they each have 4 subplots per raw feature vector
    if feature_type == "mfcc":
        plot_mfcc_features_all(first_features, feature_type)


def plot_sample_stats(stats, labels, feature_type):
    # Get first sample of each label
    first_stats = {
        label: stats[labels == label][0]
        for label in np.unique(labels)
    }
    # 6 plots
    for label, data in first_stats.items():
        plot_functions[feature_type][1](data, label, feature_type)


def plot_mistaken_features(feature1, feature2, label1, label2, feature_type):
    """
        Plot raw features for samples of two classes that were mistaken for one another

    Args:
        feature1: raw feature for first sample
        feature2: raw feature for second sample
        label1: real label of first sample
        label2: real label of second sample
        feature_type: feature type name
    """
    max_val = max(np.max(feature1), np.max(feature2))
    min_val = min(np.min(feature1), np.min(feature2))
    x_tick_interval = 10  # Show every 10th time frame
    # Convert time frame index to time units (seconds)
    time_bins = librosa.times_like(feature1, sr=44100)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    sns.heatmap(feature1, cmap="viridis", xticklabels=False, vmin=min_val, vmax=max_val)
    # Show x-axis ticks in seconds (with precision .2f)
    plt.xticks(ticks=np.arange(0, feature1.shape[1], x_tick_interval),
               labels=[f"{time_bins[i]:.2f}" for i in range(0, feature1.shape[1], x_tick_interval)])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Coefficient Index")
    plt.title(f"{feature_type} {label1} mistaken as {label2} Feature Vector Heatmap")

    plt.subplot(1, 2, 2)
    sns.heatmap(feature2, cmap="viridis", xticklabels=False, vmin=min_val, vmax=max_val)
    # Show x-axis ticks in seconds (with precision .2f)
    plt.xticks(ticks=np.arange(0, feature2.shape[1], x_tick_interval),
               labels=[f"{time_bins[i]:.2f}" for i in range(0, feature2.shape[1], x_tick_interval)])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Coefficient Index")
    plt.title(f"{feature_type} {label2} mistaken as {label1} Feature Vector Heatmap")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{feature_type}_mistaken_features.png")
    plt.close()


def plot_misclassified(stats, raw_features, y_pred, y_test, label1, label2, feature_type):
    """
        Plot an overlap of mistaken predictions of label1 and label2

    Args:
        stats: test data (statistics)
        raw_features: raw features of the test data
        y_pred: predicted test labels
        y_test: true test labels
        label1: first class (mistaken for label2)
        label2: second class (mistaken for label1)
        feature_type: feature type name

    Returns:
        The indices of the mistaken predictions, else None x 2

    """
    # Find indices of misclassified samples
    misclassified_indices = np.where(y_test != y_pred)[0]

    mistaken_stats1 = None  # Actually label1, predicted as label2
    mistaken_stats2 = None  # Actually label2, predicted as label1

    label1_transform = movement_names.index(label1)
    label2_transform = movement_names.index(label2)

    for i in misclassified_indices:
        # Found mistaken label1
        if mistaken_stats1 is None and y_test[i] == label1_transform and y_pred[i] == label2_transform:
            mistaken_stats1 = stats[i]
            mistaken_features1 = raw_features[i]
            move1_idx = i

        # Found mistaken label2
        if mistaken_stats2 is None and y_test[i] == label2_transform and y_pred[i] == label1_transform:
            mistaken_stats2 = stats[i]
            mistaken_features2 = raw_features[i]
            move2_idx = i

        # Stop when both are found
        if mistaken_stats1 is not None and mistaken_stats2 is not None:
            plot_mistaken_features(mistaken_features1, mistaken_features2, label1, label2, feature_type)
            plot_functions[feature_type][2](mistaken_stats1, mistaken_stats2, label1, label2, feature_type)
            return move1_idx, move2_idx

    return None, None


def generate_mfcc_stats_names():
    """
    Returns: Generates all the statistics names (classifier features names) for the mfcc classifiers
    """
    num_coef = 13
    frame_counts = [64, 64, 64, 67, 259]
    quarters = ["Q1", "Q2", "Q3", "Q4", "FULL"]
    stats_time = ["max", "min", "std", "median", "iqr", "mean_delta", "mean_delta2", "std_delta", "std_delta2"]
    stats_coeff = ["max", "min", "std", "median", "iqr"]
    feature_names = []

    # Per-quarter stats
    for q_idx, frame_count in enumerate(frame_counts):
        q = quarters[q_idx]
        # Time stats for each MFCC coefficient
        for stat in stats_time:
            for coef in range(num_coef):
                feature_names.append(f"{q}_coef_{coef+1}_{stat}")
        # Coefficient stats across each time frame
        for stat in stats_coeff:
            for frame in range(frame_count):
                feature_names.append(f"{q}_frame_{frame+1}_{stat}")

    return feature_names


def generate_harmonic_stats_names():
    """
    Returns: Generates all the statistics names (classifier features names) for the harmonic classifiers
    """
    # Quarter features
    quarter_feature_names = []
    for i in range(1, 5):  # quarters 1 to 4
        quarter_feature_names += [
                                     # Harmonic features (16)
                                     f"Q{i}_harmonic_centroid_max",
                                     f"Q{i}_harmonic_centroid_min",
                                     f"Q{i}_harmonic_centroid_std",
                                     f"Q{i}_harmonic_centroid_median",
                                     f"Q{i}_harmonic_bandwidth_max",
                                     f"Q{i}_harmonic_bandwidth_min",
                                     f"Q{i}_harmonic_bandwidth_std",
                                     f"Q{i}_harmonic_bandwidth_median",
                                     f"Q{i}_harmonic_rolloff_max",
                                     f"Q{i}_harmonic_rolloff_min",
                                     f"Q{i}_harmonic_rolloff_std",
                                     f"Q{i}_harmonic_rolloff_median",
                                     f"Q{i}_delta_harmonic_centroid_mean",
                                     f"Q{i}_delta_harmonic_centroid_std",
                                     f"Q{i}_delta2_harmonic_centroid_mean",
                                     f"Q{i}_delta2_harmonic_centroid_std",

                                     # Percussive features (32)
                                     f"Q{i}_percussive_onset_max",
                                     f"Q{i}_percussive_onset_min",
                                     f"Q{i}_percussive_onset_std",
                                     f"Q{i}_percussive_onset_median",
                                 ] + [
                                     f"Q{i}_percussive_contrast_{stat}_{band+1}"
                                     for stat in ["max", "min", "std", "median"]
                                     for band in range(7)
                                 ]

    # Full segment harmonic
    full_harmonic_features = [
        # 16 summary stats
        "FULL_harmonic_centroid_max", "FULL_harmonic_centroid_min", "FULL_harmonic_centroid_std",
        "FULL_harmonic_centroid_median",
        "FULL_harmonic_bandwidth_max", "FULL_harmonic_bandwidth_min", "FULL_harmonic_bandwidth_std",
        "FULL_harmonic_bandwidth_median",
        "FULL_harmonic_rolloff_max", "FULL_harmonic_rolloff_min", "FULL_harmonic_rolloff_std",
        "FULL_harmonic_rolloff_median",
        "FULL_delta_harmonic_centroid_mean", "FULL_delta_harmonic_centroid_std",
        "FULL_delta2_harmonic_centroid_mean", "FULL_delta2_harmonic_centroid_std",
    ]

    # 257 freq bin stds + 257 medians
    full_harmonic_features += [f"FULL_harmonic_std_freqbin_{i+1}" for i in range(257)]
    full_harmonic_features += [f"FULL_harmonic_median_freqbin_{i+1}" for i in range(257)]

    # 517 time frame stds + 517 medians
    full_harmonic_features += [f"FULL_harmonic_std_frame_{i+1}" for i in range(517)]
    full_harmonic_features += [f"FULL_harmonic_median_frame_{i+1}" for i in range(517)]

    # Full segment percussive
    full_percussive_features = [
        "FULL_percussive_onset_max", "FULL_percussive_onset_min",
        "FULL_percussive_onset_std", "FULL_percussive_onset_median",
    ]

    # 7 bands × 4 stats
    full_percussive_features += [
        f"FULL_percussive_contrast_{stat}_{band+1}"
        for stat in ["max", "min", "std", "median"]
        for band in range(7)
    ]

    # 257 freq bin stds + 257 medians
    full_percussive_features += [f"FULL_percussive_std_freqbin_{i+1}" for i in range(257)]
    full_percussive_features += [f"FULL_percussive_median_freqbin_{i+1}" for i in range(257)]

    # 517 time frame stds + 517 medians
    full_percussive_features += [f"FULL_percussive_std_frame_{i+1}" for i in range(517)]
    full_percussive_features += [f"FULL_percussive_median_frame_{i+1}" for i in range(517)]

    ordered_feature_names = (
            quarter_feature_names +
            full_harmonic_features +
            full_percussive_features
    )

    return ordered_feature_names


def generate_dwt_stats_names():
    """
    Returns: Generates all the statistics names (classifier features names) for the dwt classifiers
    """
    max_level = 3
    coeff_types = ["approx"] + [f"detail_{i}" for i in range(max_level, 0, -1)]
    quarters = [f"Q{i + 1}" for i in range(4)]

    base_stats = [
        "max", "min", "std", "median", "iqr",  # 5 base stats
        "mean_delta", "mean_delta2", "std_delta", "std_delta2", "zcr"  # 5 dynamics
    ]

    autocorr_stats = ["autocorr_max", "autocorr_min", "autocorr_mean", "autocorr_std"]

    feature_names = []

    # Quarter-based features
    for q in quarters:
        for coeff in coeff_types:
            for stat in base_stats:
                feature_names.append(f"{q}_{coeff}_{stat}")
            for stat in autocorr_stats:
                feature_names.append(f"{q}_{coeff}_{stat}")
            for i in range(50):
                feature_names.append(f"{q}_{coeff}_autocorr_lag{i+1}")

    # Full segment features
    for coeff in coeff_types:
        for stat in base_stats:
            feature_names.append(f"FULL_{coeff}_{stat}")
        for stat in autocorr_stats:
            feature_names.append(f"FULL_{coeff}_{stat}")
        for i in range(50):
            feature_names.append(f"FULL_{coeff}_autocorr_lag{i+1}")

    return feature_names


# Mapping feature type to classifier features names
generate_classifier_feature_names = {
    "mfcc": generate_mfcc_stats_names,
    "harmonic": generate_harmonic_stats_names,
    "dwt": generate_dwt_stats_names
}


def plot_mfcc_shap(explainer, shap_values, X_test_selected, supp):
    """
        Plots the shap values of the classifier features of mfcc, given the data

    Args:
        explainer: explainer object for the data to plot
        shap_values: shap values to plot
        X_test_selected: test samples
        supp: the support mask (to get the features we actually use), from selector.get_support()

    """
    all_feature_names = generate_mfcc_stats_names()
    selected_feature_names = [name for name, keep in zip(all_feature_names, supp) if keep]

    max_conf_sample = max(
        range(X_test_selected.shape[0]),
        key=lambda i: max(
            explainer.expected_value[c] + shap_values[i, :, c].sum()
            for c in range(6)
        )
    )

    probs = [
        explainer.expected_value[c] + shap_values[max_conf_sample, :, c].sum()
        for c in range(6)
    ]
    for i, p in enumerate(probs):
        print(f"Class {movement_names[i]}: prob = {p * 100:.2f}%")

    move_idx = max_conf_sample
    label_transform = np.argmax(probs)
    label_name = movement_names[label_transform]

    # Create DataFrame
    df = pd.DataFrame({
        "Feature Name": selected_feature_names,
        f"Value- ascent": X_test_selected[move_idx],
    })

    sample_named = pd.Series(X_test_selected[move_idx], index=selected_feature_names)

    explanation = shap.Explanation(
        values=shap_values[move_idx, :, label_transform],
        base_values=explainer.expected_value[label_transform],
        data=sample_named.values,
        feature_names=sample_named.index.tolist()
    )

    # Plot waterfall
    shap.plots.waterfall(explanation, show=False)
    plt.xlabel("SHAP Value")
    plt.title(f"SHAP-Stats: predicted class {label_name}")
    plt.tight_layout()
    plt.savefig(f"stats_predicted_{label_name}.png")
    plt.close()


def plot_mfcc_shap_confusion(explainer, shap_values, X_test_selected, supp, label1, label2, move1_idx, move2_idx):
    """
        Plots the shap values for the classifier features of two samples that were confused for one another

    Args:
        explainer: SHAP explainer object
        shap_values: SHAP values (for the X_test_selected data)
        X_test_selected: data on which the SHAP was calculated
        supp: support mask for the selected features, from selector.get_support()
        label1: true label of the first sample
        label2: true label of the second sample
        move1_idx: index of the first sample in the test set
        move2_idx: index of the second sample in the test set

    """

    label1_transform = movement_names.index(label1)
    label2_transform = movement_names.index(label2)

    all_feature_names = generate_mfcc_stats_names()
    selected_feature_names = [name for name, keep in zip(all_feature_names, supp) if keep]

    # Create DataFrame
    df = pd.DataFrame({
        "Feature Name": selected_feature_names,
        f"Value- {label1}": X_test_selected[move1_idx],
        f"Value- {label2}": X_test_selected[move2_idx]
    })

    df.to_csv(f"selected_stats_{label1} vs {label2}.csv", index=False)

    sample_named = pd.Series(X_test_selected[move1_idx], index=selected_feature_names)

    shap_label_transform = [label2_transform, label1_transform, label1_transform, label2_transform]
    shap_move_idx = [move1_idx, move1_idx, move2_idx, move2_idx]

    for idx in range(4):
        move_idx = shap_move_idx[idx]
        label_transform = shap_label_transform[idx]

        if idx % 2 == 0:
            label_checked = label1 if idx == 0 else label2
            print(f"For sample {label_checked}:")
            probs = [
                explainer.expected_value[c] + shap_values[move_idx, :, c].sum()
                for c in range(6)
            ]
            for i, p in enumerate(probs):
                print(f"Class {movement_names[i]}: prob = {p * 100:.2f}%")

        explanation = shap.Explanation(
            values=shap_values[move_idx, :, label_transform],
            base_values=explainer.expected_value[label_transform],
            data=sample_named.values,
            feature_names=sample_named.index.tolist()
        )

        shap.plots.waterfall(explanation, show=False)
        plt.xlabel("SHAP Value")

        match idx:
            case 0:
                plt.title(f"SHAP-Stats, {label1}: predicted class {label2} (Incorrect)")
                plt.tight_layout()
                plt.savefig(f"stats_predicted_{label1}_incorrect_{label2}.png")
            case 1:
                plt.title(f"SHAP-Stats, {label1}: predicted class {label1} (Correct)")
                plt.tight_layout()
                plt.savefig(f"stats_predicted_{label1}_correct_{label1}.png")
            case 2:
                plt.title(f"SHAP-Stats, {label2}: predicted class {label1} (Incorrect)")
                plt.tight_layout()
                plt.savefig(f"stats_predicted_{label2}_incorrect_{label1}.png")
            case 3:
                plt.title(f"SHAP-Stats, {label2}: predicted class {label2} (Correct)")
                plt.tight_layout()
                plt.savefig(f"stats_predicted_{label2}_correct_{label2}.png")
        plt.close()


def apply_mask(feature_type, mask):
    """
        Applies a given mask to the classifier features names of feature_type

    Args:
        feature_type: feature type name
        mask: either an index mask (support) or boolean mask

    Returns: the selected classifier features names

    """
    all_names = generate_classifier_feature_names[feature_type]()
    length = len(all_names)

    if mask.dtype == bool:
        # Already a boolean mask — validate length
        if len(mask) != length:
            raise ValueError(f"Boolean mask length {len(mask)} doesn't match expected length {length}")
        bool_mask = mask
    else:
        # Assume it"s an index mask — convert to boolean
        bool_mask = np.zeros(length, dtype=bool)
        if np.any(mask >= length) or np.any(mask < 0):
            raise IndexError(f"Index mask contains values outside valid range 0 to {length - 1}")
        bool_mask[mask] = True
    return [name for name, keep in zip(all_names, bool_mask) if keep]


def print_importances(mask, feature_type, prediction_name, top_x=15):
    """
        Prints the top_x most important classifier features names

    Args:
        mask: either an index mask (support) or boolean mask of clf features
        feature_type:
        top_x: number of classifier features to print

    """

    top_indices = mask[:15]
    top_names = apply_mask(feature_type, top_indices)

    with open(prediction_name, "a") as f:
        print(f"Top {top_x} most important clf features (globally): ", file=f)
        for i, name in enumerate(top_names):
            print(f"{i + 1}) {name}", file=f)


def plot_estimate_importances(clf, X_train, mask, feature_type, clf_type, fold_num):
    """
        Tries to estimate and plot the importances of the classifier features from the classifier (Random Forest)

    Args:
        clf: Random Forest classifier data
        X_train: train set
        mask: either an index mask (support) or boolean mask
        feature_type: feature type name
        clf_type: classifier type name
        fold_num: number of the current fold (cross-validation)

    """
    num_classes = len(clf.classes_)
    num_features = X_train.shape[1]
    class_feature_importances = np.zeros((num_classes, num_features))

    for estimator in clf.estimators_:
        tree = estimator.tree_
        for node in range(tree.node_count):
            if tree.children_left[node] != tree.children_right[node]:  # check it"s not a leaf
                feature = tree.feature[node]
                samples = tree.n_node_samples[node]
                class_distribution = tree.value[node][0]
                class_probs = class_distribution / class_distribution.sum()
                class_feature_importances[:, feature] += (samples * class_probs)

    # Normalize
    class_feature_importances = class_feature_importances / class_feature_importances.sum(axis=1, keepdims=True)
    selected_feature_names = apply_mask(feature_type, mask)
    top_x = 10  # Set this to the number of top features you want to display

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    for c in range(num_classes):
        # Get the top "x" features for the current class
        # Sort the mean absolute SHAP values for class "c" and get the indices of the top "x"
        top_indices = np.argsort(class_feature_importances[c, :])[-top_x:][::-1]

        # Get the names and values of the top features
        top_feature_names = [selected_feature_names[i] for i in top_indices]
        top_feature_values = class_feature_importances[c, top_indices]

        row, col = divmod(c, 3)
        ax = axes[row, col]

        # Plot the top "x" features for this class
        ax.barh(top_feature_names, top_feature_values, color="skyblue")
        ax.set_title(f"Top {top_x} Features for Class {movement_names[c]}")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
        ax.set_xlabel("Mean |value|")
        ax.set_ylabel("Features")
        ax.invert_yaxis()  # So the highest bar is on top

    plot_folder_path = Path(project_root / f"plots/final/{feature_type}/feature_selection/{clf_type}")
    plot_folder_path.mkdir(parents=True, exist_ok=True)

    fig.suptitle(f"Top {top_x} Features for {feature_type} {clf_type} fold {fold_num}", fontsize=20, y=1.05)
    fig.savefig(plot_folder_path / f"estimate_importance_fold_{fold_num}.png", bbox_inches="tight", pad_inches=1)
    plt.close()


def plot_shap_importances(shap_values, mask, feature_type, clf_type, fold_num, top_x=10):
    """
         Plots the top_x shap values for the classifier features over all samples (global importances per class)

    Args:
        shap_values: shap values to plot
        mask: either an index mask (support) or boolean mask
        feature_type: feature type name
        clf_type: classifier type name
        fold_num: number of the current fold (cross-validation)
        top_x:  number of classifier features to plot

    """
    mean_abs_shap = np.array([
        np.abs(shap_values[:, :, c]).mean(axis=0)  # Mean absolute SHAP per feature for class "c"
        for c in range(shap_values.shape[2])
    ])
    selected_feature_names = apply_mask(feature_type, mask)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    for c in range(shap_values.shape[2]):
        # Get the top "x" features for the current class
        # Sort the mean absolute SHAP values for class "c" and get the indices of the top "x"
        top_indices = np.argsort(mean_abs_shap[c, :])[-top_x:][::-1]

        # Get the names and values of the top features
        top_feature_names = [selected_feature_names[i] for i in top_indices]
        top_feature_values = mean_abs_shap[c, top_indices]

        row, col = divmod(c, 3)
        ax = axes[row, col]

        # Plot the top "x" features for this class
        ax.barh(top_feature_names, top_feature_values, color="skyblue")
        ax.set_title(f"Top {top_x} Features for Class {movement_names[c]}")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_ylabel("Features")
        ax.invert_yaxis()  # So the highest bar is on top

    plot_folder_path = Path(project_root / f"plots/final/{feature_type}/feature_selection/{clf_type}")
    plot_folder_path.mkdir(parents=True, exist_ok=True)

    fig.suptitle(f"Top {top_x} Features for {feature_type} {clf_type} fold {fold_num} (SHAP)", fontsize=20, y=1.05)
    fig.savefig(plot_folder_path / f"shap_importance_fold_{fold_num}.png", bbox_inches="tight", pad_inches=1)
    plt.close()


def plot_recordings():
    """

    Returns: Plots the real recordings results from the logged csv files.

    """
    for file in mydir.rglob("*.csv"):
        df = pd.read_csv(file)

        x = df.iloc[:, 0].values  # x is in the first column
        z = df.iloc[:, 1].values  # z is in the second column
        t = df.iloc[:, 2].values  # t is in the third column
        lab = df.iloc[:, 3].values  # label is in the fourth column

        label_colors = {
            "ascent": "pink",
            "hover": "red",
            "towards": "blue",
            "turn": "green",
            "away": "yellow",
            "descent": "orange"
        }

        # Create a 3D plot for x, t, z
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot3D(x, t, z, "green")
        ax.set_xlabel("x", labelpad=20)
        ax.set_ylabel("t", labelpad=20)
        ax.set_zlabel("z", labelpad=20)

        # Save the 3D plot as a PNG
        path_x_z = file.with_name(f"{file.stem}_path_x_z.png")
        plt.savefig(path_x_z)
        plt.close()

        # Create a 2D scatter plot for t vs x, colored by label
        fig, ax = plt.subplots()
        line_seg_x = {}

        for label in np.unique(lab):
            indices = np.where(lab == label)
            line_seg_x[label] = {"t": t[indices], "x": x[indices]}

        for label, segment in line_seg_x.items():
            ax.scatter(segment["t"], segment["x"], color=label_colors.get(label, "black"), s=2, label=label)
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

        ax.set_xlabel("t")
        ax.set_ylabel("x")

        # Save the t vs x plot as a PNG
        plt.tight_layout()
        path_x = file.with_name(f"{file.stem}_path_x.png")
        plt.savefig(path_x)
        plt.close()

        # Create a 2D plot for t vs z
        fig, ax = plt.subplots()
        ax.plot(t, z, "green")
        ax.set_xlabel("t", labelpad=20)
        ax.set_ylabel("z", labelpad=20)

        # Save the t vs z plot as a PNG
        path_z = file.with_name(f"{file.stem}_path_z.png")
        plt.savefig(path_z)
        plt.close()


def plot_predicted():
    """

    Returns: Plots a comparison between the predicted movement (saved in csv files) and the real logged movement.

    """

    # For every prediction in plots folder, match it to the recording in the recordings folder.
    predicted_dir = Path(project_root/"plots/final")
    for file in predicted_dir.rglob("*.csv"):
        file_name = file.stem.split("_")[0]
        if "selected" in file_name:
            continue
        real = None
        for file2 in mydir.rglob("*.csv"):
            if file2.stem.split("_")[0] == file_name:
                real = file2
                break
        if real is None:
            raise ValueError("Error: File not found")

        # Prediction
        df1 = pd.read_csv(file)
        x1 = df1.iloc[:, 0].values  # x is in the first column
        z1 = df1.iloc[:, 1].values  # z is in the second column
        t1 = df1.iloc[:, 2].values  # t is in the third column
        lab1 = df1.iloc[:, 3].values  # label is in the fourth column

        # Real
        df2 = pd.read_csv(real)
        x2 = df2.iloc[:, 0].values  # x is in the first column
        z2 = df2.iloc[:, 1].values  # z is in the second column
        t2 = df2.iloc[:, 2].values  # t is in the third column
        lab2 = df2.iloc[:, 3].values  # label is in the fourth column

        # Colors for legend
        label_colors = {
            "ascent": "pink",
            "hover": "red",
            "towards": "blue",
            "turn": "green",
            "away": "yellow",
            "descent": "orange"
        }

        # Start from 0
        if t1[0] != 0:
            t1 = np.insert(t1, 0, 0)
            x1 = np.insert(x1, 0, 0)
            z1 = np.insert(z1, 0, 0)
            lab1 = np.insert(lab1, 0, lab1[0])

        # Create a 3D plot for x, t, z
        fig = plt.figure(figsize=(8, 16))
        ax1 = fig.add_subplot(2, 1, 1, projection="3d")
        ax1.plot3D(x1, t1, z1, "green")
        ax1.set_xlabel("x (m)", labelpad=20)
        ax1.set_ylabel("t (s)", labelpad=20)
        ax1.set_zlabel("z (m)", labelpad=20)
        ax1.set_title("Prediction: X-Z axes movement over time")

        ax2 = fig.add_subplot(2, 1, 2, projection="3d")
        ax2.plot3D(x2, t2, z2, "green")
        ax2.set_xlabel("x (m)", labelpad=20)
        ax2.set_ylabel("t (s)", labelpad=20)
        ax2.set_zlabel("z (m)", labelpad=20)
        ax2.set_title("Real: X-Z axes movement over time")

        # Save the 3D plot as a PNG
        plt.tight_layout()
        path_x_z = file.with_name(f"{file.stem}_path_x_z.png")
        plt.savefig(path_x_z)
        plt.close()

        # Create a 2D scatter plot for t vs x, colored by label
        fig, (ax1, ax2) = plt.subplots(2, 1)
        line_seg_x = {}

        for i in range(len(t1) - 1):
            x_segment = [t1[i], t1[i + 1]]  # Time axis
            y_segment = [x1[i], x1[i + 1]]  # X-axis values
            color = label_colors.get(lab1[i + 1], "black")  # Default to black if label is missing
            ax1.plot(x_segment, y_segment, color=color)

            # Create a legend
            legend_patches = [mpatches.Patch(color=color, label=label) for label, color in label_colors.items()]
            ax1.legend(handles=legend_patches, title="Task Labels", loc="upper left", bbox_to_anchor=(1, 1))
            ax1.set_xlabel("t (s)")
            ax1.set_ylabel("x (m)")

        for label in np.unique(lab2):
            indices = np.where(lab2 == label)
            line_seg_x[label] = {"t": t2[indices], "x": x2[indices]}

        ax1.set_title("Prediction: X-axis movement over time")

        for label, segment in line_seg_x.items():
            ax2.scatter(segment["t"], segment["x"], color=label_colors.get(label, "black"), s=2, label=label)

        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("x (m)")
        ax2.set_title("Real: X-axis movement over time")

        # Save the t vs x plot as a PNG
        plt.tight_layout()
        path_x = file.with_name(f"{file.stem}_path_x.png")
        plt.savefig(path_x)
        plt.close()

        # Create a 2D plot for t vs z
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(t1, z1, "green")
        ax1.set_xlabel("t (s)", labelpad=20)
        ax1.set_ylabel("z (m)", labelpad=20)
        ax1.set_title("Prediction: Z-axis movement over time")

        ax2.plot(t2, z2, "green")
        ax2.set_xlabel("t (s)", labelpad=20)
        ax2.set_ylabel("z (m)", labelpad=20)
        ax2.set_title("Real: Z-axis movement over time")

        # Save the t vs z plot as a PNG
        plt.tight_layout()
        path_z = file.with_name(f"{file.stem}_path_z.png")
        plt.savefig(path_z)
        plt.close()


def plot_cms(all_cm, label_encoder, feature_type, clf_type):
    """
         Plot and save all confusion matrices

    Args:
        all_cm: vector of all cms
        label_encoder: the label encoder (for classes info)
        feature_type: feature type name
        clf_type: classifier type name

    """

    # Create dir to save plots
    cm_folder_path = Path(project_root / "plots/final/cms")
    cm_folder_path.mkdir(parents=True, exist_ok=True)

    # Plot all confusion matrices as subplots
    fig, axes = plt.subplots(1, len(all_cm), figsize=(20, 5))
    for i, cm in enumerate(all_cm):
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_, ax=ax)
        ax.set_title(f"Fold {i + 1}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle(f"Confusion Matrices for {feature_type} - Classifier {clf_type}")
    plt.tight_layout()
    plt.savefig(cm_folder_path / f"{feature_type}_{clf_type}_cms.png")
    plt.close()


if __name__ == "__main__":
    start = time.time()

    mydir = Path(project_root / "recordings")

    # Plot all real results:
    #plot_recordings()

    # Plot all prediction results:
    # plot_predicted()

    print("----Elapsed time: %.2f seconds----" % (time.time() - start))
