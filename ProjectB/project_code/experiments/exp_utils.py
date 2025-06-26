''' This file contains code to define plot and statistics functions, as well as other variables
    used in the experiments. '''


from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pywt
import seaborn as sns
from librosa.feature import delta
from scipy.signal import correlate

# List of drone movement names
movement_names = ["ascent", "away", "descent", "hover", "towards", "turn"]

# List of features
features_list = ["mfcc", "harmonic", "dwt"]

# List of classifiers
classifiers_list = ["random forest", "svm rbf", "mlp"]

# Segment number per movement type.
# Used in Experiments 1 and 2.1 only!
HOVER = 5
TURN = 5
ASCENT_DESCENT = 9
TOWARDS_AWAY = 12

# Map movement type to segment length
movement_types = {
    "hover": HOVER,
    "turn": TURN,
    "ascent": ASCENT_DESCENT,
    "descent": ASCENT_DESCENT,
    "towards": TOWARDS_AWAY,
    "away": TOWARDS_AWAY
}

MAX_HOVER = 30

# Flag dictionary (to track feature vectors plot)
# First flag: statistics vector
# Second flag: feature vector
flags = {
    "hover": [0, 0],
    "turn": [0, 0],
    "ascent": [0, 0],
    "descent": [0, 0],
    "towards": [0, 0],
    "away": [0, 0]
}

# Get the correct path, regardless of working dir
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent

OLD_DATA = False   # Before split alteration, for experiments 1-2.1
NEW_DATA = True    # After split alteration, for experiments 2.2-4


def calculate_mfcc_quarter_statistics(segment, sr=44100, num_coef=13):
    """
    Function to extract mfcc features.
    Statistics are calculated over quarters of the data (maintaining temporal contex),
     and also over the full segment.

    Args:
        segment: data per segment (1 second)
        sr: sampling rate
        num_coef: number of coefficients

    Returns:
        Statistics vector of the mfcc features of this segment.
    """

    # Compute the MFCC for the entire segment
    # shape: (13,87) = (number of coefficients, time frames)
    # time frames are calculated by 44100/512, where sr=44100, and mfcc default num_hops=512
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=num_coef)

    # Number of time frames in the MFCC array
    num_frames = mfcc.shape[1]

    # Define quarter lengths for time frames
    quarter_len = num_frames // 4

    # Create a list to hold the statistics for each quarter
    segment_features = []

    # Calculate statistics over each quarter
    # Result: list sized 4:
    #         i=0,1,2: 222 values each (9*13=117 time stats, 5*21=105 coefficients stats)
    #         i=3: 237 values (9*13=117 time stats, 5*24=120 coefficients stats)
    for i in range(4):
        start_idx = i * quarter_len
        end_idx = (i + 1) * quarter_len if i < 3 else num_frames

        # shape: (13,21), last quarter: (13,24)
        quarter_mfcc = mfcc[:, start_idx:end_idx]
        # Delta and delta-delta (quarter_mfcc)
        delta_q_mfcc = delta(quarter_mfcc)
        delta2_q_mfcc = delta(quarter_mfcc, order=2)

        segment_features.append(np.hstack([
            # Calculate statistics over time frames (axis=1)
            np.max(quarter_mfcc, axis=1), np.min(quarter_mfcc, axis=1),
            np.std(quarter_mfcc, axis=1),
            np.median(quarter_mfcc, axis=1),
            np.percentile(quarter_mfcc, 75, axis=1) - np.percentile(quarter_mfcc, 25, axis=1),  # IQR
            np.mean(delta_q_mfcc, axis=1), np.mean(delta2_q_mfcc, axis=1),
            np.std(delta_q_mfcc, axis=1), np.std(delta2_q_mfcc, axis=1),

            # Calculate statistics over coefficients (axis=0)
            np.max(quarter_mfcc, axis=0), np.min(quarter_mfcc, axis=0),
            np.std(quarter_mfcc, axis=0),
            np.median(quarter_mfcc, axis=0),
            np.percentile(quarter_mfcc, 75, axis=0) - np.percentile(quarter_mfcc, 25, axis=0),
        ]))

    # Delta and delta-delta (mfcc)
    delta_mfcc = delta(mfcc)
    delta2_mfcc = delta(mfcc, order=2)

    # Compute overall statistics
    # Result: nparray of 552 values (9*13=117 time stats, 5*87=435 coefficients stats)
    full_features = np.hstack([
        np.max(mfcc, axis=1), np.min(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.median(mfcc, axis=1), np.percentile(mfcc, 75, axis=1) - np.percentile(mfcc, 25, axis=1),
        np.mean(delta_mfcc, axis=1), np.mean(delta2_mfcc, axis=1),
        np.std(delta_mfcc, axis=1), np.std(delta2_mfcc, axis=1),

        np.max(mfcc, axis=0), np.min(mfcc, axis=0),
        np.std(mfcc, axis=0),
        np.median(mfcc, axis=0), np.percentile(mfcc, 75, axis=0) - np.percentile(mfcc, 25, axis=0),
    ])

    # Combine quarter-based and full-segment features
    # Result: nparray of size 1455 (222*3+237+552)
    return np.hstack(segment_features + [full_features])


def calculate_harmonic_quarter_statistics(segment, sr=44100):
    """
    Function to extract harmonic and percussive features.
    Statistics are calculated over quarters of the data (maintaining temporal contex),
     and also over the full segment.

    Args:
        segment: data per segment (1 second)
        sr: sampling rate

    Returns:
        Statistics vector of the harmonic and percussive features of this segment.
    """

    # Compute the STFT for the entire segment
    # We set n_fft=512, hop_length=256
    D = librosa.stft(segment, n_fft=512, hop_length=256)

    # Apply Harmonic-Percussive Source Separation (HPSS)
    # shape: (257, 517) = (frequency bins, time frames)
    # frequency bins are calculated by (n_fft // 2) + 1 (we set n_fft=512)
    # time frames are calculated by 44100/256, where sr=44100, and we set hop_length=256
    harmonic_mag, percussive_mag = librosa.decompose.hpss(np.abs(D))

    # Number of time frames in the harmonic and percussive magnitude spectrograms
    num_frames = harmonic_mag.shape[1]

    # Define quarter lengths for time frames
    quarter_len = num_frames // 4

    # Create a list to hold the statistics for each quarter
    segment_features = []

    # Loop through the four quarters
    for i in range(4):
        start_idx = i * quarter_len
        end_idx = (i + 1) * quarter_len if i < 3 else num_frames

        # Extract the harmonic and percussive magnitudes for this quarter
        # shape: (257,43)
        quarter_harmonic_mag = harmonic_mag[:, start_idx:end_idx]
        quarter_percussive_mag = percussive_mag[:, start_idx:end_idx]

        # Harmonic Features
        quarter_harmonic_centroid = librosa.feature.spectral_centroid(S=quarter_harmonic_mag, sr=sr)
        quarter_harmonic_bandwidth = librosa.feature.spectral_bandwidth(S=quarter_harmonic_mag, sr=sr)
        quarter_harmonic_rolloff = librosa.feature.spectral_rolloff(S=quarter_harmonic_mag, sr=sr)
        # Delta and delta-delta (harmonic_centroid)
        delta_quarter_harmonic_centroid = delta(quarter_harmonic_centroid)
        delta2_quarter_harmonic_centroid = delta(quarter_harmonic_centroid, order=2)

        # Percussive Features
        quarter_percussive_onset = librosa.onset.onset_strength(S=quarter_percussive_mag, sr=sr)
        quarter_percussive_contrast = librosa.feature.spectral_contrast(S=quarter_percussive_mag, sr=sr)

        # Calculate statistics over time frames (axis=1) and frequency bins (axis=0)
        # Harmonic Statistics
        # size: 16
        harmonic_stats = np.hstack([
            np.max(quarter_harmonic_centroid), np.min(quarter_harmonic_centroid),
            np.std(quarter_harmonic_centroid), np.median(quarter_harmonic_centroid),
            np.max(quarter_harmonic_bandwidth), np.min(quarter_harmonic_bandwidth),
            np.std(quarter_harmonic_bandwidth), np.median(quarter_harmonic_bandwidth),
            np.max(quarter_harmonic_rolloff), np.min(quarter_harmonic_rolloff),
            np.std(quarter_harmonic_rolloff), np.median(quarter_harmonic_rolloff),
            np.mean(delta_quarter_harmonic_centroid), np.std(delta_quarter_harmonic_centroid),
            np.mean(delta2_quarter_harmonic_centroid), np.std(delta2_quarter_harmonic_centroid)
        ])

        # Percussive Statistics
        # size: 32 (4 + 4*7)
        percussive_stats = np.hstack([
            np.max(quarter_percussive_onset), np.min(quarter_percussive_onset),
            np.std(quarter_percussive_onset), np.median(quarter_percussive_onset),
            np.max(quarter_percussive_contrast, axis=1), np.min(quarter_percussive_contrast, axis=1),
            np.std(quarter_percussive_contrast, axis=1), np.median(quarter_percussive_contrast, axis=1),
        ])

        # Combine harmonic and percussive statistics for the current quarter
        segment_features.append(np.hstack([harmonic_stats, percussive_stats]))

    # Harmonic Statistics
    harmonic_centroid = librosa.feature.spectral_centroid(S=harmonic_mag, sr=sr)
    harmonic_bandwidth = librosa.feature.spectral_bandwidth(S=harmonic_mag, sr=sr)
    harmonic_rolloff = librosa.feature.spectral_rolloff(S=harmonic_mag, sr=sr)
    # Delta and delta-delta (harmonic_centroid)
    delta_harmonic_centroid = delta(harmonic_centroid)
    delta2_harmonic_centroid = delta(harmonic_centroid, order=2)

    # Percussive Features
    percussive_onset = librosa.onset.onset_strength(S=percussive_mag, sr=sr)
    percussive_contrast = librosa.feature.spectral_contrast(S=percussive_mag, sr=sr)

    # Compute overall statistics for the entire segment (harmonic and percussive)
    # size: 876 (16 + 2*257 + 2*173)
    full_harmonic_features = np.hstack([
        np.max(harmonic_centroid), np.min(harmonic_centroid),
        np.std(harmonic_centroid), np.median(harmonic_centroid),
        np.max(harmonic_bandwidth), np.min(harmonic_bandwidth),
        np.std(harmonic_bandwidth), np.median(harmonic_bandwidth),
        np.max(harmonic_rolloff), np.min(harmonic_rolloff),
        np.std(harmonic_rolloff), np.median(harmonic_rolloff),
        np.mean(delta_harmonic_centroid), np.std(delta_harmonic_centroid),
        np.mean(delta2_harmonic_centroid), np.std(delta2_harmonic_centroid),
        np.std(harmonic_mag, axis=1), np.median(harmonic_mag, axis=1),
        np.std(harmonic_mag, axis=0), np.median(harmonic_mag, axis=0)
    ])

    # size: 892 (32 + 2*257 + 2*173)
    full_percussive_features = np.hstack([
        np.max(percussive_onset), np.min(percussive_onset),
        np.std(percussive_onset), np.median(percussive_onset),
        np.max(percussive_contrast, axis=1), np.min(percussive_contrast, axis=1),
        np.std(percussive_contrast, axis=1), np.median(percussive_contrast, axis=1),
        np.std(percussive_mag, axis=1), np.median(percussive_mag, axis=1),
        np.std(percussive_mag, axis=0), np.median(percussive_mag, axis=0)
    ])

    # Combine quarter-based and full-segment features
    # size: 1960 (4*48 + 876 + 892)
    return np.hstack(segment_features + [full_harmonic_features, full_percussive_features])


def autocorrelation_stats(autocorr):
    """
        Calculate statistics for the autocorrelation vector

    Args:
        autocorr: autocorrelation vector

    Returns: vector of the stats: max,min,mean,std and first 50 lags

    """
    max_autocorr = np.max(autocorr)
    min_autocorr = np.min(autocorr)
    mean_autocorr = np.mean(autocorr)
    std_autocorr = np.std(autocorr)

    # Peak locations (ignoring lag 0)
    # peak_lags = np.where(autocorr == max_autocorr)[0]

    # Additional lags
    autocorr_lags = autocorr[:50]
    # size: 54 (4 + 50)
    return [max_autocorr, min_autocorr, mean_autocorr, std_autocorr] + list(autocorr_lags)


def calculate_dwt_quarter_statistics(segment, wavelet="coif5", max_level=3):
    """
    Function to extract dwt features.
    Statistics are calculated over quarters of the data (maintaining temporal contex),
     and also over the full segment.

    Args:
        segment: data per segment (1 second)
        wavelet: wavelet type
        max_level: maximum level of detail

    Returns:
        Statistics vector of the dwt features of this segment.
    """

    # Compute DWT for all levels and store coefficients
    # (level 3) approx- size: 16562
    # level 1 detail- size: 66164
    # level 2 detail- size: 33096
    # level 3 detail- size: 16562
    # size: 4- list of approx, detail3, detail2, detail1
    coeffs = pywt.wavedec(segment, wavelet, level=max_level)

    quarter_features = [[] for _ in range(4)]

    segment_features = []

    # For each quarter, calculate the statistics
    # size: 256 per quarter (4*(10 + 54))
    for coeff in coeffs:
        num_frames = len(coeff)
        quarter_len = num_frames // 4
        for i in range(4):
            start_idx = i * quarter_len
            end_idx = (i + 1) * quarter_len if i < 3 else num_frames

            # Get the segment of the approximation and detail coefficients for the current quarter
            quarter_coeff = coeff[start_idx:end_idx]

            # Compute delta and delta-delta for each quarter
            delta_q_coeff = delta(quarter_coeff)
            delta2_q_coeff = delta(delta_q_coeff, order=2)

            # Zero-Crossing Rate
            zcr_q_coeff = np.mean(librosa.feature.zero_crossing_rate(quarter_coeff))

            # Autocorrelation (positive lags only)
            autocorr_q_coeff = correlate(quarter_coeff, quarter_coeff, mode="full")[len(quarter_coeff) - 1:]

            # Append extracted features (approx/ detail)
            quarter_features[i].extend([
                # Approximation Coefficients Statistics (quarter)
                np.max(quarter_coeff), np.min(quarter_coeff), np.std(quarter_coeff),
                np.median(quarter_coeff), np.percentile(quarter_coeff, 75) - np.percentile(quarter_coeff, 25),
                np.mean(delta_q_coeff), np.mean(delta2_q_coeff), np.std(delta_q_coeff), np.std(delta2_q_coeff),
                zcr_q_coeff
            ])

            quarter_features[i].extend(np.hstack(autocorrelation_stats(autocorr_q_coeff)))

    segment_features.append(np.hstack(quarter_features))

    # Compute full-segment statistics for each level
    # size: 256 (4*(10 + 54))
    full_features = []
    for coeff in coeffs:
        # Compute delta and delta-delta for whole segment
        delta_coeff = delta(coeff)
        delta2_coeff = delta(delta_coeff, order=2)

        # Zero-Crossing Rate
        zcr_coeff = np.mean(librosa.feature.zero_crossing_rate(coeff))

        # Calculate autocorrelation (only positive lags)
        autocorr_coeff = correlate(coeff, coeff, mode="full")[len(coeff) - 1:]

        # Calculate overall statistics for the entire segment (approximation and detail coefficients)
        full_features.append(np.hstack([
            # Detail Coefficients Statistics (overall)
            np.max(coeff), np.min(coeff), np.std(coeff), np.median(coeff),
            np.percentile(coeff, 75) - np.percentile(coeff, 25), np.mean(delta_coeff), np.mean(delta2_coeff),
            np.std(delta_coeff), np.std(delta2_coeff), zcr_coeff
        ]))

        full_features.append(np.hstack(autocorrelation_stats(autocorr_coeff)))

    # Combine quarter-based and full-segment features
    # size: 1280 (5*256)
    return np.hstack(segment_features + [np.hstack(full_features)])


# Mapping feature name to feature function
feature_functions = {
    "mfcc": calculate_mfcc_quarter_statistics,
    "harmonic": calculate_harmonic_quarter_statistics,
    "dwt": calculate_dwt_quarter_statistics
}


def get_features(feature_type, audio_path, label, sample_rate=44100, segment_length=3.0):
    """
        Get the feature statistics for this recording.

    Args:
        feature_type: feature type
        audio_path: path to this recording
        label: label of this recording
        sample_rate: sample rate (44100)
        segment_length: length of each segment (1 second)

    Returns:
        Statistics vector of this recording, per the selected feature type.
    """

    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    segment_samples = int(sr * segment_length)

    # Step size for 75% overlap
    step_size = int(sr * 0.75)

    # Round up to the nearest whole number of segments
    pad_length = (len(audio) + segment_samples - 1) // step_size * step_size

    # Pad the last segment if needed
    audio = np.pad(audio, (0, max(0, pad_length - len(audio))), mode="edge")

    # Calculate the number of segments (without counting partial segments)
    total_segments = (len(audio) - segment_samples) // step_size + 1
    if label == "hover":
        num_segments = min(MAX_HOVER, total_segments)
    else:
        num_segments = total_segments

    segments = [audio[i * step_size: i * step_size + segment_samples] for i in range(num_segments)]

    # Get the feature function from the dictionary
    feature_func = feature_functions[feature_type]

    # Extract and aggregate features per segment
    segment_features = [feature_func(segment) for segment in segments]

    # nparray of shape: (NUM_SEGMENTS, NUM_STATS)
    return segment_features


def paths_and_labels(flag_new=OLD_DATA):
    """
        Scans the database and returns the tuple (audio_files, labels, recording_ids) where:
        audio_files: list of file paths of every recording.

        labels: list of the label of every recording (out of 6 possible movement types).

        recording_ids: maps filename to id numbers for every recording
            (for GroupKFold, and for full recording regression)

    Args:
        flag_new: old or new split dataset
    """

    if flag_new == NEW_DATA:
        path = Path(project_root/"split/new")
    else:
        path = Path(project_root/"split/old")
    audio_files = []
    labels = []
    recording_ids = {}  # Maps filename prefix to a unique ID
    unique_id = 0

    for label_folder in path.iterdir():
        if label_folder.is_dir():  # movement types
            for subfolder in label_folder.iterdir():
                if subfolder.is_dir():
                    for file in subfolder.rglob("*.wav"):
                        audio_files.append(file)
                        labels.append(label_folder.name)
                        # Assign a unique ID per original recording (without segment number)
                        recording_name = file.stem.split("_")[0]  # Extract original recording name
                        if recording_name not in recording_ids:
                            recording_ids[recording_name] = unique_id
                            unique_id += 1

    return audio_files, labels, recording_ids


# ------Plot functions------


def plot_mfcc_stats(feature_type, exp_name, data, label):
    """
        Plot mfcc statistics

    Args:
        data: statistics vector to plot
        label: label for this statistics vector

    """
    stats_path = Path(project_root / f"plots/{exp_name}/{feature_type}/stats/")
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
        br = np.arange(1, len(coeffs) + 1) * (bar_width * 5 + spacing)

        plt.bar(br, feature_stats[0, :], color='r', width=bar_width,
                edgecolor='grey', label='Q1')
        plt.bar(br + bar_width, feature_stats[1, :], color='g', width=bar_width,
                edgecolor='grey', label='Q2')
        plt.bar(br + 2 * bar_width, feature_stats[2, :], color='b', width=bar_width,
                edgecolor='grey', label='Q3')
        plt.bar(br + 3 * bar_width, feature_stats[3, :], color='c', width=bar_width,
                edgecolor='grey', label='Q4')
        plt.bar(br + 4 * bar_width, feature_stats[4, :], color='m',
                width=bar_width, edgecolor='grey', label='Full Segment')

        plt.xticks(br + (bar_width * 2), coeffs)
        plt.title(f"{stat}", fontweight='bold', fontsize=15)
        plt.xlabel("Coefficients")
        plt.ylabel('Amplitude')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(stats_path/f"{label}_stats_over_time.png")
    plt.close()

    # Over Coefficients - Full Segment
    plt.figure(figsize=(30, 20))
    for i, stat in enumerate(stats_coeff, start=1):
        plt.subplot(2, 3, i)
        br = np.arange(1, time_frames[-1] + 1)
        plt.plot(br, stats_coeff[stat])
        plt.title(f"{stat}", fontweight='bold', fontsize=15)
        plt.xlabel('Time Frames')
        plt.ylabel('Amplitude')

    plt.suptitle("Full Segment Statistics Over Coefficients", fontsize=30, y=1)
    plt.tight_layout()
    plt.savefig(stats_path/f"{label}_stats_over_coeffs.png")
    plt.close()


def plot_mfcc_stats_all(feature_type, exp_name, data, flag=False):
    stats_path = Path(project_root / f"plots/{exp_name}/{feature_type}/stats/")
    stats_path.mkdir(parents=True, exist_ok=True)

    start_fs = 222 * 3 + 237
    features_time = ["Max Over Time", "Min Over Time", "Std Over Time", "Median Over Time",
                     "IQR Over Time", "Mean Delta Over Time", "Mean Delta-Delta Over Time",
                     "Std Delta Over Time", "Std Delta-Delta Over Time"]

    features_coeff = ["Max Over Coefficients", "Min Over Coefficients", "Std Over Coefficients",
                      "Median Over Coefficients", "IQR Over Coefficients"]
    COEFF = 13

    start_fs_coeffs = 117 + start_fs
    FULL_FRAME = 87

    colors = ["b","r","g","c","m","y"]

    stats_time = {feat: {} for feat in features_time}
    stats_coeff = {feat: {} for feat in features_coeff}

    for label, vec in data.items():
        for i, feat in enumerate(features_time):
            stats_time[feat][label] = vec[start_fs + i * COEFF: start_fs + (i + 1) * COEFF]
        for i, feat in enumerate(features_coeff):
            stats_coeff[feat][label] = vec[start_fs_coeffs + i * FULL_FRAME: start_fs_coeffs + (i + 1) * FULL_FRAME]

    # Over Time
    fig, axes = plt.subplots(3, 3, figsize=(30, 15))
    for i, ax in enumerate(axes.flat):
        stat = list(stats_time.keys())[i]
        for color_idx, label in enumerate(data):
            ax.plot(np.arange(1, COEFF+1), stats_time[stat][label], label=label, color=colors[color_idx], linewidth=2)

        ax.set_title(f"{stat}", fontweight="bold", fontsize=20)
        ax.set_xlabel("Coefficients", fontsize=20)
        ax.set_ylabel("Amplitude", fontsize=20)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    first_ax = axes[0, 0]
    handles, labels = first_ax.get_legend_handles_labels()
    first_ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1.1), fontsize=20, ncol=len(labels))

    if flag:
        fig.suptitle(f"Full Segment Statistics Over Time- Average All", fontsize=30, x=0.95, ha="right")
        fig.savefig(stats_path/f"average_all_stats_over_time.svg", format="svg")
    else:
        fig.suptitle(f"Full Segment Statistics Over Time- Sample All", fontsize=30, x=0.95, ha="right")
        fig.savefig(stats_path/f"sample_all_stats_over_time.svg", format="svg")
    plt.close()

    # Over Coefficients - Full Segment
    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    for i, ax in enumerate(axes.flat):
        if i < 5:
            stat = list(stats_coeff.keys())[i]
            for color_idx, label in enumerate(data):
                ax.plot(np.arange(1, FULL_FRAME+1), stats_coeff[stat][label], label=label, color=colors[color_idx], linewidth=2)

            ax.set_title(f"{stat}", fontweight="bold", fontsize=20)
            ax.set_xlabel("Time Frames", fontsize=20)
            ax.set_ylabel("Amplitude", fontsize=20)

    axes.flat[-1].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    first_ax = axes[0, 0]
    handles, labels = first_ax.get_legend_handles_labels()
    first_ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1.1), fontsize=20, ncol=len(labels))

    if flag:
        fig.suptitle(f"Full Segment Statistics Over Coefficients- Average All", fontsize=30, x=0.95, ha="right")
        fig.savefig(stats_path/f"average_all_stats_over_coeffs.svg", format="svg")
    else:
        fig.suptitle(f"Full Segment Statistics Over Coefficients- Sample All", fontsize=30, x=0.95, ha="right")
        fig.savefig(stats_path/f"sample_all_stats_over_coeffs.svg", format="svg")
    plt.close()


def plot_mfcc_features(feature_type, exp_name, data, label, flag_new=OLD_DATA):
    features_path = Path(project_root / f"plots/{exp_name}/{feature_type}/features/")
    features_path.mkdir(parents=True, exist_ok=True)
    if flag_new:
        x_tick_interval = 20  # Show every 20th time frame
    else:
        x_tick_interval = 10  # Show every 10th time frame
    # Convert time frame index to time units (seconds)
    time_bins = librosa.times_like(data, sr=44100)
    plt.figure(figsize=(10, 10))
    sns.heatmap(data, cmap="viridis", xticklabels=False)
    # Show x-axis ticks in seconds (with precision .2f)
    plt.xticks(ticks=np.arange(0, data.shape[1], x_tick_interval),
               labels=[f"{time_bins[i]:.2f}" for i in range(0, data.shape[1], x_tick_interval)])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Coefficient Index")
    plt.title("Feature Vector Heatmap")
    plt.savefig(features_path/f"{label}_features.png")
    plt.close()


def plot_mfcc_mistaken_full(clf_type, exp_name, data, label):
    stats_path = Path(project_root / f"plots/{exp_name}/mfcc/stats/misclassified/{clf_type}/")
    stats_path.mkdir(parents=True, exist_ok=True)
    move1, move2 = data
    label1, label2 = label
    features_time = ["Max Over Time", "Min Over Time", "Std Over Time", "Median Over Time",
                     "IQR Over Time", "Mean Delta Over Time", "Mean Delta-Delta Over Time",
                     "Std Delta Over Time", "Std Delta-Delta Over Time"]

    features_coeff = ["Max Over Coefficients", "Min Over Coefficients", "Std Over Coefficients",
                      "Median Over Coefficients", "IQR Over Coefficients"]
    COEFF = 13
    frames = [21,21,21,24,87]

    for j in range(5):
        start = 222 * j
        if j==4:
            start = 222 * 3 + 237
        stats_time = {
            f"{feature}": np.array([
                move1[start + i * COEFF: start + (i + 1) * COEFF],
                move2[start + i * COEFF: start + (i + 1) * COEFF]
            ])
            for i, feature in enumerate(features_time)
        }
        start_coeffs = start + 117
        FRAME = frames[j]

        stats_coeff = {
            f"{feature}": np.array([
                move1[start_coeffs + i * FRAME: start_coeffs + (i + 1) * FRAME],
                move2[start_coeffs + i * FRAME: start_coeffs + (i + 1) * FRAME]
            ])
            for i, feature in enumerate(features_coeff)
        }

        # Over Time
        fig, axes = plt.subplots(3, 3, figsize=(30, 15))
        for i, ax in enumerate(axes.flat):
            stat = list(stats_time.keys())[i]
            feature_stats = stats_time[stat]
            br = np.arange(1, COEFF+1)
            ax.plot(br, feature_stats[0, :], color='b', linewidth=2, label=label1)
            ax.plot(br, feature_stats[1, :], color='r', linestyle="dashed", linewidth=2, label=label2)

            ax.set_title(f"{stat}", fontweight="bold", fontsize=15)
            ax.set_xlabel("Coefficients", fontsize=15)
            ax.set_ylabel("Amplitude", fontsize=15)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        first_ax = axes[0, 0]
        handles, labels = first_ax.get_legend_handles_labels()
        first_ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1.1), fontsize=20, ncol=len(labels))

        if j==4:
            fig.suptitle(f"Full Segment Statistics Over Time- {label1} vs {label2}", fontsize=20, x=0.95, ha="right")
            fig.savefig(f"{stats_path}/FS_misclassified_{label1}_{label2}_stats_over_time.svg", format="svg")
        else:
            fig.suptitle(f"Q{j+1} Statistics Over Time- {label1} vs {label2}", fontsize=20, x=0.95, ha="right")
            fig.savefig(f"{stats_path}/q{j+1}_misclassified_{label1}_{label2}_stats_over_time.svg", format="svg")
        plt.close()

        # Over Coefficients
        fig, axes = plt.subplots(2, 3, figsize=(30, 15))
        for i, ax in enumerate(axes.flat):
            if i < 5:
                stat = list(stats_coeff.keys())[i]
                feature_stats = stats_coeff[stat]
                br = np.arange(1, FRAME+1)

                ax.plot(br, feature_stats[0, :], color='b', linewidth=2, label=label1)
                ax.plot(br, feature_stats[1, :], color='r', linestyle="dashed", linewidth=2, label=label2)

                ax.set_title(f"{stat}", fontweight="bold", fontsize=15)
                ax.set_xlabel("Time Frames", fontsize=15)
                ax.set_ylabel("Amplitude", fontsize=15)

        axes.flat[-1].set_visible(False)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        first_ax = axes[0, 0]
        handles, labels = first_ax.get_legend_handles_labels()
        first_ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1.1), fontsize=20, ncol=len(labels))

        if j==4:
            fig.suptitle(f"Full Segment Statistics Over Coefficients- {label1} vs {label2}", fontsize=20, x=0.95,
                         ha="right")
            fig.savefig(f"{stats_path}/FS_misclassified_{label1}_{label2}_stats_over_coeffs.svg", format="svg")
        else:
            fig.suptitle(f"Q{j+1} Statistics Over Coefficients- {label1} vs {label2}", fontsize=20, x=0.95, ha="right")
            fig.savefig(f"{stats_path}/q{j+1}_misclassified_{label1}_{label2}_stats_over_coeffs.svg", format="svg")
        plt.close()


def plot_mfcc_mistaken_features(clf_type, exp_name, data, label):
    features_path = Path(project_root / f"plots/{exp_name}/mfcc/features/misclassified/{clf_type}/")
    features_path.mkdir(parents=True, exist_ok=True)
    move1, move2 = data
    label1, label2 = label
    result = np.concatenate((move1.ravel(), move2.ravel()))
    vmax = max(result)
    vmin = min(result)
    x_tick_interval = 10  # Show every 10th time frame
    # Convert time frame index to time units (seconds)
    time_bins = librosa.times_like(move1, sr=44100)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    sns.heatmap(move1, cmap="viridis", xticklabels=False, vmax=vmax, vmin=vmin)
    # Show x-axis ticks in seconds (with precision .2f)
    plt.xticks(ticks=np.arange(0, move1.shape[1], x_tick_interval),
               labels=[f"{time_bins[i]:.2f}" for i in range(0, move1.shape[1], x_tick_interval)])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Coefficient Index")
    plt.title(f"Feature Vector Heatmap- Real: {label1}, Predicted: {label2}", fontweight="bold", fontsize=17)

    plt.subplot(1, 2, 2)
    sns.heatmap(move2, cmap="viridis", xticklabels=False, vmax=vmax, vmin=vmin)
    # Show x-axis ticks in seconds (with precision .2f)
    plt.xticks(ticks=np.arange(0, move2.shape[1], x_tick_interval),
               labels=[f"{time_bins[i]:.2f}" for i in range(0, move2.shape[1], x_tick_interval)])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Coefficient Index")
    plt.title(f"Feature Vector Heatmap- Real: {label1}, Predicted: {label2}", fontweight="bold", fontsize=17)
    plt.suptitle(f"Feature Vector Heatmap- {label1} vs {label2}")
    plt.tight_layout()
    plt.savefig(f"{features_path}/{label1}_vs_{label2}_features.png")
    plt.close()


def plot_mfcc_features_all(feature_type, exp_name, data):
    """
        Plot mfcc raw features for all classes

    Args:
        data: a dictionary with 6 raw feature vectors
        feature_type: feature type name

    """
    features_path = Path(project_root / f"plots/{exp_name}/mfcc/features/")
    features_path.mkdir(parents=True, exist_ok=True)

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
    plt.savefig(features_path / f"features_all_sample.png")
    plt.close()


def plot_harmonic_stats(feature_type, exp_name, data, label):
    stats_path = Path(project_root / f"plots/{exp_name}/{feature_type}/stats/")
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
            plt.bar(br, feature_stats[0, :], color='r', width=bar_width,
                    edgecolor='grey', label='max')
            plt.bar(br + bar_width, feature_stats[1, :], color='g', width=bar_width,
                    edgecolor='grey', label='min')
            plt.bar(br + 2 * bar_width, feature_stats[2, :], color='b', width=bar_width,
                    edgecolor='grey', label='std')
            plt.bar(br + 3 * bar_width, feature_stats[3, :], color='c', width=bar_width,
                    edgecolor='grey', label='median')

            plt.xticks(br + bar_width, bands)
            plt.xlabel("Frequency Bands")
        else:
            br = np.arange(1, len(quarters)+1)
            if "Delta" in feature:
                plt.bar(br, feature_stats[:, 0], color='r', width=bar_width,
                        edgecolor='grey', label='mean')
                plt.bar(br + bar_width, feature_stats[:, 1], color='g', width=bar_width,
                        edgecolor='grey', label='std')
            else:
                plt.bar(br, feature_stats[:, 0], color='r', width=bar_width,
                        edgecolor='grey', label='max')
                plt.bar(br + bar_width, feature_stats[:, 1], color='g', width=bar_width,
                        edgecolor='grey', label='min')
                plt.bar(br + 2 * bar_width, feature_stats[:, 2], color='b', width=bar_width,
                        edgecolor='grey', label='std')
                plt.bar(br + 3 * bar_width, feature_stats[:, 3], color='c', width=bar_width,
                        edgecolor='grey', label='median')

            plt.xticks(br + bar_width, quarters)
            plt.xlabel("Quarters")
        plt.ylabel('Amplitude')
        plt.title(f"{feature}", fontweight='bold', fontsize=15)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(stats_path/f"{label}_stats.png")
    plt.close()

    # Magnitude stats
    plt.figure(figsize=(20, 12))
    for i, feature in enumerate(features_mag):
        plt.subplot(3, 3, i + 1)
        if i % 4 < 2:  # Over Time
            br = np.arange(1, freq_bin_size+1)
            plt.plot(br, stats_mag[feature])
            plt.title(f"{feature}", fontweight='bold', fontsize=15)
            plt.xlabel("Frequency Bins")
            plt.ylabel('Amplitude')
        else:  # Over Frequency
            br = np.arange(1, time_frame_size+1)
            plt.plot(br, stats_mag[feature])
            plt.title(f"{feature}", fontweight='bold', fontsize=15)
            plt.xlabel("Time Frames")
            plt.ylabel('Amplitude')

    plt.suptitle("Full Segment Magnitude Statistics", fontsize=30, y=1)
    plt.tight_layout()
    plt.savefig(stats_path/f"{label}_stats_magnitude.png")
    plt.close()


def plot_heatmap(data, time_bins, freq_bins, title, subplot_idx, flag_new=OLD_DATA):
    if flag_new:
        x_tick_interval = 20  # Show every 20th time frame
    else:
        x_tick_interval = 10  # Show every 10th time frame
    y_tick_interval = 5  # Show every 5th frequency bin
    plt.subplot(2, 2, subplot_idx)
    sns.heatmap(data, cmap="viridis", cbar_kws={"label": "Amplitude"}, xticklabels=False, yticklabels=False)
    plt.xticks(ticks=np.arange(0, data.shape[1], x_tick_interval),
               labels=[f"{time_bins[i]:.2f}" for i in range(0, data.shape[1], 10)],
               rotation=45)
    plt.yticks(ticks=np.arange(0, data.shape[0], y_tick_interval),
               labels=[f"{freq_bins[i]:.0f}" for i in range(0, data.shape[0], 5)])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)


def plot_harmonic_features(feature_type, exp_name, data, label, flag_new=OLD_DATA):
    features_path = Path(project_root / f"plots/{exp_name}/{feature_type}/features/")
    features_path.mkdir(parents=True, exist_ok=True)
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
    time_bins = [librosa.times_like(harmonic_mag[:midpoint, :], sr=44100, hop_length=215),
                 librosa.times_like(harmonic_mag[midpoint:, :], sr=44100, hop_length=215)]

    plt.figure(figsize=(20, 20))
    for i, (segment, title) in enumerate(segments, start=1):
        plot_heatmap(segment, time_bins[i % 2], freq_bins[midpoint_freq_idx * (i // 3):], title, i, flag_new)
    plt.tight_layout()
    plt.savefig(features_path/f"{label}_features.png")
    plt.close()


def plot_dwt_stats(feature_type, exp_name, data, label):
    stats_path = Path(project_root / f"plots/{exp_name}/{feature_type}/stats/")
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

        plt.bar(br, feature_stats[0, :], color='r', width=bar_width,
                edgecolor='grey', label='Q1')
        plt.bar(br + bar_width, feature_stats[1, :], color='g', width=bar_width,
                edgecolor='grey', label='Q2')
        plt.bar(br + 2 * bar_width, feature_stats[2, :], color='b', width=bar_width,
                edgecolor='grey', label='Q3')
        plt.bar(br + 3 * bar_width, feature_stats[3, :], color='c', width=bar_width,
                edgecolor='grey', label='Q4')
        plt.bar(br + 4 * bar_width, feature_stats[4, :], color='m',
                width=bar_width, edgecolor='grey', label='Full Segment')

        plt.xticks(br + (bar_width * 2), coeffs)
        plt.title(f"{stat}", fontweight='bold', fontsize=15)
        plt.xlabel("Features")
        plt.ylabel("Amplitude")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(stats_path/f"{label}_stats.png")
    plt.close()

    # Autocorrelation stats
    plt.figure(figsize=(30, 20))
    for i, stat in enumerate(stats_ac, start=1):
        plt.subplot(3, 3, i)
        feature_stats = stats_ac[stat]
        br = np.arange(1, len(coeffs)+1) * (bar_width * 5 + spacing)

        plt.bar(br, feature_stats[0, :], color='r', width=bar_width,
                edgecolor='grey', label='Q1')
        plt.bar(br + bar_width, feature_stats[1, :], color='g', width=bar_width,
                edgecolor='grey', label='Q2')
        plt.bar(br + 2 * bar_width, feature_stats[2, :], color='b', width=bar_width,
                edgecolor='grey', label='Q3')
        plt.bar(br + 3 * bar_width, feature_stats[3, :], color='c', width=bar_width,
                edgecolor='grey', label='Q4')
        plt.bar(br + 4 * bar_width, feature_stats[4, :], color='m',
                width=bar_width, edgecolor='grey', label='Full Segment')

        plt.xticks(br + (bar_width * 2), coeffs)
        plt.title(f"{stat}", fontweight='bold', fontsize=15)
        plt.xlabel("Features")
        plt.ylabel("Amplitude")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Lags stats
    for i, stat in enumerate(lags, start=len(stats_ac) + 1):
        plt.subplot(3, 3, i)
        br = np.arange(1, lag_size+1)

        plt.plot(br, lags[stat])
        plt.title(f"{stat}- Full Segment {lag_size} Lags", fontweight='bold', fontsize=15)
        plt.xlabel("Lag Number")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(stats_path/f"{label}_stats_autocorrelation.png")
    plt.close()


def plot_dwt_features(feature_type, exp_name, data, label, flag_new=OLD_DATA):
    features_path = Path(project_root / f"plots/{exp_name}/{feature_type}/features/")
    features_path.mkdir(parents=True, exist_ok=True)
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
    plt.savefig(features_path/f"{label}_features.png")
    plt.close()


# Mapping feature type to plot function (according to selector)
# sel=0: plot stat, sel=1: plot feature
plot_functions = {
    "mfcc": [plot_mfcc_stats, plot_mfcc_features],
    "harmonic": [plot_harmonic_stats, plot_harmonic_features],
    "dwt": [plot_dwt_stats, plot_dwt_features]
}


def save_vecs(feature_type, exp_name, data, label, sel=0, flag_new=OLD_DATA):
    """
    Plots and saves the given vector.

    Args:
        data: statistics vector, extracted feature vector, 2 confused stat vectors
        label: the data's label (movement type)
        sel: sel=0: data = statistics vector,
             sel=1: data = features vector,
             sel=2: data = (towards, away) stat vectors
        flag: True: plotting average label1 and label2 (where label1,label2 = label)
    """

    # If this movement type has not been plotted yet
    if flags[label][sel] == 0:
        if flag_new:
            plot_functions[feature_type][sel](feature_type, exp_name, data, label, flag_new)
        else:
            plot_functions[feature_type][sel](feature_type, exp_name, data, label)
        flags[label][sel] = 1  # Set the flag of this label to 1, so this movement is not saved again.


def generate_mfcc_stats_names(flag_new=OLD_DATA):
    """
        Returns: Generates all the statistics names (classifier features names) for the mfcc classifiers
        """
    num_coef = 13
    if flag_new:
        frame_counts = [64, 64, 64, 67, 259]
    else:
        frame_counts = [21, 21, 21, 24, 87]
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


def generate_harmonic_stats_names(flag_new=OLD_DATA):
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

    freq_bins = 257
    if flag_new:
        time_frames = 517
    else:
        time_frames = 173

    # 257 freq bin stds + 257 medians
    full_harmonic_features += [f"FULL_harmonic_std_freqbin_{i+1}" for i in range(freq_bins)]
    full_harmonic_features += [f"FULL_harmonic_median_freqbin_{i+1}" for i in range(freq_bins)]

    # 517 time frame stds + 517 medians
    full_harmonic_features += [f"FULL_harmonic_std_frame_{i+1}" for i in range(time_frames)]
    full_harmonic_features += [f"FULL_harmonic_median_frame_{i+1}" for i in range(time_frames)]

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
    full_percussive_features += [f"FULL_percussive_std_freqbin_{i+1}" for i in range(freq_bins)]
    full_percussive_features += [f"FULL_percussive_median_freqbin_{i+1}" for i in range(freq_bins)]

    # 517 time frame stds + 517 medians
    full_percussive_features += [f"FULL_percussive_std_frame_{i+1}" for i in range(time_frames)]
    full_percussive_features += [f"FULL_percussive_median_frame_{i+1}" for i in range(time_frames)]

    ordered_feature_names = (
            quarter_feature_names +
            full_harmonic_features +
            full_percussive_features
    )

    return ordered_feature_names


def generate_dwt_stats_names(flag_new=OLD_DATA):
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