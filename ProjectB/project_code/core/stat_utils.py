''' This file contains code to define statistics calculations as well as other similar utils. '''

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pywt
from librosa.feature import delta
from scipy.signal import correlate

import plot_utils as pu

MAX_HOVER = 30  # Limit idle flight segments (so hover class is not too big)

# Length in seconds for each movement type
section_secs = [4, 6, 3, 34, 6, 3.5]

# Dictionary: for each movement name, the length in seconds
section_secs_dict = dict(zip(pu.movement_names, section_secs))

# Maximum height and distance
Z_AXIS = 0.5
X_AXIS = 1.0

# Actual number of segments per movement type
ASCENT_SEGMENTS = 6
AWAY_SEGMENTS = 8
DESCENT_SEGMENTS = 5
TOWARDS_SEGMENTS = 8

# Calculate the expected movement per one segment
ASCENT_RATE = Z_AXIS / ASCENT_SEGMENTS
DESCENT_RATE = Z_AXIS / DESCENT_SEGMENTS
AWAY_RATE = X_AXIS / AWAY_SEGMENTS
TOWARDS_RATE = X_AXIS / TOWARDS_SEGMENTS


# ------Feature extraction functions------


def calculate_mfcc_quarter_data(segment, sr=44100, num_coef=13):
    """
    Function to extract mfcc raw features and statistics.
    Statistics are calculated over quarters of the data (maintaining temporal contex),
     and also over the full segment.

    Args:
        segment: data per segment
        sr: sampling rate
        num_coef: number of coefficients

    Returns:
        Statistics vector of the mfcc features of this segment.
    """

    # Compute the MFCC for the entire segment
    # shape: (13,259) = (number of coefficients, time frames)
    # time frames are calculated by 132300/512, where |segment|=132300, and mfcc default num_hops=512
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


def calculate_harmonic_quarter_data(segment, sr=44100):
    """
    Function to extract harmonic and percussive raw features and statistics.
    Statistics are calculated over quarters of the data (maintaining temporal contex),
     and also over the full segment.

    Args:
        segment: data per segment
        sr: sampling rate

    Returns:
        Statistics vector of the harmonic and percussive features of this segment.
    """

    # Compute the STFT for the entire segment
    D = librosa.stft(segment, n_fft=512, hop_length=256)

    # Apply Harmonic-Percussive Source Separation (HPSS)
    # shape: (257, 517) = (frequency bins, time frames)
    # frequency bins are calculated by (n_fft // 2) + 1 (we set n_fft=512)
    # time frames are calculated by 132300/256, where |segment|=132300, and we set hop_length=256
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


def calculate_dwt_quarter_data(segment, wavelet="coif5", max_level=3):
    """
    Function to extract dwt raw features and statistics.
    Statistics are calculated over quarters of the data (maintaining temporal contex),
     and also over the full segment.

    Args:
        segment: data per segment
        wavelet: wavelet type
        max_level: highest level of detail

    Returns:
        Statistics and raw features vectors of the dwt features of this segment.
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


# Mapping feature name to feature data extraction function
feature_functions = {
    "mfcc": calculate_mfcc_quarter_data,
    "harmonic": calculate_harmonic_quarter_data,
    "dwt": calculate_dwt_quarter_data
}


def extract_segments_data(audio_section, label, segment_samples, feature_type, sr=44100):
    """
        Extracts per-segment raw features and statistics from a single section of audio.

    Args:
        audio_section: 1D numpy array for a section of audio
        label: label for this section
        segment_samples: number of samples per segment
        feature_type: feature type name
        sr: sample rate

    Returns:
        List of raw features and statistics vectors for this section
    """

    # Get function to extract features per segment
    feature_func = feature_functions[feature_type]
    step_size = int(sr * 0.75)
    pad_length = (len(audio_section) + segment_samples - 1) // step_size * step_size
    audio_section = np.pad(audio_section, (0, max(0, pad_length - len(audio_section))), mode="edge")
    total_segments = (len(audio_section) - segment_samples) // step_size + 1

    if label == "hover":
        num_segments = min(MAX_HOVER, total_segments)
    else:
        num_segments = total_segments

    stats = [feature_func(audio_section[i * step_size: i * step_size + segment_samples])
             for i in range(num_segments)]

    return stats


def get_data_split(audio_path, label, feature_type, sample_rate=44100, segment_length=3.0):
    """

    Args:
        audio_path: audio file path (section)
        label: label for section
        feature_type: feature type name
        sample_rate: audio file sample rate
        segment_length: length in seconds for each segment

    Returns: Statistics and raw features for all segments in this section

    """
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    segment_samples = int(sr * segment_length)

    segment_stats = extract_segments_data(audio, label, segment_samples, feature_type, sr)
    return segment_stats


def get_data_full_recording(audio_path, flight_type, feature_type, sample_rate=44100, segment_length=3.0):
    """

    Args:
        audio_path: audio file path (full recording)
        flight_type: the flight type for this recording (linear forward, linear back, linear middle, idle)
        feature_type: feature name
        sample_rate: audio file sample rate
        segment_length: length in seconds for each segment

    Returns: Statistics and raw features for this full recording (all segments in all sections)

    """
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    segment_samples = int(sr * segment_length)

    flight_splits = {
        "mic_end": ([0, 4, 10, 13.5, 19.5], ["ascent", "towards", "turn", "away", "descent"]),
        "mic_start": ([0, 4, 10, 13.5, 19.5], ["ascent", "away", "turn", "towards", "descent"]),
        "mic_middle": ([0, 4, 7, 10, 13.5, 16.5, 19.5], ["ascent", "towards", "away", "turn", "towards", "away", "descent"]),
        None: ([0, 4, 38], ["ascent", "hover", "descent"])
    }

    if flight_type not in flight_splits:
        raise ValueError(f"Unknown flight type '{flight_type}' for file {audio_path}")

    split_times, real_labels = flight_splits[flight_type]
    split_frames = [int(time * sr) for time in split_times]

    total_stats = []
    total_labels = []
    section_indices = []
    section_index = 0

    for i in range(len(split_frames)):
        start_idx = split_frames[i]
        end_idx = split_frames[i + 1] if i < len(split_frames) - 1 else len(audio)
        section_audio = audio[start_idx:end_idx]
        label = real_labels[i]

        section_stats = extract_segments_data(section_audio, label, segment_samples, feature_type, sr)

        total_stats.extend(section_stats)
        total_labels.extend([label] * len(section_stats))

        section_indices.extend([section_index] * len(section_stats))
        section_index += 1

    # Total flight time
    duration_secs = split_times[-1] + section_secs_dict["descent"]
    return total_stats, real_labels, section_indices, duration_secs


def paths_and_labels_split():
    """
        Scans the database and returns the tuple (audio_files, labels, recording_ids) where:
        audio_files: list of file paths of every recording.

        labels: list of the label of every recording (out of 6 possible movement types).

        recording_ids: maps filename to id numbers for every recording
            (for GroupKFold, and for full recording regression)

    """

    path = Path(pu.project_root/"split")
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

    data_folder_path = Path(pu.project_root/"data/new")
    data_folder_path.mkdir(parents=True, exist_ok=True)

    # Save recording ID mapping for future classification
    with open(data_folder_path / "recording_ids.json", "w") as f:
        json.dump(recording_ids, f)

    return audio_files, labels, recording_ids


def paths_and_labels_full():
    """
        Scans the database and returns the tuple (audio_files, labels) where:
        audio_files: list of file paths of every recording.

    """

    path = Path(pu.project_root/"recordings")
    audio_files = []

    for movement_folder in path.iterdir():
        if movement_folder.is_dir():  # movement types
            for file in movement_folder.rglob("*.wav"):
                audio_files.append(file)

    return audio_files


def create_csv(audio_path, secs, pred_labels, feature_type, clf_type, is_segment=True, section_secs=None):
    """
        Saves distance prediction by segments or by sections.

    Args:
        audio_path: path to audio file
        secs: duration of recording in seconds
        pred_labels: prediction labels
        is_segment: True if we evaluate by segments, False if we evaluate by sections
        section_secs: if is_segment is False, holds the duration of each section in seconds

    """
    t = np.linspace(0, secs, len(pred_labels))

    # For sections: add the section duration
    if not is_segment:
        t[0] = section_secs[0]

    x = np.zeros(len(t))
    z = np.zeros(len(t))

    pred = pred_labels[0]

    if pred == "ascent":
        # Update z for ascent
        z[0] = ASCENT_RATE if is_segment else Z_AXIS
    elif pred == "descent":
        # Update z for descent
        z[0] = -DESCENT_RATE if is_segment else -Z_AXIS
    elif pred == "towards":
        # Update x for towards
        x[0] = TOWARDS_RATE if is_segment else X_AXIS
    elif pred == "away":
        # Update x for away
        x[0] = -AWAY_RATE if is_segment else -X_AXIS

    for i in range(1, len(t)):
        pred = pred_labels[i]

        # For sections: add the section duration
        if not is_segment:
            t[i] = t[i - 1] + section_secs[i]

        x[i] = x[i - 1]
        z[i] = z[i - 1]

        if pred == "ascent":
            # Update z for ascent
            z[i] += ASCENT_RATE if is_segment else Z_AXIS
        elif pred == "descent":
            # Update z for descent
            z[i] -= DESCENT_RATE if is_segment else Z_AXIS
        elif pred == "towards":
            # Update x for towards
            x[i] += TOWARDS_RATE if is_segment else X_AXIS
        elif pred == "away":
            # Update x for away
            x[i] -= AWAY_RATE if is_segment else X_AXIS

    df = pd.DataFrame({
        "x_axis": x,
        "z_axis": z,
        "time": t,
        "task_name": pred_labels
    })

    if is_segment:
        folder_path = Path(pu.project_root/f"plots/final/{feature_type}/segments/{clf_type}")
    else:
        folder_path = Path(pu.project_root/f"plots/final/{feature_type}/sections/{clf_type}")
    folder_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(folder_path / f"{audio_path.stem}_{feature_type}_{clf_type}_predicted.csv", index=False)
