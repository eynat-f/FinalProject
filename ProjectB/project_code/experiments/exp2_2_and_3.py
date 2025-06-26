''' This file contains code regarding Experiment 2.2, and 3 (both 3.1 and 3.2). '''

import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pywt
import seaborn as sns
import shap
from librosa.feature import delta
from scipy.signal import correlate
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVC

import exp_utils as eu

# To plot the MFCC features (other two feature types are too heavy)
plot_mfcc_features = []


# ------Feature extraction functions------

def calculate_mfcc_quarter_statistics(feature_type, exp_name, segment, label, sr=44100, num_coef=13):
    """
    Function to extract mfcc features.
    Statistics are calculated over quarters of the data (maintaining temporal contex),
     and also over the full segment.

    Args:
        segment: data per segment (1 second)
        label: current segment label
        sr: sampling rate
        num_coef: number of coefficients

    Returns:
        Statistics vector of the mfcc features of this segment.
    """

    # Compute the MFCC for the entire segment
    # shape: (13,87) = (number of coefficients, time frames)
    # time frames are calculated by 44100/512, where sr=44100, and mfcc default num_hops=512
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=num_coef)

    plot_mfcc_features.append((mfcc))

    # Plot and save the features of this segment (one segment per label)
    if exp_name == "exp2_2":
        eu.save_vecs(feature_type, exp_name, mfcc, label, 1, eu.NEW_DATA)

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


def calculate_harmonic_quarter_statistics(feature_type, exp_name, segment, label, sr=44100):
    """
    Function to extract harmonic and percussive features.
    Statistics are calculated over quarters of the data (maintaining temporal contex),
     and also over the full segment.

    Args:
        segment: data per segment (1 second)
        label: current segment label
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

    # Plot and save the features of this segment (one segment per label)
    if exp_name == "exp2_2":
        eu.save_vecs(feature_type, exp_name, (harmonic_mag, percussive_mag), label, 1, eu.NEW_DATA)

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


def calculate_dwt_quarter_statistics(feature_type, exp_name, segment, label, wavelet="coif5", max_level=3):
    """
    Function to extract dwt features.
    Statistics are calculated over quarters of the data (maintaining temporal contex),
     and also over the full segment.

    Args:
        segment: data per segment (1 second)
        label: current segment label
        sr: sampling rate
        wavelet:

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

    # Plot and save the features of this segment (one segment per label)
    if exp_name == "exp2_2":
        eu.save_vecs(feature_type, exp_name, coeffs, label, 1, eu.NEW_DATA)

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


def get_features(audio_path, label, sample_rate=44100, segment_length=3.0):
    """
        Get the feature statistics for this recording.

    Args:
        audio_path: path to this recording
        label: label of this recording
        selector: select which feature to extract
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
        num_segments = min(eu.MAX_HOVER, total_segments)
    else:
        num_segments = total_segments

    segments = [audio[i * step_size: i * step_size + segment_samples] for i in range(num_segments)]

    # Get the feature function from the dictionary
    feature_func = feature_functions[feature_type]

    # Extract and aggregate features per segment
    segment_features = [feature_func(feature_type, exp_name, segment, label) for segment in segments]

    # nparray of shape: (NUM_SEGMENTS, NUM_STATS)
    return segment_features


# Experiment 2.2
def run_classifier_2_2():
    """
        Runs the chosen classifier and calculates classification_report and confusion_matrix.

    Args:
        sel: select clf type

    """

    # Extract important features (for other classifiers as well)
    rf = RandomForestClassifier(random_state=r_s)
    rf.fit(X_train, y_train)
    if clf_type == "random forest":
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)
        y_pred = rf.predict(X_test)

    else:  # clf type is svm or mlp
        if feature_type == "dwt":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if clf_type == "svm rbf":
            if feature_type == "mfcc":
                pca = PCA(n_components=0.97)
            else:
                pca = PCA(n_components=0.95)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)

            with open(prediction_name, "a") as f:
                print(f"Number of components selected: {pca.n_components_}", file=f)

            svm = SVC(kernel="rbf", C=10, random_state=r_s)
            svm.fit(X_train_pca, y_train)

            # Predictions
            y_pred = svm.predict(X_test_pca)
        else:  # clf type is mlp
            if feature_type == "dwt":
                mlp = MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=1000,
                                    activation="relu", solver="adam", random_state=r_s)
            else:
                mlp = MLPClassifier(hidden_layer_sizes=(400, 300, 200, 100), max_iter=1000,
                                    activation="relu", solver="adam", random_state=r_s)

            mlp.fit(X_train_scaled, y_train)

            y_pred = mlp.predict(X_test_scaled)

    # For the first split, and random forest classifier:
    # Plot misclassified and the average vector for labels: ("towards", "away", "turn")
    if fold_num == 1:
        top_n = 50
        with open(importances_name, "a") as f:
            print(f"{feature_type} top {top_n} most important features for fold {fold_num}: ", file=f)
        importances = rf.feature_importances_
        args = np.argsort(importances)[::-1]
        names = eu.generate_classifier_feature_names[feature_type](eu.NEW_DATA)
        for i in range(top_n):
            feat = args[i]
            with open(importances_name, "a") as f:
                print(f"{names[feat]}-{importances[feat]:.4f}", file=f)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    all_reports.append(report)

    # Print classification results
    with open(prediction_name, "a") as f:
        print(f"Fold {fold_num}:\n", file=f)
        print("Accuracy:", accuracy_score(y_test, y_pred), file=f)
        print("Classification Report:\n", file=f)
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_), file=f)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    all_cm.append(cm)


# Experiment 3.1
def plot_shap_waterfall(feature_type, explainer, shap_values, X_test_selected, supp):
    """
        Plots the shap values of the classifier features of the given feature type, given the data

    Args:
        feature_type: feature type
        explainer: explainer object for the data to plot
        shap_values: shap values to plot
        X_test_selected: test samples
        supp: the support mask (to get the features we actually use), from selector.get_support()

    """

    top_n = 8

    all_feature_names = eu.generate_classifier_feature_names[feature_type](eu.NEW_DATA)
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
    move_idx = max_conf_sample
    label_transform = np.argmax(probs)
    label_name = eu.movement_names[label_transform]

    with open(shap_name, "a") as f:
        print(f"For sample {label_name}:", file=f)

    for i, p in enumerate(probs):
        with open(shap_name, "a") as f:
            print(f"Class {eu.movement_names[i]}: prob = {p * 100:.2f}%", file=f)

    # Full SHAP values for this sample and class
    shap_vals = shap_values[move_idx, :, label_transform]
    base_val = explainer.expected_value[label_transform]
    input_sample = X_test_selected[move_idx, :]
    feature_names = selected_feature_names

    # Select top-N indices
    sorted_indices = np.argsort(-np.abs(shap_vals))[:top_n]
    top_shap_vals = shap_vals[sorted_indices]

    top_feature_vals = input_sample[sorted_indices]
    top_feature_names = [feature_names[i] for i in sorted_indices]

    # Adjust base value so f(x) is still correct
    residual = np.sum(shap_vals) - np.sum(top_shap_vals)
    adjusted_base_val = base_val + residual

    filtered_explanation = shap.Explanation(
        values=top_shap_vals,
        base_values=adjusted_base_val,
        data=top_feature_vals,
        feature_names=top_feature_names
    )

    # Plot waterfall
    shap.plots.waterfall(filtered_explanation, show=False)

    plt.title(f"SHAP-Stats: predicted class {label_name}")
    plt.tight_layout()
    plt.savefig(plots_path / f"shap_confident_predicted_{label_name}.png")
    plt.close()


# Experiment 3.1
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
    # Show top 8 features only!
    top_n = 8

    label1_transform = label_encoder.transform([label1])[0]
    label2_transform = label_encoder.transform([label2])[0]

    all_feature_names = eu.generate_classifier_feature_names[feature_type](eu.NEW_DATA)
    selected_feature_names = [name for name, keep in zip(all_feature_names, supp) if keep]

    shap_label_transform = [label2_transform, label1_transform, label1_transform, label2_transform]
    shap_move_idx = [move1_idx, move1_idx, move2_idx, move2_idx]

    for idx in range(4):
        move_idx = shap_move_idx[idx]
        label_transform = shap_label_transform[idx]

        if idx % 2 == 0:
            label_checked = label1 if idx == 0 else label2
            with open(shap_name, "a") as f:
                print(f"For sample {label_checked}:", file=f)
            probs = [
                explainer.expected_value[c] + shap_values[move_idx, :, c].sum()
                for c in range(6)
            ]
            for i, p in enumerate(probs):
                with open(shap_name, "a") as f:
                    print(f"Class {eu.movement_names[i]}: prob = {p * 100:.2f}%", file=f)

        # Full SHAP values for this sample and class
        shap_vals = shap_values[move_idx, :, label_transform]
        base_val = explainer.expected_value[label_transform]
        input_sample = X_test_selected[move_idx, :]
        feature_names = selected_feature_names

        # Select top-N indices
        sorted_indices = np.argsort(-np.abs(shap_vals))[:top_n]
        top_shap_vals = shap_vals[sorted_indices]

        top_feature_vals = input_sample[sorted_indices]
        top_feature_names = [feature_names[i] for i in sorted_indices]

        # Adjust base value so f(x) is still correct
        residual = np.sum(shap_vals) - np.sum(top_shap_vals)
        adjusted_base_val = base_val + residual

        filtered_explanation = shap.Explanation(
            values=top_shap_vals,
            base_values=adjusted_base_val,
            data=top_feature_vals,
            feature_names=top_feature_names
        )

        shap.plots.waterfall(filtered_explanation, show=False)

        match idx:
            case 0:
                plt.title(f"SHAP-Stats, {label1}: predicted class {label2} (Incorrect)")
                plt.tight_layout()
                plt.savefig(plots_path / f"shap_predicted_{label1}_incorrect_{label2}.png")
            case 1:
                plt.title(f"SHAP-Stats, {label1}: predicted class {label1} (Correct)")
                plt.tight_layout()
                plt.savefig(plots_path / f"shap_predicted_{label1}_correct_{label1}.png")
            case 2:
                plt.title(f"SHAP-Stats, {label2}: predicted class {label1} (Incorrect)")
                plt.tight_layout()
                plt.savefig(plots_path / f"shap_predicted_{label2}_incorrect_{label1}.png")
            case 3:
                plt.title(f"SHAP-Stats, {label2}: predicted class {label2} (Correct)")
                plt.tight_layout()
                plt.savefig(plots_path / f"shap_predicted_{label2}_correct_{label2}.png")
        plt.close()


# Experiment 3.2
def plot_shap_bar(feature_type, explainer, shap_values, X_test_selected, supp):
    """
        Plots the shap values of the classifier features of the given feature type, given the data

    Args:
        feature_type: feature type
        explainer: explainer object for the data to plot
        shap_values: shap values to plot
        X_test_selected: test samples
        supp: the support mask (to get the features we actually use), from selector.get_support()

    """

    all_feature_names = eu.generate_classifier_feature_names[feature_type](eu.NEW_DATA)
    selected_feature_names = [name for name, keep in zip(all_feature_names, supp) if keep]

    fig, axs = plt.subplots(2, 3, figsize=(20, 8))
    axs = axs.flatten()

    top_n = 8
    for class_idx, movement in enumerate(eu.movement_names):
        # Mean absolute SHAP values over all samples for this class
        class_shap_values = shap_values[:, :, class_idx]
        mean_abs_shap = np.abs(class_shap_values).mean(axis=0)

        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
        top_shap_values = class_shap_values[:, top_indices]
        top_data = X_test_selected[:, top_indices]
        top_feature_names = [selected_feature_names[i] for i in top_indices]

        explanation = shap.Explanation(
            values=top_shap_values,
            base_values=explainer.expected_value[class_idx],
            data=top_data,
            feature_names=top_feature_names
        )

        shap.plots.bar(explanation, show=False, max_display=8, ax=axs[class_idx])
        axs[class_idx].set_title(f"SHAP Importances – {movement}", fontweight="bold", fontsize=15)

    plt.suptitle(f"{feature_type} SHAP Importances", fontweight="bold", fontsize=20)
    plt.tight_layout()
    plt.savefig(plots_path / f"global_per_class.png")
    plt.close()


# Experiment 3
def run_classifier_3():
    """
        Runs the chosen classifier and calculates classification_report and confusion_matrix.

    Args:
        sel: select clf type

    """
    clf = RandomForestClassifier(random_state=r_s)
    clf.fit(X_train, y_train)

    # Extract important features (for other classifiers as well)
    if feature_type == "mfcc":
        if clf_type == "mlp":
            selector = SelectFromModel(clf, threshold="0.75* mean", prefit=True)
        else:
            selector = SelectFromModel(clf, threshold="mean", prefit=True)
    elif feature_type == "harmonic":
        if clf_type == "svm rbf":
            selector = SelectFromModel(clf, threshold="mean", prefit=True)
        else:
            selector = SelectFromModel(clf, threshold="0.75* mean", prefit=True)
    else:  # dwt
        if clf_type == "random forest":
            selector = SelectFromModel(clf, threshold="1.25* mean", prefit=True)
        elif clf_type == "svm rbf":
            selector = SelectFromModel(clf, threshold="0.75* mean", prefit=True)
        else:
            selector = SelectFromModel(clf, threshold="mean", prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    supp = selector.get_support()

    with open(prediction_name, "a") as f:
        print(f"Reduced from {X_train.shape[1]} to {X_train_selected.shape[1]}", file=f)

    if clf_type == "random forest":
        clf.fit(X_train_selected, y_train)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_selected)
        y_pred = clf.predict(X_test_selected)

    else:  # clf type is svm or mlp
        if feature_type == "dwt":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)

        if clf_type == "svm rbf":
            if feature_type == "mfcc":
                pca = PCA(n_components=0.97)
            else:
                pca = PCA(n_components=0.95)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)

            with open(prediction_name, "a") as f:
                print(f"Number of components selected: {pca.n_components_}", file=f)

            clf = SVC(kernel="rbf", C=10, random_state=r_s)
            clf.fit(X_train_pca, y_train)

            # Predictions
            y_pred = clf.predict(X_test_pca)
        else:  # clf type is mlp
            if feature_type == "dwt":
                mlp = MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=1000,
                                    activation="relu", solver="adam", random_state=r_s)
            else:
                mlp = MLPClassifier(hidden_layer_sizes=(400, 300, 200, 100), max_iter=1000,
                                    activation="relu", solver="adam", random_state=r_s)

            mlp.fit(X_train_scaled, y_train)

            y_pred = mlp.predict(X_test_scaled)

    # For the first split, and random forest classifier:
    # Plot misclassified and the average vector for labels: ("towards", "away", "turn")
    if fold_num == 1 and clf_type == "random forest":
        plot_shap_bar(feature_type, explainer, shap_values, X_test_selected, supp)
        if feature_type == "mfcc":  # exp3.1
            plot_shap_waterfall(feature_type, explainer, shap_values, X_test_selected, supp)
            label1 = "towards"
            label2 = "away"
            move1_idx = 482
            move2_idx = 180
            plot_mfcc_shap_confusion(explainer, shap_values, X_test_selected, supp,
                                     label1, label2, move1_idx, move2_idx)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    all_reports.append(report)

    # Print classification results
    with open(prediction_name, "a") as f:
        print(f"Fold {fold_num}:\n", file=f)
        print("Accuracy:", accuracy_score(y_test, y_pred), file=f)
        print("Classification Report:\n", file=f)
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_), file=f)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    all_cm.append(cm)


# Experiments list
exp_list = ["exp2_2", "exp3"]

if __name__ == "__main__":

    start = time.time()

    # Select Experiment:
    # 0: experiment 2.2, 1: experiment 3
    exp_select = 1
    exp_name = exp_list[exp_select]

    # Select classifier:
    # 0: random forest, 1: svm, 2: mlp
    sel_clf = 2
    clf_type = eu.classifiers_list[sel_clf]

    # Select feature type:
    # 0: mfcc, 1: hpss, 2: dwt
    sel_feature = 2
    feature_type = eu.features_list[sel_feature]

    audio_files, labels, recording_ids = eu.paths_and_labels(eu.NEW_DATA)

    # Path to load/save statistics/raw features arrays
    data_path = Path(eu.project_root / f"data/new/{feature_type}/train")
    data_path.mkdir(parents=True, exist_ok=True)
    stats_npz = Path(data_path / f"{feature_type}_stats_data.npz")
    features_npz = Path(data_path / f"{feature_type}_features_data.npz")

    # Paths to save plots and text output
    if exp_name == "exp3":
        plots_path = Path(eu.project_root / f"plots/{exp_name}/stats_model/{feature_type}")
        output_path = Path(eu.project_root / f"output/{exp_name}/stats_model/{feature_type}")
    else:
        plots_path = Path(eu.project_root / f"plots/{exp_name}/{feature_type}")
        output_path = Path(eu.project_root / f"output/{exp_name}/{feature_type}")
    plots_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    prediction_name = f"{output_path}/{clf_type}_predictions.txt"
    open(prediction_name, "w").close()
    if exp_name == "exp2_2":  # same importances for experiment 3, no need to duplicate results
        importances_name = f"{output_path}/{exp_name}_importances.txt"
        open(importances_name, "w").close()
    elif clf_type == "random forest" and feature_type == "mfcc":
        shap_name = f"{output_path}/shap_confidence.txt"
        open(shap_name, "w").close()

    # Encode labels
    label_encoder = LabelEncoder()

    if stats_npz.exists():
        data = np.load(stats_npz, allow_pickle=True)
        stats = data["stats"]
        all_labels = data["all_labels"]
        file_ids = data["file_ids"]
        if features_npz.exists():
            data = np.load(features_npz, allow_pickle=True)
            plot_mfcc_features = data["raw_features"]
    else:
        # Extract features and ensure segments are grouped correctly
        stats, all_labels, file_ids = [], [], []

        for file_path, label in zip(audio_files, labels):
            segment_features = get_features(file_path, label)  # Get multiple segments per file
            stats.extend(segment_features)  # Add all segments to stats list
            all_labels.extend([label] * len(segment_features))  # Same label for all segments
            file_ids.extend([recording_ids[file_path.stem.split("_")[0]]] * len(segment_features))
            # Group ID for StratifiedGroupKFold, accounts for segments in split recording, and for the full recording

        # Convert lists to arrays
        stats = np.array(stats)
        all_labels = np.array(all_labels)
        file_ids = np.array(file_ids)

        np.savez(stats_npz,
                 stats=stats,
                 all_labels=all_labels,
                 file_ids=file_ids)

        if feature_type == "mfcc":
            plot_mfcc_features = np.array(plot_mfcc_features)
            np.savez(features_npz, raw_features=plot_mfcc_features)

    numeric_labels = label_encoder.fit_transform(all_labels)

    all_reports = []
    all_cm = []
    fold_num = 1
    r_s = 1

    # StratifiedGroupKFold to split while keeping file segments together
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r_s)

    # Split while maintaining file segments together (groups=file_ids)
    for train_idx, test_idx in splitter.split(stats, numeric_labels, groups=file_ids):
        # Train/Test split
        X_train, X_test = stats[train_idx], stats[test_idx]
        y_train, y_test = numeric_labels[train_idx], numeric_labels[test_idx]

        if exp_name == "exp2_2" and feature_type == "mfcc" and fold_num == 1:
            plot_features_train = plot_mfcc_features[train_idx]

            first_feature_train = {
                label_encoder.inverse_transform([class_label])[0]: plot_features_train[y_train == class_label][0]
                for class_label in np.unique(y_train)
            }

            eu.plot_mfcc_features_all(feature_type, exp_name, first_feature_train)

        if exp_name == "exp2_2":
            run_classifier_2_2()
        else:
            run_classifier_3()

        fold_num += 1

    # Calculate and print average F1-scores across splits
    average_f1_scores = {
        class_name: np.mean([r[class_name]["f1-score"] for r in all_reports])
        for class_name in label_encoder.classes_
    }
    with open(prediction_name, "a") as f:
        print("Average F1-scores over folds:", average_f1_scores, file=f)

    # Plot all confusion matrices as subplots
    fig, axes = plt.subplots(1, len(all_cm), figsize=(20, 5))
    for i, cm in enumerate(all_cm):
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_, ax=ax)
        ax.set_title(f"Fold {i + 1}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    print("----Elapsed time: %.2f seconds----" % (time.time() - start))

    plt.tight_layout()
    plt.savefig(f"{plots_path}/{clf_type}_cm.png")
    plt.close()
