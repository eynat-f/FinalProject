''' This file contains code regarding Experiment 1, which includes sub-experiments 1.1, 1.2 and 1.3. '''

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from librosa.feature import delta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

import exp_utils as eu


def calculate_mfcc_quarter_statistics_1_2(segment, sr=44100, num_coef=13):
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
    #         i=0,1,2,3: 117 values each (9*13=117 time stats)
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
        ]))

    # Delta and delta-delta (mfcc)
    delta_mfcc = delta(mfcc)
    delta2_mfcc = delta(mfcc, order=2)

    # Compute overall statistics
    # Result: nparray of 117 values (9*13=117)
    full_features = np.hstack([
        np.max(mfcc, axis=1), np.min(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.median(mfcc, axis=1), np.percentile(mfcc, 75, axis=1) - np.percentile(mfcc, 25, axis=1),
        np.mean(delta_mfcc, axis=1), np.mean(delta2_mfcc, axis=1),
        np.std(delta_mfcc, axis=1), np.std(delta2_mfcc, axis=1),
    ])

    # Combine quarter-based and full-segment features
    # Result: nparray of size 585 (117*5)
    return np.hstack(segment_features + [full_features])


def calculate_mfcc_quarter_statistics_1_3(segment, sr=44100, num_coef=13):
    """
    Function to extract mfcc features.
    Statistics are calculated over quarters of the data (maintaining temporal contex),
     and also over the full segment. This includes coefficient-based stats.

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

            # EXP1.3
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

        # EXP1.3
        np.max(mfcc, axis=0), np.min(mfcc, axis=0),
        np.std(mfcc, axis=0),
        np.median(mfcc, axis=0), np.percentile(mfcc, 75, axis=0) - np.percentile(mfcc, 25, axis=0),
    ])

    # Combine quarter-based and full-segment features
    # Result: nparray of size 1455 (222*3+237+552)
    return np.hstack(segment_features + [full_features])


def get_features(audio_path, label, sample_rate=44100, n_mfcc=13, segment_length=1.0, max_segments=18):
    """
    Args:
        audio_path: Path to the audio file.
        label: segment class label
        sample_rate: Sampling rate for audio processing.
        n_mfcc: Number of MFCC coefficients to extract.
        segment_length: Duration of each segment in seconds.
        max_segments: Maximum number of segments to process for each audio file.

    Returns: list of Numpy arrays, where each array contains the statistics vector per segment.
    """

    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    segment_samples = int(sr * segment_length)

    # Pad the last segment if needed
    padded_length = max_segments * segment_samples
    audio = np.pad(audio, (0, max(0, padded_length - len(audio))), mode="edge")

    step_size = int(sr * 0.25)

    # Split into segments
    segments = [
        audio[step_size * i: step_size * i + segment_samples]
        for i in range(eu.movement_types[label])
    ]

    # Extract and aggregate MFCCs
    segment_features = []
    for segment in segments:

        if exp_name == "exp1_1":
            # Mfcc output-> 2d array of: (n_mfcc, time_frames).
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)

            # hstack to flatten the statistics
            # For each coefficient, we want statistics over all time frames, hence axis=1.

            segment_features.append(np.hstack([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.max(mfcc, axis=1),
                np.min(mfcc, axis=1),
            ]))

        elif exp_name == "exp1_2":
            segment_features.append(calculate_mfcc_quarter_statistics_1_2(segment, sr=44100, num_coef=13))

        else:  # exp1_3
            segment_features.append(calculate_mfcc_quarter_statistics_1_3(segment, sr=44100, num_coef=13))

    return segment_features


# Experiments list
exp_list = ["exp1_1", "exp1_2", "exp1_3"]

if __name__ == "__main__":

    # Select Experiment:
    # 0: experiment 1.1, 1: experiment 1.2, 2: experiment 1.3
    exp_select = 2
    exp_name = exp_list[exp_select]

    # Path to save plots
    plots_path = Path(eu.project_root / f"plots/exp1")
    plots_path.mkdir(parents=True, exist_ok=True)

    # Path to save text output
    output_path = Path(eu.project_root / f"output/exp1")
    output_path.mkdir(parents=True, exist_ok=True)

    output_name = f"{output_path}/{exp_name}_predictions.txt"
    open(output_name, "w").close()

    audio_files, labels, recording_ids = eu.paths_and_labels(eu.OLD_DATA)

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

    # Encode labels
    label_encoder = LabelEncoder()
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

        # Train Random Forest Classifier
        clf = RandomForestClassifier(random_state=r_s)
        clf.fit(X_train, y_train)

        # Predictions
        y_pred = clf.predict(X_test)

        # Classification report
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
        all_reports.append(report)

        with open(output_name, "a") as f:
            print(f"Fold {fold_num}:\n", file=f)
            print("Accuracy:", accuracy_score(y_test, y_pred), file=f)
            print("Classification Report:\n", file=f)
            print(classification_report(y_test, y_pred, target_names=label_encoder.classes_), file=f)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        all_cm.append(cm)

        fold_num += 1

    # Calculate and print average F1-scores across folds
    average_f1_scores = {
        class_name: np.mean([r[class_name]["f1-score"] for r in all_reports])
        for class_name in label_encoder.classes_
    }
    with open(output_name, "a") as f:
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

    plt.tight_layout()
    plt.savefig(f"{plots_path}/{exp_name}_cm.png")
    plt.close()
