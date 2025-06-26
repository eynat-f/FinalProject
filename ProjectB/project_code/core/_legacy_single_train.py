''' NOTE: This is an old experiment script. Do not use for final run.
    This file contains code to train a single classifier type from a single feature type. '''

import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVC

import plot_utils as pu
import stat_utils as su


def run_classifier():
    """
        Runs the chosen classifier and calculates classification_report and confusion_matrix.

    Args:
        sel: select clf type

    """

    # Selector to derive feature importances
    rf_selector = RandomForestClassifier(random_state=r_s, n_jobs=-1)

    # Extract important features (for other classifiers as well)
    if feature_type == "mfcc":
        selector = SelectFromModel(rf_selector, threshold="mean")
    elif feature_type == "harmonic":
        selector = SelectFromModel(rf_selector, threshold="0.75* mean")
    else:
        if clf_type == "random forest":
            selector = SelectFromModel(rf_selector, threshold="1.25* mean")
        else:
            selector = SelectFromModel(rf_selector, threshold="mean")

    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    print(f"Fold {fold_num}: Reduced from {X_train.shape[1]} to {X_train_selected.shape[1]}")

    if clf_type == "random forest":
        clf = RandomForestClassifier(random_state=r_s)
        clf.fit(X_train_selected, y_train)
        y_pred = clf.predict(X_test_selected)

    else:  # clf type is svm or mlp
        if feature_type == "dwt":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)

        if clf_type == "svm rbf":
            pca = PCA(n_components=0.97)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)

            print(f"Fold {fold_num}: Number of components selected: {pca.n_components_}")

            svm = SVC(kernel="rbf", C=10, random_state=r_s)
            svm.fit(X_train_pca, y_train)

            # Predictions
            y_pred = svm.predict(X_test_pca)
        else:  # clf type is mlp
            if feature_type == "mfcc":
                mlp = MLPClassifier(hidden_layer_sizes=(300, 200, 100), max_iter=1000,
                                    activation="relu", solver="adam", random_state=r_s)
            elif feature_type == "dwt":
                mlp = MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=1000,
                                    activation="relu", solver="adam", random_state=r_s)
            else:
                mlp = MLPClassifier(hidden_layer_sizes=(400, 300, 200, 100), max_iter=1000,
                                    activation="relu", solver="adam", random_state=r_s)

            mlp.fit(X_train_scaled, y_train)

            y_pred = mlp.predict(X_test_scaled)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    all_reports.append(report)

    # Print classification results
    print(f"Fold {fold_num}:\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    all_cm.append(cm)


if __name__ == "__main__":

    start = time.time()

    # Classifier name
    sel_clf = 0
    clf_type = pu.classifiers_list[sel_clf]

    # Feature name
    sel_feature = 2
    feature_type = pu.features_list[sel_feature]

    audio_files, labels, recording_ids = su.paths_and_labels_split()

    # Path to load/save statistics/raw features arrays
    stats_path = Path(pu.project_root / f"data/new/{feature_type}/train")
    stats_path.mkdir(parents=True, exist_ok=True)

    stats_npz = Path(stats_path / f"{feature_type}_stats_data.npz")

    # Load saved statistisc data if exists, otherwise calculate and save
    if stats_npz.exists():
        data = np.load(stats_npz, allow_pickle=True)
        stats = data["stats"]
        all_labels = data["all_labels"]
        file_ids = data["file_ids"]
    else:
        # Extract stats and ensure segments are grouped correctly
        stats, all_labels, file_ids = [], [], []

        for file_path, label in zip(audio_files, labels):
            # Get multiple segments per file
            segment_stats = su.get_data_split(file_path, label, feature_type)
            stats.extend(segment_stats)  # Add all segments to stat list
            all_labels.extend([label] * len(segment_stats))  # Same label for all segments
            file_ids.extend([recording_ids[file_path.stem.split("_")[0]]] * len(segment_stats))
            # Group ID for StratifiedGroupKFold, accounts for segments in split recording, and for the full recording

        # Convert lists to arrays
        stats = np.array(stats)
        all_labels = np.array(all_labels)
        file_ids = np.array(file_ids)

        np.savez(stats_npz,
                 stats=stats,
                 all_labels=all_labels,
                 file_ids=file_ids)

    # Encode labels
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(all_labels)

    all_reports = []
    all_cm = []
    fold_num = 1
    r_s = 1

    # StratifiedGroupKFold to split into folds while keeping file segments together
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r_s)

    # Split while maintaining file segments together (groups=file_ids)
    for train_idx, test_idx in splitter.split(stats, numeric_labels, groups=file_ids):
        # Train/Test split
        X_train, X_test = stats[train_idx], stats[test_idx]
        y_train, y_test = numeric_labels[train_idx], numeric_labels[test_idx]

        run_classifier()

        fold_num += 1

    # Calculate and print average F1-scores across folds
    average_f1_scores = {
        class_name: np.mean([r[class_name]["f1-score"] for r in all_reports])
        for class_name in label_encoder.classes_
    }
    print("Average F1-scores over folds:", average_f1_scores)

    # Plot all confusion matrices as subplots
    pu.plot_cms(all_cm, label_encoder, feature_type, clf_type)

    print("----Elapsed time: %.2f seconds----" % (time.time() - start))

