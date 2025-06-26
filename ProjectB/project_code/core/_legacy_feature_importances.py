''' NOTE: This is an old experiment script. Do not use for final run.
    This file contains code to train a single classifier type from a single feature type,
    while iterating over the number of classifier features fed to the model. '''

import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVC

import plot_utils as pu
import stat_utils as su

fold_importances = [[] for i in range(5)]


def get_importances(X_train, y_train):
    """
    Args:
        X_train: train data
        y_train: train (true) labels

    Returns: classifier features importances

    """
    rf_selector = RandomForestClassifier(random_state=r_s, n_jobs=-1)
    rf_selector.fit(X_train, y_train)
    importances = rf_selector.feature_importances_
    fold_importances[fold_idx] = np.argsort(importances)[::-1]
    return rf_selector.feature_importances_


def run_classifier(X_train_sel, X_test_sel):
    """
            Runs the chosen classifier and calculates classification_report and confusion_matrix.

        Args:
            sel: select clf type

        """
    if clf_type == "random forest":
        clf = RandomForestClassifier(random_state=r_s)
        clf.fit(X_train_sel, y_train)
        y_pred = clf.predict(X_test_sel)
    else:  # clf type is svm or mlp
        if feature_type == "dwt":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)
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

    return y_pred


def get_important_idx():
    """

    Returns: the indices of the most important classifier features

    """
    sorted_idx = np.argsort(total_importances)[::-1]
    mean_importance = np.mean(total_importances)

    if feature_type == "mfcc":
        important_idx = sorted_idx[total_importances[sorted_idx] >= 0.95 * mean_importance]
    elif feature_type == "harmonic":
        important_idx = sorted_idx[total_importances[sorted_idx] >= 1.2 * mean_importance]
    else:
        important_idx = sorted_idx[total_importances[sorted_idx] >= 0.55 * mean_importance]

    return important_idx


def clf_select_features(important_idx):
    """
        Runs the chosen classifier and calculates classification_report and confusion_matrix.

    Args:
        sel: select clf type

    """
    max_features = len(important_idx)
    # Create ranges
    range1 = list(range(1, min(11, max_features + 1)))  # 1 to 10
    range2 = list(range(10, min(101, max_features + 1), 10))  # 10 to 100
    range3 = list(range(100, max_features + 1, 20))  # 100 to max

    # Combine and deduplicate while keeping order
    if clf_type == "svm rbf":  # avoid num_features in [1,10) as the pca.num_components already reaches 1
        n_features_list = list(dict.fromkeys(range2 + range3))
    else:
        n_features_list = list(dict.fromkeys(range1 + range2 + range3))
    n_features_list.reverse()

    for n_feat in n_features_list:
        top_idx = important_idx[:n_feat]
        X_train_selected = X_train[:, top_idx]
        X_test_selected = X_test[:, top_idx]

        y_pred = run_classifier(X_train_selected, X_test_selected)

        print(f"Fold {fold_num}: Reduced from {X_train.shape[1]} to {X_train_selected.shape[1]}")

        # Classification report
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
                                       , zero_division=0)

        # Store per-class F1 scores
        for class_name in label_encoder.classes_:
            class_f1 = report[class_name]["f1-score"]
            all_class_f1_per_feat[n_feat][class_name].append(class_f1)

        print(f"Fold {fold_num} | Top {n_feat} features | Class-wise F1s:")
        for class_name in label_encoder.classes_:
            print(f"  {class_name}: {report[class_name]['f1-score']:.4f}")

    return important_idx


if __name__ == "__main__":

    start = time.time()

    # Classifier name
    sel_clf = 0
    clf_type = pu.classifiers_list[sel_clf]

    # Feature name
    sel_feature = 0
    feature_type = pu.features_list[sel_feature]

    # Assign True to overwrite existing reports_npz data
    rewrite = False

    audio_files, labels, recording_ids = su.paths_and_labels_split()

    # Path to load/save statistics vectors
    stats_path = Path(pu.project_root/f"data/new/{feature_type}/train")
    stats_path.mkdir(parents=True, exist_ok=True)

    stats_npz = Path(stats_path / f"{feature_type}_stats_data.npz")

    fold_num = 1
    r_s = 1

    # StratifiedGroupKFold to split into folds while keeping file segments together
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r_s)

    # Load saved statistisc data if exists, otherwise calculate and save
    if stats_npz.exists():
        data = np.load(stats_npz, allow_pickle=True)
        stats = data["stats"]
        all_labels = data["all_labels"]
        file_ids = data["file_ids"]
    else:
        # Extract features and ensure segments are grouped correctly
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

    clf_path = Path(stats_path / f"{clf_type}")
    clf_path.mkdir(parents=True, exist_ok=True)
    reports_npz = Path(clf_path / "reports_by_feature_num.npz")

    if reports_npz.exists():
        data = np.load(reports_npz, allow_pickle=True)
        all_class_f1_per_feat = data["reports"].item()
    else:
        all_class_f1_per_feat = defaultdict(lambda: defaultdict(list))

        rf_selector = RandomForestClassifier(random_state=r_s, n_jobs=-1)
        rf_selector.fit(stats, numeric_labels)

        total_importances = np.zeros(stats.shape[1])
        fold_idx = 0

        # Split while maintaining file segments together (groups=file_ids)
        for train_idx, test_idx in splitter.split(stats, numeric_labels, groups=file_ids):
            # Train/Test split
            X_train, X_test = stats[train_idx], stats[test_idx]
            y_train, y_test = numeric_labels[train_idx], numeric_labels[test_idx]

            total_importances += get_importances(X_train, y_train)
            fold_idx += 1

        important_idx = get_important_idx()

        for train_idx, test_idx in splitter.split(stats, numeric_labels, groups=file_ids):
            X_train, X_test = stats[train_idx], stats[test_idx]
            y_train, y_test = numeric_labels[train_idx], numeric_labels[test_idx]

            clf_select_features(important_idx)

            fold_num += 1

    # Plot F1-score vs #features for each class
    plt.figure(figsize=(18, 10))

    feature_selection_folder_path = Path(pu.project_root/f"plots/final/{feature_type}/feature_selection")
    feature_selection_folder_path.mkdir(parents=True, exist_ok=True)

    for plot_idx, class_name in enumerate(label_encoder.classes_, start=1):
        plt.subplot(2, 3, plot_idx)
        x = []
        y = []
        for n_feat, class_scores in all_class_f1_per_feat.items():
            x.append(n_feat)
            y.append(np.mean(class_scores[class_name]))
        plt.plot(x, y, marker="o", label=class_name)

        plt.xlabel("Number of Features")
        plt.ylabel("F1-score")
        plt.title(f"Class: {class_name} F1-score per number of features")
        plt.legend(title="Class")
        plt.grid(True)

    plt.suptitle(f"F1-score vs Number of Features (per class)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(feature_selection_folder_path / f"selection_{clf_type}.png")
    plt.close()

    average_f1_per_num_feat = {
        n_feat: np.mean([np.mean(f1s) for f1s in all_class_f1_per_feat[n_feat].values()])
        for n_feat in all_class_f1_per_feat
    }

    max_avg_f1 = max(average_f1_per_num_feat.values())
    threshold = 0.97 * max_avg_f1

    best_n_feat = min(
        (n_feat for n_feat, score in average_f1_per_num_feat.items() if score >= threshold),
        default=None
    )
    print(f"Best number of features: {best_n_feat}")

    if rewrite or not reports_npz.exists():
        if reports_npz.exists():
            total_importances = np.zeros(stats.shape[1])
            fold_idx = 0

            # Split while maintaining file segments together (groups=file_ids)
            for train_idx, test_idx in splitter.split(stats, numeric_labels, groups=file_ids):
                # Train/Test split
                X_train, X_test = stats[train_idx], stats[test_idx]
                y_train, y_test = numeric_labels[train_idx], numeric_labels[test_idx]

                total_importances += get_importances(X_train, y_train)
                fold_idx += 1
            important_idx = get_important_idx()
        np.savez(reports_npz, reports=dict(all_class_f1_per_feat), mask=important_idx[:best_n_feat])

    best_scores = all_class_f1_per_feat[best_n_feat]
    for class_name, class_score_list in best_scores.items():
        print(f"Mean F1-score for class {class_name}: {np.mean(class_score_list):.4f}")
        print(f"F1-scores for class {class_name}: {class_score_list}")

    print("----Elapsed time: %.2f seconds----" % (time.time() - start))
