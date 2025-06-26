''' This file contains code to train all combinations of classifier types and feature types over
    the split recordings, while iterating over the number of classifier features fed to the model.
    This is the first code that should be run (if no model has been saved yet). '''

import json
import time
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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


def run_classifier(X_train_sel, X_test_sel, save_flag=False, select_mask=None, clf_folder_path=None):
    """
            Runs the chosen classifier and calculates classification_report and confusion_matrix.

        Args:
            sel: select clf type

        """
    # Set as True to plot data for samples mistakenly predicted as the other class
    plot_mistaken = False

    # Set as True to plot SHAP info
    plot_shap = False

    if clf_type == "random forest":
        clf = RandomForestClassifier(random_state=r_s)
        clf.fit(X_train_sel, y_train)
        y_pred = clf.predict(X_test_sel)
        if save_flag:
            joblib.dump({"clf": clf, "select_mask": select_mask}, clf_folder_path / f"{clf_type}_{fold_num}.joblib")
            if plot_shap:
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_test_sel)
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

            with open(prediction_name, "a") as f:
                print(f"Number of components selected after PCA: {pca.n_components_}", file=f)

            clf = SVC(kernel="rbf", C=10, random_state=r_s)
            clf.fit(X_train_pca, y_train)

            y_pred = clf.predict(X_test_pca)
            if save_flag:
                joblib.dump({"scaler": scaler, "pca": pca, "clf": clf, "select_mask": select_mask},
                            clf_folder_path / f"{clf_type}_{fold_num}.joblib")
        else:  # clf type is mlp
            if feature_type == "mfcc":
                clf = MLPClassifier(hidden_layer_sizes=(300, 200, 100), max_iter=1000,
                                    activation="relu", solver="adam", random_state=r_s)
            elif feature_type == "dwt":
                clf = MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=1000,
                                    activation="relu", solver="adam", random_state=r_s)
            else:
                clf = MLPClassifier(hidden_layer_sizes=(400, 300, 200, 100), max_iter=1000,
                                    activation="relu", solver="adam", random_state=r_s)

            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

            if save_flag:
                joblib.dump({"scaler": scaler, "clf": clf, "select_mask": select_mask},
                            clf_folder_path / f"{clf_type}_{fold_num}.joblib")

    # Plot SHAP important features
    if plot_shap and clf_type == "random forest" and save_flag:
        pu.plot_shap_importances(shap_values, select_mask, feature_type, clf_type, fold_num)

    if plot_mistaken and save_flag and (fold_num == 1) and plot_features:
        raw_features_test = raw_features[test_idx]
        # Change labels as needed
        label1 = "towards"
        label2 = "away"
        # Plot misclassified (if exists)
        move1_idx, move2_idx = pu.plot_misclassified(X_test, raw_features_test, y_pred,
                                                     y_test, label1, label2, feature_type)

        # Plot SHAP mistaken predictions (if exist)
        if plot_shap and clf_type == "random forest" and move1_idx is not None and move2_idx is not None:
            pu.plot_mfcc_shap_confusion(explainer, shap_values, X_test_sel, select_mask,
                                        label1, label2, move1_idx, move2_idx)

    return y_pred


def clf_select_features():
    """
        Runs the chosen classifier and returns the predicted labels y_pred.

    """
    rf_selector = RandomForestClassifier(random_state=r_s, n_jobs=-1)
    rf_selector.fit(X_train, y_train)

    sorted_idx = np.argsort(total_importances)[::-1]
    mean_importance = np.mean(total_importances)

    if feature_type == "mfcc":
        important_idx = sorted_idx[total_importances[sorted_idx] >= 0.95 * mean_importance]
    elif feature_type == "harmonic":
        important_idx = sorted_idx[total_importances[sorted_idx] >= 1.2 * mean_importance]
    else:
        important_idx = sorted_idx[total_importances[sorted_idx] >= 0.55 * mean_importance]

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

        with open(prediction_name, "a") as f:
            print(f"Fold {fold_num}: Reduced from {X_train.shape[1]} to {X_train_selected.shape[1]}", file=f)

        # Classification report
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

        # Store per-class F1 scores
        for class_name in label_encoder.classes_:
            class_f1 = report[class_name]["f1-score"]
            all_class_f1_per_feat[n_feat][class_name].append(class_f1)

        with open(prediction_name, "a") as f:
            print(f"Fold {fold_num} | Top {n_feat} features | Class-wise F1s:", file=f)
            for class_name in label_encoder.classes_:
                print(f"  {class_name}: {report[class_name]['f1-score']:.4f}", file=f)

    return important_idx


if __name__ == "__main__":

    start = time.time()

    # Keep at false, as the vector (harmonic raw features) is very large
    plot_features = False

    # Create dir to save classifier information (recording ids and label encoder)
    data_folder_path = Path(pu.project_root / "data/new")
    data_folder_path.mkdir(parents=True, exist_ok=True)

    total_output_path = Path(pu.project_root / f"output/final/")
    total_output_path.mkdir(parents=True, exist_ok=True)

    total_prediction_name = f"{total_output_path}/total_split_predictions.txt"
    open(total_prediction_name, "w").close()

    audio_files, labels, recording_ids = su.paths_and_labels_split()

    all_cm = {}
    all_reports = {}

    r_s = 1  # Random state

    # StratifiedGroupKFold to split into folds while keeping file segments together
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r_s)

    # Loop over different features
    for feature_type in pu.features_list:
        fold_num = 1

        # Initialize dictionary for storing reports and cms per classifier, per feature
        all_reports[feature_type] = {clf: [] for clf in pu.classifiers_list}
        all_cm[feature_type] = {clf: [] for clf in pu.classifiers_list}

        stats_path = Path(pu.project_root / f"data/new/{feature_type}/train")
        stats_path.mkdir(parents=True, exist_ok=True)

        features_npz = Path(stats_path / f"{feature_type}_features_data.npz")
        stats_npz = Path(stats_path / f"{feature_type}_stats_data.npz")

        # Load saved statistisc data if exists, otherwise calculate and save
        if stats_npz.exists():
            data = np.load(stats_npz, allow_pickle=True)
            stats = data["stats"]
            all_labels = data["all_labels"]
            file_ids = data["file_ids"]

            if feature_type == "mfcc" and plot_features:
                data2 = np.load(features_npz, allow_pickle=True)
                raw_features = data2["raw_features"]
                pu.plot_sample_features(raw_features, all_labels, feature_type)
        else:
            # Extract features and ensure segments are grouped correctly
            stats, raw_features, all_labels, file_ids = [], [], [], []

            for file_path, label in zip(audio_files, labels):
                # Get multiple segments per file
                segment_stats, segment_features = su.get_data_split(file_path, label, feature_type)
                stats.extend(segment_stats)  # Add all segments to stat list
                raw_features.extend(segment_features)  # Add all segments to raw feature list
                all_labels.extend([label] * len(segment_stats))  # Same label for all segments
                file_ids.extend([recording_ids[file_path.stem.split("_")[0]]] * len(segment_stats))
                # Group ID for StratifiedGroupKFold, accounts for segments in split recording, and for the full recording

            # Convert lists to arrays
            stats = np.array(stats)
            raw_features = np.array(raw_features)
            all_labels = np.array(all_labels)
            file_ids = np.array(file_ids)

            np.savez(stats_npz,
                     stats=stats,
                     all_labels=all_labels,
                     file_ids=file_ids)

            if feature_type == "mfcc":
                np.savez(features_npz, raw_features=raw_features)

        # Encode labels
        label_encoder = LabelEncoder()
        numeric_labels = label_encoder.fit_transform(all_labels)

        # Calculate encoding for first feature type (identical for the rest)
        if feature_type == pu.features_list[0]:
            # Store label encoder (for later full recording regression)
            joblib.dump(label_encoder, data_folder_path / "label_encoder.pkl")

        # Loop over different classifiers
        for clf_type in pu.classifiers_list:

            all_class_f1_per_feat = defaultdict(lambda: defaultdict(list))
            fold_num = 1

            clf_path = Path(stats_path / f"{clf_type}")
            clf_path.mkdir(parents=True, exist_ok=True)
            reports_npz = Path(clf_path / "reports_by_feature_num.npz")

            if reports_npz.exists():
                data = np.load(reports_npz, allow_pickle=True)
                all_class_f1_per_feat = data["reports"].item()
                select_mask = data["mask"]
            else:
                # Declare here (if reports doesn't exist) to not overwrite the full log
                output_path = Path(pu.project_root / f"output/final/split_reduction_and_predictions/{feature_type}")
                output_path.mkdir(parents=True, exist_ok=True)

                prediction_name = f"{output_path}/{clf_type}_reduction.txt"
                open(prediction_name, "w").close()

                rf_selector = RandomForestClassifier(random_state=r_s, n_jobs=-1)
                rf_selector.fit(stats, numeric_labels)

                total_importances = np.zeros(stats.shape[1])

                # Split while maintaining file segments together (groups=file_ids)
                for fold_idx, train_idx, test_idx in enumerate(splitter.split(stats, numeric_labels, groups=file_ids),
                                                               start=1):
                    # Train/Test split
                    X_train, X_test = stats[train_idx], stats[test_idx]
                    y_train, y_test = numeric_labels[train_idx], numeric_labels[test_idx]

                    total_importances += get_importances(X_train, y_train)

                    # Store test/train split (only once per fold):
                    if feature_type == pu.features_list[0]:
                        # Create dir to save fold_info
                        fold_folder_path = Path(pu.project_root / "data/new/fold_info")
                        fold_folder_path.mkdir(parents=True, exist_ok=True)

                        fold_info = {
                            "test_files": [int(x) for x in set(file_ids[test_idx])]
                        }

                        # Train/Test split
                        with open(fold_folder_path / f"test_ids_{fold_idx}.json", "w") as f:
                            json.dump(fold_info, f)

                # Split while maintaining file segments together (groups=file_ids)
                for train_idx, test_idx in splitter.split(stats, numeric_labels, groups=file_ids):
                    # Train/Test split
                    X_train, X_test = stats[train_idx], stats[test_idx]
                    y_train, y_test = numeric_labels[train_idx], numeric_labels[test_idx]

                    important_idx = clf_select_features()

                    fold_num += 1

            # Plot F1-score vs number of features for each class
            plt.figure(figsize=(18, 10))

            feature_selection_folder_path = Path(pu.project_root / f"plots/final/{feature_type}/feature_selection")
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

            if not reports_npz.exists():
                select_mask = important_idx[:best_n_feat]
                np.savez(reports_npz, reports=dict(all_class_f1_per_feat), mask=select_mask)

            with open(prediction_name, "a") as f:
                print(f"Best number of features for {feature_type} {clf_type}: {len(select_mask)}", file=f)

            pu.print_importances(select_mask, feature_type, prediction_name)

            fold_num = 1
            for train_idx, test_idx in splitter.split(stats, numeric_labels, groups=file_ids):
                # Train/Test split
                X_train, X_test = stats[train_idx], stats[test_idx]
                y_train, y_test = numeric_labels[train_idx], numeric_labels[test_idx]

                X_train_new_selected = X_train[:, select_mask]
                X_test_new_selected = X_test[:, select_mask]

                # Create dir to save classifier
                clf_folder_path = Path(pu.project_root / f"data/new/{feature_type}/train/{clf_type}")
                clf_folder_path.mkdir(parents=True, exist_ok=True)

                clf_path = Path(clf_folder_path / f"{clf_type}_{fold_num}.joblib")

                y_pred = run_classifier(X_train_new_selected, X_test_new_selected, True, select_mask, clf_folder_path)
                report = classification_report(y_test, y_pred, target_names=label_encoder.classes_,
                                               output_dict=True, zero_division=0)

                for class_name in label_encoder.classes_:
                    class_f1 = report[class_name]["f1-score"]
                    with open(prediction_name, "a") as f:
                        print(f"Fold {fold_num}- Best F1-scores for class {class_name}: {class_f1:.4f}", file=f)

                # Store the classification report for this fold and classifier
                all_reports[feature_type][clf_type].append(report)

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                all_cm[feature_type][clf_type].append(cm)

                fold_num += 1

    # Calculate and print average F1-scores across folds
    average_f1_scores = {}
    for feature, classifier_reports in all_reports.items():
        for clf, reports in classifier_reports.items():
            f1_scores = {
                class_name: np.mean([r[class_name]["f1-score"] for r in reports])
                for class_name in label_encoder.classes_
            }
            average_f1_scores[f"{feature}_classifier_{clf}"] = f1_scores

    with open(total_prediction_name, "a") as f:
        print("Average F1-scores over folds (per feature type and classifier):", file=f)
        for classifier, scores in average_f1_scores.items():
            print(f"{classifier}:", file=f)
            for movement, score in scores.items():
                print(f"\t{movement}: {score}", file=f)

    # Plot all confusion matrices as subplots
    for feature in pu.features_list:
        for clf in pu.classifiers_list:
            cm_list = all_cm[feature][clf]

            # Plot all confusion matrices as subplots
            pu.plot_cms(cm_list, label_encoder, feature, clf)

    print("----Elapsed time: %.2f seconds----" % (time.time() - start))
