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

import exp_utils as eu


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

            with open(prediction_name, "a") as f:
                print(f"Fold {fold_num}: Number of components selected: {pca.n_components_}", file=f)

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


def clf_select_features():
    """
        Runs the chosen classifier and calculates classification_report and confusion_matrix.

    Args:
        sel: select clf type

    """
    all_class_f1_per_feat = defaultdict(dict)

    rf_selector = RandomForestClassifier(random_state=r_s, n_jobs=-1)
    rf_selector.fit(X_train, y_train)
    importances = rf_selector.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    # Start with features above mean importance
    mean_imp = np.mean(importances)

    if feature_type == "mfcc":
        important_idx = sorted_idx[importances[sorted_idx] >= mean_imp]
        n_features_list = list(range(len(important_idx), 200, -20))
    elif feature_type == "harmonic":
        important_idx = sorted_idx[importances[sorted_idx] >= mean_imp]
        n_features_list = list(range(len(important_idx), 200, -30))
    else:
        important_idx = sorted_idx[importances[sorted_idx] >= 0.75 * mean_imp]
        n_features_list = list(range(len(important_idx), 20, -15))

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
            all_class_f1_per_feat[n_feat][class_name] = class_f1

        with open(prediction_name, "a") as f:
            print(f"Fold {fold_num} | Top {n_feat} features | Class-wise F1s:", file=f)
            for class_name in label_encoder.classes_:
                print(f"  {class_name}: {report[class_name]['f1-score']:.4f}", file=f)

    # Plot F1-score vs #features for each class
    plt.figure(figsize=(18, 10))

    for plot_idx, class_name in enumerate(label_encoder.classes_, start=1):
        plt.subplot(2, 3, plot_idx)
        x = []
        y = []
        for n_feat, class_scores in all_class_f1_per_feat.items():
            x.append(n_feat)
            y.append(np.mean(class_scores[class_name]))
        plt.plot(x, y, marker='o', label=class_name)

        plt.xlabel("Number of Features")
        plt.ylabel("F1-score")
        plt.title(f"Class: {class_name} F1-score per number of features")
        plt.legend(title="Class")
        plt.grid(True)

    plt.suptitle(f"F1-score vs Number of Features (per class) for fold {fold_num}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plots_path / f"selection_fold_{fold_num}.png")
    plt.close()

    average_f1_per_feat = {
        n_feat: {
            class_name: np.mean(f1s)
            for class_name, f1s in class_scores.items()
        }
        for n_feat, class_scores in all_class_f1_per_feat.items()
    }

    best_n_feat = max(
        average_f1_per_feat,
        key=lambda n: np.mean(list(average_f1_per_feat[n].values()))
    )
    with open(prediction_name, "a") as f:
        print(f"Best number of features for fold {fold_num}: {best_n_feat}", file=f)

        best_scores = all_class_f1_per_feat[best_n_feat]
        for class_name, class_score_list in best_scores.items():
            print(f"F1-scores for class {class_name}: {class_score_list:.4f}", file=f)


if __name__ == "__main__":

    start = time.time()

    exp_name = "exp4_1"

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

    # Path to save plots
    plots_path = Path(eu.project_root / f"plots/{exp_name}/{feature_type}/{clf_type}")
    plots_path.mkdir(parents=True, exist_ok=True)

    # Path to save text output
    output_path = Path(eu.project_root / f"output/{exp_name}/{feature_type}")
    output_path.mkdir(parents=True, exist_ok=True)

    prediction_name = f"{output_path}/{clf_type}_reduction_and_predictions.txt"
    open(prediction_name, "w").close()

    if stats_npz.exists():
        data = np.load(stats_npz, allow_pickle=True)
        stats = data["stats"]
        all_labels = data["all_labels"]
        file_ids = data["file_ids"]
    else:
        # Extract features and ensure segments are grouped correctly
        stats, all_labels, file_ids = [], [], []

        for file_path, label in zip(audio_files, labels):
            segment_stats = eu.get_features(feature_type, file_path, label)  # Get multiple segments per file
            stats.extend(segment_stats)  # Add all segments to stats list
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

    fold_num = 1
    r_s = 1

    rf_selector = RandomForestClassifier(random_state=r_s, n_jobs=-1)
    rf_selector.fit(stats, numeric_labels)

    # StratifiedGroupKFold to split while keeping file segments together
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r_s)

    # Split while maintaining file segments together (groups=file_ids)
    for train_idx, test_idx in splitter.split(stats, numeric_labels, groups=file_ids):
        # Train/Test split
        X_train, X_test = stats[train_idx], stats[test_idx]
        y_train, y_test = numeric_labels[train_idx], numeric_labels[test_idx]

        clf_select_features()

        fold_num += 1

    print("----Elapsed time: %.2f seconds----" % (time.time() - start))
