''' This file contains code regarding Experiment 3.1, specifically the raw feature-based model. '''

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVC

import exp_utils as eu


# ------Feature extraction functions------

def generate_mfcc_feature_names():
    """

    Returns: Names of MFCC features

    """
    coefs = 13
    frames = 259

    feature_names = [f"coef_{c+1}_frame_{f+1}" for c in range(coefs) for f in range(frames)]

    return feature_names


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

    all_feature_names = generate_mfcc_feature_names()
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
                plt.title(f"SHAP-Features, {label1}: predicted class {label2} (Incorrect)")
                plt.tight_layout()
                plt.savefig(plots_path/f"shap_predicted_{label1}_incorrect_{label2}.png")
            case 1:
                plt.title(f"SHAP-Features, {label1}: predicted class {label1} (Correct)")
                plt.tight_layout()
                plt.savefig(plots_path/f"shap_predicted_{label1}_correct_{label1}.png")
            case 2:
                plt.title(f"SHAP-Features, {label2}: predicted class {label1} (Incorrect)")
                plt.tight_layout()
                plt.savefig(plots_path/f"shap_predicted_{label2}_incorrect_{label1}.png")
            case 3:
                plt.title(f"SHAP-Features, {label2}: predicted class {label2} (Correct)")
                plt.tight_layout()
                plt.savefig(plots_path/f"shap_predicted_{label2}_correct_{label2}.png")
        plt.close()


def run_classifier():
    """
        Runs the chosen classifier and calculates classification_report and confusion_matrix.

    Args:
        sel: select clf type

    """

    rf = RandomForestClassifier(random_state=r_s)
    rf.fit(X_train, y_train)

    # Extract important features (for other classifiers as well)
    if feature_type == "mfcc":
        if clf_type == "mlp":
            selector = SelectFromModel(rf, threshold="0.75* mean", prefit=True)
        else:
            selector = SelectFromModel(rf, threshold="mean", prefit=True)
    elif feature_type == "harmonic":
        if clf_type == "svm rbf":
            selector = SelectFromModel(rf, threshold="mean", prefit=True)
        else:
            selector = SelectFromModel(rf, threshold="0.75* mean", prefit=True)
    else:   # dwt
        if clf_type == "random forest":
            selector = SelectFromModel(rf, threshold="1.25* mean", prefit=True)
        elif clf_type == "svm rbf":
            selector = SelectFromModel(rf, threshold="0.75* mean", prefit=True)
        else:
            selector = SelectFromModel(rf, threshold="mean", prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    supp = selector.get_support()

    with open(prediction_name, "a") as f:
        print(f"Reduced from {X_train.shape[1]} to {X_train_selected.shape[1]}", file=f)

    if clf_type == "random forest":
        rf.fit(X_train_selected, y_train)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test_selected)
        y_pred = rf.predict(X_test_selected)

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

    # For the first fold, and random forest classifier:
    # Plot misclassified and the average vector for labels: ("towards", "away", "turn")

    if fold_num == 1 and clf_type == "random forest":
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
        label1 = "towards"
        move1_idx = 482
        label2 = "away"
        move2_idx = 180
        plot_mfcc_shap_confusion(explainer, shap_values, X_test_selected, supp, label1, label2, move1_idx, move2_idx)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    all_reports.append(report)

    # Print classification results
    with open(prediction_name, "a") as f:
        print(f"Split {fold_num}:\n", file=f)
        print("Accuracy:", accuracy_score(y_test, y_pred), file=f)
        print("Classification Report:\n", file=f)
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_), file=f)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    all_cm.append(cm)


if __name__ == "__main__":

    start = time.time()

    exp_name = "exp3"

    # Select classifier:
    # 0: random forest, 1: svm, 2: mlp
    sel_clf = 0
    clf_type = eu.classifiers_list[sel_clf]

    feature_type = "mfcc"

    audio_files, labels, recording_ids = eu.paths_and_labels(eu.NEW_DATA)

    # Path to load/save statistics/raw features arrays
    data_path = Path(eu.project_root / f"data/new/{feature_type}/train")
    data_path.mkdir(parents=True, exist_ok=True)
    stats_npz = Path(data_path / f"{feature_type}_stats_data.npz")
    features_npz = Path(data_path / f"{feature_type}_features_data.npz")

    # Path to save plots
    plots_path = Path(eu.project_root / f"plots/{exp_name}/features_model/{feature_type}")
    plots_path.mkdir(parents=True, exist_ok=True)

    # Path to save text output
    output_path = Path(eu.project_root / f"output/{exp_name}/features_model/{feature_type}")
    output_path.mkdir(parents=True, exist_ok=True)

    prediction_name = f"{output_path}/{clf_type}_predictions.txt"
    open(prediction_name, "w").close()
    importances_name = f"{output_path}/{exp_name}_importances.txt"
    open(importances_name, "w").close()
    if clf_type == "random forest":
        shap_name = f"{output_path}/shap_confidence.txt"
        open(shap_name, "w").close()

    if features_npz.exists() and stats_npz.exists():
        data = np.load(features_npz, allow_pickle=True)
        features = data["raw_features"]
        features = features.reshape(features.shape[0], -1)  # from flat vector to 2d raw features
        # Load labels and ids from stats npz file
        data = np.load(stats_npz, allow_pickle=True)
        all_labels = data["all_labels"]
        file_ids = data["file_ids"]
    else:
        raise Exception("Please run exp2_2_and_3 first, to generate.")

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
    for train_idx, test_idx in splitter.split(features, numeric_labels, groups=file_ids):
        # Train/Test split
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = numeric_labels[train_idx], numeric_labels[test_idx]

        run_classifier()

        fold_num += 1

    # Calculate and print average F1-scores across splits
    average_f1_scores = {
        class_name: np.mean([r[class_name]["f1-score"] for r in all_reports])
        for class_name in label_encoder.classes_
    }
    with open(prediction_name, "a") as f:
        print("Average F1-scores over splits:", average_f1_scores, file=f)

    # Plot all confusion matrices as subplots
    fig, axes = plt.subplots(1, len(all_cm), figsize=(20, 5))
    for i, cm in enumerate(all_cm):
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_, ax=ax)
        ax.set_title(f"Split {i + 1}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    print("----Elapsed time: %.2f seconds----" % (time.time() - start))

    plt.tight_layout()
    plt.savefig(f"{plots_path}/exp3_1_cm.png")
    plt.close()
