''' This file contains code to test all combinations of classifier types and feature types over
    the full recordings, using the saved models. This is the second code that should be run. '''

import json
import time
from collections import defaultdict, Counter
from pathlib import Path

import joblib
import numpy as np

import plot_utils as pu
import stat_utils as su

if __name__ == "__main__":

    start = time.time()

    # Get the audio files from the full recordings
    audio_files = su.paths_and_labels_full()

    # Path to data folder
    data_folder_path = Path(pu.project_root/"data/new")
    data_folder_path.mkdir(parents=True, exist_ok=True)

    # Path to save text output
    total_output_path = Path(pu.project_root / f"output/final/")
    total_output_path.mkdir(parents=True, exist_ok=True)

    total_prediction_name = f"{total_output_path}/total_full_predictions.txt"
    open(total_prediction_name, "w").close()

    label_encoder = joblib.load(data_folder_path / "label_encoder.pkl")

    # Get ids for group cross-validation
    with open(data_folder_path / "recording_ids.json", "r") as f:
        recording_ids = json.load(f)

    # Get microphone positions for each flight
    with open(data_folder_path / "mic_position_map.json", "r") as f:
        pos_labels = json.load(f)

    # Get information for each fold (test ids)
    fold_folder_path = Path(pu.project_root/"data/new/fold_info")
    fold_folder_path.mkdir(parents=True, exist_ok=True)

    # Segment and section results
    segment_res = {
        feature_type: {
            clf_type: [] for clf_type in pu.classifiers_list
        }
        for feature_type in pu.features_list
    }
    section_res = {
        feature_type: {
            clf_type: [] for clf_type in pu.classifiers_list
        }
        for feature_type in pu.features_list
    }
    weighted_section_res = {
        feature_type: {
            clf_type: [] for clf_type in pu.classifiers_list
        }
        for feature_type in pu.features_list
    }

    for feature_type in pu.features_list:

        counter_tests = 0

        for fold_num in range(1, 6):  # 5 folds

            # Load train/test fold info
            with open(fold_folder_path / f"test_ids_{fold_num}.json", "r") as f:
                fold_info = json.load(f)

            counter_tests += len(fold_info["test_files"])

            section_offset = 0

            all_stats = []
            all_indices = []
            all_true_labels = []
            file_metadata = []
            file_section_start_indices = []
            file_segment_lengths = []

            # Path to load/save statistics vectors
            stats_path = Path(pu.project_root/f"data/new/{feature_type}/test")
            stats_path.mkdir(parents=True, exist_ok=True)

            stats_npz = Path(stats_path / f"{feature_type}_fold{fold_num}.npz")

            if stats_npz.exists():
                data = np.load(stats_npz, allow_pickle=True)
                all_stats = data["stats"]
                all_indices = data["indices"]
                all_true_labels = data["labels"]
                file_metadata = data["metadata"]
                file_section_start_indices = data["section_start"]
                file_segment_lengths = data["segment_length"]
            else:
                for file_path in audio_files:
                    file_name = file_path.stem.split("_")[0]
                    if recording_ids[file_name] in fold_info["test_files"]:
                        flight_type = pos_labels.get(file_name)
                        test_stats, section_labels, indices, num_secs = su.get_data_full_recording(file_path,
                                                                                            flight_type, feature_type)
                        all_stats.extend(test_stats)
                        all_indices.extend([i + section_offset for i in indices])
                        all_true_labels.extend(section_labels)
                        section_offset += len(section_labels)

                        # Ensures labels retain order!
                        unique_labels = list(dict.fromkeys(section_labels))

                        section_secs = [su.section_secs_dict[label] for label in unique_labels]

                        file_metadata.append((file_path, num_secs, section_secs))
                        file_section_start_indices.append(section_offset - len(section_labels))
                        file_segment_lengths.append(len(indices))

                all_stats = np.array(all_stats)
                all_indices = np.array(all_indices)
                all_true_labels = np.array(all_true_labels)
                file_metadata = np.array(file_metadata, dtype=object)
                file_section_start_indices = np.array(file_section_start_indices)
                file_segment_lengths = np.array(file_segment_lengths)

                np.savez(stats_npz, stats=all_stats,
                         indices=all_indices,
                         labels=all_true_labels,
                         metadata=file_metadata,
                         section_start=file_section_start_indices,
                         segment_length=file_segment_lengths)

            for clf_type in pu.classifiers_list:

                clf_folder_path = Path(pu.project_root/f"data/new/{feature_type}/train/{clf_type}")
                clf_folder_path.mkdir(parents=True, exist_ok=True)

                # Load trained classifier and label encoder
                model_data = joblib.load(clf_folder_path / f"{clf_type}_{fold_num}.joblib")
                clf = model_data["clf"]
                select_mask = model_data["select_mask"]
                if clf_type == "svm rbf":
                    pca = model_data["pca"]
                if clf_type != "random forest":
                    scaler = model_data["scaler"]

                all_stats_selected = all_stats[:, select_mask]

                if clf_type == "random forest":
                    y_pred = clf.predict(all_stats_selected)
                elif clf_type == "svm rbf":
                    y_pred = clf.predict(pca.transform(scaler.transform(all_stats_selected)))
                else:
                    y_pred = clf.predict(scaler.transform(all_stats_selected))

                pred_labels = label_encoder.inverse_transform(y_pred)

                true_segment_labels = [all_true_labels[i] for i in all_indices]
                pred_segment_labels = [pred_labels[i] for i in range(len(all_indices))]

                correct_segments = sum(t == p for t, p in zip(true_segment_labels, pred_segment_labels))
                segment_res[feature_type][clf_type].append(100 * correct_segments / len(pred_segment_labels))

                # Section accuracy
                section_segments = defaultdict(list)
                for i, section_idx in enumerate(all_indices):
                    section_segments[section_idx].append(i)

                correct_sections = 0
                correct_weighted = 0
                total_segments = len(pred_segment_labels)
                pred_section_labels = []

                for section_idx in sorted(section_segments):
                    segment_ids = section_segments[section_idx]
                    segment_preds = [pred_segment_labels[i] for i in segment_ids]
                    true_label = all_true_labels[section_idx]

                    # Majority vote with tie-breaking
                    majority_label = max(Counter(segment_preds).items(), key=lambda x: (x[1], x[0] == true_label))[0]
                    pred_section_labels.append(majority_label)

                    if majority_label == true_label:
                        correct_sections += 1
                        correct_weighted += len(segment_ids)

                section_res[feature_type][clf_type].append(100 * correct_sections / len(section_segments))
                weighted_section_res[feature_type][clf_type].append(100 * correct_weighted / total_segments)

                segment_start = 0

                for i, (file_path, num_secs, section_secs) in enumerate(file_metadata):
                    segment_len = file_segment_lengths[i]
                    section_start = file_section_start_indices[i]
                    section_end = section_start + len(section_secs)

                    segment_preds = pred_segment_labels[segment_start:segment_start + segment_len]
                    section_preds = pred_section_labels[section_start:section_end]

                    su.create_csv(file_path, num_secs, segment_preds, feature_type, clf_type, True)
                    su.create_csv(file_path, num_secs, section_preds, feature_type, clf_type, False, section_secs)

                    segment_start += segment_len

        with open(total_prediction_name, "a") as f:
            for clf_type in pu.classifiers_list:
                segment_average = np.sum(segment_res[feature_type][clf_type]) / 5
                section_average = np.sum(section_res[feature_type][clf_type]) / 5
                weighted_section_average = np.sum(weighted_section_res[feature_type][clf_type]) / 5
                print(f"Average segment prediction rate for {feature_type} {clf_type} over all folds: "
                      f"{segment_average:.2f}", file=f)
                print(f"Average section prediction rate for {feature_type} {clf_type} over all folds: "
                      f"{section_average:.2f}", file=f)
                print(f"Average weighted section prediction rate for {feature_type} {clf_type} over all folds: "
                      f"{weighted_section_average:.2f}\n", file=f)
            print("\n", file=f)

    print("----Elapsed time: %.2f seconds----" % (time.time() - start))
