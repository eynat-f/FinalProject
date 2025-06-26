''' NOTE: This is an old experiment script. Do not use for final run.
    This file contains code to test a single classifier type from a single feature type. '''

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

    idle_path = Path(pu.project_root/"recordings/noise_no_move/01.07-15-33-28.068.wav")
    linear_path = Path(pu.project_root/"recordings/noise_move/vertical/01.07-17-57-48.461.wav")

    # Classifier name
    sel_clf = 0
    clf_type = pu.classifiers_list[sel_clf]

    # Feature name
    sel_feature = 0
    feature_type = pu.features_list[sel_feature]

    audio_files = su.paths_and_labels_full()

    data_folder_path = Path(pu.project_root/"data/new")
    data_folder_path.mkdir(parents=True, exist_ok=True)

    label_encoder = joblib.load(data_folder_path / "label_encoder.pkl")

    with open(data_folder_path / "recording_ids.json", "r") as f:
        recording_ids = json.load(f)

    with open(data_folder_path / "mic_position_map.json", "r") as f:
        pos_labels = json.load(f)

    fold_folder_path = Path(pu.project_root/"data/new/fold_info")
    fold_folder_path.mkdir(parents=True, exist_ok=True)

    all_reports = []
    all_cm = []

    clf_folder_path = Path(pu.project_root/f"data/new/{feature_type}/train/{clf_type}")
    clf_folder_path.mkdir(parents=True, exist_ok=True)

    counter_tests = 0
    segment_res = []
    section_res = []
    weighted_section_res = []

    for fold_num in range(1, 6):  # 5 folds

        idle_sample_res = None
        idle_idx = 0

        # Load train/test split info
        with open(fold_folder_path / f"test_ids_{fold_num}.json", "r") as f:
            fold_info = json.load(f)

        counter_tests += len(fold_info["test_files"])

        # Load trained classifier and label encoder
        model_data = joblib.load(clf_folder_path / f"{clf_type}_{fold_num}.joblib")
        clf = model_data["clf"]
        select_mask = model_data["select_mask"]
        if sel_clf == 1:
            pca = model_data["pca"]
        if sel_clf != 0:
            scaler = model_data["scaler"]

        section_offset = 0

        all_stats = []
        all_indices = []
        all_true_labels = []
        file_metadata = []
        file_section_start_indices = []
        file_segment_lengths = []

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

        all_stats_selected = all_stats[:, select_mask]
        if sel_clf == 0:
            y_pred = clf.predict(all_stats_selected)
        elif sel_clf == 1:
            y_pred = clf.predict(pca.transform(scaler.transform(all_stats_selected)))
        else:
            y_pred = clf.predict(scaler.transform(all_stats_selected))

        pred_labels = label_encoder.inverse_transform(y_pred)

        true_segment_labels = [all_true_labels[i] for i in all_indices]
        pred_segment_labels = [pred_labels[i] for i in range(len(all_indices))]

        correct_segments = sum(t == p for t, p in zip(true_segment_labels, pred_segment_labels))
        segment_res.append(100 * correct_segments / len(pred_segment_labels))

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

        section_res.append(100 * correct_sections / len(section_segments))
        weighted_section_res.append(100 * correct_weighted / total_segments)

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

            if idle_path == file_path:
                idle_pred_segments_labels = segment_preds
                idle_true_segments_labels = true_segment_labels[segment_start:segment_start + segment_len]

                idle_seg_acc = sum(t == p for t, p in zip(idle_true_segments_labels, idle_pred_segments_labels))

                idle_seg_acc_percent = idle_seg_acc / len(idle_true_segments_labels)

                idle_pred_segments_labels = np.array(idle_pred_segments_labels)
                idle_true_segments_labels = np.array(idle_true_segments_labels)
                idle_wrong_true = idle_true_segments_labels[idle_pred_segments_labels != idle_true_segments_labels]
                idle_wrong_pred = idle_pred_segments_labels[idle_pred_segments_labels != idle_true_segments_labels]
                idle_wrong_pairs = list(zip(idle_wrong_true, idle_wrong_pred))

                print(f"Idle Segment Accuracy: {idle_seg_acc_percent * 100:.2f}%, ")
                print(f"Incorrect Predictions: {idle_wrong_pairs}")

                idle_pred_section_labels = section_preds
                idle_true_sections_labels = [all_true_labels[section_start + i]
                                             for i in range(section_end - section_start)]

                idle_sec_acc = sum(t == p for t, p in zip(idle_true_sections_labels, idle_pred_section_labels))
                idle_sec_acc_percent = idle_sec_acc / len(idle_true_sections_labels)

                print(f"Idle Section Accuracy: (Prediction, Actual):{idle_sec_acc_percent * 100:.2f}%")

            elif linear_path == file_path:
                linear_pred_segments_labels = segment_preds
                linear_true_segments_labels = true_segment_labels[segment_start:segment_start + segment_len]

                linear_seg_acc = sum(t == p for t, p in zip(linear_true_segments_labels, linear_pred_segments_labels))
                linear_wrong = linear_true_segments_labels[linear_pred_segments_labels != linear_true_segments_labels]
                linear_seg_acc_percent = linear_seg_acc / len(linear_true_segments_labels)

                linear_pred_segments_labels = np.array(linear_pred_segments_labels)
                linear_true_segments_labels = np.array(linear_true_segments_labels)
                linear_wrong_true = linear_true_segments_labels[linear_pred_segments_labels != linear_true_segments_labels]
                linear_wrong_pred = linear_pred_segments_labels[linear_pred_segments_labels != linear_true_segments_labels]
                linear_wrong_pairs = list(zip(linear_wrong_true, linear_wrong_pred))

                print(f"Linear Segment Accuracy: {linear_seg_acc_percent * 100:.2f}%")
                print(f"Incorrect Predictions: (Prediction, Actual):{linear_wrong_pairs}")

                linear_pred_section_labels = section_preds
                linear_true_sections_labels = [all_true_labels[section_start + i]
                                             for i in range(section_end - section_start)]

                linear_sec_acc = sum(t == p for t, p in zip(linear_true_sections_labels, linear_pred_section_labels))
                linear_sec_acc = linear_sec_acc / len(linear_true_sections_labels)

                print(f"Linear Section Accuracy: {linear_sec_acc * 100:.2f}%")

    segment_average = np.sum(segment_res) / 5
    section_average = np.sum(section_res) / 5
    weighted_section_average = np.sum(weighted_section_res) / 5
    print(f"Average segment prediction rate for {feature_type} {clf_type} over all folds: {segment_average:.2f}")
    print(f"Average section prediction rate for {feature_type} {clf_type} over all folds: {section_average:.2f}")
    print(f"Average weighted section prediction rate for {feature_type} {clf_type} over all folds: "
          f"{weighted_section_average:.2f}")

    print("----Elapsed time: %.2f seconds----" % (time.time() - start))
