# Final Print Results

This folder contains saved print results from running the final code (experiment 4.2).

## Folder Structure

- `split_reduction_and_predictions/` - All results from the train code (`project_code/core/main_train_all.py`).
  * `dwt/` - All results for DWT feature type.
  * `harmonic/` - All results for HPSS feature type.
  * `mfcc/` - All results for MFCC feature type.
- `total_full_predictions.txt` - The final average prediction rates (over 5 folds), calculated for the full recordings.
Obtained from the test code (`project_code/core/main_test_all.py`)
- `total_split_predictions.txt` - The final average prediction rates (over 5 folds), calculated for the split recordings.

### Within each folder in split_reduction_and_predictions/:
- `mlp_predictions.txt` - Prediction rates for all 5 folds achieved from MLP classifier, for each feature number reduction step.
- `random forest_predictions.txt` - Prediction rates for all 5 folds achieved from Random Forest classifier, for each feature number reduction step.
- `svm rbf_predictions.txt` - Prediction rates for all 5 folds achieved from SVM classifier, for each feature number reduction step.