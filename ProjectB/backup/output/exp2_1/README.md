# Experiment 2.1

This folder contains saved print results from running experiment 2.1.

## Folder Structure

- `dwt/` - All results for DWT feature type.
- `harmonic/` - All results for HPSS feature type.
- `mfcc/` - All results for MFCC feature type.

### Within each folder:
- `exp2_1_importances.txt` - Top 50 highest feature importances for fold 1 (from Random Forest _importances).
- `mlp_predictions.txt` - Prediction rates for all 5 folds achieved from MLP classifier.
- `random forest_predictions.txt` - Prediction rates for all 5 folds achieved from Random Forest classifier.
- `svm rbf_predictions.txt` - Prediction rates for all 5 folds achieved from SVM classifier.