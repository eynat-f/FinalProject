# Experiment 4.1

This folder contains saved print results from running experiment 4.1.

## Folder Structure

- `dwt/` - All results for DWT feature type.
- `harmonic/` - All results for HPSS feature type.
- `mfcc/` - All results for MFCC feature type.

### Within each folder:
- `mlp_predictions.txt` - Prediction rates for all 5 folds achieved from MLP classifier, for each feature number reduction step.
- `random forest_predictions.txt` - Prediction rates for all 5 folds achieved from Random Forest classifier, for each feature number reduction step.
- `svm rbf_predictions.txt` - Prediction rates for all 5 folds achieved from SVM classifier, for each feature number reduction step.