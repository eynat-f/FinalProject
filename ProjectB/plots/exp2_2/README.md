# Experiment 2.2

This folder contains saved plots results from running experiment 2.2.

## Folder Structure

- `dwt/` - All results for DWT feature type.
- `harmonic/` - All results for HPSS feature type.
- `mfcc/` - All results for MFCC feature type.

### Within each feature type folder:
- `features/` - For each movement type:
  * `<movement_type>_features.png` - Raw features plot of a sample of the movement type.
- `mlp_cm.png` - Confusion matrices for all 5 folds achieved from MLP classifier.
- `random forest_cm.png` - Confusion matrices for all 5 folds achieved from Random Forest classifier.
- `svm rbf_cm.png` - Confusion matrices for all 5 folds achieved from SVM classifier.

### Additionally, for `mfcc/` :

- `features/` -
  * `features_all_samples.png` - MFCC raw features plots of samples of all movement types (same scale, for side-by-side comparison).


