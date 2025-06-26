# Experiment 3

This folder contains saved plots results from running experiment 3.

## Folder Structure

- `features_model/` - Model based on raw-features vector.
  * `mfcc/` - All results for MFCC feature type.
- `stats_model/` - Model based on statistics vector.
  * `dwt/` - All results for DWT feature type.
  * `harmonic/` - All results for HPSS feature type.
  * `mfcc/` - All results for MFCC feature type.

### Within the `stats_model/` folder, for each sub-folder:
- `global_per_class.png` - Global SHAP: most important features for each class.
- `mlp_cm.png` - Confusion matrices for all 5 folds achieved from MLP classifier.
- `random forest_cm.png` - Confusion matrices for all 5 folds achieved from Random Forest classifier.
- `svm rbf_cm.png` - Confusion matrices for all 5 folds achieved from SVM classifier.

### Additionally:
- For the `features_model/mfcc/` folder:
  * `exp3_1_cm.png` - Confusion matrices for all 5 folds achieved from Random Forest classifier.

- For both `features_model/mfcc/` and `stats_model/mfcc/` folders:
  * `shap_predicted_away_correct_away.png` - Local SHAP: confidence in prediction of sample 'away' as 'away'.
  * `shap_predicted_away_incorrect_towards.png` - Local SHAP: confidence in prediction of sample 'away' as 'towards'.
  * `shap_predicted_towards_correct_towards.png` - Local SHAP: confidence in prediction of sample 'towards' as 'towards'.
  * `shap_predicted_towards_incorrect_away.png` - Local SHAP: confidence in prediction of sample 'towards' as 'away'.
