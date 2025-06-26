# Experiment 3

This folder contains saved print results from running experiment 3.

## Folder Structure

- `features_model/` - Model based on raw-features vector.
  * `mfcc/` - All results for MFCC feature type.
- `stats_model/` - Model based on statistics vector.
  * `dwt/` - All results for DWT feature type.
  * `harmonic/` - All results for HPSS feature type.
  * `mfcc/` - All results for MFCC feature type.

### Within all folders:
- `mlp_predictions.txt` - Prediction rates for all 5 folds achieved from MLP classifier.
- `random forest_predictions.txt` - Prediction rates for all 5 folds achieved from Random Forest classifier.
- `svm rbf_predictions.txt` - Prediction rates for all 5 folds achieved from SVM classifier.

### Additionally:
- For the `features_model/mfcc/` folder:
  * `exp3_importances.txt` - Top 50 highest feature importances for fold 1 (from Random Forest _importances).
  * `shap_confidence.txt` - The SHAP confidence rate for the two examined samples: 'towards' and 'away' samples that were misclassified as the other class.

- For the `stats_model/mfcc/` folder:
  * `shap_confidence.txt` - The SHAP confidence rate for the three examined samples:
    * The most confidently predicted sample.
    * Same 'towards' and 'away' samples as in `features_model/mfcc/shap_confidence.txt`, classified correctly by this model.
