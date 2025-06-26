# Experiment 2.1

This folder contains saved plots results from running experiment 2.1.

## Folder Structure

- `dwt/` - All results for DWT feature type.
  * `stats/` - For each movement type:
    * `<movement_type>_stats.png` - DWT (regular) statistics plot of a sample of the movement type.
    * `<movement_type>_stats_autocorrelation.png` - DWT auto-correlation statistics plot of a sample of the movement type.
- `harmonic/` - All results for HPSS feature type.
  * `stats/` - For each movement type:
    * `<movement_type>_stats.png` - HPSS (regular) statistics plot of a sample of the movement type.
    * `<movement_type>_stats_magnitude.png` - HPSS magnitude statistics (full segment) plot of a sample of the movement type.
- `mfcc/` - All results for MFCC feature type.
  * `stats/` - For each movement type:
    * `<movement_type>_stats_over_coeffs.png` - Plot of MFCC statistics per time frame (full segment) of a sample of the movement type.
    * `<movement_type>_stats_over_time.png` - Plot of MFCC statistics per coefficient of a sample of the movement type.


### Within each feature type folder:
- `features/` - For each movement type:
  * `<movement_type>_features.png` - Raw features plot of a sample of the movement type.
- `mlp_cm.png` - Confusion matrices for all 5 folds achieved from MLP classifier.
- `random forest_cm.png` - Confusion matrices for all 5 folds achieved from Random Forest classifier.
- `svm rbf_cm.png` - Confusion matrices for all 5 folds achieved from SVM classifier.

### Additionally, for `mfcc/` :

- `features/` -
  * `misclassified/`
    * `mlp/towards_vs_turn_features.png` - Plot of samples of 'towards' and 'turn' that were misclassified as the other class, when using MLP classifier.
    * `random forest/towards_vs_turn_features.png` - Plot of samples of 'towards' and 'turn' that were misclassified as the other class, when using Random Forest classifier.
    * `svm rbf/towards_vs_turn_features.png` - Plot of samples of 'towards' and 'turn' that were misclassified as the other class, when using SVM classifier.
  * `features_all_samples.png` - MFCC raw features plots of samples of all movement types (same scale, for side-by-side comparison).
- `stats/` -
  * `misclassified/mlp/`, `misclassified/random forest/` and `misclassified/svm rb/` - For each quarter and for the full segment:
    * `<quarter_number/full_segment>_misclassified_turn_towards_stats_over_coeffs` - Plot of MFCC statistics per time frame of samples of 'towards' and 'turn' that were misclassified as the other class, for each classifier type.
    * `<quarter_number/full_segment>_misclassified_turn_towards_stats_over_time` - Plot of MFCC statistics per coefficient of samples of 'towards' and 'turn' that were misclassified as the other class, for each classifier type.
  * `average_all_stats_over_coeffs.png` - Plot of MFCC statistics per time frame (full segment) of the average of the train samples (fold 1).
  * `average_all_stats_over_time.png` - Plot of MFCC statistics per coefficient of the average of the train samples (fold 1).
  * `sample_all_stats_over_coeffs.png` - Plot of MFCC statistics per time frame (full segment) of samples of all movement types (overlay).
  * `sample_all_stats_over_time.png` - Plot of MFCC statistics per coefficient of samples of all movement types (overlay).


