# Final Plot Results

This folder contains saved plots results from running the final code (experiment 4.2).

## Folder Structure

- `cms/` - 
  * `<feature_type>_<classifier_type>_cms` - Confusion matrices for all 5 folds achieved from each feature type and classifier type combination.
- `dwt/` - All results for DWT feature type.
- `harmonic/` - All results for HPSS feature type.
- `mfcc/` - All results for MFCC feature type.

### Within each feature type folder:

- `feature_selection/` -
  * `selection_mlp` - Prediction rate for each movement type as a function of feature number, for MLP classifier (averaged over folds).
  * `selection_random forest` - Prediction rate for each movement type as a function of feature number, for Random Forest classifier (averaged over folds).
  * `selection_svm rbf` - Prediction rate for each movement type as a function of feature number, for SVM classifier (averaged over folds).
- `sections/` - 
  * `mlp/`, `random forest/` , `svm rbf/` - for each full-recording:
    * `<recording_name>_<feature_type>_<classifier_type>_predicted.csv` - CSV of estimated drone location according to section-based prediction of movement types, saved in csv.
    * `<recording_name>_<feature_type>_<classifier_type>_predicted_path_x.png` - Plot of estimated drone x-axis path according to section-based prediction of movement types, plotted against real results.
    * `<recording_name>_<feature_type>_<classifier_type>_predicted_path_x_z.png` - Plot of estimated drone x-z-axes 3D path according to section-based prediction of movement types, plotted against real results.
    * `<recording_name>_<feature_type>_<classifier_type>_predicted_path_z.png` - Plot of estimated drone z-axis path according to section-based prediction of movement types, plotted against real results.
- `segments/` - 
  * `mlp/`, `random forest/` , `svm rbf/` - for each full-recording:
    * `<recording_name>_<feature_type>_<classifier_type>_predicted.csv` - CSV of estimated drone location according to segment-based prediction of movement types, saved in csv.
    * `<recording_name>_<feature_type>_<classifier_type>_predicted_path_x.png` - Plot of estimated drone x-axis path according to segment-based prediction of movement types, plotted against real results.
    * `<recording_name>_<feature_type>_<classifier_type>_predicted_path_x_z.png` - Plot of estimated drone x-z-axes 3D path according to segment-based prediction of movement types, plotted against real results.
    * `<recording_name>_<feature_type>_<classifier_type>_predicted_path_z.png` - Plot of estimated drone z-axis path according to segment-based prediction of movement types, plotted against real results.