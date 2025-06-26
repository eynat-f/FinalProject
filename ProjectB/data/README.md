# Data

This folder contains precomputed models and statistics and features vectors used by the code to reduce runtime.

## Folder Structure

- `new/` - Data for experiments from 2.2, including the final tests.
  * `train/` - Data for training.
  * `test/` - Data for testing.
- `old/` - Data for experiments up to 2.1 (including).

### Within each folder:

- `dwt/` or `dwt/train` -
  * `dwt_stats_data.npz` - Saved numpy object to store DWT statistics data.
- `harmonic/` or `harmonic/train` - 
  * `harmonic_stats_data.npz` - Saved numpy object to store HPSS statistics data.
- `mfcc/` or `mfcc/train` - 
  * `mfcc_stats_data.npz` - Saved numpy object to store MFCC statistics data.
  * `mfcc_stats_data.npz` - Saved numpy object to store MFCC raw feature data.
- `split_info/` - 
  * `test_ids_1.json` - Stores the test file ids for fold 1.
  * `test_ids_2.json` - Stores the test file ids for fold 2.
  * `test_ids_3.json` - Stores the test file ids for fold 3.
  * `test_ids_4.json` - Stores the test file ids for fold 4.
  * `test_ids_5.json` - Stores the test file ids for fold 5.
- `label_encoder.pkl` - Saved label encoder.
- `mic_position_map.json` - Mapping of full recording name to the position of the microphone in the flight.
- `recording_ids.json` - Mapping of full recording name to the recording id.

### Additionally, for every feature type folder in `new/` :

- `train/`
  * `mlp/` , `random forest/` and `svm rbf/` -
    * `<classifier_type>_1.joblib` - Saved model for classifier type, fold 1.
    * `<classifier_type>_2.joblib` - Saved model for classifier type, fold 2.
    * `<classifier_type>_3.joblib` - Saved model for classifier type, fold 3.
    * `<classifier_type>_4.joblib` - Saved model for classifier type, fold 4.
    * `<classifier_type>_5.joblib` - Saved model for classifier type, fold 5.
  * `reports_by_feature_num.npz` - Saved feature reduction information (classification results and chosen features).
- `test/`
  * `<feature_type>_fold1.npz` - Saved statistics for the full recordings of the test samples of fold 1.
  * `<feature_type>_fold2.npz` - Saved statistics for the full recordings of the test samples of fold 2.
  * `<feature_type>_fold3.npz` - Saved statistics for the full recordings of the test samples of fold 3.
  * `<feature_type>_fold4.npz` - Saved statistics for the full recordings of the test samples of fold 4.
  * `<feature_type>_fold5.npz` - Saved statistics for the full recordings of the test samples of fold 5.