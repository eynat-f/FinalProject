# Core Code

The scripts corresponding to the final experiment described in the paper.


## Final Scripts
- `main_train_all.py` - Train models on all feature-classifier combinations - split recordings.
- `main_test_all.py` - Test models on all feature-classifier combinations - full recordings.

### Utilities
- `plot_utils.py`, `stat_utils.py` - Shared functions, for plotting and general calculations.

### Legacy
- `_legacy_single_train.py` - Early single-feature training (obsolete).
- `_legacy_single_test.py` - Early test script.
- `_legacy_feature_importance.py` - Feature ranking experiments.

### Archived
- `_archive_adjust_csv_once.py` - One-off script to fix CSVs (do not use).

## Usage

The scripts in this folder correspond to the final results (meaning Experiment 4.2).
- First run `main_train_all.py` to train and save the models.
- Then run `main_test_all.py` to test over the full recordings.