# Project B - Drone Movement Classification from Audio

This project focuses on classifying drone movement types (ascending, descending, hovering, moving away/towards the microphone and turning), using only an audio recordings database.
The repository contains both the code used throughout the experimental process and the finalized version for reproducing results or further exploration.

## Project Structure

- `backup/` - A copy of both saved plots (`plots/`) and prints output (`output/`) for all experiments including the final results. Designed to maintain the original results in case the project code is re-run.
- [`data/`](data/README.md) - All saved vectors, models and other data objects used throughout the project.
- [`output/`](output/README.md) - All saved predictions and prints for all experiments, including the final results.
- [`plots/`](plots/README.md) - All saved plots for all experiments, including the final results.
- `project_code/` - Main code folder, from which the code is run.
- [`recordings/`](recordings/README.md) - All full recordings data.
- [`split/`](split/README.md) - All split recordings data.

## Setup Instructions

To run this project, follow the steps below.

#### 1. Set up the Python environment by installing the required packages:
```bash
pip install -r requirements.txt
```
This will enable you to run all experiments and the final code.

##### 2. SHAP Plot Reproduction (Experiment 3)

To replicate our SHAP plots exactly as shown in Experiment 3, you'll need to manually modify the SHAP library source code:
1. Locate the installed SHAP package:

```bash
pip show shap
```

2. Navigate to the `shap/plots/` directory.
3. Replace the files `_bar.py` and `_waterfall.py` with the modified versions found in: `project_code/experiments/shap_modified`.

## Usage

Test execution is divided into two categories: **experimental** test and **final** tests. Follow the instructions provided in the corresponding README files:

For experiments: [`project_code/experiments`](project_code/experiments/README.md)

For final tests: [`project_code/core`](project_code/core/README.md)
