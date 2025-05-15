
# Exo Planet Detection: Kepler Mission Data Analysis & Classification

<!-- Badges Section -->
<p align="center">
  <a href="https://github.com/AayushBadola/Exo-Planet-Detection/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python Version">
  </a>
  <img src="https://img.shields.io/badge/Building%20-orange" alt="Project Status">
  <img src="https://img.shields.io/badge/platform-windows%20%7C%20macos%20%7C%20linux-lightgrey" alt="Platform Agnostic">
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
  </a>
  <a href="https://hydra.cc/">
    <img src="https://img.shields.io/badge/configured%20with-Hydra-89A1F0" alt="Configured with Hydra">
  </a>
  <a href="https://mlflow.org/">
    <img src="https://img.shields.io/badge/tracked%20with-MLflow-019FEF" alt="Tracked with MLflow">
  </a>
  <!-- Add more relevant badges as the project evolves, e.g., build status, code coverage -->
</p>

<p align="center">
  <em>A machine learning pipeline for detecting exoplanets from Kepler Space Telescope data.</em>
</p>

---

**Author:** Aayush Badola <br>
**Repository:** [https://github.com/AayushBadola/Exo-Planet-Detection.git](https://github.com/AayushBadola/Exo-Planet-Detection.git)

---

## Table of Contents

1.  [Introduction](#introduction)
    *   [Project Goal](#project-goal)
    *   [Dataset Overview](#dataset-overview)
    *   [Key Features](#key-features)
2.  [Project Architecture](#project-architecture)
    *   [Directory Structure](#directory-structure)
    *   [Technology Stack](#technology-stack)
    *   [Workflow Overview](#workflow-overview)
3.  [Setup and Installation](#setup-and-installation)
    *   [Prerequisites](#prerequisites)
    *   [Cloning the Repository](#cloning-the-repository)
    *   [Setting up the Virtual Environment](#setting-up-the-virtual-environment)
    *   [Installing Dependencies](#installing-dependencies)
    *   [Data Acquisition](#data-acquisition)
4.  [Configuration Management with Hydra](#configuration-management-with-hydra)
    *   [Overview of Hydra](#overview-of-hydra)
    *   [Main Configuration File (`conf/config.yaml`)](#main-configuration-file-confconfigyaml)
    *   [Configuration Groups](#configuration-groups)
        *   [`data_paths`](#data_paths-configuration)
        *   [`preprocessing`](#preprocessing-configuration)
        *   [`model`](#model-configuration)
    *   [Overriding Configuration Parameters](#overriding-configuration-parameters)
5.  [Data Processing Pipeline](#data-processing-pipeline)
    *   [Data Loading (`src/data_loader.py`)](#data-loading-srcdata_loaderpy)
    *   [Exploratory Data Analysis (EDA) (`notebooks/01_eda_and_preprocessing.ipynb`)](#exploratory-data-analysis-eda-notebooks01_eda_and_preprocessingipynb)
        *   [Initial Data Inspection](#initial-data-inspection)
        *   [Target Variable Analysis](#target-variable-analysis)
        *   [Missing Value Analysis](#missing-value-analysis)
        *   [Feature Distributions and Correlations](#feature-distributions-and-correlations)
    *   [Preprocessing Strategy (`src/preprocessor.py`)](#preprocessing-strategy-srcpreprocessorpy)
        *   [Column Name Cleaning](#column-name-cleaning)
        *   [Target Variable Encoding](#target-variable-encoding)
        *   [Feature Dropping](#feature-dropping)
        *   [Handling Categorical Features](#handling-categorical-features)
        *   [Missing Value Imputation](#missing-value-imputation)
        *   [Feature Scaling](#feature-scaling)
        *   [Polynomial Features (Optional)](#polynomial-features-optional)
        *   [Train-Test Split](#train-test-split)
    *   [Saving Preprocessing Artifacts](#saving-preprocessing-artifacts)
6.  [Model Training and Evaluation](#model-training-and-evaluation)
    *   [Model Selection (`conf/model/*.yaml`)](#model-selection-confmodelyaml)
        *   [Random Forest Classifier](#random-forest-classifier)
    *   [Training Process (`src/model_trainer.py`)](#training-process-srcmodel_trainerpy)
    *   [Model Evaluation (`src/model_trainer.py`, `notebooks/02_model_training_and_evaluation.ipynb`)](#model-evaluation-srcmodel_trainerpy-notebooks02_model_training_and_evaluationipynb)
        *   [Metrics Used](#metrics-used)
        *   [Classification Report](#classification-report)
        *   [Confusion Matrix](#confusion-matrix)
        *   [ROC AUC and PR AUC](#roc-auc-and-pr-auc)
    *   [Feature Importance Analysis](#feature-importance-analysis)
    *   [Saving the Trained Model](#saving-the-trained-model)
7.  [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
    *   [MLflow Overview](#mlflow-overview)
    *   [Setting up MLflow Tracking URI](#setting-up-mlflow-tracking-uri)
    *   [Logging Parameters, Metrics, and Artifacts](#logging-parameters-metrics-and-artifacts)
    *   [Using the MLflow UI](#using-the-mlflow-ui)
    *   [Model Registry (Conceptual)](#model-registry-conceptual)
8.  [Running the Pipeline](#running-the-pipeline)
    *   [Main Execution Script (`main.py`)](#main-execution-script-mainpy)
    *   [Command Line Execution](#command-line-execution)
    *   [Interpreting Log Outputs](#interpreting-log-outputs)
    *   [Expected Output Artifacts](#expected-output-artifacts)
9.  [Prediction Demo](#prediction-demo)
    *   [Loading Trained Model and Artifacts (`src/predict.py`)](#loading-trained-model-and-artifacts-srcpredictpy)
    *   [Preprocessing New Data for Prediction](#preprocessing-new-data-for-prediction)
    *   [Making Predictions](#making-predictions)
10. [Hyperparameter Tuning (Conceptual with Optuna)](#hyperparameter-tuning-conceptual-with-optuna)
    *   [Optuna Overview](#optuna-overview)
    *   [Configuration for Tuning (`conf/config.yaml`)](#configuration-for-tuning-confconfigyaml)
    *   [Integration (Placeholder)](#integration-placeholder)
11. [Code Structure and Modularity](#code-structure-and-modularity)
    *   [`src/` Directory Modules](#src-directory-modules)
    *   [Utility Functions](#utility-functions)
12. [Testing (Conceptual)](#testing-conceptual)
    *   [Unit Tests](#unit-tests)
    *   [Integration Tests](#integration-tests)
13. [Contribution Guidelines](#contribution-guidelines)
14. [Future Work and Potential Improvements](#future-work-and-potential-improvements)
15. [License](#license)
16. [Acknowledgements](#acknowledgements)
17. [Contact Author](#contact-author)

---

## 1. Introduction

This project is dedicated to the fascinating challenge of identifying exoplanets – planets orbiting stars beyond our solar system. Leveraging machine learning techniques and publicly available data from NASA's Kepler Space Telescope, this pipeline aims to build a robust classifier capable of distinguishing promising exoplanet candidates from instrumental noise and other astrophysical phenomena.

### 1.1 Project Goal

The primary objective is to develop an end-to-end, reproducible machine learning pipeline that can:
*   Process and clean raw Kepler exoplanet candidate data.
*   Train a classification model to predict the disposition of Kepler Objects of Interest (KOIs).
*   Evaluate model performance using appropriate metrics.
*   Track experiments, parameters, and model artifacts for reproducibility and comparison.
*   Provide a framework for easy configuration and potential extension with new models or features.

### 1.2 Dataset Overview

*   **Dataset Name:** Kepler Objects of Interest (KOI) Cumulative Table
*   **Source:** [NASA Exoplanet Archive](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results) 
   
*   **Description:** This table contains photometric time series data features and derived physical parameters for thousands of KOIs observed by the Kepler mission. The data includes features like orbital period, transit duration, transit depth, planetary radius, stellar parameters, and various false positive flags.
*   **Target Variable:** `koi_disposition` (e.g., CONFIRMED, CANDIDATE, FALSE POSITIVE).

<details>
<summary><strong>Illustrative Key Dataset Columns</strong></summary>

| Column Name        | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `kepid`            | Kepler ID of the target star.                                               |
| `kepoi_name`       | Kepler Object of Interest (KOI) name.                                       |
| `koi_disposition`  | Final disposition of the KOI. **This is the target variable.**                |
| `koi_period`       | Orbital period of the KOI (days).                                           |
| `koi_duration`     | Transit duration (hours).                                                   |
| `koi_depth`        | Transit depth (parts per million, ppm).                                     |
| `koi_prad`         | Planetary radius (Earth radii).                                             |
| `koi_fpflag_nt`    | Not Transit-Like False Positive Flag (0 or 1).                              |
| `koi_fpflag_ss`    | Stellar Eclipse False Positive Flag (0 or 1).                               |
| ...                | *(Numerous other features and their associated error columns)*              |

> **Note:** Refer to `notebooks/01_eda_and_preprocessing.ipynb` and the [NASA Exoplanet Archive data column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html) for a complete list and detailed descriptions.

</details>

The raw data file (`cumulative.csv`) is expected to be placed in the `data/` directory.

### 1.3 Key Features

*   **Modular Pipeline:** Code organized into logical modules for data loading, preprocessing, training, and prediction.
*   **Configuration Driven:** Utilizes [Hydra](https://hydra.cc/) for managing all configurations, allowing for easy modification of parameters and experiment setups via YAML files and command-line overrides.
*   **Experiment Tracking:** Integrated with [MLflow](https://mlflow.org/) to log parameters, metrics, model artifacts, and source code for each run, ensuring reproducibility and facilitating comparison between experiments.
*   **Comprehensive Preprocessing:** Includes steps for cleaning column names, target encoding, feature dropping, missing value imputation (median strategy), and feature scaling (StandardScaler).
*   **Random Forest Classifier:** Implements a robust ensemble learning model for classification.
*   **Automated Evaluation:** Generates standard classification metrics (accuracy, precision, recall, F1-score), ROC AUC, PR AUC, confusion matrices, and feature importance plots.
*   **Reproducibility:** Designed for high reproducibility through version-controlled code, managed dependencies (`requirements.txt`), tracked experiments, and versioned models.
*   **Jupyter Notebooks:** Provides notebooks for Exploratory Data Analysis (`01_eda_and_preprocessing.ipynb`) and initial model development/evaluation strategy (`02_model_training_and_evaluation.ipynb`).
*   **Prediction Capability:** Includes functionality to load a trained model and make predictions on new (or sample) data.

---

## 2. Project Architecture

### 2.1 Directory Structure

The project adheres to a conventional structure for scalable machine learning projects:

```
📁 Exo-Planet-Detection/
├── 📁 conf/                    # Hydra configuration files
│   ├── 📁 data_paths/
│   │   └── default.yaml
│   ├── 📁 model/
│   │   └── random_forest.yaml
│   ├── 📁 preprocessing/
│   │   └── default.yaml
│   └── config.yaml            # Main configuration file
├── 📁 data/                    # Raw data (e.g., cumulative.csv)
├── 📁 logs/                    # Pipeline execution logs
├── 📁 models/                  # Saved trained models & preprocessing artifacts
├── 📁 notebooks/               # Jupyter notebooks for EDA & experimentation
├── 📁 reports/                 # Generated reports & plots (e.g., confusion matrix)
├── 📁 src/                     # Source code for the pipeline modules
├── .gitignore                 # Specifies intentionally untracked files
├── main.py                    # Main script to orchestrate the pipeline
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```
*(For a detailed breakdown, see the `main.py` script which references these paths via Hydra configuration.)*

### 2.2 Technology Stack

| Category                 | Technology / Library                                      | Purpose                                            |
|--------------------------|-----------------------------------------------------------|----------------------------------------------------|
| **Language**             | Python 3.9+                                               | Core programming language                          |
| **Data Handling**        | Pandas, NumPy                                             | Data manipulation, numerical operations            |
| **Machine Learning**     | Scikit-learn                                              | ML algorithms, preprocessing, metrics              |
| **Configuration**        | Hydra                                                     | Flexible and hierarchical configuration management |
| **Experiment Tracking**  | MLflow                                                    | Logging, model management, reproducibility         |
| **Serialization**        | Joblib                                                    | Saving/loading Python objects (models, scalers)    |
| **Visualization**        | Matplotlib, Seaborn                                       | Plotting and statistical visualization             |
| **Notebooks**            | Jupyter Notebook / Lab                                    | EDA, prototyping                                   |
| **Code Formatting**      | Black                                                     | Consistent Python code style                       |
| **Development Environment** | VSCode (Tested On)                                     | Integrated Development Environment                 |

### 2.3 Workflow Overview

The pipeline executes the following major steps sequentially:

```mermaid
graph TD
    NodeA[Start Run main.py]
    NodeB{Load Configuration Hydra}
    NodeC[Setup Logging and MLflow]
    NodeD[Load Raw Data]
    NodeE[Preprocess Data]
    NodeF[Split Data Train Test]
    NodeG[Train Model]
    NodeH[Log Model and Artifacts MLflow]
    NodeI[Evaluate Model]
    NodeJ[Log Metrics and Plots MLflow]
    NodeK[Feature Importance Analysis]
    NodeL[Prediction Demo Optional]
    NodeM[End Pipeline Complete]

    NodeA --> NodeB
    NodeB --> NodeC
    NodeC --> NodeD
    NodeD --> NodeE
    NodeE --> NodeF
    NodeF --> NodeG
    NodeG --> NodeH
    NodeH --> NodeI
    NodeI --> NodeJ
    NodeJ --> NodeK
    NodeK --> NodeL
    NodeL --> NodeM

    subgraph "Data Preparation Stage"
        NodeD
        NodeE
        NodeF
    end

    subgraph "Modeling and Tracking Stage"
        NodeG
        NodeH
        NodeI
        NodeJ
        NodeK
    end

---

## 3. Setup and Installation

### 3.1 Prerequisites

*   Python (version 3.9 or higher recommended). You can download it from [python.org](https://www.python.org/downloads/).
*   `pip` (Python package installer, usually comes with Python).
*   `git` (for cloning the repository).
*   A virtual environment manager (e.g., `venv`, `conda`). Using `venv` is demonstrated below.

### 3.2 Cloning the Repository

```bash
git clone https://github.com/AayushBadola/Exo-Planet-Detection.git
cd Exo-Planet-Detection
```

### 3.3 Setting up the Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment (e.g., named 'venv')
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
You should see `(venv)` at the beginning of your terminal prompt.

### 3.4 Installing Dependencies

All required Python packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```
This will install Pandas, NumPy, Scikit-learn, Hydra, MLflow, Matplotlib, Seaborn, Joblib, and Prettytable.

### 3.5 Data Acquisition

1.  **Download the Dataset:**
    The primary dataset is the Kepler Objects of Interest (KOI) Cumulative Table.
    *   If using the version from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative), download the CSV file.
    *   *(If you are using a specific Kaggle version, provide the link and instructions here.)*
2.  **Place the Data:**
    Rename the downloaded CSV file to `cumulative.csv` (if it's not already named that) and place it inside the `data/` directory at the root of the project.
    ```
    Exo-Planet-Detection/
    └── data/
        └── cumulative.csv
    ```
    The pipeline is configured to look for this specific file path by default (`data_paths.raw_data_filename` in Hydra config).

---

## 4. Configuration Management with Hydra

This project utilizes [Hydra](https://hydra.cc/) for flexible and powerful configuration management. All pipeline parameters, paths, and settings are defined in YAML files located in the `conf/` directory.

### 4.1 Overview of Hydra

Hydra allows you to:
*   Compose configurations from multiple files.
*   Override any configuration parameter from the command line.
*   Manage different configurations for various environments or experiments (e.g., development, production, different model types).
*   Automatically track configurations and create structured output directories.

### 4.2 Main Configuration File (`conf/config.yaml`)

This is the entry point for the configuration. It defines default settings and includes other configuration groups.

```yaml
# conf/config.yaml
defaults:
  - data_paths: default
  - preprocessing: default
  - model: random_forest
  - _self_ # Allows defining top-level keys here

# General project settings
project_name: "ExoplanetDetectionKepler"
random_state: 42
test_size: 0.2
# ... other global settings ...

# MLflow settings
mlflow_experiment_name: "Exoplanet Detection Pipeline"
mlflow_tracking_uri: "sqlite:///mlflow.db"
# ... other MLflow settings ...

# Paths (relative to base_dir, which is project root)
base_dir: ??? # Dynamically set by main.py using Hydra's get_original_cwd()
data_dir_rel: "data"
model_dir_rel: "models"
# ... other relative paths ...

# Resolved absolute paths (populated by src/config_utils.py at runtime)
data_dir: ???
model_dir: ???
# ... other absolute paths ...
model_path: ???
scaler_path: ???
# ... paths for all artifacts ...

# Prediction demo settings
prediction_sample_size: 5
run_prediction_demo: true # Control whether to run the demo

# Tuning settings (for Optuna, conceptual)
tuning:
  cv_folds: 5
  # ... other tuning params ...
```

### 4.3 Configuration Groups

Configurations are modularized into groups, located in subdirectories within `conf/`.

#### 4.3.1 `data_paths` Configuration
*   File: `conf/data_paths/default.yaml`
*   Purpose: Specifies data file names.
    ```yaml
    # conf/data_paths/default.yaml
    raw_data_filename: "cumulative.csv"
    # Potentially paths for processed data, etc.
    ```

#### 4.3.2 `preprocessing` Configuration
*   File: `conf/preprocessing/default.yaml`
*   Purpose: Defines parameters for data preprocessing.
    ```yaml
    # conf/preprocessing/default.yaml
    target_column: "koi_disposition"
    positive_labels: ["CONFIRMED", "CANDIDATE"]
    negative_label: "FALSE POSITIVE"
    features_to_drop:
      - "rowid"
      # ... other features ...
    apply_polynomial_features: false
    polynomial_degree: 2
    ```

#### 4.3.3 `model` Configuration
*   File: `conf/model/random_forest.yaml` (Example for one model type)
*   Purpose: Defines model type and its hyperparameters.
    ```yaml
    # conf/model/random_forest.yaml
    name: "RandomForestClassifier"
    params:
      n_estimators: 100
      max_depth: 20
      # ... other hyperparameters ...
      class_weight: "balanced"
    ```
    To use a different model, you could create, for example, `conf/model/logistic_regression.yaml` and then run the pipeline with `model=logistic_regression` override.

### 4.4 Overriding Configuration Parameters

Hydra's power comes from its ability to override any configuration parameter from the command line.

**Examples:**

*   Change the number of estimators for the Random Forest model:
    ```bash
    python main.py model.params.n_estimators=200
    ```
*   Use a different random state and test size:
    ```bash
    python main.py random_state=123 test_size=0.3
    ```
*   Run without the prediction demo:
    ```bash
    python main.py run_prediction_demo=false
    ```
*   Specify a different model configuration (if you have `conf/model/xgb.yaml`):
    ```bash
    python main.py model=xgb
    ```

Hydra also creates output directories for each run (by default in `outputs/YYYY-MM-DD/HH-MM-SS/`) containing the full configuration (`.hydra/config.yaml`), overrides, and any logs if not redirected. This project redirects primary logs to the `logs/` directory.

---

## 5. Data Processing Pipeline

The data processing pipeline is crucial for preparing the raw Kepler data for machine learning. It involves several steps orchestrated by `src/preprocessor.py` and informed by EDA done in `notebooks/01_eda_and_preprocessing.ipynb`.

### 5.1 Data Loading (`src/data_loader.py`)
The `load_data` function in `src/data_loader.py` is responsible for:
*   Reading the `cumulative.csv` file using Pandas.
*   Handling potential comments (lines starting with `#`) often found in astronomical data files.
*   Basic error handling for file not found or empty data.

```python
# Illustrative snippet from src/data_loader.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame | None:
    logger.info(f"Attempting to load data from: {file_path}")
    try:
        df = pd.read_csv(file_path, comment='#')
        # ... (error handling and fallback) ...
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    # ... (exception handling) ...
```

### 5.2 Exploratory Data Analysis (EDA) (`notebooks/01_eda_and_preprocessing.ipynb`)
The `01_eda_and_preprocessing.ipynb` notebook details the initial exploration of the dataset. Key EDA steps include:

#### 5.2.1 Initial Data Inspection
*   Loading the data.
*   Displaying `df.head()`, `df.info()`, `df.describe()`.
*   Checking data types and identifying numerical vs. categorical features.

#### 5.2.2 Target Variable Analysis
*   Analyzing the distribution of the `koi_disposition` target variable.
*   Visualizing class balance.
*   Deciding on the mapping for binary classification (e.g., `CONFIRMED` and `CANDIDATE` to 1, `FALSE POSITIVE` to 0).

#### 5.2.3 Missing Value Analysis
*   Calculating the percentage of missing values for each feature.
*   Visualizing missingness patterns.
*   Informing the imputation strategy (e.g., median imputation for numerical features with NaNs).

#### 5.2.4 Feature Distributions and Correlations
*   Plotting histograms or density plots for numerical features to understand their distributions (skewness, outliers).
*   Generating a correlation heatmap to identify highly correlated features.

### 5.3 Preprocessing Strategy (`src/preprocessor.py`)
The `preprocess_data` function in `src/preprocessor.py` implements the following steps based on insights from EDA and configuration:

#### 5.3.1 Column Name Cleaning
*   Strips leading/trailing whitespace from column names for consistency.

#### 5.3.2 Target Variable Encoding
*   Maps the multi-class `koi_disposition` column to a binary target variable based on `cfg.preprocessing.positive_labels` and `cfg.preprocessing.negative_label`.
*   Samples not matching these labels are typically dropped.

#### 5.3.3 Feature Dropping
*   Removes columns specified in `cfg.preprocessing.features_to_drop`. These usually include:
    *   Identifiers (e.g., `rowid`, `kepid`, `kepoi_name`).
    *   Redundant or leaky features (e.g., `koi_pdisposition`, `koi_score` if it's a pre-computed model output).
    *   Textual comment columns.
    *   Columns with very high cardinality or no predictive value identified during EDA.

#### 5.3.4 Handling Categorical Features
*   The current strategy primarily focuses on numerical data. Any remaining object-type columns (after initial drops) are typically dropped.
*   *Future extension: Could involve one-hot encoding or other categorical encoding techniques if valuable categorical features are identified.*

#### 5.3.5 Missing Value Imputation
*   Uses `sklearn.impute.SimpleImputer` with a `median` strategy for all numerical features. Median is chosen for its robustness to outliers.
*   The fitted imputer is saved.

#### 5.3.6 Feature Scaling
*   Applies `sklearn.preprocessing.StandardScaler` to all numerical features after imputation. This standardizes features by removing the mean and scaling to unit variance.
*   The fitted scaler is saved.

#### 5.3.7 Polynomial Features (Optional)
*   If `cfg.preprocessing.apply_polynomial_features` is true, `sklearn.preprocessing.PolynomialFeatures` can be applied to a subset of numerical features to generate interaction terms and polynomial terms up to `cfg.preprocessing.polynomial_degree`.
*   *Note: Careful selection of features for polynomial expansion is important to avoid excessive dimensionality.*
*   The fitted transformer is saved if used.

#### 5.3.8 Train-Test Split
*   The `split_data` function (also in `src/preprocessor.py`) splits the processed features (X) and target (y) into training and testing sets using `sklearn.model_selection.train_test_split`.
*   Stratification (`stratify=y`) is used to maintain class proportions in both sets, which is important for imbalanced datasets.
*   `test_size` and `random_state` are controlled via Hydra configuration.

### 5.4 Saving Preprocessing Artifacts
The following artifacts are saved to the directory specified by `cfg.model_dir` (typically `models/`):
*   `imputer.joblib`: The fitted `SimpleImputer`.
*   `scaler.joblib`: The fitted `StandardScaler`.
*   `poly_features_transformer.joblib` (if polynomial features are applied).
*   `training_columns.joblib`: A list of column names present in the training data after all preprocessing. This is crucial for ensuring consistency when preprocessing new data for prediction.

---

## 6. Model Training and Evaluation

This section details the model training approach, the chosen model, and how its performance is evaluated.

### 6.1 Model Selection (`conf/model/*.yaml`)

The model type and its hyperparameters are defined in the `conf/model/` directory. The project currently focuses on:

#### 6.1.1 Random Forest Classifier
*   **Configuration:** `conf/model/random_forest.yaml`
*   **Rationale:** Random Forests are powerful ensemble models that:
    *   Perform well on a variety of tasks.
    *   Are relatively robust to outliers and non-linear data.
    *   Can handle high-dimensional data.
    *   Provide feature importance measures.
    *   Less sensitive to feature scaling than some other models (though scaling is still applied for consistency and potential use with other models).
*   **Key Hyperparameters (configurable via Hydra):**
    *   `n_estimators`: Number of trees in the forest.
    *   `max_depth`: Maximum depth of each tree.
    *   `min_samples_split`: Minimum number of samples required to split an internal node.
    *   `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
    *   `class_weight`: Set to `"balanced"` to help address class imbalance in the target variable.
    *   `n_jobs`: Set to `-1` to use all available CPU cores for training.
    *   `random_state`: Ensures reproducibility.

### 6.2 Training Process (`src/model_trainer.py`)
The `train_model` function in `src/model_trainer.py`:
1.  Instantiates the model (e.g., `RandomForestClassifier`) using parameters from the Hydra configuration (`cfg.model.params`).
2.  Logs model parameters to MLflow.
3.  Fits the model on the training data (`X_train`, `y_train`).
4.  Saves the trained model object to a `.joblib` file (e.g., `models/randomforestclassifier_model.joblib`).
5.  Logs the saved model file as an artifact to MLflow.
6.  Also logs the model using `mlflow.sklearn.log_model`, which packages it in MLflow's native format with signature and input example for easier deployment and serving.

### 6.3 Model Evaluation (`src/model_trainer.py`, `notebooks/02_model_training_and_evaluation.ipynb`)
The `evaluate_model` function in `src/model_trainer.py` assesses the trained model's performance on the unseen test set (`X_test`, `y_test`). The `02_model_training_and_evaluation.ipynb` notebook provides an interactive environment for these steps.

#### 6.3.1 Metrics Used
A suite of standard classification metrics is employed:
*   **Accuracy:** Overall proportion of correctly classified instances.
*   **Precision:** Ability of the classifier not to label as positive a sample that is negative (TP / (TP + FP)). Crucial for minimizing false alarms if the cost of a false positive is high.
*   **Recall (Sensitivity):** Ability of the classifier to find all the positive samples (TP / (TP + FN)). Crucial for minimizing missed detections if the cost of a false negative is high.
*   **F1-Score:** The harmonic mean of precision and recall, providing a balance between the two.
*   **ROC AUC (Area Under the Receiver Operating Characteristic Curve):** Measures the model's ability to distinguish between classes across all thresholds. An AUC of 1.0 represents a perfect classifier, while 0.5 represents a random classifier.
*   **PR AUC (Area Under the Precision-Recall Curve):** Particularly informative for imbalanced datasets, as it focuses on the performance regarding the positive class.

All these metrics are calculated for each class and as macro/weighted averages, then logged to MLflow.

#### 6.3.2 Classification Report
A detailed text-based classification report from `sklearn.metrics.classification_report` is generated and logged, showing precision, recall, and F1-score for each class.

#### 6.3.3 Confusion Matrix
A confusion matrix is generated to visualize the model's predictions against the actual labels (True Positives, True Negatives, False Positives, False Negatives).
*   The matrix is printed to the console.
*   A plot of the confusion matrix is saved to the `reports/` directory (e.g., `reports/confusion_matrix.png`) and logged as an artifact to MLflow.

#### 6.3.4 ROC AUC and PR AUC
*   The ROC curve is plotted (True Positive Rate vs. False Positive Rate).
*   The Precision-Recall curve is plotted.
*   *Currently, these plots are primarily generated within the notebook; they could be added as saved artifacts from `model_trainer.py` in the future.*

### 6.4 Feature Importance Analysis
For tree-based models like Random Forest, feature importances can be extracted.
*   The `get_feature_importances` function in `src/model_trainer.py` calculates and displays the top N most important features.
*   A bar plot visualizing these importances is saved to the `reports/` directory (e.g., `reports/feature_importances.png`) and logged as an artifact to MLflow. This helps in understanding which features are most influential in the model's decisions.

### 6.5 Saving the Trained Model
The primary trained model object is saved as a `.joblib` file in the `cfg.model_dir` (e.g., `models/randomforestclassifier_model.joblib`). This allows for easy reloading for future predictions or analysis. It is also logged to MLflow using `mlflow.sklearn.log_model` in a more structured format.

---

## 7. Experiment Tracking with MLflow

[MLflow](https://mlflow.org/) is integral to this project for managing the machine learning lifecycle, ensuring reproducibility, and facilitating model comparison.

### 7.1 MLflow Overview
MLflow provides four main components:
*   **MLflow Tracking:** An API and UI for logging parameters, code versions, metrics, and output files when running machine learning code and for later visualizing the results. **This is the primary component used in this project.**
*   **MLflow Projects:** A standard format for packaging reusable data science code.
*   **MLflow Models:** A convention for packaging machine learning models in multiple "flavors" and a variety of tools to help deploy them.
*   **MLflow Model Registry:** A centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of an MLflow Model.

### 7.2 Setting up MLflow Tracking URI
The tracking URI tells MLflow where to store experiment data. It's configured in `conf/config.yaml`:
```yaml
mlflow_tracking_uri: "sqlite:///mlflow.db"
```
This setup uses a local SQLite database (`mlflow.db` created in the project root) for metadata and a local `mlruns/` directory (also in the project root) for storing artifacts (files). For more collaborative or production environments, this URI can be pointed to a remote server.

The experiment name is also configured:
```yaml
mlflow_experiment_name: "Exoplanet Detection Pipeline"
```
The `main.py` script uses `mlflow.set_experiment()` to ensure runs are logged under this specific experiment name. If the experiment doesn't exist, MLflow creates it.

### 7.3 Logging Parameters, Metrics, and Artifacts
Throughout the `main.py` pipeline and its helper modules (`model_trainer.py`), MLflow logging functions are used:
*   **`mlflow.log_param()` / `mlflow.log_params()`:** Used to log:
    *   Key configuration parameters from Hydra (e.g., model hyperparameters, preprocessing settings, data paths).
    *   Information like the number of features after preprocessing.
*   **`mlflow.log_metric()` / `mlflow.log_metrics()`:** Used to log:
    *   Dataset statistics (e.g., raw data rows/columns).
    *   All evaluation metrics (accuracy, precision, recall, F1, ROC AUC, PR AUC for each class and averages).
*   **`mlflow.log_artifact()` / `mlflow.log_artifacts()`:** Used to log:
    *   The saved `.joblib` model file.
    *   The saved preprocessing artifacts (`scaler.joblib`, `imputer.joblib`, `training_columns.joblib`).
    *   Generated plots (confusion matrix, feature importances).
    *   The source code of the run can also be implicitly logged by MLflow.
*   **`mlflow.sklearn.log_model()`:** Logs the scikit-learn model in MLflow's native format, including its signature (inferred input/output schema) and an input example, making it easier to understand and redeploy.
*   **`mlflow.set_tag()`:** Used to set tags for a run, such as `pipeline_status` (`success`, `failed_data_load`, etc.).

### 7.4 Using the MLflow UI
Once runs have been executed, you can inspect them using the MLflow UI:
1.  Ensure your virtual environment is activated.
2.  Navigate to the project root directory in your terminal.
3.  Run the command:
    ```bash
    mlflow ui
    ```
4.  Open your web browser and go to `http://127.0.0.1:5000` (or the address shown in the terminal).
5.  In the UI, you can:
    *   Select the "Exoplanet Detection Pipeline" experiment.
    *   View a list of all runs.
    *   Click on a specific run to see its details:
        *   Parameters logged.
        *   Metrics logged (with options to view charts).
        *   Artifacts stored (models, plots, other files can be downloaded).
        *   Tags associated with the run.
    *   Compare multiple runs side-by-side.

### 7.5 Model Registry (Conceptual)
While not fully implemented with registration steps in this project, the use of `mlflow.sklearn.log_model` lays the groundwork for using MLflow's Model Registry. The registry allows for versioning models, transitioning them through stages (e.g., Staging, Production), and managing their lifecycle in a more formal way. This would be a valuable next step for operationalizing the model.

---

## 8. Running the Pipeline

### 8.1 Main Execution Script (`main.py`)
The entire pipeline is orchestrated by the `main.py` script. It leverages Hydra for configuration and calls the various modules in `src/` to perform each step.

The typical flow within `main.py`:
1.  Hydra decorator (`@hydra.main`) loads and composes configuration.
2.  `src.config_utils.setup_config` resolves paths and creates directories.
3.  `src.logger_utils.setup_logging` initializes logging.
4.  MLflow tracking URI and experiment are set.
5.  An MLflow run is started (`with mlflow.start_run()`).
6.  Parameters are logged.
7.  Data is loaded (`src.data_loader.load_data`).
8.  Data is preprocessed (`src.preprocessor.preprocess_data`), and artifacts are saved and logged.
9.  Data is split (`src.preprocessor.split_data`).
10. Model is trained (`src.model_trainer.train_model`), saved, and logged.
11. Model is evaluated (`src.model_trainer.evaluate_model`), metrics and plots are logged.
12. Feature importances are analyzed and logged (`src.model_trainer.get_feature_importances`).
13. (Optional) Prediction demo is run using `src.predict`.
14. Pipeline status is tagged in MLflow.

### 8.2 Command Line Execution
After completing the [Setup and Installation](#setup-and-installation) steps:

1.  **Activate the virtual environment:**
    ```bash
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
2.  **Run the pipeline with default configurations:**
    ```bash
    python main.py
    ```
3.  **Run with overridden configurations (Hydra syntax):**
    ```bash
    # Example: Change number of estimators and test size
    python main.py model.params.n_estimators=150 test_size=0.25

    # Example: Turn off the prediction demo
    python main.py run_prediction_demo=false
    ```
    Refer to the [Hydra section](#configuration-management-with-hydra) for more on overrides.

### 8.3 Interpreting Log Outputs
*   **Console Output:** `main.py` (and its modules) will print informational messages, warnings, and errors to the console. The `--- MAIN.PY DEBUG: ---` lines (if uncommented) provide verbose step-by-step execution details. MLflow also prints some informational messages about experiment creation and database setup.
*   **Log File (`logs/pipeline.log`):** A more detailed log, including timestamps and log levels, is saved here. This is useful for debugging issues after a run.

### 8.4 Expected Output Artifacts
Upon successful completion of a run, the following directories and files will be created or updated:

*   **`models/`:**
    *   `randomforestclassifier_model.joblib` (or model name from config)
    *   `imputer.joblib`
    *   `scaler.joblib`
    *   `training_columns.joblib`
    *   `poly_features_transformer.joblib` (if applicable)
*   **`reports/`:**
    *   `confusion_matrix.png`
    *   `feature_importances.png`
*   **`logs/`:**
    *   `pipeline.log`
*   **`mlruns/` (MLflow artifacts):**
    *   A subdirectory for each experiment (e.g., `mlruns/0/` for Default, `mlruns/1/` for "Exoplanet Detection Pipeline").
    *   Inside each experiment directory, a subdirectory for each run (named by run ID).
    *   Inside each run directory:
        *   `meta.yaml` (run metadata)
        *   `params/` (logged parameters)
        *   `metrics/` (logged metrics)
        *   `tags/` (logged tags)
        *   `artifacts/` (logged artifacts, including subfolders like `model_files/`, `preprocessing_artifacts/`, `evaluation_plots/`, and the `mlflow_sklearn_{model_name}/` directory).
*   **`mlflow.db` (MLflow metadata):** The SQLite database storing experiment and run metadata.
*   **`outputs/YYYY-MM-DD/HH-MM-SS/` (Hydra default output, if not disabled):** Contains a `.hydra/` subdirectory with `config.yaml`, `hydra.yaml`, `overrides.yaml`.

---

## 9. Prediction Demo

The pipeline includes an optional demonstration of how to use the trained model and preprocessing artifacts to make predictions on new or sample data. This is controlled by the `run_prediction_demo` flag in `conf/config.yaml` (defaults to `true`).

### 9.1 Loading Trained Model and Artifacts (`src/predict.py`)
The `src/predict.py` module contains helper functions:
*   `load_trained_model()`: Loads the saved `.joblib` model file using the path from `cfg.model_path`.
*   `load_preprocessing_artifacts()`: Loads the `scaler.joblib`, `imputer.joblib`, `training_columns.joblib`, and optionally `poly_features_transformer.joblib` using paths from the configuration.

In `main.py`, for the demo, these artifacts are loaded directly from the paths of the artifacts *just created* in the same run to ensure consistency.

### 9.2 Preprocessing New Data for Prediction
The `predict.preprocess_for_prediction()` function is crucial. It takes raw new data (as a Pandas DataFrame) and applies the *exact same* preprocessing steps used during training, utilizing the *loaded* (fitted) artifacts:
1.  Cleans column names.
2.  Drops features that were dropped during training (based on `cfg.preprocessing.features_to_drop` and ensuring the target column is not present or is removed).
3.  Drops any purely categorical features (current strategy).
4.  Applies the loaded imputer to fill missing values in numerical features.
5.  Applies the loaded polynomial features transformer (if used during training) to the appropriate columns.
6.  Applies the loaded scaler to numerical features.
7.  **Critically, it aligns the columns of the new data to match the `training_columns.joblib` list**, adding missing columns (e.g., with zeros) and ensuring the correct order. This is vital for the model to receive input in the expected format.

### 9.3 Making Predictions
The `predict.make_prediction()` function:
1.  Takes the trained model object and the fully preprocessed new data.
2.  Calls `model.predict()` to get class predictions.
3.  Calls `model.predict_proba()` to get class probabilities.

The demonstration in `main.py` then:
*   Selects a few random samples from the original raw dataset (or test set indices).
*   Retrieves their actual labels for comparison.
*   Passes these raw samples through `predict.preprocess_for_prediction()`.
*   Makes predictions using `predict.make_prediction()`.
*   Prints a beautified table comparing the Original Index, Actual Label, Predicted Label, Confidence Score (probability of the predicted class), and the raw probability of the positive class (P(Exoplanet=1)).

**Example Output of Prediction Demo:**
```
Sample Predictions (Actual vs. Predicted with Confidence):

+----------------+-----------------+-------------------+--------------+------------------+
| Original Index | Actual Label    | Predicted Label   | Confidence   | P(Exoplanet=1)   |
+----------------+-----------------+-------------------+--------------+------------------+
|           6947 | Not Exoplanet   | Not Exoplanet     |       0.9601 |           0.0399 |
|            281 | Exoplanet       | Exoplanet         |       0.9827 |           0.9827 |
|           4158 | Exoplanet       | Exoplanet         |       0.9588 |           0.9588 |
|           3828 | Exoplanet       | Exoplanet         |       0.9061 |           0.9061 |
|           8670 | Not Exoplanet   | Not Exoplanet     |       0.9612 |           0.0388 |
+----------------+-----------------+-------------------+--------------+------------------+
```

---

## 10. Hyperparameter Tuning (Conceptual with Optuna)

While full hyperparameter tuning with [Optuna](https://optuna.org/) is not yet deeply integrated into the main automated pipeline, the configuration structure in `conf/config.yaml` includes a `tuning:` section, laying the groundwork for it.

### 10.1 Optuna Overview
Optuna is an automatic hyperparameter optimization framework, particularly designed for machine learning. It uses algorithms like Tree-structured Parzen Estimator (TPE) to efficiently search for optimal hyperparameters.

### 10.2 Configuration for Tuning (`conf/config.yaml`)
The `tuning:` section in `conf/config.yaml` currently holds placeholder parameters:
```yaml
tuning:
  cv_folds: 5               # Number of cross-validation folds
  cv_n_jobs: -1             # Parallel jobs for CV
  n_trials: 50              # Number of Optuna trials to run
  direction: "maximize"     # e.g., "maximize" ROC AUC or "minimize" log loss
  metric_to_optimize: "cv_roc_auc_mean" # Metric to optimize during CV
  # timeout_seconds: 3600   # Optional timeout for the tuning process
```

### 10.3 Integration (Placeholder)
A typical Optuna integration would involve:
1.  Defining an `objective` function that takes an Optuna `trial` object.
2.  Inside the `objective` function:
    *   Suggest hyperparameters using `trial.suggest_float()`, `trial.suggest_int()`, `trial.suggest_categorical()`.
    *   Instantiate the model with these suggested hyperparameters.
    *   Perform cross-validation (e.g., using `sklearn.model_selection.cross_val_score`) on the training data.
    *   Return the metric to be optimized (e.g., mean ROC AUC from CV).
3.  Creating an Optuna `study` object (specifying `direction` and optionally a `sampler` and `pruner`).
4.  Running `study.optimize(objective, n_trials=cfg.tuning.n_trials, timeout=cfg.tuning.timeout_seconds)`.
5.  Retrieving the best trial's parameters (`study.best_trial.params`) and best value.
6.  Optionally, saving the Optuna study for later analysis (e.g., to a SQLite DB).

This functionality could be added as a separate script or integrated into `main.py` triggered by a specific Hydra flag (e.g., `python main.py +run_tuning=true`). The best parameters found could then be updated in the default model configuration.

---

## 11. Code Structure and Modularity

The project emphasizes a modular design for better organization, maintainability, and reusability.

### 11.1 `src/` Directory Modules

*   **`__init__.py`:** Makes `src` a Python package.
*   **`config_utils.py`:**
    *   `setup_config()`: Takes the initial Hydra `DictConfig` object, resolves relative paths to absolute paths based on `cfg.base_dir` (project root), creates necessary output directories (`data/`, `models/`, `reports/`, `logs/`), and populates configuration keys for all critical file paths. This ensures all other modules work with absolute paths.
*   **`data_loader.py`:**
    *   `load_data()`: Handles loading the raw CSV data, including basic error checking and parsing options.
*   **`logger_utils.py`:**
    *   `setup_logging()`: Configures Python's `logging` module to output logs to both the console (stdout) and a file (`logs/pipeline.log`), with specified formatting and log levels.
*   **`model_trainer.py`:**
    *   `get_model_instance()`: Creates a model object based on `cfg.model.name` and `cfg.model.params`.
    *   `train_model()`: Orchestrates model training, saving the model, and logging parameters/model to MLflow.
    *   `evaluate_model()`: Calculates and logs various evaluation metrics, generates and saves/logs confusion matrix plots.
    *   `get_feature_importances()`: Calculates and logs/saves feature importance plots for tree-based models.
    *   `sanitize_metric_name()`: Helper to make metric names from `classification_report` MLflow-compatible.
*   **`predict.py`:**
    *   `load_trained_model()`: Loads a saved `.joblib` model.
    *   `load_preprocessing_artifacts()`: Loads saved scaler, imputer, etc.
    *   `preprocess_for_prediction()`: Applies all necessary preprocessing steps to new raw data, ensuring consistency with training data using loaded artifacts. This is a critical function for reliable predictions.
    *   `make_prediction()`: Uses the loaded model and preprocessed data to generate class predictions and probabilities.
*   **`preprocessor.py`:**
    *   `clean_column_names()`: Basic utility.
    *   `engineer_features()`: Placeholder for future feature engineering steps.
    *   `preprocess_data()`: The core function that takes raw data and the configuration, performs all preprocessing steps (target encoding, dropping, imputation, scaling, optional polynomial features), and saves the fitted transformers (scaler, imputer). It returns the processed `X`, `y`, and a dictionary of fitted transformers.
    *   `split_data()`: Splits data into training and testing sets with stratification.

### 11.2 Utility Functions
Helper functions are generally co-located within the module where they are most relevant (e.g., `sanitize_metric_name` in `model_trainer.py`, path resolution in `config_utils.py`).

---

## 12. Testing (Conceptual)

Formal testing is a crucial part of robust software development. While this project currently does not have an extensive test suite, here's a conceptual outline:

### 12.1 Unit Tests
*   **Location:** Would typically reside in a `tests/unit/` directory.
*   **Framework:** [pytest](https://docs.pytest.org/) is a popular choice.
*   **Targets:**
    *   Test individual functions in `src/data_loader.py` (e.g., handling of commented lines, empty files).
    *   Test functions in `src/preprocessor.py` (e.g., target encoding logic, correct application of imputer/scaler on mock data, column alignment).
    *   Test functions in `src/model_trainer.py` (e.g., metric sanitization, correct instantiation of models from config).
    *   Test functions in `src/predict.py` (e.g., ensuring `preprocess_for_prediction` correctly transforms mock data based on mock artifacts).
    *   Test `src/config_utils.py` path resolution with mock configurations.

### 12.2 Integration Tests
*   **Location:** Would typically reside in a `tests/integration/` directory.
*   **Purpose:** Test the interaction between different components of the pipeline.
*   **Examples:**
    *   A test that runs a miniature version of the `main.py` pipeline with a very small mock dataset and checks if output artifacts are created and if MLflow logging occurs.
    *   A test that checks if a saved model and artifacts can be loaded correctly by `src/predict.py` to make predictions.

*Adding a test suite would significantly enhance the project's reliability and maintainability.*

---

## 13. Contribution Guidelines

Currently, this is a solo project. However, for future collaboration:
1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (e.g., `git checkout -b feature/new-model` or `git checkout -b fix/preprocessing-bug`).
3.  Make your changes, adhering to the existing code style (Black for formatting).
4.  Add unit tests for any new functionality.
5.  Ensure all existing tests pass.
6.  Update `README.md` or other documentation if your changes affect usage or architecture.
7.  Commit your changes with clear, descriptive commit messages.
8.  Push your branch to your fork (`git push origin feature/your-feature`).
9.  Open a Pull Request against the `main` branch of the original repository, detailing your changes.

---

## 14. Future Work and Potential Improvements

This project provides a solid foundation. Potential areas for future development include:
*   **Advanced Feature Engineering:** Explore creating more sophisticated features from existing ones (e.g., ratios, interactions beyond simple polynomials, domain-specific features based on astrophysics).
*   **More Model Exploration:** Experiment with other classification algorithms (e.g., XGBoost, LightGBM, Support Vector Machines, simple Neural Networks). Add their configurations to `conf/model/`.
*   **Full Hyperparameter Tuning Integration:** Implement a robust Optuna-based hyperparameter tuning script that can be easily triggered, updating the best model parameters.
*   **Cross-Validation:** Integrate k-fold cross-validation more deeply into the training and evaluation process for more robust performance estimation, especially during hyperparameter tuning.
*   **Advanced Imbalance Handling:** Explore techniques like SMOTE (Synthetic Minority Over-sampling Technique) or ADASYN if `class_weight='balanced'` is insufficient.
*   **Error Analysis:** Perform a deeper dive into misclassified samples to understand model weaknesses and guide feature engineering or model adjustments.
*   **CI/CD Pipeline:** Set up a Continuous Integration/Continuous Deployment pipeline (e.g., using GitHub Actions) to automate testing, linting, and potentially model deployment.
*   **Model Deployment:** Package the model for serving (e.g., as a REST API using FastAPI or Flask, or deploying directly from MLflow).
*   **Data Versioning:** Implement data versioning (e.g., using DVC) if the dataset is expected to change frequently.
*   **Comprehensive Test Suite:** Develop unit and integration tests as outlined in the [Testing section](#testing-conceptual).
*   **Enhanced Visualizations:** Add more interactive visualizations to the EDA notebooks or create a separate dashboard (e.g., using Streamlit or Dash) for exploring results.
*   **Scalability:** For larger datasets or more complex models, explore optimizations or distributed computing frameworks.

---

## 15. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Aayush Badola

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 16. Acknowledgements

*   **NASA Exoplanet Archive:** For providing the Kepler Objects of Interest data.
*   The developers of Scikit-learn, Pandas, NumPy, Hydra, MLflow, and other open-source libraries used in this project.
*   The Kepler Science Team for their monumental effort in collecting and processing the data.

---

## 17. Contact Author

**Aayush Badola**

*   <a href="https://www.linkedin.com/in/aayush-badola-0a7b2b343/" target="_blank">
        <img src="https://img.shields.io/badge/LinkedIn-Aayush%20Badola-0077B5?style=for-the-badge&logo=linkedin" alt="LinkedIn Profile">
    </a>
*   <a href="https://github.com/AayushBadola" target="_blank">
        <img src="https://img.shields.io/badge/GitHub-AayushBadola-181717?style=for-the-badge&logo=github" alt="GitHub Profile">
    </a>
*   <a href="mailto:aayush.badola2@gmail.com">
        <img src="https://img.shields.io/badge/Email-aayush.badola2%40gmail.com-D14836?style=for-the-badge&logo=gmail" alt="Email">
    </a>

<br>
<p align="center">
  Made with ❤️ and ☕
</p>

---
