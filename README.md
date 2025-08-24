# 🏗️ Airbnb Price Prediction Project

## Project Overview

This repository contains a modular, reproducible pipeline for predicting Airbnb listing prices using multimodal data (tabular, text, spatial, and optionally image features). The project is structured for clarity, scalability, and ease of collaboration, following best practices for modern machine learning workflows.

---

## 📁 Project Structure

```
Airbnb-price-prediction_02/
├── README.md              # Project documentation
├── setup.py               # Python package configuration
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
├── data/                  # Project data (raw & processed)
├── notebooks/             # Jupyter notebooks (analysis, prototyping)
├── src/                   # Source code (modular package)
└── venv/                  # Virtual environment (not tracked by git)
```

---

### 📊 `data/` — Project Data

```
data/
├── raw/                  # Original data files (from Inside Airbnb)
│   ├── listings.gz
│   ├── calendar.gz
│   ├── reviews.gz
│   ├── neighbourhoodsCSV.csv
│   └── neighbourhoodsJSON.geojson
├── processed/            # Cleaned/generated data
│   ├── listings_clean.csv
│   ├── calendar_clean.csv
│   └── features/         # Engineered features
└── README.md             # Data documentation
```
**Purpose:** Store all raw, processed, and feature data. Large files are ignored by git.

---

### 📓 `notebooks/` — Analysis & Experimentation

```
notebooks/
├── 01_data_exploration.ipynb         # Initial data exploration
├── 02_feature_engineering.ipynb      # Basic feature engineering
├── 03_baseline_models.ipynb          # Baseline models (Linear, RF, etc.)
├── 04_categorical_encoding.ipynb     # Categorical variable encoding
├── 05_spatial_features.ipynb         # Spatial features (distances, clustering)
├── 06_multimodal_integration.ipynb   # Multimodal integration (tabular + text + spatial)
└── 07_final_evaluation.ipynb         # Final evaluation & comparison
```
**Purpose:** Interactive analysis, prototyping, and results presentation.

---

### 💻 `src/` — Source Code (Python Package)

```
src/
├── __init__.py           # Makes src a Python package
├── config.py             # Configuration (paths, parameters)
├── data/                 # Data processing pipeline
│   ├── load.py           # Data loading
│   ├── preprocess.py     # Data cleaning/preprocessing
│   └── split.py          # Train/test/validation split
├── features/             # Feature engineering modules
│   ├── tabular.py        # Numeric features
│   ├── text.py           # Text features (sentiment, keywords)
│   ├── spatial.py        # Spatial features (distances, clustering)
│   ├── images.py         # Image features (if available)
│   └── categorical.py    # Categorical encoding
├── models/               # Machine learning models
│   ├── base_model.py     # Base class for all models
│   ├── linear.py         # Linear, Ridge, Lasso, Elastic Net
│   ├── nn.py             # Neural networks (MLP, hybrid)
│   └── multimodal.py     # Multimodal model (combines all types)
└── scripts/              # CLI scripts
    ├── run_preprocessing.py  # Full preprocessing pipeline
    ├── train_model.py        # Model training
    └── evaluate.py           # Model evaluation
```
**Purpose:** Modular, reusable code for all data, feature, and modeling steps.

---

## 🚦 Workflow

1. **Data Exploration:**
   - Use `01_data_exploration.ipynb` to understand and visualize the raw data.
2. **Preprocessing:**
   - Clean and prepare data using `src/data/` modules and `02_feature_engineering.ipynb`.
3. **Feature Engineering:**
   - Engineer tabular, text, spatial, and categorical features using `src/features/` and related notebooks.
4. **Modeling:**
   - Train and compare baseline and advanced models using `src/models/` and `03_baseline_models.ipynb`.
5. **Multimodal Integration:**
   - Combine multiple data modalities in `06_multimodal_integration.ipynb` and `src/models/multimodal.py`.
6. **Evaluation:**
   - Evaluate and compare models in `07_final_evaluation.ipynb` and with CLI scripts.

---

## 🛠️ Usage

- **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  # or for development
  pip install -e .[dev,notebook]
  ```
- **Run scripts:**
  ```bash
  python src/scripts/run_preprocessing.py
  python src/scripts/train_model.py
  python src/scripts/evaluate.py
  ```
- **Work interactively:**
  Open and run the Jupyter notebooks in order for step-by-step analysis.

---

## 🤝 Contributing

- Please keep code modular and well-documented.
- Add new features as separate modules or notebooks.
- Large data files should **not** be committed to git (see `.gitignore`).
- For questions or improvements, open an issue or pull request.

---

## 📚 References
- [Inside Airbnb Open Data](http://insideairbnb.com/)
- Project template and structure inspired by best practices in ML engineering.

---

**Good luck and happy modeling!**# Airbnb_PricePrediction
