# RetailGenius – Customer Churn Prediction

## Project Overview

This repository contains the complete implementation of the **AI Project Methodology – Graded Project (2025–2026)**. The objective of the project is to design, implement, and explain an end-to-end AI pipeline for **customer churn prediction** for a fictional e-commerce company called **RetailGenius**.

The project is structured according to industry best practices and covers:

* Functional framing of an AI project (Part 1)
* Technical implementation with ML best practices (Part 2)
* Explainable AI using SHAP (Part 3)

The focus of this project is **methodology, reproducibility, and explainability**, rather than model performance optimization.

---

## Project Structure

```
retailgenius-churn-prediction/
│
├── data/
│   ├── raw/                # Original dataset (Excel / CSV)
│   └── processed/          # Cleaned and feature-engineered data
│
├── src/
│   ├── data_preprocessing.py   # Data loading and cleaning
│   ├── feature_engineering.py  # Feature preparation
│   ├── train.py                # Model training with MLflow
│   ├── evaluate.py             # Model evaluation
│   └── xai_shap.py             # Explainable AI (SHAP)
│
├── models/
│   └── churn_model.pkl     # Trained model artifact
│
├── mlruns/                 # MLflow experiment tracking
│
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── .gitignore
```

---

## Dataset

The project uses an **E-Commerce Customer Churn Dataset**, which contains historical customer information including:

* Demographic attributes
* Purchase behavior
* Engagement metrics

**Target variable:**

* `Churn` (binary: churned / not churned)

The dataset exhibits class imbalance, which is a common characteristic of churn prediction problems.

---

## Environment Setup

### Requirements

* Python 3.10 (recommended)
* pip

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

All commands should be executed from the project root directory.

### 1. Data Preprocessing

```bash
python src/data_preprocessing.py
```

* Loads the raw dataset
* Handles missing values
* Saves cleaned data to `data/processed/`

---

### 2. Feature Engineering

```bash
python src/feature_engineering.py
```

* Encodes categorical variables
* Produces a machine-learning-ready dataset

---

### 3. Model Training & Experiment Tracking

```bash
python src/train.py
```

* Trains a Random Forest classifier
* Logs parameters, metrics, and model artifacts using **MLflow**
* Saves the trained model locally

To launch the MLflow UI:

```bash
mlflow ui
```

Then open:

```
http://127.0.0.1:5000
```

---

### 4. Model Evaluation

```bash
python src/evaluate.py
```

* Evaluates the trained model on a test set
* Displays classification metrics

---

### 5. Explainable AI (SHAP)

```bash
python src/xai_shap.py
```

* Generates SHAP explanations
* Produces global and local explainability plots:

  * Summary plot
  * Mean feature importance plot
  * Waterfall plot
  * Force plot

Generated plots are saved as `.png` files in the project directory.

---

## Explainability and Business Value

SHAP explanations are used to:

* Identify the most influential features driving customer churn
* Provide transparency into model predictions
* Support business teams in designing targeted retention strategies

Both **global** (dataset-level) and **local** (individual prediction) explanations are included.

---

## Reproducibility & Best Practices

This project follows best practices for production-ready AI systems:

* Modular code structure
* Reproducible pipelines
* Dependency management via `requirements.txt`
* Experiment tracking with MLflow
* Model persistence and versioning

---

## Repository & Submission

* The complete project code, models, and MLflow artifacts are available in this repository
* The repository has been shared with the instructor for evaluation
* The final report (Parts 1, 2, and 3) is submitted via the group’s dedicated drive

---

## Author

Aishik Das
AI Project Methodology – EPITA International Programs

---

## References

* Scikit-learn documentation
* MLflow documentation
* SHAP documentation
* CRISP-DM methodology resources
* Public e-commerce churn datasets
