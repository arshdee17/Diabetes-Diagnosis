# Diabetes-Diagnosis
Dataset: Pima Indians Diabetes Diagnosis 🩺

This repository contains a complete machine learning pipeline to predict the onset of diabetes in Pima Indian women based on diagnostic measurements. 

## 📋 Project Overview
The goal of this project is to build and optimize a binary classifier that predicts whether a patient has diabetes (Outcome 1) or not (Outcome 0). The dataset consists of 768 entries with 8 medical predictor variables.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Environment:** Google Colab
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn

## 🚀 Workflow

### 1. Data Quality & EDA
* **Imputation:** Identified "hidden" missing values (zeros) in Glucose, Blood Pressure, BMI, etc., and replaced them with median values grouped by Outcome.
* **Visualization:** Utilized Histograms to understand distributions and a Correlation Heatmap to identify key predictors.

### 2. Baseline Model
* **Algorithm:** Logistic Regression
* **Pre-processing:** StandardScaler was applied to normalize feature scales.
* **Result:** Provided a baseline accuracy of ~76%, but highlighted a need for better recall in positive cases.

### 3. Model Improvement (Optimized Model)
* **Algorithm:** Random Forest Classifier
* **Feature Selection:** Used `SelectKBest` to focus on the top 6 most impactful features.
* **Tuning:** Performed Hyperparameter tuning via `GridSearchCV` to optimize `n_estimators` and `max_depth`.
* **Impact:** Significant improvement in the F1-score and Recall, making the model more reliable for clinical clinical application.

## 📊 Results
The final model identifies **Glucose**, **BMI**, and **Age** as the most significant predictors of diabetes in this population.

## 📂 Repository Structure
* `diabetes.csv`: The raw dataset.
* `Binary_Classification.ipynb`: The complete Python analysis and model code.
* `README.md`: Project documentation.
