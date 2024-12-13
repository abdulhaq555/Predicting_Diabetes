# Diabetes Prediction Using Machine Learning Models

This project focuses on predicting the likelihood of diabetes using machine learning algorithms. The dataset contains features related to diabetes risk factors, such as blood sugar levels, BMI, age, and family history, along with a categorical target variable representing the diagnosis result (diabetic or not). We use three machine learning models for prediction: **Logistic Regression**, **Random Forest**, and **Gradient Boosting**. The models are optimized using **RandomizedSearchCV** for hyperparameter tuning, and their performance is evaluated and compared.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objective](#objective)
3. [Dataset](#dataset)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Models Used](#models-used)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Results](#results)
8. [Installation Instructions](#installation-instructions)
9. [Usage](#usage)
10. [License](#license)

## Project Overview

This project aims to classify whether an individual has diabetes based on various health indicators. The project involves performing **Exploratory Data Analysis (EDA)**, training machine learning models, and evaluating their performance. The models used in this project include Logistic Regression, Random Forest, and Gradient Boosting, which are fine-tuned using **RandomizedSearchCV** to improve their performance.

## Objective

- Perform **Exploratory Data Analysis (EDA)** to understand the dataset's structure.
- Train machine learning models for diabetes prediction.
- Tune hyperparameters using **RandomizedSearchCV** for optimal performance.
- Compare the performance of multiple models.

## Dataset

The dataset used in this project contains various features representing health indicators and a target variable "Diagnosis" which indicates whether the individual has diabetes or not. The dataset includes independent variables such as:
- Glucose level
- BMI (Body Mass Index)
- Age
- Blood pressure, skin thickness, insulin levels, etc.

### Key Features:
- **Independent Variables**: Various health indicators (e.g., glucose, BMI, age).
- **Target Variable**: Diagnosis (binary classification: Diabetic or Not Diabetic).

## Exploratory Data Analysis (EDA)

The following steps were taken during the EDA phase:
1. **Data Cleaning**:
   - Handled missing values by imputing or removing rows/columns.
   - Encoded categorical data for compatibility with models.
2. **Data Visualization**:
   - Visualized the distribution of features using histograms.
   - Used a heatmap to visualize the correlation between features and the target variable.
3. **Class Imbalance Check**:
   - Checked for class imbalance in the target variable and applied techniques like oversampling/undersampling if necessary.
4. **Feature Engineering**:
   - Scaled continuous features using **StandardScaler**.
   - Encoded categorical variables using **One-Hot Encoding**.

## Models Used

### 1. **Logistic Regression**
- A simple and interpretable model that works well for binary classification.
- Hyperparameters tuned using **RandomizedSearchCV**:
  - **C** (Regularization strength)
  - **solver** (Optimization algorithm)
  - **max_iter** (Maximum number of iterations for convergence)

### 2. **Random Forest**
- An ensemble model combining multiple decision trees to reduce overfitting.
- Hyperparameters tuned using **RandomizedSearchCV**:
  - **n_estimators** (Number of trees)
  - **max_depth** (Maximum depth of trees)
  - **min_samples_split** (Minimum samples required to split a node)
  - **min_samples_leaf** (Minimum samples at leaf nodes)

### 3. **Gradient Boosting**
- A boosting technique where trees are added sequentially to correct previous errors.
- Hyperparameters tuned using **RandomizedSearchCV**:
  - **n_estimators** (Number of boosting stages)
  - **learning_rate** (Controls the contribution of each weak learner)
  - **max_depth** (Maximum depth of each tree)
  - **min_samples_split** and **min_samples_leaf** (Minimum samples for splitting/leaf nodes)

## Hyperparameter Tuning

**RandomizedSearchCV** was used for hyperparameter tuning to optimize the models' performance. It randomly samples parameter combinations and evaluates their performance using cross-validation, providing a more efficient approach compared to exhaustive search methods like **GridSearchCV**.

Key Parameters Tuned:
- **n_estimators**, **learning_rate**, **max_depth**, and **min_samples_split** for Random Forest and Gradient Boosting.
- **C**, **solver**, and **max_iter** for Logistic Regression.

## Results

The performance of each model was evaluated based on accuracy. The results before and after hyperparameter tuning are as follows:
- **Logistic Regression**:
  - Accuracy (before tuning): 0.693
  - Accuracy (after tuning): 0.693
- **Random Forest**:
  - Accuracy (before tuning): 0.643
  - Accuracy (after tuning): 0.677
- **Gradient Boosting**:
  - Accuracy (before tuning): 0.630
  - Accuracy (after tuning): 0.693

