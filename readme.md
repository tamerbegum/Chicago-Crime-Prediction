# Chicago Crime Analysis and Prediction

## Project Overview
This project analyzes crime data from Chicago and builds machine learning models to predict arrests. It includes exploratory data analysis, data preprocessing, and the implementation of various machine learning algorithms.

## Files in the Project
1. `ExploratoryDataAnalysis.py`: Performs initial data exploration and visualization.
2. `FactorizedSetXGB&CAT.py`: Implements XGBoost and CatBoost models with feature engineering.
3. `preprocessing.py`: Handles data cleaning, feature engineering, and preparation for modeling.
4. `MachineLearning.py`: Implements and compares various machine learning models.

## Key Features
- Data cleaning and preprocessing of Chicago crime data
- Exploratory data analysis with visualizations
- Feature engineering and selection
- Implementation of multiple machine learning models:
  - K-Nearest Neighbors (KNN)
  - Decision Trees
  - Random Forest
  - Logistic Regression
  - CatBoost
  - AdaBoost
  - XGBoost
  - Neural Networks
- Model evaluation and comparison using metrics like accuracy and AUC
- Hyperparameter tuning for optimal model performance
- Visualization of model results and feature importance

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- CatBoost
- TensorFlow
- SHAP (for model interpretability)

## Setup and Installation
1. Clone the repository
2. Install required packages:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost tensorflow shap
3. Ensure you have the Chicago crime dataset files in the project directory.

## Usage
1. Run `ExploratoryDataAnalysis.py` for initial data insights.
2. Execute `preprocessing.py` to prepare the data for modeling.
3. Use `MachineLearning.py` to train and evaluate different models.
