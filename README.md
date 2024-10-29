# My-first-Kaggle-Competition
Final project Machine Learning Fundamentals and Applications

https://www.kaggle.com/competitions/ml-fundamentals-and-applications-2024-10-01

## Customer Churn Prediction Project

## Project Overview

This project aims to predict customer churn for a telecommunications company using anonymized marketing data. The primary goal is to develop an effective predictive model that can identify which customers are likely to switch service providers. This final project consolidates all the skills learned during the ML: Fundamentals and Applications course, requiring us to work with both categorical and numerical features, paying particular attention to the challenges of class imbalance.

## Dataset Description

The dataset provided for this project is anonymized and contains both numerical and categorical features related to customer behavior. The target variable indicates whether a customer is likely to churn (leave the service) or stay. The features include customer information such as demographic details, service usage, billing information, and customer support interactions.

The data is split into three subsets:
- **Training Set**: Used for model training.
- **Testing Set**: Used for model evaluation during the development process.
- **Validation Set**: Used for final predictions and scoring on Kaggle.

## Key Project Steps

1. **Exploratory Data Analysis (EDA)**
   - Conducted in-depth EDA to understand feature distributions, relationships, and the target class imbalance.
   - Missing values were identified and imputed appropriately. Categorical variables were analyzed for unique categories.

2. **Data Preprocessing**
   - **Categorical Encoding**: Applied One-Hot Encoding for categorical variables with low cardinality and Target Encoding for high-cardinality features.
   - **Normalization**: Normalized numerical features to ensure consistent scaling, improving model performance.
   - **Class Balancing**: Addressed the imbalance in churn and non-churn classes using SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class.

3. **Feature Selection and Dimensionality Reduction**
   - Employed Recursive Feature Elimination (RFE) to identify important features.
   - Conducted experiments with PCA (Principal Component Analysis) to reduce data dimensionality and improve model performance.

4. **Model Selection and Training**
   - Evaluated multiple machine learning algorithms, including Random Forest, Gradient Boosting, and XGBoost, using cross-validation.
   - Tuned hyperparameters using GridSearchCV to optimize model performance.
   - Built an ML pipeline with scikit-learn's `Pipeline` to integrate preprocessing, class balancing, and modeling seamlessly.

5. **Evaluation and Metrics**
   - The primary evaluation metric was **Balanced Accuracy**, which addresses class imbalance better by considering recall of each class.
   - Used k-fold cross-validation to obtain robust metrics and ensure that the model generalizes well to unseen data.

6. **Final Model and Kaggle Submission**
   - Retrained the final model on combined training and testing datasets to leverage all available data.
   - Generated predictions for the validation dataset and saved the results in the required `.csv` format for submission to Kaggle.

## Instructions for Running the Code

To reproduce the results or modify the model, follow these steps:

1. **Dependencies**: Ensure that the following Python libraries are installed:
   - `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `matplotlib`, and `seaborn`.

   You can install these dependencies with:
    ```bash
    conda install -c conda-forge pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
   ```
   or
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
   ```

3. **Running the Notebook**:
   - The notebook (`Notebook_Private_Score_0.8544.ipynb`) contains all the code from data preprocessing to model training and evaluation.
   - Open the notebook using Jupyter Notebook or any compatible IDE and run all cells sequentially to reproduce the final model.

4. **Generating Predictions for Kaggle**:
   - After running all cells, a CSV file named `submission.csv` will be generated in the working directory, which contains the customer indices and the predicted churn values.

## Acknowledgements

I would like to extend my gratitude to the course teacher, **Volodymyr Holomb** (https://github.com/woldemarg), and mentor, **Yehor Yanvarov** (https://github.com/herbvertuher), for their invaluable support.

---

**Author**: M Matviiuk
**Date**: 28th October 2024
**Final Kaggle Score**: Balanced Accuracy = 0.8544

