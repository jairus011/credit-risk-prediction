# Credit Risk Prediction Using Machine Learning

## Executive Summary

This project developed an end-to-end machine learning solution for predicting customer credit risk using the German Credit dataset. The primary objective was to identify customers who are likely to default on their loans while minimizing the rejection of creditworthy applicants. Three machine learning classification models—Logistic Regression, Decision Tree, and Random Forest—were trained and evaluated to determine the most suitable approach for credit risk prediction.

The project followed a complete machine learning workflow, beginning with data understanding and exploratory data analysis, followed by data preprocessing, model development, performance evaluation, and business interpretation. Among the evaluated models, the Random Forest Classifier demonstrated the strongest overall performance by achieving the highest Accuracy, Precision, F1-Score, and ROC-AUC while maintaining improved detection of high-risk borrowers compared to Logistic Regression.

Although the Random Forest outperformed the other models, the evaluation also highlighted opportunities for further improvement before deployment in a production banking environment. Overall, the project demonstrates a practical application of machine learning to financial risk management and provides a strong foundation for future model optimization.

# Business Problem

## Background

Financial institutions are exposed to significant financial risk whenever loans are issued to customers who may fail to meet their repayment obligations. While extending credit generates revenue through interest, approving loans for high-risk applicants can result in loan defaults, increased operational costs, and reduced profitability. Conversely, rejecting creditworthy applicants may lead to missed business opportunities and reduced customer satisfaction.

Traditional credit assessment methods often rely on manual evaluation and predefined business rules. However, as the volume of loan applications continues to increase, financial institutions are increasingly adopting machine learning techniques to improve the speed, consistency, and accuracy of credit risk assessment.

---

## Problem Statement

The challenge addressed in this project is to develop a machine learning model capable of classifying loan applicants as either **Good** or **Bad** credit risks based on their demographic, financial, and loan-related characteristics. The objective is to support lending decisions by reducing the likelihood of approving high-risk borrowers while minimizing the rejection of customers who are likely to repay their loans successfully.

---

## Business Objectives

This project was developed to achieve the following objectives:

- Develop a reliable machine learning pipeline for credit risk prediction.
- Compare multiple classification algorithms to identify the most suitable model.
- Improve the detection of high-risk borrowers while maintaining acceptable prediction accuracy.
- Provide interpretable insights that can support credit approval decisions.
- Demonstrate how machine learning can assist financial institutions in reducing loan default risk while maintaining profitable lending practices.

# Dataset Overview

## Dataset Description

This project uses the German Credit dataset, a widely used benchmark dataset for credit risk classification. The dataset contains information about loan applicants, including demographic characteristics, financial status, employment information, and loan details. The objective is to predict whether a customer represents a Good or Bad credit risk.

After preprocessing, the dataset contained predictor variables describing each customer and a binary target variable (**Risk**) indicating the customer's creditworthiness.

The major feature categories include:

- Customer demographics (Age, Sex)
- Employment characteristics (Job)
- Financial information (Saving accounts, Checking account)
- Loan characteristics (Credit amount, Duration, Purpose)
- Housing information (Housing)

These variables provide valuable information for understanding customer borrowing behaviour and estimating credit risk.

---

# Exploratory Data Analysis Findings

Exploratory Data Analysis (EDA) was conducted to understand the characteristics of the dataset before model development.

The analysis revealed that:

- The dataset contained a larger proportion of Good credit customers than Bad credit customers, indicating a moderately imbalanced classification problem.
- Numerical variables such as Credit amount and Duration showed right-skewed distributions with several realistic high-value observations.
- Loan duration appeared to have a stronger relationship with credit risk than Age and Credit amount.
- Several categorical variables, including Checking account, Saving accounts, Job, Housing, and Purpose, demonstrated varying relationships with customer credit risk.
- Correlation analysis indicated no severe multicollinearity among numerical variables, allowing all major numerical features to be retained for modelling.

The findings from EDA guided the preprocessing decisions and feature selection process.

---

# Data Preprocessing Findings

The preprocessing stage prepared the raw dataset for machine learning by improving data quality and ensuring compatibility with classification algorithms.

The following preprocessing steps were performed:

- Removed unnecessary identifier columns.
- Handled missing values by introducing an **Unknown** category instead of deleting valuable customer records.
- Applied Binary Encoding to binary variables.
- Applied Ordinal Encoding to ordered categorical variables.
- Applied One-Hot Encoding to nominal categorical variables.
- Split the dataset into training (80%) and testing (20%) sets using stratified sampling.
- Standardized numerical variables using StandardScaler.
- Saved the processed datasets for reproducible model development.

These preprocessing steps reduced the risk of data leakage while preserving valuable customer information.

---

# Model Development Findings

Three supervised machine learning classification algorithms were developed and evaluated.

## Logistic Regression

Logistic Regression served as the baseline model due to its simplicity, interpretability, and suitability for binary classification problems. Although it achieved reasonable overall accuracy, it struggled to detect risky borrowers because of its relatively low recall for the Bad credit class.

## Decision Tree

The Decision Tree improved the detection of risky borrowers by learning nonlinear decision rules. However, its performance suggested signs of overfitting, resulting in reduced precision and lower generalization compared to the ensemble model.

## Random Forest

Random Forest combined multiple Decision Trees to improve predictive stability and reduce overfitting. Among the evaluated models, it achieved the strongest overall performance by balancing predictive accuracy with improved identification of risky borrowers.

---

# Model Comparison

The three machine learning models demonstrated different strengths and weaknesses.

| Metric | Logistic Regression | Decision Tree | Random Forest |
|--------|:-------------------:|:-------------:|:-------------:|
| Accuracy | 0.7300 | 0.7000 | **0.7650** |
| Precision | 0.6250 | 0.5000 | **0.6857** |
| Recall | 0.2500 | **0.5000** | 0.4000 |
| F1-Score | 0.3571 | 0.5000 | **0.5053** |
| ROC-AUC | 0.6720 | 0.6429 | **0.7474** |

Although the Decision Tree achieved the highest Recall, Random Forest provided the strongest overall balance between correctly identifying risky borrowers and minimizing unnecessary rejection of creditworthy applicants.

---

# Business Recommendations

Based on the evaluation results, the Random Forest Classifier is recommended as the preferred baseline model for credit risk prediction.

The model demonstrated:

- The highest overall Accuracy.
- The highest Precision.
- The highest F1-Score.
- The highest ROC-AUC.
- Better generalization than a single Decision Tree.

However, the current model should not be deployed directly into a production banking environment. Additional optimization is required to improve the detection of high-risk borrowers while maintaining acceptable precision.

---

# Limitations

Several limitations were identified during this project:

- The German Credit dataset contains only 1,000 customer records.
- The dataset is moderately imbalanced, making the identification of risky borrowers more challenging.
- Default hyperparameters were used without optimization.
- Limited feature engineering was performed.
- External financial and behavioural variables were unavailable.

These limitations restrict the predictive performance of the current baseline models.

---

# Future Improvements

Future work should focus on improving both predictive performance and model robustness through:

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Addressing class imbalance using techniques such as SMOTE or class weighting.
- Optimizing the probability threshold for lending decisions.
- Developing additional engineered financial features.
- Evaluating advanced ensemble algorithms such as XGBoost, LightGBM, and CatBoost.
- Validating the models using larger and more recent banking datasets.

---

# Conclusion

This project successfully demonstrated an end-to-end machine learning workflow for credit risk prediction, covering data understanding, preprocessing, model development, evaluation, and business interpretation.

Among the evaluated algorithms, Random Forest emerged as the strongest baseline model by achieving the best overall balance across multiple evaluation metrics. Although additional optimization is required before deployment, the project provides a strong technical and business foundation for developing more advanced credit risk prediction systems.

Beyond model performance, this project highlights the importance of combining technical evaluation with business reasoning when selecting machine learning models for financial decision-making. Effective credit risk prediction is not solely about maximizing accuracy, but about balancing profitability, financial risk, and responsible lending practices.