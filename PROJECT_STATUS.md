# Project Status

## Purpose
Predict customer credit risk using the German Credit Dataset and compare interpretable baseline classifiers for lending-risk analysis.

## What works
- exploratory data analysis
- missing-value handling
- categorical encoding
- feature scaling
- stratified train/test split
- Logistic Regression
- Decision Tree
- Random Forest
- multi-metric evaluation
- business interpretation
- documented project structure and figures

## Model snapshot
Random Forest is the strongest overall baseline in the recorded comparison:
- Accuracy: 76.5%
- Precision: 68.6%
- Recall: 40.0%
- F1: 50.5%
- ROC-AUC: 74.7%

Decision Tree has higher recall at 50.0%, so model choice still depends on the cost of missed risky borrowers versus false positives.

## Deployment
Not currently deployed.

Deployment is optional, not mandatory. Before hosting a prediction UI, the training and inference preprocessing should be packaged into one saved pipeline so the deployed app cannot drift from the evaluated workflow.

## API
No API is integrated.

A FastAPI endpoint would only be justified if another application actually needs programmatic credit scoring. A simple Streamlit demo can use the saved pipeline directly.

## Portfolio role
Flagship banking ML project and one of the strongest projects to feature publicly.

## Next improvements
- remove duplicated text at the top of README
- verify that the saved model artifact exactly matches the documented preprocessing
- create a single inference pipeline
- add a small test set for input validation
- decide between Streamlit deployment or FastAPI + small frontend only after inference packaging is clean
- add a short model limitations / responsible-use section
