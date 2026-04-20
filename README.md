# Final Project

## Project Title
Interpretable vs Black-Box Models for Predicting Death Events in Heart Failure Patients

## Group Members
- Tala Alkabbani
- Salem Nasereddin

## Project Overview
This project compares interpretable machine learning models and black-box models for predicting death events in heart failure patients.

The main goal is to see whether simpler and more understandable models, like logistic regression and decision trees, can perform close to more complex models, like random forests and gradient boosting.

This project is about more than just accuracy. Since this is a healthcare-related problem, it is also important to understand how the model makes predictions. A model that is easier to explain may be more useful in real-life medical settings.

## Why This Project Matters
Heart failure is a serious condition, and predicting which patients may be at higher risk can be helpful.

In healthcare, predictions should be used carefully. The purpose of this project is not to replace doctors or medical judgment. Instead, the goal is to explore whether machine learning can support decision-making in a way that is both useful and understandable.

We also want to compare performance fairly and look at which clinical features seem most important.

## Research Question
Can interpretable machine learning models predict death events in heart failure patients nearly as well as black-box models?

## Dataset
We are using the Heart Failure Clinical Records Dataset.

The dataset includes patient health information such as:
- age
- anaemia
- creatinine phosphokinase
- diabetes
- ejection fraction
- high blood pressure
- platelets
- serum creatinine
- serum sodium
- sex
- smoking
- time

Target variable:
- `DEATH_EVENT`

## Models
The models planned for this project are:

### Interpretable Models
- Logistic Regression — TBD
- Decision Tree — TBD

### Black-Box Models
- Random Forest — TBD
- Gradient Boosting — TBD

## Evaluation
The models will be compared using:
- Accuracy — TBD
- Precision — TBD
- Recall — TBD
- F1-score — TBD
- ROC-AUC — TBD
- Confusion Matrix — TBD

Because this is a healthcare task, recall is especially important, since missing a high-risk patient could be harmful.

## Project Structure
```text
final_project/
│
├── data/
├── notebooks/
├── src/
├── results/
├── paper/
├── requirements.txt
└── README.md