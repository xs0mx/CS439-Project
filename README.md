# Interpretable vs Black-Box Models for Predicting Death Events in Heart Failure Patients

## Group Members
- Tala Alkabbani
- Salem Nasereddin

## Project Overview
This project compares interpretable machine learning models and black-box models for predicting death events in heart failure patients.

The main question is whether simpler models, such as Logistic Regression and Decision Tree, can perform close to more complex models, such as Random Forest and Gradient Boosting.

## Dataset
The project uses the Heart Failure Clinical Records Dataset.

- Rows: 299 patients
- Target variable: `DEATH_EVENT`
- Negative class: 203 patients
- Positive class: 96 patients

Main features include age, anaemia, creatinine phosphokinase, diabetes, ejection fraction, high blood pressure, platelets, serum creatinine, serum sodium, sex, smoking, and time.

## Models
The models compared are:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Each model was tested with two feature sets:

- With `time`
- Without `time`

## Evaluation Metrics
The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Recall is important in this project because missing a high-risk patient is a serious issue.

## Main Results

| Model | Feature Set | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | With time | 0.800 | 0.733 | 0.579 | 0.647 | 0.858 |
| Logistic Regression | Without time | 0.700 | 0.526 | 0.526 | 0.526 | 0.741 |
| Decision Tree | With time | 0.783 | 0.636 | 0.737 | 0.683 | 0.778 |
| Decision Tree | Without time | 0.717 | 0.538 | 0.737 | 0.622 | 0.744 |
| Random Forest | With time | 0.833 | 0.846 | 0.579 | 0.688 | 0.899 |
| Random Forest | Without time | 0.733 | 0.615 | 0.421 | 0.500 | 0.797 |
| Gradient Boosting | With time | 0.833 | 0.800 | 0.632 | 0.706 | 0.868 |
| Gradient Boosting | Without time | 0.700 | 0.538 | 0.368 | 0.438 | 0.766 |

## Main Findings
Random Forest with time had the highest ROC-AUC.

Decision Tree had the highest recall on the test split.

Models generally performed better when the `time` feature was included.

Important predictors included:

- time
- serum creatinine
- ejection fraction
- age
- serum sodium

## Project Structure

```text
final_project/
├── data/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_models.ipynb
│   ├── 03_v2_models.ipynb
│   ├── 04_interpretability.ipynb
│   └── 05_final_results.ipynb
├── paper/
├── results/
│   ├── figures/
│   ├── metrics/
│   └── tables/
└── README.md
```

## How to Run
Run the notebooks in this order:

```text
01_eda.ipynb
02_preprocessing.ipynb
03_models.ipynb
03_v2_models.ipynb
04_interpretability.ipynb
05_final_results.ipynb
```

## Libraries Used
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Notes
Because this project uses a small dataset and the models were run multiple times during development, some reported values may differ slightly across notebooks, figures, commits, or reruns. These differences can happen because train-test splits, cross-validation folds, and model randomness may produce slightly different results each time the code is executed.

The overall project idea and conclusions remain the same: we compare interpretable models and ensemble models for heart-failure mortality prediction, with recall treated as the most important metric because missing high-risk patients is costly. The results should be interpreted as exploratory rather than exact deployment-level performance, especially because the dataset contains only 299 observations.