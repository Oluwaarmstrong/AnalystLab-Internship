# AnalystLab Africa Internship — Week 6: Feature Engineering and Model Optimization

## Project Overview

Week 6 focused on **feature engineering, feature transformation, feature selection, and model optimization** using the Titanic dataset.

The goal was to improve the prediction of passenger survival by creating more meaningful features, transforming variables into a format suitable for machine learning, identifying important features, and testing whether hyperparameter tuning could improve the Logistic Regression model.

The work also included comparing the baseline model with a feature-selected model and a tuned model using classification metrics.

---

## 1. Dataset

The project used the **Titanic dataset**. https://www.kaggle.com/competitions/titanic/data

The original dataset contained:

- **891 rows**
- **12 columns**

The original variables included:

- PassengerId
- Survived
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

The target variable was **Survived**.

---

## 2. Feature Engineering

Several new features were created from the existing Titanic variables.

### FamilySize

FamilySize was created by combining the number of siblings/spouses and parents/children travelling with the passenger.

```python
FamilySize = SibSp + Parch + 1
```

### IsAlone

IsAlone was created to identify passengers travelling alone.

```text
1 = Travelling alone
0 = Travelling with family
```

### Title

Passenger titles were extracted from the `Name` column.

The extracted titles initially included several different categories. The common titles were retained:

- Mr
- Miss
- Mrs
- Master

All other titles were grouped into:

```text
Rare
```

### AgeGroup

Passenger ages were divided into five groups:

- Child
- Teenager
- Young Adult
- Adult
- Senior

The age boundaries used were:

```text
0–12       Child
12–18      Teenager
18–35      Young Adult
35–60      Adult
60–100     Senior
```

### HasCabin

HasCabin was created to indicate whether cabin information was available.

```text
1 = Cabin information available
0 = Cabin information unavailable
```

---

## 3. Missing Value Handling

Missing values were checked before modelling.

The original dataset contained:

- 177 missing Age values
- 687 missing Cabin values
- 2 missing Embarked values

Age was filled using the median age:

```python
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
```

Embarked was filled using the most frequent value:

```python
titanic["Embarked"] = titanic["Embarked"].fillna(
    titanic["Embarked"].mode()[0]
)
```

Cabin was not filled directly. Instead, the `HasCabin` feature was created to capture whether cabin information existed.

After the missing Age and Embarked values were handled, the remaining Cabin missingness was represented through `HasCabin`.

---

## 4. Encoding

Categorical variables were transformed using one-hot encoding.

The following variables were encoded:

```python
["Sex", "Embarked", "Title", "AgeGroup"]
```

The encoding was performed using:

```python
pd.get_dummies(..., drop_first=True)
```

The resulting dataset contained **24 columns** before the variables that were not needed for modelling were removed.

The following columns were removed from the model input:

- Survived
- PassengerId
- Name
- Ticket
- Cabin

This left **19 features** for the machine learning model.

---

## 5. Model Features

The final 19 features used for modelling were:

1. Pclass
2. Age
3. SibSp
4. Parch
5. Fare
6. FamilySize
7. IsAlone
8. HasCabin
9. Sex_male
10. Embarked_Q
11. Embarked_S
12. Title_Miss
13. Title_Mr
14. Title_Mrs
15. Title_Rare
16. AgeGroup_Teenager
17. AgeGroup_Young Adult
18. AgeGroup_Adult
19. AgeGroup_Senior

The feature matrix had the following shape:

```text
X shape: (891, 19)
y shape: (891,)
```

The feature values were converted to integer type before modelling.

---

## 6. Train-Test Split

The data was divided into training and testing sets using an 80/20 split.

The split used:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Stratification was used so that the distribution of the target classes was maintained between the training and testing sets.

---

# 7. Feature Selection

A Random Forest classifier was trained to examine feature importance.

The Random Forest model was created with:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

The feature importance results were then sorted from highest to lowest.

### Most Important Features

The highest feature importance values were:

| Feature | Importance |
|---|---:|
| Age | 0.200341 |
| Fare | 0.169900 |
| Sex_male | 0.132473 |
| Title_Mr | 0.120543 |
| Pclass | 0.072825 |
| HasCabin | 0.051107 |
| FamilySize | 0.045664 |
| SibSp | 0.034200 |
| Title_Mrs | 0.033508 |
| Title_Miss | 0.033314 |
| Embarked_S | 0.023191 |
| Parch | 0.020492 |

Features with importance below 0.02 were excluded from the feature-selection comparison.

The 12 selected features were:

```python
[
    "Age",
    "Fare",
    "Sex_male",
    "Title_Mr",
    "Pclass",
    "HasCabin",
    "FamilySize",
    "SibSp",
    "Title_Mrs",
    "Title_Miss",
    "Embarked_S",
    "Parch"
]
```

A feature importance visualization was also created and saved as:

```text
Feature_Importance_Chart.png
```

---

# 8. Feature Scaling

Because Logistic Regression was used, the numerical features were standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler was fitted only on the training data and then applied to the test data.

The same scaling approach was also applied to the 12-feature dataset when evaluating the feature-selected model.

---

# 9. Baseline Logistic Regression

The baseline Logistic Regression model was created with:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

The model was trained using the full set of **19 features**.

### Baseline Results

| Metric | Score |
|---|---:|
| Accuracy | 83.24% |
| Precision | 80.00% |
| Recall | 75.36% |
| F1 Score | 77.61% |
| ROC-AUC | 87.58% |

The confusion matrix was:

```text
[[97 13]
 [17 52]]
```

This baseline model became the reference point for the feature-selection and hyperparameter-tuning experiments.

---

# 10. Feature-Selected Logistic Regression

The Logistic Regression model was then trained using only the 12 features selected from the Random Forest feature importance analysis.

### Results

| Metric | Score |
|---|---:|
| Accuracy | 81.56% |
| Precision | 77.27% |
| Recall | 73.91% |
| F1 Score | 75.56% |
| ROC-AUC | 86.28% |

### Feature Selection Decision

The model using all 19 features performed better than the model using the selected 12 features across all five evaluation metrics.

Therefore, the 19 features were retained for further model optimization.

This showed that some features with lower individual importance could still contribute useful information when combined with other predictors.

---

# 11. Hyperparameter Tuning

GridSearchCV was used to determine whether different Logistic Regression parameters could improve the baseline model.

The parameter grid was:

```python
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"]
}
```

Five-fold cross-validation was used with accuracy as the scoring metric.

```python
GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
```

### Best Parameters

GridSearchCV identified:

```python
{
    "C": 0.1,
    "penalty": "l2",
    "solver": "liblinear"
}
```

### Best Cross-Validation Score

```text
0.8188614202698711
```

The best estimator was then used to generate predictions on the test data.

---

# 12. Tuned Logistic Regression Results

The tuned Logistic Regression model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 81.56% |
| Precision | 79.03% |
| Recall | 71.01% |
| F1 Score | 74.81% |
| ROC-AUC | 86.59% |

Compared with the baseline model, the tuned model performed worse on the test set.

---

# 13. Model Performance Comparison

The final comparison was between:

- Baseline Logistic Regression
- Tuned Logistic Regression

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline Logistic Regression | 83.24% | 80.00% | 75.36% | 77.61% | 87.58% |
| Tuned Logistic Regression | 81.56% | 79.03% | 71.01% | 74.81% | 86.59% |

The baseline model consistently performed better across all evaluation metrics.

### Conclusion from the Comparison

Although GridSearchCV found a different parameter combination that performed well during cross-validation, the tuned model did not perform better on the final test set.

The baseline Logistic Regression model using all 19 features was therefore retained as the best-performing model from this experiment.

---

# 14. Visualizations

Two main visualizations were created during the model evaluation stage.

### Feature Importance Chart

This chart showed the importance of the 19 model features according to the Random Forest classifier.

Saved as:

```text
Feature_Importance_Chart.png
```

### Baseline vs Tuned Model Performance

A bar chart was created to compare the baseline and tuned Logistic Regression models across:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Saved as:

```text
baseline_vs_tuned_logistic_regression.png
```

### Confusion Matrix

A heatmap was created for the baseline Logistic Regression model.

Saved as:

```text
baseline_logistic_regression_confusion_matrix.png
```

---

# 15. Model Saving

The baseline Logistic Regression model was selected as the final model because it achieved the best test-set performance.

It was saved using Joblib:

```python
jb.dump(
    logistic_model,
    "final_titanic_logistic_regression.pkl"
)
```

The optimized model was also saved separately:

```python
jb.dump(
    grid_search.best_estimator_,
    "optimized_logistic_regression.pkl"
)
```

The baseline model was retained as the preferred model because it performed better on the test data.

---

# 16. Technologies Used

The notebook used:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

Scikit-learn tools used included:

- `train_test_split`
- `GridSearchCV`
- `StandardScaler`
- `RandomForestClassifier`
- `LogisticRegression`
- Classification metrics
- Confusion matrix

---

# 17. Key Lessons

The main lessons from Week 6 were:

1. **Feature engineering can add useful information to a dataset.** FamilySize, IsAlone, Title, AgeGroup, and HasCabin provided additional information beyond the original variables.

2. **Feature importance does not automatically determine the best feature set.** Although Random Forest identified 12 relatively important features, the Logistic Regression model performed better when all 19 features were retained.

3. **Feature scaling is important for Logistic Regression.** StandardScaler was used to standardize the model inputs.

4. **Hyperparameter tuning does not guarantee better test performance.** GridSearchCV identified the best parameter combination through cross-validation, but the tuned model performed worse on the held-out test set.

5. **Model selection should be based on actual evaluation results.** The baseline Logistic Regression model was retained because it achieved the best overall test performance.

---

# Conclusion

Week 6 focused on improving the Titanic survival prediction workflow through feature engineering, feature transformation, feature selection, and model optimization.

The project created five new features, handled missing values, encoded categorical variables, reduced the data to 19 model features, and used Random Forest feature importance to investigate which variables were most influential.

A baseline Logistic Regression model using all 19 features achieved **83.24% accuracy and 87.58% ROC-AUC**. Feature selection reduced performance, while GridSearchCV also produced a tuned model that performed worse on the final test set.

Based on the experimental results, the **baseline Logistic Regression model using all 19 features** was retained as the final model and saved as:

```text
final_titanic_logistic_regression.pkl
```
