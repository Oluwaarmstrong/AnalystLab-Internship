# Week 6: Feature Engineering & Model Optimization

## Project Overview

Week 6 focused on improving the machine learning models developed in
previous weeks through feature engineering, data transformation, feature
selection, and hyperparameter tuning.

The main goal was to understand how preparing the data differently and
optimizing model parameters can affect machine learning performance.

## 1. Feature Engineering

### Titanic Dataset: https://www.kaggle.com/competitions/titanic/data

The following features were created:

-   **FamilySize** --- calculated as `SibSp + Parch + 1`.
-   **IsAlone** --- identifies passengers travelling alone (`1`) or with
    family (`0`).
-   **Title** --- extracted from passenger names. Common titles such as
    `Mr`, `Miss`, `Mrs`, and `Master` were retained, while less common
    titles were grouped as `Rare`.
-   **AgeGroup** --- grouped passengers into Child, Teenager, Young
    Adult, Adult, and Senior.
-   **HasCabin** --- indicates whether cabin information was available.

### House Prices Dataset

The existing housing variables were prepared for modelling and
transformed where necessary so that they could be used by the machine
learning algorithms.

## 2. Data Transformation

Categorical variables were converted into numerical representations
using encoding techniques.

-   **Label Encoding** was used where categorical values needed
    numerical representation.
-   **One-Hot Encoding** was used for categorical variables with
    multiple categories.
-   **Scaling and normalization** were applied where appropriate to
    place numerical variables on comparable scales.

## 3. Feature Selection

Feature selection was used to identify useful variables and reduce
unnecessary information.

The approaches explored included:

-   Correlation-based feature selection
-   Tree-based feature importance
-   Recursive Feature Elimination (RFE)

For the Titanic dataset, the Random Forest feature importance analysis
identified **Age, Fare, Sex_male, Title_Mr, and Pclass** among the most
important features.

## 4. Model Comparison

For the House Prices regression task, the models compared were:

-   Linear Regression
-   Decision Tree
-   Random Forest
-   Gradient Boosting

The main evaluation metrics were MAE, MSE, RMSE, and R².

  Model                          MAE           RMSE       R²
  ------------------- -------------- -------------- --------
  Linear Regression       970,043.40   1,324,506.96   0.6529
  Decision Tree         1,195,266.06   1,625,669.90   0.4771
  Random Forest         1,022,560.05   1,401,496.84   0.6114
  Gradient Boosting       960,578.78   1,299,761.15   0.6658

Gradient Boosting produced the strongest initial result, with the lowest
RMSE and highest R² among the models compared.

## 5. Hyperparameter Tuning

Grid Search was used to test different Gradient Boosting parameter
combinations.

The best parameters identified were:

``` python
{
    "learning_rate": 0.05,
    "max_depth": 2,
    "min_samples_split": 2,
    "n_estimators": 100
}
```

The best cross-validation score was:

``` text
-1081553.1217
```

Random Search was also explored as another approach to hyperparameter
optimization.

## 6. Performance After Optimization

The tuned Gradient Boosting model produced:

  Metric     Tuned Gradient Boosting
  -------- -------------------------
  MAE                   1,021,988.20
  MSE                  1.9379 × 10¹²
  RMSE                  1,392,107.66
  R²                          0.6166

The tuned model did not outperform the original Gradient Boosting model
on the final test set. The original Gradient Boosting model achieved an
RMSE of **1,299,761.15** and an R² of **0.6658**.

This demonstrated that hyperparameter tuning does not automatically
guarantee better test-set performance.

## 7. Titanic Logistic Regression

The baseline Logistic Regression model achieved:

  Metric         Score
  ----------- --------
  Accuracy      83.24%
  Precision     80.00%
  Recall        75.36%
  F1 Score      77.61%
  ROC-AUC       87.58%

The tuned Logistic Regression model did not outperform the baseline
across the main evaluation metrics. The baseline model was therefore
retained as the preferred model for the later deployment work.

## 8. Key Findings

1.  Feature engineering created additional information that could be
    used by the models.
2.  Feature selection helped identify variables that contributed
    strongly to predictions.
3.  Gradient Boosting produced the best initial regression performance
    among the models compared for the House Prices dataset.
4.  Hyperparameter tuning did not automatically improve final test
    performance.
5.  Model selection was based on actual evaluation results rather than
    assuming that a tuned model would always perform better.
6.  The baseline Titanic Logistic Regression model remained the
    preferred classification model for deployment.

## 9. Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Jupyter Notebook

Scikit-learn techniques used included StandardScaler, Label Encoding,
One-Hot Encoding, Random Forest, Gradient Boosting, Recursive Feature
Elimination (RFE), Grid Search, Random Search, and model evaluation
metrics.

## Conclusion

Week 6 focused on improving the machine learning workflow through
feature engineering, data transformation, feature selection, and model
optimization.

The work showed that better features and parameter tuning can influence
model performance, but optimization must always be validated on unseen
data. The final model should be selected based on its actual performance
rather than simply choosing the model that has undergone the most
tuning.
