# Week 5 – Advanced Machine Learning & Model Optimization

## Overview

Week 5 focused on applying advanced machine learning algorithms and model optimization techniques to the **House Price Prediction dataset** used earlier in the internship.

The main goal was to build multiple regression models, compare their performance, and use hyperparameter tuning to determine whether the performance of a selected model could be improved.

This week provided practical experience with:

* Baseline machine learning models
* Decision Trees
* Random Forest
* Gradient Boosting
* Hyperparameter tuning
* Model evaluation and comparison

---

## Dataset

The dataset contains information about houses and their corresponding prices.

Link to dataset: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

The main variables include:

* `price` – Target variable representing the house price
* `area` – Size of the house
* `bedrooms` – Number of bedrooms
* `bathrooms` – Number of bathrooms
* `stories` – Number of floors
* `mainroad` – Whether the property is connected to the main road
* `guestroom` – Whether the property has a guest room
* `basement` – Whether the property has a basement
* `hotwaterheating` – Whether hot water heating is available
* `airconditioning` – Whether air conditioning is available
* `parking` – Number of parking spaces
* `prefarea` – Whether the property is located in a preferred area
* `furnishingstatus` – Furnishing status of the property

The target variable was **`price`**, making this a **regression problem**.

---

## 1. Data Preparation

The dataset was first inspected to understand its structure and identify potential data-quality issues.

The preparation process included:

1. Loading the dataset using Pandas.
2. Inspecting the data structure and data types.
3. Checking for missing values.
4. Checking for duplicate records.
5. Encoding categorical variables.
6. Preparing the features and target variable.
7. Splitting the dataset into training and testing sets.
8. Applying feature scaling where required.

The categorical variables were converted into numerical representations so that they could be used by the machine learning algorithms.

---

## 2. Baseline Model – Linear Regression

Linear Regression was used as the baseline model.

The purpose of the baseline model was to provide a reference point against which the more advanced models could be compared.

### Baseline Results

| Metric   |                Result |
| -------- | --------------------: |
| MAE      |           ₦970,043.40 |
| MSE      | ₦1,754,318,687,330.66 |
| RMSE     |         ₦1,324,506.96 |
| R² Score |                0.6529 |

The model achieved an **R² score of approximately 0.65**, meaning that the model explained about 65% of the variation in house prices in the test dataset.

---

## 3. Decision Tree Regression

A Decision Tree Regressor was trained to capture potentially nonlinear relationships between the house features and prices.

### Results

| Metric   |                Result |
| -------- | --------------------: |
| MAE      |         ₦1,195,266.06 |
| MSE      | ₦2,642,802,637,614.68 |
| RMSE     |         ₦1,625,669.90 |
| R² Score |                0.4771 |

The Decision Tree performed worse than the Linear Regression baseline.

Its R² score of approximately **0.48** indicates that it explained less variation in house prices than the baseline model.

---

## 4. Random Forest Regression

Random Forest was then introduced as an ensemble model consisting of multiple decision trees.

The purpose was to determine whether combining several decision trees would produce better predictions than using a single Decision Tree.

### Results

| Metric   |                Result |
| -------- | --------------------: |
| MAE      |         ₦1,022,560.05 |
| MSE      | ₦1,964,193,399,645.33 |
| RMSE     |         ₦1,401,496.84 |
| R² Score |                0.6114 |

Random Forest performed better than the individual Decision Tree but still did not outperform the Linear Regression baseline.

---

## 5. Gradient Boosting Regression

Gradient Boosting was also tested.

Unlike Random Forest, which builds trees independently and combines their predictions, Gradient Boosting builds models sequentially, with each new model attempting to improve upon the errors made by the previous models.

### Results

| Metric   |                Result |
| -------- | --------------------: |
| MAE      |           ₦960,578.78 |
| MSE      | ₦1,689,379,037,287.40 |
| RMSE     |         ₦1,299,761.15 |
| R² Score |                0.6658 |

Gradient Boosting produced the best results among the initial models tested.

It achieved:

* The **lowest MAE**
* The **lowest RMSE**
* The **highest R² score**

This made Gradient Boosting the strongest candidate for hyperparameter tuning.

---

## 6. Model Comparison

The models were compared using MAE, MSE, RMSE, and R².

| Model                   |             MAE |                       MSE |              RMSE |         R² |
| ----------------------- | --------------: | ------------------------: | ----------------: | ---------: |
| Linear Regression       |     ₦970,043.40 |     ₦1,754,318,687,330.66 |     ₦1,324,506.96 |     0.6529 |
| Decision Tree           |   ₦1,195,266.06 |     ₦2,642,802,637,614.68 |     ₦1,625,669.90 |     0.4771 |
| Random Forest           |   ₦1,022,560.05 |     ₦1,964,193,399,645.33 |     ₦1,401,496.84 |     0.6114 |
| **Gradient Boosting**   | **₦960,578.78** | **₦1,689,379,037,287.40** | **₦1,299,761.15** | **0.6658** |
| Tuned Gradient Boosting |   ₦1,021,988.20 |     ₦1,937,963,730,283.18 |     ₦1,392,107.66 |     0.6166 |

### Key Observation

Gradient Boosting was the best-performing model before hyperparameter tuning.

Its R² score of **0.6658** was slightly higher than the Linear Regression score of **0.6529**, while its RMSE and MAE were also lower.

---

## 7. Hyperparameter Tuning

Since Gradient Boosting produced the best initial results, it was selected for hyperparameter tuning.

`GridSearchCV` was used to test different combinations of hyperparameters.

The parameters considered included:

* `n_estimators`
* `learning_rate`
* `max_depth`
* `min_samples_split`

### Best Parameters

The GridSearchCV process identified the following combination:

```text
learning_rate = 0.05
max_depth = 2
min_samples_split = 2
n_estimators = 100
```

The best cross-validation score was:

```text
-1,081,553.12
```

The negative value occurs because GridSearchCV was using a loss/error metric where **lower error is better**, so Scikit-learn represents the score as a negative value.

---

## 8. Tuned Gradient Boosting Results

After applying the selected hyperparameters, the tuned Gradient Boosting model produced:

| Metric   |          Tuned Result |
| -------- | --------------------: |
| MAE      |         ₦1,021,988.20 |
| MSE      | ₦1,937,963,730,283.18 |
| RMSE     |         ₦1,392,107.66 |
| R² Score |                0.6166 |

Interestingly, the tuned model performed worse on the test set than the original Gradient Boosting model.

---

## 9. Original vs Tuned Gradient Boosting

| Metric   | Original Gradient Boosting | Tuned Gradient Boosting |
| -------- | -------------------------: | ----------------------: |
| MAE      |            **₦960,578.78** |           ₦1,021,988.20 |
| MSE      |  **₦1,689,379,037,287.40** |   ₦1,937,963,730,283.18 |
| RMSE     |          **₦1,299,761.15** |           ₦1,392,107.66 |
| R² Score |                 **0.6658** |                  0.6166 |

The original Gradient Boosting model outperformed the tuned model across all four evaluation metrics.

This was an important finding because **hyperparameter tuning does not automatically guarantee better performance on unseen test data**.

---

## 10. Model Evaluation

The following metrics were used to evaluate the regression models:

### MAE – Mean Absolute Error

MAE measures the average absolute difference between the predicted and actual house prices.

A lower MAE indicates that predictions are, on average, closer to the actual values.

### MSE – Mean Squared Error

MSE calculates the average squared difference between predicted and actual values.

Because the errors are squared, larger errors have a greater impact on this metric.

### RMSE – Root Mean Squared Error

RMSE is the square root of MSE and is expressed in the same unit as the target variable.

For this project, RMSE represents the typical size of the prediction error in terms of house price.

### R² Score

R² measures how much of the variation in house prices is explained by the model.

A higher R² indicates better explanatory and predictive performance.

---

## 11. Key Findings

1. **Gradient Boosting was the best initial model.** It achieved the highest R² score of 0.6658 and the lowest RMSE of approximately ₦1.30 million.

2. **Decision Tree performed the worst.** Its R² score of 0.4771 was substantially lower than the other models.

3. **Random Forest improved upon the individual Decision Tree**, demonstrating the benefit of combining multiple trees, but it still did not outperform Linear Regression or Gradient Boosting.

4. **Gradient Boosting slightly outperformed Linear Regression**, suggesting that capturing nonlinear relationships provided some additional predictive value.

5. **Hyperparameter tuning did not improve the final test performance.** The tuned Gradient Boosting model had a lower R² score and higher prediction errors than the original model.

6. **The original Gradient Boosting model was selected as the best-performing model** for this week's experiment.

---

## 12. Final Model

Based on the test-set results, the original **Gradient Boosting Regressor** was selected as the best-performing model.

### Final Performance

* **MAE:** ₦960,578.78
* **RMSE:** ₦1,299,761.15
* **R² Score:** 0.6658

The model explains approximately **66.6% of the variation in house prices** in the test dataset.

---

## 13. Conclusion

Week 5 provided practical experience with advanced machine learning algorithms, ensemble methods, model comparison, and hyperparameter tuning.

The experiment demonstrated that more complex models do not necessarily perform better in every situation. While Gradient Boosting initially produced the strongest results, hyperparameter tuning actually reduced its test-set performance.

This highlights the importance of evaluating models on unseen data rather than assuming that a tuned model will always outperform its baseline version.

Overall, the **Gradient Boosting Regressor** provided the best performance for the House Price Prediction task and was selected as the final model for this week's project.

---

## Tools and Libraries

The project was implemented using Python and the following libraries:

* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-learn** – Machine learning models, preprocessing, evaluation, and hyperparameter tuning

---

## Project Structure

```text
Week-5/
│
├── README.md
├── week_5_advanced_machine_learning.ipynb
│
└── visuals/
    ├── model_comparison.png
    └── ...
```

---

## Skills Demonstrated

Through this project, I practiced:

* Regression modeling
* Decision Tree Regression
* Random Forest Regression
* Gradient Boosting Regression
* Model comparison
* Hyperparameter tuning
* GridSearchCV
* Cross-validation
* Regression evaluation metrics
* Data preprocessing
* Model selection
* Interpreting machine learning results
