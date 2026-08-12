# 🚢 AnalysLab Africa Week 7 — Model Deployment & Real-World Application

---

## 📌 Project Description

This project focuses on deploying a machine learning model developed to predict whether a passenger survived the Titanic disaster.

The model was developed using the Titanic dataset and trained to identify patterns between passenger characteristics and survival outcomes. After evaluating the models, the baseline Logistic Regression model with the highest accuracy was selected for deployment.

The trained model was saved using **Joblib** so that it could be reused without retraining. A **Flask API** was developed to receive passenger information and return survival predictions. A **Streamlit application** was also created to provide a simple interface for interacting with the prediction system.

The main objective of this project is to demonstrate how a machine learning model developed in a Jupyter Notebook can be converted into a usable application.

---

## ❓ Problem Statement

The Titanic dataset contains information about passengers who travelled on the RMS Titanic, including their age, sex, passenger class, family information, fare, embarkation point, and cabin information.

The objective of this project is to predict whether a passenger survived based on these characteristics.

The target variable is:
- `1` — Survived
- `0` — Did not survive

The model uses passenger information together with engineered features such as `FamilySize`, `IsAlone`, `Title`, `AgeGroup`, and `HasCabin`.

---

## 🤖 Model Used

The model used for deployment is a **Logistic Regression** classifier.

The baseline Logistic Regression model was selected because it achieved the highest accuracy among the models evaluated during the model development process.

### Model Configuration

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

### Model Performance

The selected model achieved the following results before deployment:

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 83.24% |
| Precision | 80.00% |
| Recall    | 75.36% |
| F1 Score  | 77.61% |
| ROC-AUC   | 87.58% |

The trained model was saved using Joblib as:

```
final_titanic_logistic_regression.pkl
```

The saved model can be loaded and reused without retraining.

---

## 🛠️ Technologies Used

The project was developed using:

| Technology       | Purpose                                  |
|------------------|------------------------------------------|
| Python           | Main programming language                |
| Pandas           | Data manipulation and preprocessing      |
| NumPy            | Numerical operations                     |
| Scikit-learn     | Machine learning                         |
| Joblib           | Saving and loading the trained model     |
| Flask            | Building the prediction API              |
| Streamlit        | Creating the web interface               |
| Requests         | Sending requests to the Flask API        |
| Jupyter Notebook | Development environment                  |

---

## 🔌 API Endpoints

The Flask application provides one prediction endpoint.

### `POST /predict`

This endpoint receives passenger information and returns the predicted survival outcome.

**Local endpoint:**
```
http://127.0.0.1:5000/predict
```

**Request method:** `POST`

### Input Format

The API accepts passenger information in JSON format.

**Example request:**

```json
{
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S",
    "Title": "Mr",
    "HasCabin": 0
}
```

### Input Variables

| Field    | Description                                        |
|----------|----------------------------------------------------|
| Pclass   | Passenger class                                    |
| Sex      | Passenger sex                                      |
| Age      | Passenger age                                      |
| SibSp    | Number of siblings or spouses aboard               |
| Parch    | Number of parents or children aboard               |
| Fare     | Passenger fare                                     |
| Embarked | Port of embarkation                                |
| Title    | Passenger title                                    |
| HasCabin | Indicates whether cabin information is available   |

The API processes these values and creates the required features before passing the data to the trained model.

### Output Format

The API returns a JSON response containing the predicted class and a readable result.

**Example response:**

```json
{
    "prediction": 0,
    "result": "Did not survive"
}
```

The prediction values represent:
- `1` — Survived
- `0` — Did not survive

A successful prediction request returns a status code of `200`.

---

## 🧪 API Testing

The Flask API was tested using sample passenger information.

A successful request returned a status code of `200`.

**Example response:**

```json
{
    "prediction": 0,
    "result": "Did not survive"
}
```

This confirmed that the API could successfully:
- Receive passenger information
- Process the input
- Load the saved model
- Generate a prediction
- Return the result

---

## 🖥️ Streamlit Application

A Streamlit application was created to provide a simple user interface for the prediction system.

The application allows users to enter passenger information such as:
- Passenger class
- Sex
- Age
- Number of siblings or spouses
- Number of parents or children
- Fare
- Port of embarkation
- Passenger title
- Cabin availability

The application sends the entered information to the Flask prediction API and displays the returned survival prediction.

> The age input is configured to accept whole numbers because passenger age is represented in years.

---

## 📁 Project Structure

```
Week 7/
│
├── Week_7_Model_Deployment.ipynb
├── app.py
├── streamlit_app.py
├── final_titanic_logistic_regression.pkl
└── README.md
```

### File Descriptions

| File                                      | Description                                                                 |
|-------------------------------------------|-----------------------------------------------------------------------------|
| `Week_7_Model_Deployment.ipynb`           | Contains the development and deployment process completed during Week 7     |
| `app.py`                                  | Contains the Flask application and prediction endpoint                      |
| `streamlit_app.py`                        | Contains the Streamlit user interface                                       |
| `final_titanic_logistic_regression.pkl`   | Contains the saved Logistic Regression model used to generate predictions   |
| `README.md`                               | Contains the project documentation and instructions for running the app     |

---

## ⚙️ Setup Instructions

### 1. Install Python

Python must be installed on the computer before running the project.

### 2. Install the Required Libraries

Run the following command in the terminal:

```bash
pip install pandas numpy scikit-learn joblib flask streamlit requests
```

### 3. Place the Project Files in the Same Directory

Make sure the following files are available in the project directory:
- `app.py`
- `streamlit_app.py`
- `final_titanic_logistic_regression.pkl`

> The saved model file is required because the Flask API loads it to generate predictions.

---

```
## ▶️ How to Run the Project

### Step 1 — Start the Streamlit Application

Open a terminal in the project directory and run:

```bash
streamlit run streamlit_app.py
```

Streamlit will provide a local URL, normally:

```
http://localhost:8501
```

Open the URL in a web browser.

### Step 2 — Enter Passenger Information

Enter the required passenger information through the Streamlit interface.

The application collects information such as:

- Passenger class
- Sex
- Age
- Number of siblings/spouses
- Number of parents/children
- Fare
- Port of embarkation
- Passenger title
- Cabin availability

### Step 3 — Generate a Prediction

Click the **Predict Survival** button.

The Streamlit application processes the passenger information, creates the required features, and passes the data directly to the saved Logistic Regression model.

The model then returns the predicted survival outcome, which is displayed in the application.

---

## 🌐 Flask API

The Flask API developed during this project can also be run separately for API testing.

To start the Flask API, open a terminal in the project directory and run:

```bash
python app.py
```

The Flask API will run locally at:

```
http://127.0.0.1:5000
```

The prediction endpoint is:

```
http://127.0.0.1:5000/predict
```

> The Flask API is included as part of the deployment work and can be used to test predictions through an API. However, the Streamlit application does not depend on the local Flask server and loads the saved model directly.

---

## 🔄 Deployment Workflow

The Streamlit application follows this workflow:

```
User
  ↓
Streamlit Application
  ↓
Input Processing
  ↓
Feature Engineering
  ↓
Saved Logistic Regression Model
  ↓
Prediction
  ↓
Streamlit Result
```

The Flask API follows a separate workflow for API-based testing:

```
User / API Request
  ↓
Flask /predict Endpoint
  ↓
Input Processing
  ↓
Feature Engineering
  ↓
Saved Logistic Regression Model
  ↓
Prediction
  ↓
JSON ResponseS
```
```


---

## ✅ Conclusion

The Titanic survival prediction model was successfully prepared for real-world use.

The baseline Logistic Regression model with the highest accuracy was saved using Joblib and integrated into a Flask API. The API was successfully tested and returned predictions with a status code of `200`.

A Streamlit application was also created to provide a simple interface through which users can enter passenger information and receive survival predictions.

This project demonstrates the transition from developing and evaluating a machine learning model in Jupyter Notebook to making the trained model accessible through an API and web application.
