
from flask import Flask, request, jsonify
import joblib
import pandas as pd


# Creating the Flask application.
app = Flask(__name__)


# Loading the saved Logistic Regression model.
model = joblib.load("final_titanic_logistic_regression.pkl")


def preprocess_input(data):

    # Converting the input data into a DataFrame.
    df = pd.DataFrame([data])

    # Creating the FamilySize feature.
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Identifying passengers travelling alone.
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Creating age groups from the passenger's age.
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=[
            "Child",
            "Teenager",
            "Young Adult",
            "Adult",
            "Senior"
        ]
    )

    # Converting categorical variables into numerical features.
    df = pd.get_dummies(
        df,
        columns=["Sex", "Embarked", "Title", "AgeGroup"],
        drop_first=True
    )

    # Defining the features required by the model.
    required_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "IsAlone",
        "HasCabin",
        "Sex_male",
        "Embarked_Q",
        "Embarked_S",
        "Title_Miss",
        "Title_Mr",
        "Title_Mrs",
        "Title_Rare",
        "AgeGroup_Teenager",
        "AgeGroup_Young Adult",
        "AgeGroup_Adult",
        "AgeGroup_Senior"
    ]

    # Adding missing features with zero values.
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0

    # Keeping the same feature order used during training.
    df = df[required_features]

    return df


@app.route("/predict", methods=["POST"])
def predict():

    # Receiving passenger information from the request.
    data = request.get_json()

    # Processing the passenger information.
    processed_data = preprocess_input(data)

    # Generating the survival prediction.
    prediction = model.predict(processed_data)[0]

    # Converting the prediction into a readable result.
    result = "Survived" if prediction == 1 else "Did not survive"

    # Returning the prediction as JSON.
    return jsonify({
        "prediction": int(prediction),
        "result": result
    })


# Running the Flask application locally.
if __name__ == "__main__":
    app.run(debug=True)
