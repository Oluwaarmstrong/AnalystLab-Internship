
import streamlit as st
import pandas as pd
import joblib

# I configured the Streamlit page for the Titanic prediction application.
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢"
)

# I displayed the title and a short description of the application.
st.title("🚢 Titanic Survival Predictor")
st.write("Enter the passenger information to generate a survival prediction.")

# I loaded the saved Logistic Regression model without retraining it.
model = joblib.load("final_titanic_logistic_regression.pkl")

# I collected the passenger class.
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

# I collected the passenger's sex.
sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

# I collected the passenger's age as a whole number.
age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=25,
    step=1
)

# I collected the number of siblings or spouses.
sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

# I collected the number of parents or children.
parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

# I collected the passenger fare.
fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=30.0
)

# I collected the passenger's port of embarkation.
embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)

# I collected the passenger's title.
title = st.selectbox(
    "Title",
    ["Mr", "Miss", "Mrs", "Master", "Rare"]
)

# I collected whether cabin information was available.
has_cabin = st.selectbox(
    "Has Cabin",
    [0, 1]
)

# I generated a prediction when the user selected the prediction button.
if st.button("Predict Survival"):

    # I created a DataFrame containing the passenger information.
    passenger_data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "Title": title,
        "HasCabin": has_cabin
    }])

    # I created the FamilySize feature used during model training.
    passenger_data["FamilySize"] = (
        passenger_data["SibSp"] +
        passenger_data["Parch"] +
        1
    )

    # I created the IsAlone feature.
    passenger_data["IsAlone"] = (
        passenger_data["FamilySize"] == 1
    ).astype(int)

    # I grouped the passenger's age into the same categories used during training.
    passenger_data["AgeGroup"] = pd.cut(
        passenger_data["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=[
            "Child",
            "Teenager",
            "Young Adult",
            "Adult",
            "Senior"
        ]
    )

    # I converted the categorical variables into numerical features.
    passenger_data = pd.get_dummies(
        passenger_data,
        columns=["Sex", "Embarked", "Title", "AgeGroup"],
        drop_first=True
    )

    # I defined the 19 features expected by the trained model.
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

    # I added missing features with a value of zero.
    for feature in required_features:
        if feature not in passenger_data.columns:
            passenger_data[feature] = 0

    # I arranged the features in the exact order expected by the model.
    passenger_data = passenger_data[required_features]

    # I generated the survival prediction using the saved model.
    prediction = model.predict(passenger_data)[0]

    # I displayed the prediction to the user.
    if prediction == 1:
        st.success("The model predicts that the passenger survived.")
    else:
        st.error("The model predicts that the passenger did not survive.")
