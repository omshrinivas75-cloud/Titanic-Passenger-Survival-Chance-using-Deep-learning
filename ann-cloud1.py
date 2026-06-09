import streamlit as st
import pandas as pd
import pickle
import tensorflow
from tensorflow.keras.models import load_model

# Load model
model = load_model("model.h5")

# Title
st.title("Passenger Survival Chance in the Titanic Journey")

# Inputs
pclass = st.slider("Enter Passenger Class", 1, 3)

sex = st.selectbox(
    "Enter Passenger Gender",
    ["male", "female"]
)

sibsp = st.slider(
    "Enter Number of Siblings/Spouses",
    0,
    8
)

parch = st.slider(
    "Enter Number of Parents/Children",
    0,
    8
)

fare = st.number_input(
    "Enter Passenger Fare",
    min_value=0.0
)

embarked = st.selectbox(
    "Enter Boarding Station",
    ["Sounthampton", "Chebourg", "Queenstown"]
)

# Load encoders and scaler
with open("Label_encoder.pkl", "rb") as f1:
    label_encoder = pickle.load(f1)

with open("onehot_encoder.pkl", "rb") as f2:
    onehot_encoder = pickle.load(f2)

with open("scaler_encoder.pkl", "rb") as f3:
    scaler = pickle.load(f3)


def predict_survival():
    # Create dataframe
    df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])

    # Label Encoding
    df["Sex"] = label_encoder.transform(df["Sex"])

    # One-Hot Encoding
    embarked_encoded = onehot_encoder.transform(
        df[["Embarked"]]
    ).toarray()

    embarked_df = pd.DataFrame(
        embarked_encoded,
        columns=["Southampton", "Chebourg", "Queenstown"]
    )

    # Remove original Embarked column
    df = pd.concat(
        [df.drop(columns=["Embarked"]), embarked_df],
        axis=1
    )

    # Scale numerical columns
    df[["Pclass", "SibSp", "Parch", "Fare"]] = scaler.transform(
        df[["Pclass", "SibSp", "Parch", "Fare"]]
    )

    # Prediction
    prediction = model.predict(df, verbose=0)
    probability = float(prediction[0][0])

    return probability


if st.button("Predict Survival Chance"):

    probability = predict_survival()

    st.write(
        f"Probability of Survival: {probability:.4f}"
    )

    if probability > 0.5:
        st.success("The Passenger is likely to Survive.")
    else:
        st.error("The Passenger is unlikely to Survive.")
