
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# Load saved models
@st.cache_resource
def load_model_data():
    return joblib.load("wine_quality_models.pkl")

model_data = load_model_data()
models = model_data["models"]
feature_columns = model_data["feature_columns"]

# App heading
st.title("🍷 Wine Quality Predictor")
st.write(
    "Enter the chemical properties of a wine to predict "
    "whether it belongs to the good-quality or low-quality class."
)

st.info(
    "Good Quality = quality score ≥ 6. "
    "Low Quality = quality score < 6."
)

# Model selection
model_name = st.selectbox(
    "Choose a classification model",
    list(models.keys())
)

st.subheader("Wine Characteristics")

# Input form
with st.form("wine_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        wine_type = st.selectbox(
            "Wine type", ["red", "white"]
        )
        fixed_acidity = st.number_input(
            "Fixed acidity", min_value=0.0, value=7.0
        )
        volatile_acidity = st.number_input(
            "Volatile acidity", min_value=0.0, value=0.5
        )
        citric_acid = st.number_input(
            "Citric acid", min_value=0.0, value=0.3
        )
        residual_sugar = st.number_input(
            "Residual sugar", min_value=0.0, value=2.5
        )
        chlorides = st.number_input(
            "Chlorides", min_value=0.0, value=0.07
        )
        free_sulfur_dioxide = st.number_input(
            "Free sulfur dioxide", min_value=0.0, value=15.0
        )

    with col2:
        total_sulfur_dioxide = st.number_input(
            "Total sulfur dioxide", min_value=0.0, value=46.0
        )
        density = st.number_input(
            "Density", min_value=0.0, value=0.996
        )
        ph = st.number_input(
            "pH", min_value=0.0, max_value=14.0, value=3.3
        )
        sulphates = st.number_input(
            "Sulphates", min_value=0.0, value=0.6
        )
        alcohol = st.number_input(
            "Alcohol (%)", min_value=0.0, value=10.0
        )

    submitted = st.form_submit_button(
        "Predict Wine Quality",
        type="primary"
    )

if submitted:
    # Build the input row using the training feature names
    input_values = {
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": ph,
        "sulphates": sulphates,
        "alcohol": alcohol,
        "wine_type_white": int(wine_type == "white")
    }

    input_df = pd.DataFrame(
        [[input_values[col] for col in feature_columns]],
        columns=feature_columns
    )

    model = models[model_name]
    prediction = int(model.predict(input_df)[0])

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("🍷 Predicted class: Good Quality")
    else:
        st.warning("Predicted class: Low Quality")

    st.caption(
        f"Model used: {model_name}. "
        "This is a machine-learning classification result, "
        "not a professional assessment of the wine."
    )
