"""
Heart Disease Prediction - Demo App
----------------------------------------------------------
A small Streamlit web app built from the "Heart Disease Prediction" notebook.
It loads the trained Support Vector Classifier (heart_model.pkl) and recreates
the exact same LabelEncoders (fit on the same fixed category lists) used during
training, so predictions match the notebook exactly.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

# ---------------------------------------------------------------------------
# Load the trained model (same file the notebook produced)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "heart_model.pkl"

# Fixed category lists - identical to Step 5.1 of the notebook
CHEST_PAIN = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
VESSELS = ["Zero", "One", "Two", "Three"]
THAL = ["Normal", "Fixed Defect", "Reversible Defect"]
EXANG = ["No", "Yes"]


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    """
    Recreates the exact same LabelEncoders used in the notebook.
    LabelEncoder assigns codes based on the sorted order of the category
    list, so fitting on the same fixed lists reproduces the same codes
    deterministically - no need to pickle/ship the encoders separately.
    """
    encoders = {}
    encoders["ChestPain"] = LabelEncoder().fit(CHEST_PAIN)
    encoders["Vessels"] = LabelEncoder().fit(VESSELS)
    encoders["Thal"] = LabelEncoder().fit(THAL)
    encoders["ExerciseAngina"] = LabelEncoder().fit(EXANG)
    return encoders


model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("🫀 Heart Disease Predictor")
st.markdown(
    """
    This is a teaching demo built from the **Heart Disease Prediction** notebook.
    Enter a patient's details below and the trained Support Vector Classifier (SVC)
    will predict whether the patient has heart disease.

    *Model accuracy on held-out test data: **80%** (60 patients, UCI Cleveland dataset).*
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    chest_pain = st.selectbox("Chest Pain Type", CHEST_PAIN, index=0)
    thal = st.selectbox("Thalassemia (Thallium Stress Test Result)", THAL, index=0)
with col2:
    vessels = st.selectbox("Major Vessels Colored by Fluoroscopy", VESSELS, index=0)
    exang = st.selectbox("Exercise-Induced Angina", EXANG, index=0)

if st.button("Predict Heart Disease", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "ChestPain": [chest_pain],
        "Vessels": [vessels],
        "Thal": [thal],
        "ExerciseAngina": [exang],
    })

    encoded = pd.DataFrame({
        "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
        "Vessels": encoders["Vessels"].transform(user_input["Vessels"]),
        "Thal": encoders["Thal"].transform(user_input["Thal"]),
        "ExerciseAngina": encoders["ExerciseAngina"].transform(user_input["ExerciseAngina"]),
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE")
    else:
        st.success("### ✅ NO HEART DISEASE")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Built for teaching purposes from a small (297-patient) UCI Cleveland sample. "
    "This is an educational demo and not medical advice."
)