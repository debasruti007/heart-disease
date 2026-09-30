"""
=============================================================================
 webapp/app.py

 Streamlit form for the heart disease risk screener. This file NEVER
 loads the model itself - it only calls the FastAPI backend (api/main.py)
 over HTTP, as required by the brief ("the model must be served behind
 your own API endpoint").

 Run the API first:   uvicorn api.main:app --reload
 Then run this file:  streamlit run webapp/app.py
=============================================================================
"""

import os
import requests
import streamlit as st

API_URL = os.environ.get("HEART_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Heart Disease Risk Screener", page_icon="\u2764\ufe0f")

st.title("Heart Disease Risk Screener")

# Mandatory disclaimer - shown prominently, before any input is taken.
st.warning(
    "**This tool is a screening aid, not a medical diagnosis.** "
    "It does not replace professional medical advice. Please consult "
    "a qualified healthcare professional for any health concerns."
)

st.write("Enter the patient's clinical details below.")


def call_api(payload: dict) -> dict:
    """
    Sends the form data to the FastAPI backend and returns its JSON
    response. Raises requests.RequestException if the API is unreachable,
    and includes the API's own error detail if the input was rejected.
    """
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ValueError(f"API rejected the request: {detail}")
    return response.json()


# -----------------------------------------------------------------------
# The form. Number inputs use the SAME ranges as the API's validation
# (see PatientInput in api/main.py), so bad input is usually caught here
# before it's even submitted - the API still re-checks everything itself.
# -----------------------------------------------------------------------
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        cp = st.selectbox(
            "Chest pain type", options=[
                ("Typical angina", 1), ("Atypical angina", 2),
                ("Non-anginal pain", 3), ("Asymptomatic", 4)],
            format_func=lambda x: x[0])[1]
        trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=50, max_value=250, value=130)
        chol = st.number_input("Serum cholesterol (mg/dl)", min_value=0, max_value=700, value=230)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl?", options=[("No", 0), ("Yes", 1)],
                            format_func=lambda x: x[0])[1]
        restecg = st.selectbox(
            "Resting ECG result", options=[
                ("Normal", 0), ("ST-T wave abnormality", 1), ("Left ventricular hypertrophy", 2)],
            format_func=lambda x: x[0])[1]

    with col2:
        thalach = st.number_input("Max heart rate achieved", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Exercise-induced angina?", options=[("No", 0), ("Yes", 1)],
                              format_func=lambda x: x[0])[1]
        oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0,
                                   value=1.0, step=0.1)
        slope = st.selectbox(
            "Slope of peak exercise ST segment", options=[
                ("Upsloping", 1), ("Flat", 2), ("Downsloping", 3)],
            format_func=lambda x: x[0])[1]
        ca_known = st.checkbox("Number of major vessels (ca) is known", value=False)
        ca = st.number_input("Major vessels colored by fluoroscopy (0-3)", min_value=0, max_value=3,
                              value=0, disabled=not ca_known)
        thal_known = st.checkbox("Thalassemia result is known", value=False)
        thal = st.selectbox("Thalassemia", options=[("Normal", 3), ("Fixed defect", 6),
                                                      ("Reversible defect", 7)],
                             format_func=lambda x: x[0], disabled=not thal_known)
        thal = thal[1] if thal_known else None

    submitted = st.form_submit_button("Assess risk")

if submitted:
    payload = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope,
        "ca": ca if ca_known else None,
        "thal": thal,
    }

    try:
        result = call_api(payload)
    except requests.exceptions.RequestException:
        st.error(
            "Could not reach the prediction API. Make sure it is running "
            "(`uvicorn api.main:app --reload`) and try again."
        )
    except ValueError as e:
        st.error(str(e))
    else:
        st.subheader("Result")

        band = result["risk_band"]
        band_color = {"Low": "green", "Moderate": "orange", "High": "red"}.get(band, "gray")
        st.markdown(f"**Risk probability:** {result['risk_probability']:.1%}")
        st.markdown(f"**Risk band:** :{band_color}[{band}]")
        if result["flagged_for_review"]:
            st.markdown("**This patient is flagged for clinical review.**")

        st.subheader("What influenced this result")
        for feature in result["top_contributing_features"]:
            arrow = "\u2b06\ufe0f" if feature["direction"] == "increased risk" else "\u2b07\ufe0f"
            st.write(f"{arrow} **{feature['feature']}** - {feature['direction']} "
                     f"(impact: {feature['impact']:.3f})")

        st.info(result["disclaimer"])