import streamlit as st
import pandas as pd
import joblib

model = joblib.load("loan_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🏦 Loan Approval Prediction System")
st.write("Predict whether a loan application is likely to be approved using Machine Learning.")

st.sidebar.title("About")

st.sidebar.info("""
This application predicts whether a loan will be approved based on applicant details.

**Machine Learning Models Used**
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Developed using Streamlit.
""")

# -----------------------------
# User Inputs
# -----------------------------
gender = st.selectbox("Gender", ["Male", "Female"])

married = st.selectbox("Married", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

education = st.selectbox("Education", ["Graduate", "Not Graduate"])

self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0,
    value=0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=150
)

loan_term = st.number_input(
    "Loan Amount Term",
    min_value=0,
    value=360
)

credit_history = st.selectbox(
    "Credit History",
    [1, 0]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

gender = 1 if gender == "Male" else 0

married = 1 if married == "Yes" else 0

education = 0 if education == "Graduate" else 1

self_employed = 1 if self_employed == "Yes" else 0

dependents = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3+": 3
}[dependents]

property_area = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}[property_area]

input_data = pd.DataFrame([[
    gender,
    married,
    dependents,
    education,
    self_employed,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    property_area
]], columns=[
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area"
])

if st.button("Predict Loan Status"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success("✅ Congratulations! Loan is likely to be Approved.")
    else:
        st.error("❌ Loan is likely to be Rejected.")

    st.write(f"### Approval Probability: **{probability[0][1]*100:.2f}%**")

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;">
        <h4>Developed by Rohit Prasad</h4>
        <p>AI & Machine Learning Project | Streamlit Application</p>
    </div>
    """,
    unsafe_allow_html=True
)