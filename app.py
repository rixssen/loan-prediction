import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("final.pkl")

# Title
st.title("Loan Approval Prediction App")

# Input fields
married = st.selectbox("Married", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
loan_amount = st.number_input("Loan Amount", min_value=0)
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0)

# Encode inputs (assuming LabelEncoding like: "No"=0, "Yes"=1 for 'Married', etc.)
married_encoded = 1 if married == "Yes" else 0
dependents_encoded = {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents]

# Prediction
if st.button("Predict"):
    input_data = np.array([[married_encoded, dependents_encoded, loan_amount,
                            applicant_income, coapplicant_income, loan_amount_term]])
    
    prediction = model.predict(input_data)[0]
    result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"
    st.success(result)


