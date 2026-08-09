import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and column list
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Telco Customer Churn Prediction")
st.write("Enter customer details to predict churn risk.")

# --- Input fields ---
tenure = st.slider("Tenure (months with company)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges (₹)", 0, 150, 70)
total_charges = st.number_input("Total Charges (₹)", 0.0, 10000.0, 1000.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])

if st.button("Predict Churn Risk"):
    # Build a single-row dataframe matching training data format
    input_dict = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': "No",
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': "No",
        'DeviceProtection': "No",
        'TechSupport': tech_support,
        'StreamingTV': "No",
        'StreamingMovies': "No",
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encode the same way as training
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Align columns with training data (fill missing dummy columns with 0)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict using the Random Forest model (no scaling needed for this model)
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ This customer is LIKELY to churn (Risk: {probability*100:.1f}%)")
    else:
        st.success(f"✅ This customer is UNLIKELY to churn (Risk: {probability*100:.1f}%)")
