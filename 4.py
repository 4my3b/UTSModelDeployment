import pickle
import pandas as pd
import streamlit as st
import time

with open("xgb_best_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("xgb_label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
with open("xgb_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("xgb_ordinal_encoder.pkl", "rb") as f:
    ordinal_encoder = pickle.load(f)
with open("xgb_feature_order.pkl", "rb") as f:
    feature_order = pickle.load(f)

scale_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'cb_person_cred_hist_length', 'credit_score']
ord_col = 'person_education'

st.title("FinPred")
st.markdown("##### FinPred is a machine learning-powered application designed to predict whether a loan application will be approved or rejected. By analyzing key factors such as age, income, credit history, and loan purpose, FinPred helps financial institutions make fast, accurate, and data-driven decisions.")
st.write("Please input the data")

test_cases = {
    "Test Case 1": {
        'person_age': 28,
        'person_income': 60000.0,
        'person_home_ownership': 'RENT',
        'person_emp_exp': 5,
        'loan_intent': 'MEDICAL',
        'loan_amnt': 20000.0,
        'loan_int_rate': 13.88,
        'cb_person_cred_hist_length': 6,
        'person_gender': 'male',
        'person_education': 'Bachelor',
        'previous_loan_defaults_on_file': 'No',
        'credit_score': 632
    },
    "Test Case 2": {
        'person_age': 28,
        'person_income': 86000.0,
        'person_home_ownership': 'MORTGAGE',
        'person_emp_exp': 6,
        'loan_intent': 'EDUCATION',
        'loan_amnt': 9000.0,
        'loan_int_rate': 10.48,
        'cb_person_cred_hist_length': 6,
        'person_gender': 'female',
        'person_education': 'Associate',
        'previous_loan_defaults_on_file': 'Yes',
        'credit_score': 633
    }
}

col1, col2 = st.columns(2)
if col1.button("Example 1"):
    for k, v in test_cases["Test Case 1"].items():
        st.session_state[k] = v
if col2.button("Example 2"):
    for k, v in test_cases["Test Case 2"].items():
        st.session_state[k] = v

col_a, col_b, col_c = st.columns(3)

with col_a:
    person_age = st.number_input("Age", value=st.session_state.get("person_age", 30), step=1)
    person_home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'], index=['RENT', 'OWN', 'MORTGAGE', 'OTHER'].index(st.session_state.get("person_home_ownership", 'RENT')))
    loan_amnt = st.number_input("Loan Amount", value=st.session_state.get("loan_amnt", 10000.0))
    person_gender = st.selectbox("Gender", ['male', 'female'], index=['male', 'female'].index(st.session_state.get("person_gender", 'male')))

with col_b:
    person_income = st.number_input("Income", value=st.session_state.get("person_income", 50000.0))
    person_emp_exp = st.number_input("Work Experience (years)", value=st.session_state.get("person_emp_exp", 5), step=1)
    loan_int_rate = st.number_input("Loan Interest Rate (%)", value=st.session_state.get("loan_int_rate", 10.0))
    person_education = st.selectbox("Education", ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'], index=['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'].index(st.session_state.get("person_education", 'Bachelor')))

with col_c:
    loan_intent = st.selectbox("Loan Purpose", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'], index=['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'].index(st.session_state.get("loan_intent", 'EDUCATION')))
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", value=st.session_state.get("cb_person_cred_hist_length", 5), step=1)
    default_val = st.session_state.get("previous_loan_defaults_on_file", 'No')
    index = ['Yes', 'No'].index(default_val) if default_val in ['Yes', 'No'] else 1
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File?", ['Yes','No'], index=index)
    credit_score = st.number_input("Credit Score", value=st.session_state.get("credit_score", 650), step=1)

if st.button("Predict"):
    user_input = {
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'person_emp_exp': person_emp_exp,
        'loan_intent': loan_intent,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'person_gender': person_gender,
        'person_education': person_education,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file,
        'credit_score': credit_score
    }

    with st.spinner("Processing..."):
        time.sleep(2)
        input_df = pd.DataFrame([user_input])

        for col, le in label_encoders.items():
            input_df[col] = le.transform(input_df[col])

        input_df[[ord_col]] = ordinal_encoder.transform(input_df[[ord_col]])
        input_df[scale_cols] = scaler.transform(input_df[scale_cols])
        input_df = input_df[feature_order]

        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        prediction = int(prediction)
        prediction_proba = [float(p) for p in prediction_proba]

    status_map = {0: "Denied", 1: "Approved"}
    if prediction == 1:
        st.success(f"Loan Status: {status_map[prediction]}")
    else:
        st.error(f"Loan Status: {status_map[prediction]}")

    st.write("Prediction Probability:")
    prob_df = pd.DataFrame({
        "Status": [status_map[i] for i in range(len(prediction_proba))],
        "Probability": [f"{p*100:.2f}%" for p in prediction_proba]
    })
    st.write(prob_df)
