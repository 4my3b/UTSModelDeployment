import pickle
import pandas as pd
import streamlit as st

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

st.title("Prediksi Status Pinjaman")
st.write("Masukkan data untuk memprediksi status pinjaman.")

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
if col1.button("Gunakan Test Case 1"):
    for k, v in test_cases["Test Case 1"].items():
        st.session_state[k] = v
if col2.button("Gunakan Test Case 2"):
    for k, v in test_cases["Test Case 2"].items():
        st.session_state[k] = v

person_age = st.number_input("Usia", value=st.session_state.get("person_age", 30), step=1)
person_income = st.number_input("Pendapatan", value=st.session_state.get("person_income", 50000.0))
person_home_ownership = st.selectbox("Status kepemilikan rumah", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'], index=['RENT', 'OWN', 'MORTGAGE', 'OTHER'].index(st.session_state.get("person_home_ownership", 'RENT')))
person_emp_exp = st.number_input("Lama pengalaman kerja (tahun)", value=st.session_state.get("person_emp_exp", 5), step=1)
loan_intent = st.selectbox("Tujuan pinjaman", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'], index=['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'].index(st.session_state.get("loan_intent", 'EDUCATION')))
loan_amnt = st.number_input("Jumlah pinjaman", value=st.session_state.get("loan_amnt", 10000.0))
loan_int_rate = st.number_input("Tingkat bunga pinjaman (%)", value=st.session_state.get("loan_int_rate", 10.0))
cb_person_cred_hist_length = st.number_input("Lama riwayat kredit (tahun)", value=st.session_state.get("cb_person_cred_hist_length", 5), step=1)
person_gender = st.selectbox("Jenis kelamin", ['male', 'female'], index=['male', 'female'].index(st.session_state.get("person_gender", 'male')))
person_education = st.selectbox("Pendidikan", ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'], index=['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'].index(st.session_state.get("person_education", 'Bachelor')))
default_val = st.session_state.get("previous_loan_defaults_on_file", 'No')
index = ['Yes', 'No'].index(default_val) if default_val in ['Yes', 'No'] else 1
previous_loan_defaults_on_file = st.selectbox("Ada gagal bayar sebelumnya?", ['Yes','No'], index=index)

credit_score = st.number_input("Skor kredit", value=st.session_state.get("credit_score", 650), step=1)

if st.button("Prediksi"):
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
        st.success(f"Prediksi Status Pinjaman: {status_map[prediction]}")
    else:
        st.error(f"Prediksi Status Pinjaman: {status_map[prediction]}")

    st.write("Probabilitas Prediksi:")
    prob_df = pd.DataFrame({
        "Status": [status_map[i] for i in range(len(prediction_proba))],
        "Probabilitas": [f"{p*100:.2f}%" for p in prediction_proba]
    })
    st.write(prob_df)
