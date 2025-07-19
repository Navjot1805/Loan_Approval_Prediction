import streamlit as st
import pandas as pd
import pickle as pk

# Load model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

# Title
st.header('Loan Prediction App')

# Input fields
np_of_dep = st.slider('Choose Number of Dependents', 0, 5)
grad = st.selectbox('Choose Education', ['Graduated', 'Not Graduated'])
self_emp = st.selectbox('Self Employed?', ['Yes', 'No'])
AnnualK = st.slider('Choose Annual Income', 0, 10000000)
Loan_Amount = st.slider('Choose Loan Amount', 0, 10000000)
Loan_Dur = st.slider('Choose Loan Duration (Years)', 0, 20)
Cibil = st.slider('Choose CIBIL Score', 0, 1000)
Assets = st.slider('Choose Total Assets', 0, 10000000)

# Encoding categorical features
grad_s = 0 if grad == 'Graduated' else 1
emp_s = 0 if self_emp == 'No' else 1

# Predict
if st.button('Predict'):
    # Create DataFrame with actual values
    input_data = pd.DataFrame([[np_of_dep, grad_s, emp_s, AnnualK, Loan_Amount, Loan_Dur, Cibil, Assets]],
        columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets']
    )

    # Scale input
    scaled_input = scaler.transform(input_data)

    # Predict
    result = model.predict(scaled_input)

    if result[0] == 1:
        st.success('✅ Loan will be Approved!')
    else:
        st.error('❌ Loan will NOT be Approved.')
