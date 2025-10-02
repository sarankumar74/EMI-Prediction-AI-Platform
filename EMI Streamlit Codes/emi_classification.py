import streamlit as st
import pandas as pd
import pickle

with open("EMI Classification Model.pkl", "rb") as f:
    model = pickle.load(f)


categorical_cols = {
    'Gender': ['Male', 'Female', 'Other'],
    'Marital Status': ['Yes', 'No'],
    'Education': ['High School','Graduate', 'Post Graduate'],
    'Employment Type': ['Self Employed', 'Private', 'Government', 'Business'],
    'Company Type': ['Private', 'Government', 'Business'],
    'House Type': ['Owned', 'Rented'],
    'Emi Scenario': ['Yes', 'No'],
    'Existing Loans': ['Yes', 'No']
}

numerical_cols = [
    'Age', 'Monthly Salary', 'Years Of Employment',
    'Monthly Rent', 'Family Size', 'Other Monthly Expenses',
    'Credit Score', 'Current Emi Amount', 'Requested Amount',
    'Emergency Fund', 'Bank Balance'
]


st.set_page_config(
    page_title="EMI Eligibility Predictor",
    page_icon="🎉",
    layout="centered"
)

st.markdown("<h1 style='text-align: center; color: #4B0082;'> ⭐ EMI Eligibility Prediction</h1>", unsafe_allow_html=True)
st.write("Fill in the details below to predict **EMI Eligibility**:")


user_input = {}

st.subheader("Check Your are Eligible 💳.")
for col, options in categorical_cols.items():
    user_input[col] = st.selectbox(f"{col}", options)

st.subheader("Enter Detail")
for col in numerical_cols:
    user_input[col] = st.number_input(f"{col}", min_value=0, value=0)

input_df = pd.DataFrame([user_input])


if st.button("📈 Predict EMI Eligibility"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"✨ Predicted Classification: **{prediction}**")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
else:
    st.info("Enter details and click Predict to see the result.")


with st.expander("📄 View Input Data"):
    st.dataframe(input_df)
