import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder
import streamlit as st

from Tensorflow.keras.models import load_model   # ✅ FIX

model = load_model('churn_model.h5')

#label the encode the scaled data
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender   = pickle.load(file)

with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)    


 ##Streamlit app
st.title("Rajesh Nayak - Customer Churn Prediction App")

st.write("Please enter the following details to predict if a customer will churn or not:")

# Create input fields for user to enter data
geography = st.selectbox("Geography", onehot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", min_value=0.0, value=10000.0)
creadit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)


input_data = pd.DataFrame({
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Balance': [balance],
    'CreditScore': [creadit_score],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


geography_encoded = onehot_encoder.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder.get_feature_names_out(['Geography'])) 

#combine the input data with the encoded geography data
input_data = pd.concat([input_data.reset_index(drop=True), geography_encoded_df], axis=1)

# 🔥 CRITICAL STEP: match training columns
input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

#scale the input data
input_data_scaled = scaler.transform(input_data)
#make prediction
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]
if st.button("Predict"):
    if prediction_probability > 0.5:
        st.write(f"⚠️ Customer is likely to LEAVE the bank ({prediction_probability*100:.1f}%)")
    else:
        st.write(f"✅ Customer is likely to STAY with the bank ({(1 - prediction_probability)*100:.1f}%)")
