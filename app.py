import streamlit as st
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

#loading encdoers and scalers 
with open('onehot_encoder_geo.pkl','rb') as file:
    en_geo = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    en_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

#strimlit app
st.title("Costumer Churn Prediction")

#user input 
geography = st.selectbox('Geography',en_geo.categories_[0]) 
gender = st.selectbox('Gender',en_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('credit_score')
estimated_salary = st.number_input('estimated_salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Nmumber of products',1,4)
has_cr_card = st.selectbox('Has credit card',[0,1])
is_active_member = st.selectbox('Is Active member',[0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [en_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = en_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=en_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.') 