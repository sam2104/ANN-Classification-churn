# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# import pandas as pd
# import pickle

# # Load the trained model
# model = tf.keras.models.load_model('regression_model.h5')

# # Load the encoders and scaler
# with open('label_encoder_gender.pkl', 'rb') as file:
#     label_encoder_gender = pickle.load(file)

# with open('onehot_encoder_geo.pkl', 'rb') as file:
#     onehot_encoder_geo = pickle.load(file)

# with open('scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)


# ## streamlit app
# st.title('Estimated Salary PRediction')

# # User input
# geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
# gender = st.selectbox('Gender', label_encoder_gender.classes_)
# age = st.slider('Age', 18, 92)
# balance = st.number_input('Balance')
# credit_score = st.number_input('Credit Score')
# exited = st.number_input('Exited')
# tenure = st.slider('Tenure', 0, 10)
# num_of_products = st.slider('Number of Products', 1, 4)
# has_cr_card = st.selectbox('Has Credit Card', [0, 1])
# is_active_member = st.selectbox('Is Active Member', [0, 1])

# # Prepare the input data
# input_data = pd.DataFrame({
#     'CreditScore': [credit_score],
#     'Gender': [label_encoder_gender.transform([gender])[0]],
#     'Age': [age],
#     'Tenure': [tenure],
#     'Balance': [balance],
#     'NumOfProducts': [num_of_products],
#     'HasCrCard': [has_cr_card],
#     'IsActiveMember': [is_active_member],
#     'Exited': [exited]
# })

# # One-hot encode 'Geography'
# geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
# geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# # Combine one-hot encoded columns with input data
# input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# # Scale the input data
# input_data_scaled = scaler.transform(input_data)


# # Predict churn
# prediction = model.predict(input_data_scaled)
# prediction_salary = prediction[0][0]

# st.write(f'Predicted Estimated Salary: {prediction_salary:.2f}')
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import matplotlib.pyplot as plt


st.set_page_config(page_title="Estimated Salary Predictor", layout="centered", page_icon="💼")

# ---- Header Image ----
# st.markdown("<h1 style='text-align: center; color: #00adb5;'>🧠 Estimated Salary Prediction</h1>", unsafe_allow_html=True)
st.markdown("""
    <style>
        
        h1 {
            color: #00adb5;
        }
        .stButton button {
            background-color: #00adb5;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)
# ---- Sidebar ----
st.sidebar.title("Input Customer Features")
st.sidebar.markdown("Customize input to predict salary")

# 🎯 Title 
st.markdown("<h1 style='text-align: center;'>💼 Estimated Salary Prediction</h1>", unsafe_allow_html=True)


# --- Dark Mode Toggle ---
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")

if dark_mode:
    st.markdown(
        """
        <style>
        /* Overall app background */
        .stApp {
            background-color: #121212;
            color: #E0E0E0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #BB86FC;
            font-weight: 700;
        }

        /* Sidebar background */
        .css-1d391kg {
            background-color: #1F1B24;
        }

        /* Text inputs, number inputs, sliders */
        input, .css-1v0mbdj, .css-1aumxhk, .st-b7 {
            background-color: #2C2C2E !important;
            color: #E0E0E0 !important;
            border: 1px solid #3A3A3C !important;
            border-radius: 6px !important;
        }

        /* Selectbox dropdown */
        div[role="combobox"] > div {
            background-color: #2C2C2E !important;
            color: #E0E0E0 !important;
        }

        /* Buttons */
        button, .st-b7 {
            background-color: #BB86FC !important;
            color: #121212 !important;
            font-weight: 600;
            border-radius: 8px !important;
            border: none !important;
            padding: 8px 18px !important;
            box-shadow: 0 4px 6px rgba(187, 134, 252, 0.4);
            transition: background-color 0.3s ease;
        }

        button:hover, .st-b7:hover {
            background-color: #9a67ea !important;
            box-shadow: 0 6px 8px rgba(154, 103, 234, 0.6);
            cursor: pointer;
        }

        /* Progress bar */
        .stProgress > div > div > div {
            background-color: #BB86FC !important;
        }

        /* Dataframe tables */
        .css-1q8dd3e.edgvbvh3 {
            background-color: #1E1E1E !important;
            color: #E0E0E0 !important;
        }

        /* Links */
        a {
            color: #BB86FC !important;
            text-decoration: underline;
        }

        /* Scrollbar (optional) */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: #BB86FC;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #1E1E1E;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: white;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# Load model
model = tf.keras.models.load_model('regression_model.h5')

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ---- Input Fields in Sidebar ----
geography = st.sidebar.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('⚧ Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('🎂 Age', 18, 92)
balance = st.sidebar.number_input('💰 Balance', min_value=0.0)
credit_score = st.sidebar.number_input('💳 Credit Score', min_value=0.0)
exited = st.sidebar.selectbox('Exited', [0, 1])
tenure = st.sidebar.slider('📆 Tenure', 0, 10)
num_of_products = st.sidebar.slider('📦 Number of Products', 1, 4)
has_cr_card = st.sidebar.selectbox('💳 Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('✅ Is Active Member', [0, 1])

# ---- Prepare Data ----
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)

# ---- Prediction ----
if st.button("🔍 Predict Salary"):
    prediction = model.predict(input_data_scaled)
    prediction_salary = prediction[0][0]
    st.success(f"💰 **Estimated Salary:** ₹ {prediction_salary:,.2f}")

st.markdown("---")
st.header("📥 Batch Prediction from CSV")

uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    df_batch = pd.read_csv(uploaded_file)

    try:
        # Drop unused columns
        columns_to_drop = ['CustomerId', 'RowNumber', 'Surname', 'EstimatedSalary']
        for col in columns_to_drop:
            if col in df_batch.columns:
                df_batch.drop(col, axis=1, inplace=True)

        # Encode Gender
        df_batch['Gender'] = label_encoder_gender.transform(df_batch['Gender'])

        # One-hot encode Geography
        geo_transformed = onehot_encoder_geo.transform(df_batch[['Geography']]).toarray()
        geo_cols = onehot_encoder_geo.get_feature_names_out(['Geography'])
        df_geo = pd.DataFrame(geo_transformed, columns=geo_cols)

        # Drop Geography and combine
        df_batch = df_batch.drop('Geography', axis=1).reset_index(drop=True)
        df_combined = pd.concat([df_batch, df_geo], axis=1)

        # Scale
        df_scaled = scaler.transform(df_combined)

        # Predict
        predictions = model.predict(df_scaled)
        df_batch['PredictedSalary'] = predictions.flatten()

        # Display
        st.subheader("📋 Predictions")
        st.dataframe(df_batch)

        # Optional Chart
        st.subheader("📊 Salary Distribution")
        fig_hist = px.histogram(df_batch, x='PredictedSalary', nbins=20, title="Distribution of Predicted Salaries")
        st.plotly_chart(fig_hist)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")


# Bar chart using Plotly
st.subheader("📊 Input Feature Summary")

features_bar = input_data.T.reset_index()
features_bar.columns = ['Feature', 'Value']
fig_bar = px.bar(features_bar, x='Feature', y='Value', color='Feature', title="Entered Features Overview")
st.plotly_chart(fig_bar, use_container_width=True)

# ---- Footer ----
st.markdown("""
---
<center>
Made with ❤️ using Streamlit | Model: TensorFlow ANN<br>
By vemu samhita
</center>
""", unsafe_allow_html=True)