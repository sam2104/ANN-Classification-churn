# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# import pandas as pd
# import pickle

# # Load the trained model
# model = tf.keras.models.load_model('model.h5')

# # Load the encoders and scaler
# with open('label_encoder_gender.pkl', 'rb') as file:
#     label_encoder_gender = pickle.load(file)

# with open('onehot_encoder_geo.pkl', 'rb') as file:
#     onehot_encoder_geo = pickle.load(file)

# with open('scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)


# ## streamlit app
# st.title('Customer Churn PRediction')

# # User input
# geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
# gender = st.selectbox('Gender', label_encoder_gender.classes_)
# age = st.slider('Age', 18, 92)
# balance = st.number_input('Balance')
# credit_score = st.number_input('Credit Score')
# estimated_salary = st.number_input('Estimated Salary')
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
#     'EstimatedSalary': [estimated_salary]
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
# prediction_proba = prediction[0][0]

# st.write(f'Churn Probability: {prediction_proba:.2f}')

# if prediction_proba > 0.5:
#     st.write('The customer is likely to churn.')
# else:
#     st.write('The customer is not likely to churn.')
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Page config
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

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

# Load model and encoders
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.image("https://cdn-icons-png.flaticon.com/512/2921/2921822.png", width=100)


# --- TITLE
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Use this app to predict whether a customer will churn based on their profile.")

# --- SIDEBAR: Single Customer Input
st.sidebar.header("🔍 Predict Single Customer")
geography = st.sidebar.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('⚧ Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('🎂 Age', 18, 92)
balance = st.sidebar.number_input('💰 Balance', step=100.0)
credit_score = st.sidebar.number_input('💳 Credit Score', step=1.0)
estimated_salary = st.sidebar.number_input('📈 Estimated Salary', step=100.0)
tenure = st.sidebar.slider('📆 Tenure (Years)', 0, 10)
num_of_products = st.sidebar.slider('📦 Number of Products', 1, 4)
has_cr_card = st.sidebar.selectbox('💳 Has Credit Card?', [0, 1])
is_active_member = st.sidebar.selectbox('✅ Is Active Member?', [0, 1])

if st.sidebar.button("Predict Churn"):
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    proba = prediction[0][0]

    st.subheader("🔔 Prediction Result")
    st.metric("Churn Probability", f"{proba:.2f}")
    st.progress(int(proba * 100))

    if proba > 0.5:
        st.error("🔴 The customer is likely to churn.")
    else:
        st.success("🟢 The customer is not likely to churn.")

# --- BATCH UPLOAD
st.markdown("### 📁 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Drop irrelevant columns before preprocessing
        cols_to_drop = ['CustomerId', 'Exited', 'RowNumber', 'Surname']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        # Encode gender
        df['Gender'] = label_encoder_gender.transform(df['Gender'])

        # One-hot encode geography
        geo_encoded = onehot_encoder_geo.transform(df[['Geography']]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

        df_processed = pd.concat([df.drop(columns=['Geography']).reset_index(drop=True), geo_encoded_df], axis=1)
        df_scaled = scaler.transform(df_processed)

        # Predict
        predictions = model.predict(df_scaled)
        df['Churn Probability'] = predictions[:, 0]
        df['Prediction'] = (df['Churn Probability'] > 0.5).astype(int)

        st.success("✅ Batch predictions complete!")
        st.dataframe(df)

        counts = df['Prediction'].value_counts()
        st.markdown("### Batch Prediction Summary")
        st.write(counts)

        fig2, ax2 = plt.subplots()
        ax2.pie(counts, labels=['Not Churn', 'Churn'], autopct='%1.1f%%', colors=['#03dac6', '#b00020'])
        st.pyplot(fig2)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", csv, file_name="churn_batch_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# --- FOOTER
st.markdown("""
---
<center>
Made with ❤️ using Streamlit | Model: TensorFlow ANN<br>
By vemu samhita
</center>
""", unsafe_allow_html=True)
