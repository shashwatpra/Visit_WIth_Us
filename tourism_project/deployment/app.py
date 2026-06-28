"""
Streamlit app for Tourism Package Purchase Prediction.
1. Download the model from Hugging Face Model Hub.
2. Load the model using joblib.
3. Build Streamlit UI for user input.
4. Collect user input features.
5. Prepare input data as a DataFrame.
6. Predict on button click and display result.
"""
import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# --- Load Model ---
HF_TOKEN = os.environ.get('HF_TOKEN')
MODEL_REPO = os.environ.get('MODEL_REPO', 'your-hf-username/tourism-package-model')

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename='model.pkl',
        token=HF_TOKEN
    )
    return joblib.load(model_path)

model = load_model()

# --- Streamlit UI ---
st.title('Tourism Package Purchase Predictor')
st.markdown('Predict whether a customer will purchase the **Wellness Tourism Package**.')

with st.form('prediction_form'):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=18, max_value=100, value=35)
        city_tier = st.selectbox('City Tier', [1, 2, 3])
        duration_of_pitch = st.number_input('Duration of Pitch (mins)', min_value=0, max_value=120, value=20)
        occupation = st.selectbox('Occupation', ['Salaried', 'Self Employed', 'Free Lancer', 'Large Business', 'Small Business'])
        gender = st.selectbox('Gender', ['Male', 'Female'])
        number_of_person_visiting = st.number_input('Number of Persons Visiting', min_value=1, max_value=10, value=2)

    with col2:
        number_of_followups = st.number_input('Number of Follow-ups', min_value=0, max_value=10, value=3)
        product_pitched = st.selectbox('Product Pitched', ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'])
        preferred_property_star = st.selectbox('Preferred Property Star', [3, 4, 5])
        marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
        number_of_trips = st.number_input('Number of Trips/Year', min_value=0, max_value=20, value=3)
        passport = st.selectbox('Passport', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

    with col3:
        pitch_satisfaction_score = st.slider('Pitch Satisfaction Score', 1, 5, 3)
        own_car = st.selectbox('Own Car', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        number_of_children_visiting = st.number_input('Children Visiting (under 5)', min_value=0, max_value=5, value=0)
        designation = st.selectbox('Designation', ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
        monthly_income = st.number_input('Monthly Income', min_value=0, max_value=100000, value=25000)
        type_of_contact = st.selectbox('Type of Contact', ['Company Invited', 'Self Enquiry'])

    submitted = st.form_submit_button('Predict')

if submitted:
    input_data = pd.DataFrame([{
        'Age': age,
        'TypeofContact': type_of_contact,
        'CityTier': city_tier,
        'DurationOfPitch': duration_of_pitch,
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': number_of_person_visiting,
        'NumberOfFollowups': number_of_followups,
        'ProductPitched': product_pitched,
        'PreferredPropertyStar': preferred_property_star,
        'MaritalStatus': marital_status,
        'NumberOfTrips': number_of_trips,
        'Passport': passport,
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'OwnCar': own_car,
        'NumberOfChildrenVisiting': number_of_children_visiting,
        'Designation': designation,
        'MonthlyIncome': monthly_income
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f'✅ Customer is likely to **purchase** the package! (Probability: {probability:.1%})')
    else:
        st.warning(f'❌ Customer is **unlikely to purchase** the package. (Probability: {probability:.1%})')
