import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="harasar/churn-model", filename="best_churn_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI
st.title("🧳 Tourism Package Purchase Prediction App")
st.write("This app predicts whether a customer will purchase the tourism package (`ProdTaken`) based on their details.")

# Collect user inputs
Age = st.number_input("Age (in years)", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Female", "Male", "Fe Male"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, max_value=10, value=2)
ProductPitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Divorced", "Married", "Unmarried"])
NumberOfTrips = st.number_input("Number of Trips per year", min_value=0, max_value=20, value=1)
Passport = st.selectbox("Passport", ["Yes", "No"])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=20000)

# Convert inputs into dataframe
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])


# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
result = "will BUY the package 🎉" if prediction == 1 else "will NOT buy the package 🙁"
    st.write(f"Based on the provided details, the customer **{result}**.")
    st.write(f"Probability of buying: {prediction_proba:.2%}")
