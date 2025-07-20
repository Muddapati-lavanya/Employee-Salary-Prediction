import streamlit as st
import joblib
import pandas as pd

# Load model and encoder
model = joblib.load("model_with_age_gender.joblib")
encoder = joblib.load("encoder_with_age_gender.joblib")

st.title("💼 Salary Prediction App")
st.markdown("Fill the details below to predict your expected salary.")

# Input fields
education = st.selectbox("Education", ["High School", "Diploma", "Associate's", "Bachelor's", "Master's", "PhD"])
job_title = st.selectbox("Job Title", [
    "Intern", "Developer", "Software Engineer", "Data Scientist", "Researcher",
    "Analyst", "Consultant", "UI/UX Designer", "ML Engineer", "IT Support",
    "AI Researcher", "DevOps Engineer"
])

# Allow user to type a location
location = st.text_input("Location (type your own location)", placeholder="e.g., Berlin, Nairobi, Dubai")

experience = st.selectbox("Experience", ["0 years", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "9 years", "10 years", "20+ years"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 65, 25)

# Predict button
if st.button("Predict Salary"):
    if location.strip() == "":
        st.error("❗ Please enter a location.")
    else:
        input_df = pd.DataFrame([{
            "education": education,
            "job_title": job_title,
            "location": location.strip(),
            "experience": experience,
            "gender": gender,
            "age": age
        }])

        # Transform and predict
        input_encoded = encoder.transform(input_df)
        predicted_salary = model.predict(input_encoded)[0]

        st.success(f"💰 Predicted Salary: ${int(predicted_salary):,}")
