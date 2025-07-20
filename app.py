import pandas as pd
import joblib

# Load the saved model and encoder
model = joblib.load("model.joblib")
encoder = joblib.load("encoder.joblib")

# Example user input (you can later take this from form/UI)
user_input = {
    "education": ["Master's"],
    "job_title": ["Data Scientist"],
    "location": ["San Francisco"],
    "experience": ["4 years"]
}

# Convert to DataFrame
input_df = pd.DataFrame(user_input)

# Encode input using the saved encoder
encoded_input = encoder.transform(input_df[["education", "job_title", "location", "experience"]])

# Predict
predicted_salary = model.predict(encoded_input)

print(f"💰 Predicted Salary: ${predicted_salary[0]:,.2f}")
