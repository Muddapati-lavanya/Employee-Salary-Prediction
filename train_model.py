import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Extended training data with more location options
data = pd.DataFrame([
    {"education": "High School", "job_title": "Intern", "location": "India", "experience": "0 years", "gender": "Female", "age": 21, "salary": 25000},
    {"education": "Diploma", "job_title": "Developer", "location": "London", "experience": "2 years", "gender": "Male", "age": 25, "salary": 40000},
    {"education": "Bachelor's", "job_title": "Software Engineer", "location": "New York", "experience": "5 years", "gender": "Male", "age": 30, "salary": 80000},
    {"education": "Master's", "job_title": "Data Scientist", "location": "San Francisco", "experience": "7 years", "gender": "Female", "age": 32, "salary": 120000},
    {"education": "PhD", "job_title": "Researcher", "location": "Boston", "experience": "10 years", "gender": "Female", "age": 38, "salary": 150000},
    {"education": "Associate's", "job_title": "Analyst", "location": "Remote", "experience": "3 years", "gender": "Male", "age": 28, "salary": 60000},
    {"education": "High School", "job_title": "Consultant", "location": "Austin", "experience": "20+ years", "gender": "Male", "age": 45, "salary": 95000},
    {"education": "Bachelor's", "job_title": "UI/UX Designer", "location": "Berlin", "experience": "4 years", "gender": "Female", "age": 29, "salary": 70000},
    {"education": "Master's", "job_title": "ML Engineer", "location": "Tokyo", "experience": "6 years", "gender": "Male", "age": 33, "salary": 110000},
    {"education": "Diploma", "job_title": "IT Support", "location": "Toronto", "experience": "3 years", "gender": "Male", "age": 27, "salary": 50000},
    {"education": "PhD", "job_title": "AI Researcher", "location": "Paris", "experience": "9 years", "gender": "Female", "age": 36, "salary": 140000},
    {"education": "Bachelor's", "job_title": "DevOps Engineer", "location": "Sydney", "experience": "5 years", "gender": "Male", "age": 31, "salary": 85000}
])

# Features and label
X = data[["education", "job_title", "location", "experience", "gender", "age"]]
y = data["salary"]

# Preprocessing
encoder = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["education", "job_title", "location", "experience", "gender"]),
    ("num", StandardScaler(), ["age"])
])

X_encoded = encoder.fit_transform(X)

# Train model
model = RandomForestRegressor()
model.fit(X_encoded, y)

# Save model and encoder
joblib.dump(model, "model_with_more_locations.joblib")
joblib.dump(encoder, "encoder_with_more_locations.joblib")

print("✅ Model and encoder saved with more location options.")
