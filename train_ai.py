import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("data/FINAL_COMBINED_REAL_APPROX_DATASET_2025.csv")

# Input features
X = df[
    [
        "Satellite_Altitude_km",
        "Ground_Satellite_Distance_km",
        "Elevation_Angle_deg",
        "Temperature_C",
        "Humidity_percent",
        "Visibility_km",
        "Cloud_Cover_percent",
        "Day_Night"
    ]
]

# Outputs
y_transmission = df["Transmission"]
y_noise = df["Hybrid_Noise"]

# Train-test split
X_train, X_test, y_trans_train, y_trans_test = train_test_split(
    X, y_transmission, test_size=0.2, random_state=42
)

_, _, y_noise_train, y_noise_test = train_test_split(
    X, y_noise, test_size=0.2, random_state=42
)

# Models
trans_model = RandomForestRegressor(n_estimators=200, random_state=42)
noise_model = RandomForestRegressor(n_estimators=200, random_state=42)

# Train
trans_model.fit(X_train, y_trans_train)
noise_model.fit(X_train, y_noise_train)

# Save models
joblib.dump(trans_model, "models/transmission_model.pkl")
joblib.dump(noise_model, "models/noise_model.pkl")

print("AI models trained and saved successfully!")
