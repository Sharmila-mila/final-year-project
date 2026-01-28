import requests
import pandas as pd
import joblib
import datetime
import sys

# ===============================
# LOAD TRAINED AI MODEL
# ===============================
try:
    noise_model = joblib.load("models/noise_prediction_model.pkl")
except Exception as e:
    print("❌ Error loading AI model:", e)
    sys.exit()

# ===============================
# WEATHER API CONFIG
# ===============================
API_KEY = "f7b7a4d2dc0c8dc3cdba0b389dd342f3"   # 🔑 inside quotes
CITY = "Chennai,IN"

url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

# ===============================
# CALL WEATHER API
# ===============================
response = requests.get(url)
data = response.json()

if response.status_code != 200:
    print("❌ Weather API Error:", data)
    sys.exit()

# ===============================
# EXTRACT WEATHER DATA
# ===============================
temp = data["main"]["temp"]
pressure = data["main"]["pressure"]
wind_speed = data["wind"]["speed"]
wind_bearing = data["wind"].get("deg", 0)
clouds = data["clouds"]["all"]

# Day / Night (optional info)
current_time = datetime.datetime.utcnow().timestamp()
sunrise = data["sys"]["sunrise"]
sunset = data["sys"]["sunset"]
day_night = 1 if sunrise < current_time < sunset else 0

# ===============================
# APPROXIMATE MISSING FEATURES
# (because API doesn’t give all)
# ===============================
temp_min = temp - 2
temp_max = temp + 2

# simple radiation proxy
global_radiation = 5 if clouds < 50 else 2

# fixed satellite indicators (can be improved later)
satellite_activity = 0.5
norad_density = 0.4

# ===============================
# PREPARE AI INPUT
# ===============================
sample = pd.DataFrame([[
    temp,
    temp_min,
    temp_max,
    pressure,
    global_radiation,
    wind_speed,
    wind_bearing,
    satellite_activity,
    norad_density
]], columns=[
    "temp_mean(c)",
    "temp_min(c)",
    "temp_max(c)",
    "Pressure",
    "global_radiation",
    "Wind_Speed",
    "Wind_Bearing",
    "satellite_activity_index",
    "norad_density_index"
])

# ===============================
# AI NOISE PREDICTION
# ===============================
predicted_noise = noise_model.predict(sample)[0]

# ===============================
# OUTPUT
# ===============================
print("✅ Weather → AI Prediction Successful\n")
print("City:", CITY)
print("Temperature (°C):", temp)
print("Pressure (hPa):", pressure)
print("Wind Speed (m/s):", wind_speed)
print("Cloud Cover (%):", clouds)
print("Time:", "Day" if day_night else "Night")
print("\n🔮 Predicted Hybrid Noise:", round(predicted_noise, 3))
