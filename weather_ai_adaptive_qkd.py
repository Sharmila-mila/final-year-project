import requests
import pandas as pd
import joblib
import datetime
import sys

# ===============================
# LOAD AI MODEL
# ===============================
noise_model = joblib.load("models/noise_prediction_model.pkl")

# ===============================
# WEATHER API
# ===============================
API_KEY = "f7b7a4d2dc0c8dc3cdba0b389dd342f3"
CITY = "Chennai,IN"

url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
response = requests.get(url)
data = response.json()

if response.status_code != 200:
    print("❌ Weather API Error:", data)
    sys.exit()

# ===============================
# WEATHER DATA
# ===============================
temp = data["main"]["temp"]
pressure = data["main"]["pressure"]
wind_speed = data["wind"]["speed"]
wind_bearing = data["wind"].get("deg", 0)
clouds = data["clouds"]["all"]

# Day / Night
current_time = datetime.datetime.utcnow().timestamp()
sunrise = data["sys"]["sunrise"]
sunset = data["sys"]["sunset"]
day_night = "Day" if sunrise < current_time < sunset else "Night"

# ===============================
# APPROXIMATE FEATURES
# ===============================
temp_min = temp - 2
temp_max = temp + 2
global_radiation = 5 if clouds < 50 else 2
satellite_activity = 0.5
norad_density = 0.4

# ===============================
# AI INPUT
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
# ADAPTIVE QKD LOGIC
# ===============================
base_skr = 1.0  # normalized

if predicted_noise > 0.7:
    modulation = "LOW"
    final_skr = 0.6 * base_skr
elif predicted_noise >= 0.4:
    modulation = "MEDIUM"
    final_skr = base_skr
else:
    modulation = "HIGH"
    final_skr = 1.2 * base_skr

# ===============================
# OUTPUT
# ===============================
print("\n✅ WEATHER-AWARE ADAPTIVE QKD RESULT\n")
print("City:", CITY)
print("Time:", day_night)
print("Temperature (°C):", temp)
print("Predicted Hybrid Noise:", round(predicted_noise, 3))
print("Chosen Modulation:", modulation)
print("Adjusted SKR:", round(final_skr, 3))
