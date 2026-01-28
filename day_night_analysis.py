import pandas as pd
import joblib

# ===============================
# LOAD AI MODEL
# ===============================
noise_model = joblib.load("models/noise_prediction_model.pkl")

# ===============================
# BASE WEATHER VALUES (FIXED)
# ===============================
temp = 28.0
pressure = 1015
wind_speed = 3.0
wind_bearing = 180
satellite_activity = 0.5
norad_density = 0.4

results = []

# ===============================
# DAY vs NIGHT CASES
# ===============================
cases = {
    "Day": {
        "clouds": 30,
        "global_radiation": 5
    },
    "Night": {
        "clouds": 30,
        "global_radiation": 1
    }
}

for time_period, params in cases.items():

    temp_min = temp - 2
    temp_max = temp + 2

    sample = pd.DataFrame([[
        temp,
        temp_min,
        temp_max,
        pressure,
        params["global_radiation"],
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

    predicted_noise = noise_model.predict(sample)[0]

    # Adaptive QKD logic
    if predicted_noise > 0.7:
        skr = 0.6
    elif predicted_noise >= 0.4:
        skr = 1.0
    else:
        skr = 1.2

    results.append([time_period, predicted_noise, skr])

# ===============================
# DISPLAY RESULTS
# ===============================
df = pd.DataFrame(results, columns=["Time", "Predicted Noise", "Adjusted SKR"])

print("\n🌞🌙 DAY vs NIGHT ANALYSIS RESULT\n")
print(df)
