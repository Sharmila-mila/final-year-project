import pandas as pd
import joblib
import matplotlib.pyplot as plt
import math

# ===============================
# LOAD TRAINED AI MODEL
# ===============================
noise_model = joblib.load("models/noise_prediction_model.pkl")

# ===============================
# CONSTANT PARAMETERS
# ===============================
pressure = 1015
wind_bearing = 180
norad_density = 0.4

hours = []
noise_values = []
skr_values = []

# ===============================
# 24-HOUR SIMULATION
# ===============================
for hour in range(24):

    # 🌞🌙 Day / Night radiation
    if 6 <= hour <= 18:
        global_radiation = 5   # Daytime
    else:
        global_radiation = 1   # Nighttime

    # 🌡️ Temperature variation (diurnal cycle)
    temp_mean = 26 + 6 * math.sin((hour - 6) * math.pi / 12)
    temp_min = temp_mean - 2
    temp_max = temp_mean + 2

    # 🌬️ Wind speed variation
    wind_speed = 2 + abs(math.sin(hour / 3))

    # 🛰️ Satellite activity variation
    satellite_activity = 0.3 + 0.4 * abs(math.cos(hour / 4))

    # ===============================
    # PREPARE AI INPUT
    # ===============================
    sample = pd.DataFrame([[
        temp_mean,
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
    # CONTINUOUS ADAPTIVE SKR
    # ===============================
    skr = max(0.2, 1.2 - predicted_noise)

    # Store results
    hours.append(hour)
    noise_values.append(predicted_noise)
    skr_values.append(skr)

# ===============================
# PLOT RESULTS
# ===============================
plt.figure()
plt.plot(hours, skr_values, marker='o', label="Adjusted SKR")
plt.xlabel("Hour of Day")
plt.ylabel("Adjusted SKR")
plt.title("SKR vs Time (24-Hour AI-Assisted Simulation)")
plt.grid(True)
plt.legend()
plt.show()
