import streamlit as st
import requests
import pandas as pd
import joblib
import datetime
import math
import plotly.graph_objects as go
import plotly.express as px

from finite_size import finite_size_skr

# ===============================
# PAGE CONFIG & STYLING
# ===============================
st.set_page_config(
    page_title="Satellite QKD Simulator",
    layout="wide",
    page_icon="🛰️"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stMetric {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #41444D;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    h1, h2, h3 {
        color: #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛰️ AI-Assisted Satellite QKD Simulator")
st.markdown("### Real-time Weather Integration & Noise Prediction")
st.divider()

# ===============================
# LOAD AI MODEL
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("models/noise_prediction_model.pkl")

try:
    noise_model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ===============================
# SIDEBAR - CONTROLS
# ===============================
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    
    API_KEY = st.text_input("🔑 OpenWeather API Key", type="password")
    CITY = st.text_input("🌍 Target City", value="Chennai,IN")
    
    st.subheader("📊 Quantum Parameters")
    N = st.slider(
        "Pulse Count (Finite-size)",
        min_value=100_000,
        max_value=10_000_000,
        step=100_000,
        value=1_000_000,
        help="Number of quantum signals sent for key generation."
    )
    
    st.markdown("---")
    run_btn = st.button("🚀 Run Simulation", type="primary")

# ===============================
# MAIN APP LOGIC
# ===============================
if run_btn:
    if not API_KEY:
        st.sidebar.error("⚠️ Please enter a Weather API Key.")
        st.stop()
        
    with st.spinner("Fetching weather data & simulating quantum channel..."):
        # ===============================
        # WEATHER API CALL
        # ===============================
        url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
        except Exception as e:
            st.error(f"Connection Error: {e}")
            st.stop()

        if response.status_code != 200:
            st.error(f"Weather API Error: {data.get('message', 'Unknown error')}")
            st.stop()

        # Extract Weather Data
        temp = float(data["main"]["temp"])
        pressure = float(data["main"]["pressure"])
        wind_speed = float(data["wind"]["speed"])
        wind_bearing = float(data["wind"].get("deg", 0))
        clouds = float(data["clouds"]["all"])
        
        current_time = datetime.datetime.utcnow().timestamp()
        sunrise = data["sys"]["sunrise"]
        sunset = data["sys"]["sunset"]
        is_day = sunrise < current_time < sunset
        current_day_night = "Day" if is_day else "Night"

        # ===============================
        # METRICS DISPLAY
        # ===============================
        st.subheader(f"📍 Live Weather: {CITY.split(',')[0]}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Temperature", f"{temp:.1f} °C", delta=None)
        col2.metric("Pressure", f"{pressure} hPa")
        col3.metric("Wind Speed", f"{wind_speed} m/s")
        col4.metric("Cloud Cover", f"{clouds} %")
        
        # ===============================
        # AI PREDICTION
        # ===============================
        # Feature Engineering
        temp_min = temp - 2
        temp_max = temp + 2
        global_radiation = 5 if is_day else 1
        satellite_activity = 0.5
        norad_density = 0.4
        
        feature_cols = [
            "temp_mean(c)", "temp_min(c)", "temp_max(c)", "Pressure",
            "global_radiation", "Wind_Speed", "Wind_Bearing",
            "satellite_activity_index", "norad_density_index"
        ]
        
        sample = pd.DataFrame([[
            temp, temp_min, temp_max, pressure,
            global_radiation, wind_speed, wind_bearing,
            satellite_activity, norad_density
        ]], columns=feature_cols)

        noise_now = float(noise_model.predict(sample)[0])
        if pd.isna(noise_now) or noise_now < 0: noise_now = 1.0

        # Calculation
        asym_skr_now = max(0.2, 1.2 - noise_now)
        finite_skr_now = finite_size_skr(asym_skr_now, N)

        st.divider()
        st.subheader("🧠 AI Channel Analysis")
        
        m1, m2, m3 = st.columns(3)
        
        # Color coding for noise
        noise_color = "normal"
        if noise_now < 0.6: noise_color = "normal" # Streamlit metric doesn't natively support custom colors easily, but delta helps
        
        m1.metric("Predicted Hybrid Noise", f"{noise_now:.3f}", delta="-Low" if noise_now < 0.6 else "+High", delta_color="inverse")
        m2.metric("Asymptotic SKR (Ideal)", f"{asym_skr_now:.3f} bits/pulse")
        m3.metric(f"Finite-Size SKR (N={N/1e6:.1f}M)", f"{finite_skr_now:.3f} bits/pulse", delta="Real-World Estimate")

        # ===============================
        # DAY vs NIGHT COMPARISON
        # ===============================
        st.divider()
        st.subheader("🌞🌙 Day vs Night Analysis")
        
        comparison = []
        for label, rad in [("Day", 5), ("Night", 1)]:
            # Create sample for this specific condition
            s_dn = pd.DataFrame([[
                temp, temp_min, temp_max, pressure,
                rad, wind_speed, wind_bearing,
                satellite_activity, norad_density
            ]], columns=feature_cols)

            noise_dn = float(noise_model.predict(s_dn)[0])
            if pd.isna(noise_dn) or noise_dn < 0: noise_dn = 1.0

            asym_dn = max(0.2, 1.2 - noise_dn)
            fin_dn = finite_size_skr(asym_dn, N)

            comparison.append([label, f"{noise_dn:.3f}", f"{fin_dn:.3f} bits/pulse"])

        df_compare = pd.DataFrame(
            comparison,
            columns=["Condition", "Predicted Noise", "Finite-Size SKR"]
        )
        
        # Display as a clean dataframe/table
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        # ===============================
        # 24H SIMULATION PLOT
        # ===============================
        st.subheader("📈 24-Hour Forecast")
        
        today_date = datetime.date.today().strftime("%Y-%m-%d")
        hours_list = list(range(24))
        finite_skr_list = []
        noise_list = []
        
        # Simulation Loop
        for hour in hours_list:
            rad = 5 if 6 <= hour <= 18 else 1
            temp_h = 26 + 6 * math.sin((hour - 6) * math.pi / 12)
            wind_h = 2 + abs(math.sin(hour / 3))
            sat_h = 0.3 + 0.4 * abs(math.cos(hour / 4))
            
            s_hour = pd.DataFrame([[
                temp_h, temp_h-2, temp_h+2, pressure, rad, wind_h, wind_bearing, sat_h, norad_density
            ]], columns=feature_cols)
            
            n_val = float(noise_model.predict(s_hour)[0])
            n_val = 1.0 if (pd.isna(n_val) or n_val < 0) else n_val
            
            askr = max(0.2, 1.2 - n_val)
            fskr = finite_size_skr(askr, N)
            
            finite_skr_list.append(round(fskr, 3))
            noise_list.append(round(n_val, 3))

        # Plotly Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours_list, 
            y=finite_skr_list, 
            mode='lines+markers',
            name='Finite-Size SKR',
            line=dict(color='#00CC96', width=3)
        ))
        
        # Add background shading for Day/Night
        fig.add_vrect(x0=0, x1=6, fillcolor="black", opacity=0.1, layer="below", line_width=0, annotation_text="Night", annotation_position="top left")
        fig.add_vrect(x0=18, x1=23, fillcolor="black", opacity=0.1, layer="below", line_width=0, annotation_text="Night", annotation_position="top right")
        
        fig.update_layout(
            title="Projected Secret Key Rate over 24 Hours",
            xaxis_title="Hour of Day",
            yaxis_title="SKR (bits/pulse)",
            template="plotly_dark",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # DATA DOWNLOAD
        # ===============================
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_{timestamp}.csv"
        
        results_df = pd.DataFrame({
            "Hour": hours_list,
            "Predicted_Noise": noise_list,
            "Finite_Size_SKR": finite_skr_list
        })
        results_df["Date"] = today_date
        
        # Save locally - append mode usually better but here specific files
        results_df.to_csv(filename, index=False)
        
        with open(filename, "rb") as f:
            st.download_button(
                label="⬇️ Download Simulation Data",
                data=f,
                file_name=filename,
                mime="text/csv",
                key='download-csv'
            )
else:
    st.info("👈 Enter your API Key and details in the sidebar, then click 'Run Simulation' to start.")
