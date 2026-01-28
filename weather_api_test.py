import requests
import pandas as pd
import joblib
import datetime

noise_model = joblib.load("models/noise_prediction_model.pkl")

API_KEY = "f7b7a4d2dc0c8dc3cdba0b389dd342f3"
CITY = "Chennai,IN"
