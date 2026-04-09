import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# LOAD MODEL + PREPROCESSORS

model = load_model("ann_model.keras")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# PAGE CONFIG
st.set_page_config(
    page_title="Traffic AI System",
    page_icon="🚦",
    layout="centered"
)

# HEADER
st.title("🚦Traffic Prediction System")
st.divider()

# INPUT SECTION
st.subheader("📊 Enter Traffic Details")

col1, col2 = st.columns(2)

with col1:
    car = st.number_input("🚗 Car Count", min_value=0, value=10)
    bus = st.number_input("🚌 Bus Count", min_value=0, value=2)

with col2:
    bike = st.number_input("🏍 Bike Count", min_value=0, value=5)
    truck = st.number_input("🚚 Truck Count", min_value=0, value=3)

hour = st.slider("⏰ Hour of Day", 0, 23, 12)

day = st.selectbox(
    "📅 Day of Week",
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)

st.divider()

# FEATURE ENGINEERING (SAME AS TRAINING)

total = car + bike + bus + truck + 1e-6

car_ratio = car / total
bike_ratio = bike / total
bus_ratio = bus / total
truck_ratio = truck / total

time_sin = np.sin(2 * np.pi * hour / 24)
time_cos = np.cos(2 * np.pi * hour / 24)

days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
day_encoded = [1 if d == day else 0 for d in days]  # drop first if used drop_first

# final feature vector
features = np.array([[
    car, bike, bus, truck,
    car_ratio, bike_ratio, bus_ratio, truck_ratio,
    time_sin, time_cos,
    *day_encoded
]])

# PREDICTION

if st.button("🚦 Predict Traffic Situation"):

    # scale input
    features_scaled = scaler.transform(features)

    # predict
    prediction = model.predict(features_scaled)

    pred_class = np.argmax(prediction)
    result = le.inverse_transform([pred_class])[0].lower()


    # OUTPUT UI
    if result == "heavy":
      st.error("🔴 Heavy Traffic")
    elif result == "high":
      st.warning("🟠 High Traffic")
    elif result == "normal":
      st.info("🔵 Normal Traffic")
    else:
      st.success("🟢 Low Traffic")

st.divider()
