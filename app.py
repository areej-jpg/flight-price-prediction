
import os
import joblib
import gdown
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="centered"
)

MODEL_URL = (
    "https://drive.google.com/uc?"
    "export=download&id=1g8LKSCay5IUS9HMsi3fxSZpEKd0qv5x4"
)
MODEL_PATH = "flight_model.joblib"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        result = gdown.download(
            MODEL_URL,
            MODEL_PATH,
            quiet=False
        )

        if result is None:
            raise RuntimeError(
                "Model download failed. Check Google Drive sharing permissions."
            )

    return joblib.load(MODEL_PATH)


st.title("✈️ Flight Price Prediction")
st.write(
    "Enter your flight details to estimate the ticket price."
)

try:
    with st.spinner("Loading the prediction model..."):
        model = load_model()
except Exception as e:
    st.error("Could not load the model. Please check the model file and sharing permissions.")
    st.exception(e)
    st.stop()


with st.form("flight_form"):

    airline = st.selectbox(
        "Airline",
        [
            "AirAsia",
            "Air_India",
            "GO_FIRST",
            "Indigo",
            "SpiceJet",
            "Vistara"
        ]
    )

    source_city = st.selectbox(
        "Departure City",
        [
            "Bangalore",
            "Chennai",
            "Delhi",
            "Hyderabad",
            "Kolkata",
            "Mumbai"
        ]
    )

    destination_city = st.selectbox(
        "Destination City",
        [
            "Bangalore",
            "Chennai",
            "Delhi",
            "Hyderabad",
            "Kolkata",
            "Mumbai"
        ]
    )

    departure_time = st.selectbox(
        "Departure Time",
        [
            "Early_Morning",
            "Morning",
            "Afternoon",
            "Evening",
            "Night",
            "Late_Night"
        ]
    )

    arrival_time = st.selectbox(
        "Arrival Time",
        [
            "Early_Morning",
            "Morning",
            "Afternoon",
            "Evening",
            "Night",
            "Late_Night"
        ]
    )

    stops = st.selectbox(
        "Number of Stops",
        ["zero", "one", "two_or_more"]
    )

    flight_class = st.selectbox(
        "Class",
        ["Economy", "Business"]
    )

    duration = st.number_input(
        "Flight Duration (hours)",
        min_value=0.5,
        max_value=50.0,
        value=2.0,
        step=0.5
    )

    days_left = st.number_input(
        "Days Until Departure",
        min_value=1,
        max_value=49,
        value=7,
        step=1
    )

    submitted = st.form_submit_button(
        "Predict Flight Price",
        use_container_width=True
    )


if submitted:

    if source_city == destination_city:
        st.warning(
            "Departure and destination cities must be different."
        )
    else:
        input_data = pd.DataFrame([{
            "airline": airline,
            "source_city": source_city,
            "departure_time": departure_time,
            "stops": stops,
            "arrival_time": arrival_time,
            "destination_city": destination_city,
            "class": flight_class,
            "duration": duration,
            "days_left": days_left
        }])

        try:
            prediction = model.predict(input_data)[0]

            st.success("Predicted Flight Price")
            st.subheader(f"₹ {prediction:,.2f}")

            st.caption(
                "This is an estimated price based on the trained model."
            )

        except Exception as e:
            st.error(
                "Prediction failed. The input columns or categories "
                "may not match the trained model."
            )
            st.exception(e)
