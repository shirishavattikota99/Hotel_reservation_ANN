import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and preprocessor
model = load_model("model1.keras")
preprocessor = joblib.load("Preprocessor1.pkl")

st.set_page_config(page_title="Hotel Reservation Cancellation Prediction")

st.title("🏨 Hotel Reservation Cancellation Prediction")
st.write("Predict whether a hotel reservation is likely to be canceled.")

# ===========================
# User Inputs
# ===========================

no_of_adults = st.number_input("Number of Adults", min_value=0, value=2)
no_of_children = st.number_input("Number of Children", min_value=0, value=0)
no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=1)
no_of_week_nights = st.number_input("Week Nights", min_value=0, value=2)

type_of_meal_plan = st.selectbox(
    "Meal Plan",
    ["Meal Plan 1","Meal Plan 2","Meal Plan 3","Not Selected"]
)

required_car_parking_space = st.selectbox(
    "Car Parking Required",
    [0,1]
)

room_type_reserved = st.selectbox(
    "Room Type",
    [
        "Room_Type 1",
        "Room_Type 2",
        "Room_Type 3",
        "Room_Type 4",
        "Room_Type 5",
        "Room_Type 6",
        "Room_Type 7"
    ]
)

lead_time = st.number_input("Lead Time", min_value=0, value=30, step=50)

arrival_year = st.number_input("Arrival Year", value=2018)

arrival_month = st.selectbox(
    "Arrival Month",
    list(range(1,13))
)

market_segment_type = st.selectbox(
    "Market Segment",
    [
        "Online",
        "Offline",
        "Corporate",
        "Complementary",
        "Aviation"
    ]
)

repeated_guest = st.selectbox(
    "Repeated Guest",
    [0,1]
)

no_of_previous_cancellations = st.number_input(
    "Previous Cancellations",
    min_value=0,
    value=0
)

no_of_previous_bookings_not_canceled = st.number_input(
    "Previous Successful Bookings",
    min_value=0,
    value=0
)

avg_price_per_room = st.number_input(
    "Average Price Per Room",
    min_value=0,
    value=100,
    step=200
)

no_of_special_requests = st.number_input(
    "Special Requests",
    min_value=0,
    value=0
)

# ===========================
# Prediction
# ===========================

if st.button("🔍 Predict Booking Status"):

    input_df = pd.DataFrame({
        "no_of_adults":[no_of_adults],
        "no_of_children":[no_of_children],
        "no_of_weekend_nights":[no_of_weekend_nights],
        "no_of_week_nights":[no_of_week_nights],
        "type_of_meal_plan":[type_of_meal_plan],
        "required_car_parking_space":[required_car_parking_space],
        "room_type_reserved":[room_type_reserved],
        "lead_time":[lead_time],
        "arrival_year":[arrival_year],
        "arrival_month":[arrival_month],
        "market_segment_type":[market_segment_type],
        "repeated_guest":[repeated_guest],
        "no_of_previous_cancellations":[no_of_previous_cancellations],
        "no_of_previous_bookings_not_canceled":[no_of_previous_bookings_not_canceled],
        "avg_price_per_room":[avg_price_per_room],
        "no_of_special_requests":[no_of_special_requests]
    })

    transformed = preprocessor.transform(input_df)

    probability = model.predict(transformed)[0][0]

    confidence = probability if probability >= 0.5 else (1-probability)

    st.subheader("Prediction Result")

    st.progress(float(confidence))

    st.write(f"### Prediction Confidence : **{confidence*100:.2f}%**")

    if probability >= 0.5:

        st.snow()

        st.markdown("""
        <div style='background:#ffebee;
                    padding:25px;
                    border-radius:15px;
                    border-left:8px solid red'>
            <h2 style='color:red;'>❌ Booking is Likely to be Canceled</h2>
            <p>The customer has a higher probability of cancelling this reservation.</p>
        </div>
        """, unsafe_allow_html=True)

        st.warning("💡 Suggestion: Send reminder emails, provide discounts, or offer flexible cancellation policies.")

    else:

        st.balloons()

        st.markdown("""
        <div style='background:#e8f5e9;
                    padding:25px;
                    border-radius:15px;
                    border-left:8px solid green'>
            <h2 style='color:green;'>✅ Booking is Likely to be Confirmed</h2>
            <p>The customer is expected to complete the stay successfully.</p>
        </div>
        """, unsafe_allow_html=True)

        st.success("🎉 Great! This reservation has a low cancellation risk.")

    st.metric(
        label="Cancellation Probability",
        value=f"{probability*100:.2f}%"
    )