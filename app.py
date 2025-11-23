# import streamlit as st
# import joblib
# import numpy as np

# model, FEATURE_COLUMNS = joblib.load("rf_model_4features.pkl")

# # Example manual input: [Rooms, Bedroom2, Bathroom, Car]
# features = np.array([[3, 3, 2, 2]])

# prediction = model.predict(features)
# print("Predicted price:", prediction[0])
import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    model, feature_columns = joblib.load("rf_model_4features.pkl")
    return model, feature_columns

model, FEATURE_COLUMNS = load_model()

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 House Price Prediction")
st.write("This app predicts the price of a house in based on its features." )

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    rooms = st.number_input("Number of Rooms", min_value=0, step=1, value=3)
    bedroom2 = st.number_input("Number of Bedrooms (Bedroom2)", min_value=0, step=1, value=3)

with col2:
    bathroom = st.number_input("Number of Bathrooms", min_value=0, step=1, value=2)
    car = st.number_input("Car Spaces", min_value=0, step=1, value=1)

# Build feature array in the SAME order as training
input_features = np.array([[rooms, bedroom2, bathroom, car]])

st.markdown("### Input Summary")
st.write(
    {
        "Rooms": rooms,
        "Bedroom2": bedroom2,
        "Bathroom": bathroom,
        "Car": car,
    }
)

st.markdown("---")

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔮 Predict Price"):
    # (Optional) shape safety check
    if input_features.shape[1] != model.n_features_in_:
        st.error(
            f"Model expects {model.n_features_in_} features, "
            f"but got {input_features.shape[1]}"
        )
    else:
        prediction = model.predict(input_features)[0]
        st.success(f"Estimated Price: **${prediction:,.2f}**")

        st.caption("Model: RandomForestRegressor (4 numeric features)")

# -----------------------------
# Extra: show model info
# -----------------------------
with st.expander("Show model details"):
    st.write("**Feature columns used for training:**")
    st.write(FEATURE_COLUMNS)
    st.write("`rf_model_4features.pkl` is loaded from disk using `joblib`.")

