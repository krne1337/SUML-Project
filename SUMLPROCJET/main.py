import streamlit as st
import numpy as np
import pickle

# Load the trained RandomForest model
model = pickle.load(open('finalized_model.sav', 'rb'))

# Streamlit app title
st.title("Breast Cancer Prediction")

# Creating input fields for the features
st.write("Please enter the following features to get the breast cancer prediction")

radius_mean = st.number_input("Radius Mean", min_value=0.0, format="%.2f")
texture_mean = st.number_input("Texture Mean", min_value=0.0, format="%.2f")
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, format="%.2f")
area_mean = st.number_input("Area Mean", min_value=0.0, format="%.2f")
smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, format="%.2f")
compactness_mean = st.number_input("Compactness Mean", min_value=0.0, format="%.2f")
concavity_mean = st.number_input("Concavity Mean", min_value=0.0, format="%.2f")
concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, format="%.2f")
symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, format="%.2f")
radius_worst = st.number_input("Radius Worst", min_value=0.0, format="%.2f")
texture_worst = st.number_input("Texture Worst", min_value=0.0, format="%.2f")
perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, format="%.2f")
area_worst = st.number_input("Area Worst", min_value=0.0, format="%.2f")
smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, format="%.2f")
compactness_worst = st.number_input("Compactness Worst", min_value=0.0, format="%.2f")
concavity_worst = st.number_input("Concavity Worst", min_value=0.0, format="%.2f")
concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, format="%.2f")
symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, format="%.2f")
fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, format="%.2f")

# Prediction button
if st.button("Predict"):
    features = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                          compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                          radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
                          compactness_worst, concavity_worst, concave_points_worst, symmetry_worst,
                          fractal_dimension_worst]])
    prediction = model.predict(features)
    prediction_prob = model.predict_proba(features)

    st.write(f"Prediction: {'Malignant' if prediction[0] == 'M' else 'Benign'}")
    st.write(f"Prediction Probability: {prediction_prob[0][1] if prediction[0] == 'M' else prediction_prob[0][0]}")

# Run this app with `streamlit run app.py` in your command line
