import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained predictive model
model = joblib.load("predictive_model_rf.pkl")

# Define numerical columns for input
numerical_columns = [
    "open_issues_count",
    "past_due_actions_count",
    "open_risks_count",
    "open_ranked_risks_count",
    "active_changes_count",
]

# Mapping for GYR status
status_mapping = {0: "Green", 1: "Yellow", 2: "Red"}

# Dashboard Title
st.title("What-If Analysis Dashboard")
st.markdown("Use this dashboard to explore synthetic scenarios and predict GYR status.")

# Sidebar for user inputs
st.sidebar.header("Adjust Project Parameters")
inputs = {}
for col in numerical_columns:
    inputs[col] = st.sidebar.slider(
        f"{col.replace('_', ' ').capitalize()}",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
    )

# Convert inputs into a DataFrame for model prediction
input_data = pd.DataFrame([inputs])

# Prediction
if st.button("Predict GYR Status"):
    prediction = model.predict(input_data)
    predicted_status = status_mapping[prediction[0]]
    st.success(f"Predicted GYR Status: **{predicted_status}**")
    st.write("### Input Parameters:")
    st.write(input_data)

# Option to view the synthetic dataset
if st.checkbox("View Synthetic Dataset"):
    synthetic_data = pd.read_csv("synthetic_project_data.csv")
    st.write(synthetic_data)

# Footer
st.markdown("---")
st.markdown("**Project Workflow Step**: What-If Analysis on the Dashboard")

