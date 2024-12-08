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

# Latent Space Visualization
import matplotlib.pyplot as plt

if st.checkbox("Show Latent Space Visualization"):
    z_mean = encoder.predict(X_train)[0]  # Get latent space means
    tsne = TSNE(n_components=2)  # Reduce dimensions to 2D for visualization
    z_mean_2d = tsne.fit_transform(z_mean)
    
    plt.figure(figsize=(10, 8))
    for status, color in zip(["Green", "Yellow", "Red"], ["green", "yellow", "red"]):
        mask = synthetic_data['overall_gyr_status'] == status
        plt.scatter(
            z_mean_2d[mask, 0], z_mean_2d[mask, 1], 
            label=status, alpha=0.7, s=50, c=color
        )
    plt.title("Latent Space Visualization")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.legend()
    st.pyplot(plt)


if st.checkbox("Show Parameter vs. Status Graph"):
    import seaborn as sns

    selected_param = st.selectbox("Select a parameter:", numerical_columns)
    sns.boxplot(
        x="overall_gyr_status", 
        y=selected_param, 
        data=synthetic_data, 
        palette={"Green": "green", "Yellow": "yellow", "Red": "red"}
    )
    plt.title(f"Parameter '{selected_param}' Distribution Across GYR Status")
    plt.xlabel("GYR Status")
    plt.ylabel(selected_param.replace('_', ' ').capitalize())
    st.pyplot(plt)

if st.checkbox("Show Parameter Distribution"):
    selected_param = st.selectbox("Select a parameter to view distribution:", numerical_columns)
    sns.histplot(synthetic_data[selected_param], kde=True, color="blue", bins=20)
    plt.title(f"Distribution of '{selected_param}'")
    plt.xlabel(selected_param.replace('_', ' ').capitalize())
    plt.ylabel("Frequency")
    st.pyplot(plt)

if st.checkbox("Compare Multiple Scenarios"):
    st.markdown("### Define Multiple Scenarios")
    scenario_inputs = []
    for i in range(3):  # Allow up to 3 scenarios
        st.markdown(f"#### Scenario {i+1}")
        scenario = {}
        for col in numerical_columns:
            scenario[col] = st.slider(
                f"{col.replace('_', ' ').capitalize()} (Scenario {i+1})",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
            )
        scenario_inputs.append(pd.DataFrame([scenario]))

    # Predict for all scenarios
    predictions = [model.predict(scenario) for scenario in scenario_inputs]
    statuses = [status_mapping[pred[0]] for pred in predictions]

    # Plot comparison chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [f"Scenario {i+1}" for i in range(len(scenario_inputs))],
        [0 if s == "Green" else 1 if s == "Yellow" else 2 for s in statuses],
        color=["green" if s == "Green" else "yellow" if s == "Yellow" else "red" for s in statuses]
    )
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Green", "Yellow", "Red"])
    ax.set_title("Scenario Comparison")
    ax.set_xlabel("Scenarios")
    ax.set_ylabel("Predicted GYR Status")
    st.pyplot(fig)

