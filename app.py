
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.datasets import load_breast_cancer
from collections import Counter

# --------------------------------------------------
# LOAD DATASET AND MODELS
# --------------------------------------------------

data = load_breast_cancer()

feature_names = list(data.feature_names)

lr_model = joblib.load("logistic_regression_pipeline.pkl")
dt_model = joblib.load("decision_tree_pipeline.pkl")
rf_model = joblib.load("random_forest_pipeline.pkl")

models = {
    "Logistic Regression": lr_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model
}

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🔬",
    layout="wide"
)

st.title("Breast Cancer Prediction System")

st.write(
    "Enter tumour measurements manually or upload a CSV file. "
    "All three machine learning models will analyse the same input."
)

st.warning(
    "This application is an educational machine learning project "
    "and should not be used as a medical diagnostic tool "
    "to make medical decisions."
)

# --------------------------------------------------
# GROUP THE 30 FEATURES
# --------------------------------------------------

mean_features = [
    f for f in feature_names if f.startswith("mean ")
]

error_features = [
    f for f in feature_names if f.endswith(" error")
]

worst_features = [
    f for f in feature_names if f.startswith("worst ")
]

feature_groups = {
    "Mean Measurements": mean_features,
    "Standard Error Measurements": error_features,
    "Worst Measurements": worst_features
}

# Use dataset medians as starting values for manual entry
feature_medians = dict(
    zip(feature_names, np.median(data.data, axis=0))
)

# --------------------------------------------------
# INPUT METHODS
# --------------------------------------------------

manual_tab, csv_tab = st.tabs([
    "Enter Measurements Manually",
    "Upload Measurements Using CSV"
])

input_data = None

# --------------------------------------------------
# OPTION 1: MANUAL INPUT
# --------------------------------------------------

with manual_tab:

    st.subheader("Enter Tumour Measurements")

    st.caption(
        "Enter all 30 measurements. The fields are divided "
        "into three sections to make data entry easier."
    )

    with st.form("manual_input_form"):

        manual_values = {}

        for group_name, features in feature_groups.items():

            with st.expander(group_name, expanded=False):

                columns = st.columns(2)

                for i, feature in enumerate(features):

                    with columns[i % 2]:

                        manual_values[feature] = st.number_input(
                            feature,
                            min_value=0.0,
                            value=float(feature_medians[feature]),
                            step=0.01,
                            format="%.4f",
                            key=f"manual_{feature}"
                        )

        manual_submit = st.form_submit_button(
            "Predict Using All Three Models",
            use_container_width=True
        )

    if manual_submit:

        input_data = pd.DataFrame(
            [manual_values],
            columns=feature_names
        )

# --------------------------------------------------
# OPTION 2: CSV UPLOAD
# --------------------------------------------------

with csv_tab:

    st.subheader("Upload Tumour Measurements")

    st.write(
        "Upload a CSV file containing one sample and all 30 "
        "required feature columns."
    )

    st.caption(
        "The column names must match the original dataset's "
        "feature names."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if st.button(
        "Predict Using Uploaded CSV",
        use_container_width=True
    ):

        if uploaded_file is None:

            st.error("Please upload a CSV file first.")

        else:

            try:

                csv_data = pd.read_csv(uploaded_file)

                missing_features = [
                    f for f in feature_names
                    if f not in csv_data.columns
                ]

                if missing_features:

                    st.error(
                        "The CSV is missing these features: "
                        + ", ".join(missing_features)
                    )

                elif len(csv_data) != 1:

                    st.error(
                        "Please upload a CSV containing exactly "
                        "one sample (one data row)."
                    )

                else:

                    sample_data = csv_data[feature_names].apply(
                        pd.to_numeric,
                        errors="coerce"
                    )

                    if sample_data.isna().any().any():

                        st.error(
                            "The CSV contains missing or "
                            "non-numeric measurements."
                        )

                    elif not np.isfinite(
                        sample_data.to_numpy()
                    ).all():

                        st.error(
                            "The CSV contains invalid "
                            "infinite values."
                        )

                    elif (sample_data < 0).any().any():

                        st.error(
                            "Measurements cannot be negative."
                        )

                    else:

                        input_data = sample_data

            except Exception as e:

                st.error(f"Could not read the CSV file: {e}")

# --------------------------------------------------
# RUN ALL THREE MODELS
# --------------------------------------------------

if input_data is not None:

    st.divider()

    st.header("Prediction Results")

    individual_predictions = {}
    individual_probabilities = {}

    for model_name, model in models.items():

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        classes = model.classes_

        probability_dict = dict(
            zip(classes, probabilities)
        )

        individual_predictions[model_name] = prediction

        individual_probabilities[model_name] = probability_dict

    # --------------------------------------------------
    # HARD VOTING
    # --------------------------------------------------

    votes = Counter(individual_predictions.values())

    hard_prediction = votes.most_common(1)[0][0]

    # --------------------------------------------------
    # SOFT VOTING
    # --------------------------------------------------

    class_names = ["Malignant", "Benign"]

    average_probabilities = {}

    for class_name in class_names:

        average_probabilities[class_name] = np.mean([
            individual_probabilities[model_name][class_name]
            for model_name in models
        ])

    soft_prediction = max(
        average_probabilities,
        key=average_probabilities.get
    )

    # --------------------------------------------------
    # CONSOLIDATED RESULTS
    # --------------------------------------------------

    st.subheader("Consolidated Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Majority Decision**")

        if hard_prediction == "Malignant":
            st.error(hard_prediction)
        else:
            st.success(hard_prediction)

        st.write(
            f"{votes[hard_prediction]} out of 3 models "
            "voted for this result."
        )

    with col2:

        st.markdown("**Average Probability**")

        if soft_prediction == "Malignant":
            st.error(soft_prediction)
        else:
            st.success(soft_prediction)

        st.write(
            f"Average predicted probability: "
            f"{average_probabilities[soft_prediction] * 100:.2f}%"
        )

    # --------------------------------------------------
    # MODEL AGREEMENT
    # --------------------------------------------------

    st.subheader("Model Agreement")

    if len(set(individual_predictions.values())) == 1:

        st.success(
            "All three models agree on the prediction."
        )

    else:

        st.warning(
            "The models disagree. Their individual predictions "
            "should be examined carefully."
        )

    # --------------------------------------------------
    # INDIVIDUAL MODEL PREDICTIONS
    # --------------------------------------------------

    st.subheader("Individual Model Results")

    results = []

    for model_name in models:

        prediction = individual_predictions[model_name]

        probabilities = individual_probabilities[model_name]

        results.append({
            "Model": model_name,
            "Prediction": prediction,
            "Malignant Probability (%)":
                round(probabilities["Malignant"] * 100, 2),
            "Benign Probability (%)":
                round(probabilities["Benign"] * 100, 2)
        })

    results_df = pd.DataFrame(results)

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # PROBABILITY COMPARISON
    # --------------------------------------------------

    st.subheader("Probability Comparison")

    probability_df = pd.DataFrame(
        individual_probabilities
    ).T

    probability_df = probability_df[
        ["Malignant", "Benign"]
    ] * 100

    probability_df.columns = [
        "Malignant (%)",
        "Benign (%)"
    ]

    st.bar_chart(probability_df)

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    st.divider()

    st.subheader("Summary")

    st.write(
        f"Majority Decision result: **{hard_prediction}**"
    )

    st.write(
        f"Average Probability result: **{soft_prediction}**"
    )

    st.write(
        f"Malignant average probability: "
        f"**{average_probabilities['Malignant'] * 100:.2f}%**"
    )

    st.write(
        f"Benign average probability: "
        f"**{average_probabilities['Benign'] * 100:.2f}%**"
    )

    st.caption(
        "The probabilities are model-estimated values and "
        "may not be calibrated. The voting results are "
        "experimental machine learning outputs, not medical advice."
    )
