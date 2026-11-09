# CODEALPHA INTERNSHIP - TASK 4
# DIABETES PREDICTION WEB APPLICATION

import streamlit as st
import pandas as pd
import joblib
import os

# Load trained model
model = joblib.load("diabetes_model.pkl")

# Patient records file
FILE_NAME = "patient_records.csv"

# Dataset feature names
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Diabetes Prediction System")
st.write("Machine Learning Based Diabetes Prediction")

st.info(
    "This application is an educational project. "
    "Its predictions are not medical diagnoses."
)

# Load patient records
def load_records():
    if os.path.exists(FILE_NAME):
        return pd.read_csv(FILE_NAME)

    return pd.DataFrame(columns=[
        "Patient ID",
        "Patient Name",
        "Age",
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Prediction"
    ])


# Save patient records
def save_records(df):
    df.to_csv(FILE_NAME, index=False)


# Navigation
menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Predict Diabetes",
        "Patient Records",
        "Update Patient Details",
        "Delete Patient Record"
    ]
)


# HOME PAGE
if menu == "Home":

    st.header("Welcome to Diabetes Prediction System")

    st.write("""
    This system uses Machine Learning algorithms to
    classify diabetes outcomes using medical measurements.

    Features:

    - Patient Registration
    - Diabetes Prediction
    - Patient Record Management
    - Update Patient Details
    - Prediction History
    """)


# PREDICTION PAGE
elif menu == "Predict Diabetes":

    st.header("Patient Registration and Diabetes Prediction")

    with st.form("patient_form"):

        st.subheader("Patient Information")

        patient_id = st.text_input("Patient ID")
        patient_name = st.text_input("Patient Name")

        st.subheader("Medical Information")

        col1, col2 = st.columns(2)

        with col1:

            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=20,
                value=0
            )

            glucose = st.number_input(
                "Glucose",
                min_value=0.0,
                max_value=300.0,
                value=120.0
            )

            blood_pressure = st.number_input(
                "Blood Pressure",
                min_value=0.0,
                max_value=200.0,
                value=70.0
            )

            skin_thickness = st.number_input(
                "Skin Thickness",
                min_value=0.0,
                max_value=100.0,
                value=20.0
            )

        with col2:

            insulin = st.number_input(
                "Insulin",
                min_value=0.0,
                max_value=900.0,
                value=80.0
            )

            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=100.0,
                value=25.0
            )

            diabetes_pedigree = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.5
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=30
            )

        submitted = st.form_submit_button(
            "Predict Diabetes"
        )

    if submitted:

        if not patient_id.strip() or not patient_name.strip():

            st.error("Please enter Patient ID and Patient Name.")

        else:

            input_data = pd.DataFrame([[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age
            ]], columns=FEATURES)

            prediction = model.predict(input_data)[0]

            probability = model.predict_proba(input_data)[0][1]

            if prediction == 1:
                result = "Diabetes Predicted"
            else:
                result = "No Diabetes Predicted"

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error(result)
            else:
                st.success(result)

            st.write(
                "Model-estimated probability for the diabetes class:",
                round(probability * 100, 2),
                "%"
            )

            # Save patient information
            records = load_records()

            new_record = {
                "Patient ID": patient_id.strip(),
                "Patient Name": patient_name.strip(),
                "Age": age,
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": diabetes_pedigree,
                "Prediction": result
            }

            # Update existing record if Patient ID exists
            records = records[
                records["Patient ID"].astype(str)
                != patient_id.strip()
            ]

            records = pd.concat(
                [records, pd.DataFrame([new_record])],
                ignore_index=True
            )

            save_records(records)

            st.success("Patient details saved successfully!")


# PATIENT RECORDS PAGE
elif menu == "Patient Records":

    st.header("Patient Records")

    records = load_records()

    if records.empty:

        st.warning("No patient records available.")

    else:

        st.dataframe(records, use_container_width=True)

        st.write("Total Patients:", len(records))


# UPDATE PATIENT DETAILS PAGE
elif menu == "Update Patient Details":

    st.header("Update Patient Information")

    records = load_records()

    if records.empty:

        st.warning("No patient records available.")

    else:

        patient_ids = records["Patient ID"].astype(str).tolist()

        selected_id = st.selectbox(
            "Select Patient ID",
            patient_ids
        )

        patient = records[
            records["Patient ID"].astype(str) == selected_id
        ].iloc[0]

        with st.form("update_form"):

            name = st.text_input(
                "Patient Name",
                value=str(patient["Patient Name"])
            )

            updated_age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=int(patient["Age"])
            )

            updated_glucose = st.number_input(
                "Glucose",
                min_value=0.0,
                max_value=300.0,
                value=float(patient["Glucose"])
            )

            updated_bp = st.number_input(
                "Blood Pressure",
                min_value=0.0,
                max_value=200.0,
                value=float(patient["BloodPressure"])
            )

            updated_skin = st.number_input(
                "Skin Thickness",
                min_value=0.0,
                max_value=100.0,
                value=float(patient["SkinThickness"])
            )

            updated_insulin = st.number_input(
                "Insulin",
                min_value=0.0,
                max_value=900.0,
                value=float(patient["Insulin"])
            )

            updated_bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=100.0,
                value=float(patient["BMI"])
            )

            updated_pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=20,
                value=int(patient["Pregnancies"])
            )

            updated_dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=float(patient["DiabetesPedigreeFunction"])
            )

            update_button = st.form_submit_button(
                "Update Patient Details"
            )

        if update_button:

            updated_data = pd.DataFrame([[
                updated_pregnancies,
                updated_glucose,
                updated_bp,
                updated_skin,
                updated_insulin,
                updated_bmi,
                updated_dpf,
                updated_age
            ]], columns=FEATURES)

            updated_prediction = model.predict(updated_data)[0]

            if updated_prediction == 1:
                updated_result = "Diabetes Predicted"
            else:
                updated_result = "No Diabetes Predicted"

            records.loc[
                records["Patient ID"].astype(str) == selected_id,
                "Patient Name"
            ] = name

            updated_values = {
                "Age": updated_age,
                "Pregnancies": updated_pregnancies,
                "Glucose": updated_glucose,
                "BloodPressure": updated_bp,
                "SkinThickness": updated_skin,
                "Insulin": updated_insulin,
                "BMI": updated_bmi,
                "DiabetesPedigreeFunction": updated_dpf,
                "Prediction": updated_result
            }

            for column, value in updated_values.items():

                records.loc[
                    records["Patient ID"].astype(str) == selected_id,
                    column
                ] = value

            save_records(records)

            st.success("Patient details updated successfully!")
            st.write("Updated Model Prediction:", updated_result)

# DELETE PATIENT RECORD PAGE
elif menu == "Delete Patient Record":

    st.header("Delete Patient Record")

    records = load_records()

    if records.empty:

        st.warning("No patient records available.")

    else:

        patient_ids = records["Patient ID"].astype(str).tolist()

        selected_id = st.selectbox(
            "Select Patient ID",
            patient_ids
        )

        patient = records[
            records["Patient ID"].astype(str) == selected_id
        ].iloc[0]

        st.write("Patient Name:", patient["Patient Name"])

        confirm_delete = st.checkbox(
            "I confirm that I want to delete this patient record"
        )

        if st.button("Delete Patient Record"):

            if confirm_delete:

                records = records[
                    records["Patient ID"].astype(str) != selected_id
                ]

                save_records(records)

                st.success("Patient record deleted successfully!")

                st.rerun()

            else:

                st.warning("Please confirm before deleting.")