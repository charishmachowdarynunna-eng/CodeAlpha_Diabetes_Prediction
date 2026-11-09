Diabetes Prediction System

CodeAlpha Machine Learning Internship -- Task 4

1. Project Overview

Diabetes Prediction System is a learning project that uses selected attributes to estimate whether a record is classified as diabetic or non‑diabetic by a trained model.

The project has a Streamlit web application where users can type in patient details see model predictions look at saved records edit patient details and delete records.

Important: Diabetes Prediction System is for learning and demonstration. Diabetes Prediction System is not a device and must not be used to diagnose diabetes or make healthcare decisions. Use sample records for demonstrations; do not enter real patients personal or medical information.

2. Objectives

-. Prepare a diabetes dataset.

- Train and compare multiple machine‑learning classification algorithms.

-. Save the best‑performing model based on the evaluation used in the training script.

- Build a web interface for entering sample data and viewing predictions.

- Demonstrate basic patient‑record management: viewing, updating and deleting records.

3. Dataset

The training script downloads the diabetes dataset from:

https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv

- Pregnancies

- Glucose

- BloodPressure

- SkinThickness

- Insulin

- BMI

- DiabetesPedigreeFunction

- Age

The target column is Outcome (0 or 1).

The dataset is commonly known as the Pima Indians Diabetes dataset. The dataset represents a study population and has limitations; its results may not generalize to other populations. The models predictions should not be interpreted as conclusions.

4. Machine‑Learning Algorithms

The training script compares:

- Regression – a classification algorithm that estimates the probability of a class.

- Random Forest – an ensemble of decision trees whose predictions are combined.

- Support Vector Machine (SVM) – a model that finds a decision boundary between classes.

The script splits the data into training and testing sets applies preprocessing evaluates the models and saves the model selected by the script.

5. Data Preprocessing and Evaluation

The training script:

- Replaces zero values in selected columns (Glucose, BloodPressure, SkinThickness, Insulin and BMI) with missing values.

- Uses median imputation for missing values.

- Scales features for Logistic Regression and SVM.

- Splits the dataset into training and testing sets.

- Prints accuracy and a classification report for each model.

- Creates a model‑accuracy comparison chart and a confusion‑matrix image for the selected model.

The training script prints exact evaluation results when diabetes_prediction.py runs. Refer to those results of guessing accuracy.

6. Application Features

The Streamlit application (app.py) provides:

- Home: project introduction.

- Predict Diabetes: enter an ID, name and feature values to obtain a model prediction.

- Patient Records: view saved sample records.

- Update Patient Details: edit a selected record and recalculate its prediction.

- Delete Patient Record: select and remove a record with a confirmation checkbox.

Records are stored locally in patient_records.csv. This is a demonstration storage method, not a secure medical‑record system.

7. Project Files

Codealpha_Diabetes_Prediction/

├── diabetes_prediction.py

├── app.py

├── diabetes_model.pkl

├── diabetes_model_comparison.png

├── diabetes_confusion_matrix.png

├── patient_records.csv # created when records are saved

└── README.md

8. Requirements

- Python

- NumPy

- Pandas

- Matplotlib

- Scikit‑learn

- Joblib

- Streamlit

Install the required packages from the VS Code terminal:

python -m pip install numpy pandas matplotlib scikit-learn joblib streamlit

9. How to Run the Project

Step 1: Open the project folder

Codealpha_Diabetes_Prediction in VS Code.

Step 2: Train the model

Run:

python diabetes_prediction.py

This downloads the dataset, trains. Evaluates the models and creates the model and evaluation images. I hope you find this useful.

Step 3: Start the web application

Run:

python -m streamlit run app.py

Streamlit will show an address in the terminal. Open that address, in a browser to use the application. I hope you enjoy using the app.

Note: Run the training script before starting the app if

diabetes_model.pkl has not yet been created.

10. Limitations

- demonstration is a demonstration, not a clinically validated

prediction system.

- The dataset is limited and may not represent all populations.

- Model performance on a held-out test set does not guarantee

predictions for new people.

- Patient records are stored in a local CSV file and are not protected

like records in a secure healthcare system.

- Do not use real personal or medical information in demonstration

demonstration.

11. Conclusion

project demonstrates a machine-learning workflow: dataset

preparation, model training, model comparison, evaluation saving a

trained model and connecting it to a Streamlit application. Project also

demonstrates local record-management features, for educational

purposes.

