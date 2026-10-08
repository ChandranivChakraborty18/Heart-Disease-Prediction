Heart Health AI

A simple Streamlit application using a machine-learning model trained on structured heart-disease CSV data.

Structure

heart_health_ai/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    └── heart_risk_model.pkl

Model

Put the trained heart_risk_model.pkl file inside the models folder.

The dataset itself is not needed when running the app.

Run

pip install -r requirements.txt
streamlit run app.py

Input Features

age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.

Important

This project is for educational purposes. The output is an estimated machine-learning risk prediction and is not a medical diagnosis.