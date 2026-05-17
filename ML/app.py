import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀")
st.title("🫀 Heart Disease Predictor")

@st.cache_data
def train_model():
    data = pd.read_csv("heart.csv")

    X = data.drop('target', axis=1)
    y = data['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    acc = accuracy_score(y_test, rf.predict(X_test))
    return rf, scaler, acc

model, scaler, acc = train_model()

st.write(f"Model Accuracy: {acc*100:.1f}%")

st.sidebar.header("Patient Info")

age      = st.sidebar.slider("Age", 20, 80, 50)
sex      = st.sidebar.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
cp       = st.sidebar.selectbox("Chest Pain Type", [0,1,2,3])
trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
chol     = st.sidebar.slider("Cholesterol", 100, 600, 200)
fbs      = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
restecg  = st.sidebar.selectbox("Resting ECG", [0,1,2])
thalach  = st.sidebar.slider("Max Heart Rate", 70, 210, 150)
exang    = st.sidebar.selectbox("Exercise Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
oldpeak  = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
slope    = st.sidebar.selectbox("ST Slope", [0,1,2])
ca       = st.sidebar.selectbox("Major Vessels", [0,1,2,3])
thal     = st.sidebar.selectbox("Thal", [1,2,3])

inp = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                  restecg, thalach, exang, oldpeak, slope, ca, thal]],
                  columns=['age','sex','cp','trestbps','chol','fbs',
                           'restecg','thalach','exang','oldpeak','slope','ca','thal'])
inp_scaled = scaler.transform(inp)

if st.button("Predict"):
    result = model.predict(inp_scaled)[0]
    prob = model.predict_proba(inp_scaled)[0][1]

    if result == 1:
        st.error(f"⚠️ High risk — {prob*100:.1f}%")
    else:
        st.success(f"✅ Low risk — {prob*100:.1f}%")