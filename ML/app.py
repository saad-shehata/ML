import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀")
st.title("🫀 Heart Disease Predictor")

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "heart.csv"))

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Preprocessing", "Model Training", "Prediction"])


if page == "Data Preprocessing":
    st.header("Data Preprocessing")

    st.subheader("Raw Data")
    st.dataframe(df)

    st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    st.write(f"Missing values: {df.isnull().sum().sum()}")

    X = df.drop('target', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    st.subheader("Data After Scaling (StandardScaler)")
    st.write("StandardScaler transforms each feature to have mean = 0 and std = 1")
    st.dataframe(scaled_df)

    st.subheader("Before vs After Scaling — Age")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Before:")
        st.write(df['age'].describe().round(2))
    with col2:
        st.write("After:")
        st.write(scaled_df['age'].describe().round(2))


elif page == "Model Training":
    st.header("Model Training")

    if 'n_trees' not in st.session_state:
        st.session_state.n_trees = 100
    if 'max_depth' not in st.session_state:
        st.session_state.max_depth = 5

    n_trees = st.slider("Number of Trees", min_value=10, max_value=300, value=st.session_state.n_trees, step=10, key='n_trees')
    max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=st.session_state.max_depth, key='max_depth')

    if st.button("Train Model"):
        st.session_state.n_trees = n_trees
        st.session_state.max_depth = max_depth

    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=st.session_state.n_trees, max_depth=st.session_state.max_depth, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    acc = accuracy_score(y_test, preds)

    st.write(f"Trees: {st.session_state.n_trees} | Max Depth: {st.session_state.max_depth}")
    st.write(f"Accuracy: {acc*100:.1f}%")

    report = classification_report(y_test, preds, output_dict=True)
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(report).transpose().style.format(precision=2))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    with col2:
        st.subheader("Feature Importance")
        feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
        fig, ax = plt.subplots()
        feat_imp.sort_values().plot(kind='barh', color='steelblue', ax=ax)
        st.pyplot(fig)


elif page == "Prediction":
    st.header("Prediction")

    @st.cache_data
    def get_model():
        X = df.drop('target', axis=1)
        y = df['target']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        return rf, scaler

    model, scaler = get_model()

    st.subheader("Enter Patient Info")

    col1, col2, col3 = st.columns(3)

    with col1:
        age      = st.slider("Age", 20, 80, 50)
        sex      = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
        cp       = st.selectbox("Chest Pain Type", [0,1,2,3])
        trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
        chol     = st.slider("Cholesterol", 100, 600, 200)

    with col2:
        fbs      = st.selectbox("Fasting Blood Sugar > 120", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        restecg  = st.selectbox("Resting ECG", [0,1,2])
        thalach  = st.slider("Max Heart Rate", 70, 210, 150)
        exang    = st.selectbox("Exercise Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

    with col3:
        oldpeak  = st.slider("ST Depression", 0.0, 6.0, 1.0)
        slope    = st.selectbox("ST Slope", [0,1,2])
        ca       = st.selectbox("Major Vessels", [0,1,2,3])
        thal     = st.selectbox("Thal", [1,2,3])

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
