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

import os
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "heart.csv"))

@st.cache_data
def train_model():
    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    acc = accuracy_score(y_test, rf.predict(X_test))
    return rf, scaler, X_train, X_test, y_train, y_test, acc

model, scaler, X_train, X_test, y_train, y_test, acc = train_model()

# section 1
st.header("1. Data Visualization")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.countplot(x='target', hue='target', data=df, palette='Set2', legend=False, ax=ax)
    ax.set_title('Heart Disease Count')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Disease', 'Disease'])
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='age', hue='target', kde=True, palette='Set2', ax=ax)
    ax.set_title('Age Distribution')
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots()
    sns.boxplot(x='target', y='chol', hue='target', data=df, palette='Set2', legend=False, ax=ax)
    ax.set_title('Cholesterol Levels')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Disease', 'Disease'])
    st.pyplot(fig)

with col4:
    fig, ax = plt.subplots()
    sns.scatterplot(x='age', y='thalach', hue='target', data=df, palette='Set2', ax=ax)
    ax.set_title('Heart Rate vs Age')
    st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)

# section 2
st.header("2. Model Training")

st.write(f"Model: Random Forest with 100 trees")
st.write(f"Train/Test split: 80/20")
st.write(f"Accuracy: {acc*100:.1f}%")

preds = model.predict(X_test)

report = classification_report(y_test, preds, output_dict=True)
st.subheader("Classification Report")
st.dataframe(pd.DataFrame(report).transpose().style.format(precision=2))

col5, col6 = st.columns(2)

with col5:
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

with col6:
    feat_imp = pd.Series(model.feature_importances_, index=df.drop('target', axis=1).columns)
    fig, ax = plt.subplots()
    feat_imp.sort_values().plot(kind='barh', color='steelblue', ax=ax)
    ax.set_title('Feature Importance')
    st.pyplot(fig)

# section 3
st.header("3. Prediction")

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
