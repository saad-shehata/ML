import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart.csv")

# quick look at the data
print(df.head())
print(df.shape)

# disease count
sns.countplot(x='target', hue='target', data=df, palette='Set2', legend=False)
plt.title('Heart Disease Count')
plt.xticks([0,1], ['No Disease', 'Disease'])
plt.show()

# age histogram
sns.histplot(data=df, x='age', hue='target', kde=True, palette='Set2')
plt.title('Age Distribution')
plt.show()

# heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()

# cholesterol
sns.boxplot(x='target', y='chol', hue='target', data=df, palette='Set2', legend=False)
plt.title('Cholesterol')
plt.xticks([0,1], ['No Disease', 'Disease'])
plt.show()

# heart rate vs age
sns.scatterplot(x='age', y='thalach', hue='target', data=df, palette='Set2')
plt.title('Heart Rate vs Age')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(len(X_train), len(X_test))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

preds = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# confusion matrix
sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# which features mattered most
feat_imp = pd.Series(rf.feature_importances_, index=df.drop('target', axis=1).columns)
feat_imp.sort_values().plot(kind='barh', color='steelblue', figsize=(7,5))
plt.title('Feature Importance')
plt.show()
#  streamlit run C:\Users\saads\PycharmProjects\ML\app.py [ARGUMENTS]