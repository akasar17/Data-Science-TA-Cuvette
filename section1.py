# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

# Load dataset
df = pd.read_csv('/mnt/data/StudentsPerformance.csv')

# Display first few rows
print("🔹 First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\n🔹 Missing values:")
print(df.isnull().sum())

# Check for duplicates
print("\n🔹 Duplicates:", df.duplicated().sum())

# Create a 'pass/fail' column (average score ≥ 50 is a pass)
df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
df['pass_fail'] = df['average_score'].apply(lambda x: 1 if x >= 50 else 0)

# Drop the average_score column (not needed for model input)
df.drop('average_score', axis=1, inplace=True)

# Encode categorical variables
label_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# EDA: Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Distribution of Pass/Fail
sns.countplot(x='pass_fail', data=df)
plt.title("Pass/Fail Distribution")
plt.show()

# Split data
X = df.drop('pass_fail', axis=1)
y = df['pass_fail']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
log_preds = logreg.predict(X_test)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Evaluation function
def evaluate_model(name, y_true, y_pred):
    print(f"\n🔹 Evaluation for {name}:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# Evaluate both models
evaluate_model("Logistic Regression", y_test, log_preds)
evaluate_model("Random Forest", y_test, rf_preds)
