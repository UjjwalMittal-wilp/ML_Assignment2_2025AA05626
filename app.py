import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Import Models
from model.LogisticRegression import train_model as train_lr
from model.DecisionTreeClassifier import train_model as train_dt
from model.KNeighborsClassifier import train_model as train_knn
from model.GaussianNB import train_model as train_nb
from model.RandomForestClassifier import train_model as train_rf
from model.XGBClassifier import train_model as train_xgb

st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide"
)

st.title("Breast Cancer Classification System")
st.markdown("### Machine Learning Model Comparison Dashboard")

st.sidebar.header("Settings")

model_name = st.sidebar.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File (Optional)",
    type=["csv"]
)

csv_url = "https://github.com/UjjwalMittal-wilp/ML_Assignment2_2025AA05626/blob/main/Test%20Data.csv"

st.sidebar.markdown(f"[Test CSV File]({csv_url})")

# Load DataSet
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target) # 0 = Malignant, 1 = Benign
    return X, y

X, y = load_data()

# Test data received
X["target"] = y
corr = X.corr()["target"].sort_values(ascending=False)
# print(corr)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Dictionary
model_trainers = {
    "Logistic Regression": train_lr,
    "Decision Tree": train_dt,
    "K-Nearest Neighbors": train_knn,
    "Naive Bayes": train_nb,
    "Random Forest": train_rf,
    "XGBoost": train_xgb
}

# Conditional Scale
if model_name in ["Logistic Regression", "K-Nearest Neighbors"]:
    X_train_used = X_train_scaled
    X_test_used = X_test_scaled
else:
    X_train_used = X_train
    X_test_used = X_test

# Train
model = model_trainers[model_name](X_train_used, y_train)

# Predictions
y_pred = model.predict(X_test_used)
y_prob = model.predict_proba(X_test_used)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

# Display Metrics
st.subheader(f"Evaluation Metrics — {model_name}")

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{accuracy:.4f}")
c2.metric("AUC Score", f"{auc:.4f}")
c3.metric("Precision", f"{precision:.4f}")

c4, c5, c6 = st.columns(3)
c4.metric("Recall", f"{recall:.4f}")
c5.metric("F1 Score", f"{f1:.4f}")
c6.metric("MCC", f"{mcc:.4f}")


# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"]
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Classification Report
st.subheader("📄 Classification Report")

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# Prediction on Uploaded CSV

if uploaded_file is not None:
    st.subheader("📂 Uploaded File Predictions")

    input_df = pd.read_csv(uploaded_file)
    st.write("Preview:", input_df.head())

    if model_name in ["Logistic Regression", "K-Nearest Neighbors"]:
        input_scaled = scaler.transform(input_df)
        preds = model.predict(input_scaled)
    else:
        preds = model.predict(input_df)

    input_df["Prediction"] = np.where(
        preds == 1,
        "Malignant",
        "Benign"
    )

    st.write("Results:", input_df)