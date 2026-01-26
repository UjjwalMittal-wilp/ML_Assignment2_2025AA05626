import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Import Models
from model.LogisticRegression import train_model as train_lr
from model.DecisionTreeClassifier import train_model as train_dt
from model.KNeighborsClassifier import train_model as train_knn
from model.GaussianNB import train_model as train_nb
from model.RandomForestClassifier import train_model as train_rf
from model.XGBClassifier import train_model as train_xgb

# Evaluation Metrics
def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

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

#Implementation
models = [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]

evaluation_result = pd.DataFrame()

for model_name in models:
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

    # Metrics
    print(model_name)
    print(evaluate(y_test, y_pred, y_prob))