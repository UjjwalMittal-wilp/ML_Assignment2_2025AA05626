from xgboost import XGBClassifier

def train_model(X_train, y_train):
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.001,
        max_depth=4,
        eval_metric='logloss',
        random_state=42
    )
    xgb.fit(X_train, y_train)
    return xgb