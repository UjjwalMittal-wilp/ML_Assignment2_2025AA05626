from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    return rf