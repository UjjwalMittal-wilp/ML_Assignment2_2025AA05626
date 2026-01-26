from sklearn.linear_model import LogisticRegression

def train_model(X_train_scaled, y_train) :
    lrModel = LogisticRegression(max_iter=1000)
    lrModel.fit(X_train_scaled, y_train)
    return lrModel