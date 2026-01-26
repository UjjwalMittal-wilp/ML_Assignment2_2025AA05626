from sklearn.naive_bayes import GaussianNB

def train_model(X_train, y_train) :
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return nb