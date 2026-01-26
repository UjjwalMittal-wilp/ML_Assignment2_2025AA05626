from sklearn.neighbors import KNeighborsClassifier

def train_model(X_train_scaled, y_train):
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(X_train_scaled, y_train)
    return knn