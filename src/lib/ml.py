import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def evaluate_model(model, X, y, test_size=0.3, n_iterations=1000):
    accuracies = []
    all_y_test = []
    all_y_pred = []
    model.fit(X, y)  # train model only once
    for _ in range(n_iterations):
        _, X_test, _, y_test = train_test_split(X, y, test_size=test_size)
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
        all_y_test.append(y_test)
        all_y_pred.append(y_pred)
    return np.mean(accuracies), all_y_test, all_y_pred

def evaluate_svm(X, y, test_size=0.3, n_iterations=1000):
    svm_model = SVC()
    return evaluate_model(svm_model, X, y, test_size, n_iterations)

def evaluate_random_forest(X, y, test_size=0.3, n_estimators=100, n_iterations=1000):
    rf_model = RandomForestClassifier(n_estimators=n_estimators)
    return evaluate_model(rf_model, X, y, test_size, n_iterations)

def evaluate_knn(X, y, test_size=0.3, n_neighbors=5, n_iterations=1000):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    return evaluate_model(knn_model, X, y, test_size, n_iterations)
