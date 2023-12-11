from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_and_test_svm(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    accuracy = svm_model.score(X_test, y_test)
    return accuracy

def train_and_test_random_forest(X, y, test_size=0.3, n_estimators=100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    accuracy = rf_model.score(X_test, y_test)
    return accuracy

def train_and_test_knn(X, y, test_size=0.3, n_neighbors=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    accuracy = knn_model.score(X_test, y_test)
    return accuracy