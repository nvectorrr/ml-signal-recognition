import numpy as np
from src.lib.signals_generator import generate_ofdm_signal, generate_am_signal, generate_fm_signal, generate_qam_signal
from src.lib.ml import evaluate_svm, evaluate_random_forest, evaluate_knn
from src.lib.plots import plot_ofdm_signal, plot_am_signal, plot_fm_signal, plot_qam_signal, plot_predictions

# generate and plot signals
ofdm_data = generate_ofdm_signal()
plot_ofdm_signal(ofdm_data)

am_data = generate_am_signal()
plot_am_signal(am_data)

fm_data = generate_fm_signal()
plot_fm_signal(fm_data)

qam_data = generate_qam_signal()
plot_qam_signal(qam_data)

# create X_train and Y_train
X = np.concatenate([ofdm_data, am_data, fm_data, qam_data])
y = np.array([1] * len(ofdm_data) + [2] * len(am_data) + [3] * len(fm_data) + [4] * len(qam_data))  # 1 - OFDM, 2 - AM, 3 - FM, 4 - QAM

# num of iterations
n_iterations = 1000

# train and test
accuracy_svm, y_test_svm, y_pred_svm = evaluate_svm(X, y, n_iterations=n_iterations)
plot_predictions(y_test_svm, y_pred_svm, "SVM Predictions")
print(f'SVM average accuracy: {accuracy_svm}')

accuracy_rf, y_test_rf, y_pred_rf = evaluate_random_forest(X, y, n_iterations=n_iterations)
plot_predictions(y_test_rf, y_pred_rf, "RFC Predictions")
print(f'RFC average accuracy: {accuracy_rf}')

accuracy_knn, y_test_knn, y_pred_knn = evaluate_knn(X, y, n_iterations=n_iterations)
plot_predictions(y_test_knn, y_pred_knn, "KNN Predictions")
print(f'KNN average accuracy: {accuracy_knn}')
