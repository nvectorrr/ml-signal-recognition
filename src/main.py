import numpy as np
from src.lib.signals_generator import generate_ofdm_signal, generate_am_signal, generate_fm_signal, generate_qam_signal
from src.lib.ml import train_and_test_svm, train_and_test_random_forest, train_and_test_knn
from src.lib.plots import plot_ofdm_signal, plot_am_signal, plot_fm_signal, plot_qam_signal

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

# train and test
accuracy_svm = train_and_test_svm(X, y)
print(f'Точность SVM: {accuracy_svm}')

accuracy_rf = train_and_test_random_forest(X, y)
print(f'Точность Random Forest: {accuracy_rf}')

accuracy_knn = train_and_test_knn(X, y)
print(f'Точность KNN: {accuracy_knn}')
