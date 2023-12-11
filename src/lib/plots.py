import matplotlib.pyplot as plt
import numpy as np

def plot_signal(t, signal, title, xlabel, ylabel):
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_ofdm_signal(ofdm_data, Fs=10000):
    N = len(ofdm_data) // 2
    t = np.arange(0, N / Fs, 1 / Fs)
    plot_signal(t, ofdm_data[:N, 0], "OFDM Signal", "Time (s)", "Amplitude")

def plot_am_signal(am_data, Fs=10000):
    N = len(am_data) // 2
    t = np.arange(0, N / Fs, 1 / Fs)
    plot_signal(t, am_data[:N, 0], "AM Signal", "Time (s)", "Amplitude")

def plot_fm_signal(fm_data, Fs=10000):
    N = len(fm_data) // 2
    t = np.arange(0, N / Fs, 1 / Fs)
    plot_signal(t, fm_data[:N, 0], "FM Signal", "Time (s)", "Amplitude")

def plot_qam_signal(qam_data, Fs=10000):
    N = len(qam_data) // 2
    t = np.arange(0, N / Fs, 1 / Fs)
    plot_signal(t, qam_data[:N, 0], "QAM Signal", "Time (s)", "Amplitude")
