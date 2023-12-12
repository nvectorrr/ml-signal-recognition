import numpy as np
from scipy.fft import ifft
from scipy.signal import chirp

def generate_ofdm_signal(N=64, Ncp=16):
    data = np.random.randint(0, 2, N)  # random data
    modData = np.exp(2j * np.pi * data / 4)  # 4-QAM mod
    ofdm_signal = ifft(modData, N)  # IFFT
    ofdm_signal = np.concatenate([ofdm_signal[-Ncp:], ofdm_signal])  # add cycle prefix
    return np.column_stack([np.real(ofdm_signal), np.imag(ofdm_signal)])

def generate_ofdm_signal_bpsk(N=64, Ncp=16):
    data = np.random.randint(0, 2, N)  # random data
    bpsk_modulated = 2*data - 1  # BPSK mod
    ofdm_signal = ifft(bpsk_modulated, N)  # IFFT
    ofdm_signal = np.concatenate([ofdm_signal[-Ncp:], ofdm_signal])  # add cycle prefix
    return np.column_stack([np.real(ofdm_signal), np.imag(ofdm_signal)])

def generate_am_signal(Fc=1000, Fs=10000, duration=1):
    t = np.arange(0, duration, 1 / Fs)  # time
    carrier = np.cos(2 * np.pi * Fc * t)  # carrier
    message = np.cos(2 * np.pi * 10 * t)  # info signal
    am_signal = (1 + message) * carrier  # AM mod
    return np.column_stack([am_signal, np.zeros_like(am_signal)])

def generate_fm_signal(Fc=1000, Fs=10000, duration=1, f0=100, f1=300):
    t = np.arange(0, duration, 1 / Fs)
    fm_signal = chirp(t, f0=f0, f1=f1, t1=duration, method='linear') * np.cos(2 * np.pi * Fc * t)
    return np.column_stack([fm_signal, np.zeros_like(fm_signal)])

def generate_qam_signal(Fc=1000, Fs=10000, duration=1, M=16):
    t = np.arange(0, duration, 1 / Fs)
    data = np.random.randint(0, M, int(Fs * duration))
    I = np.cos(2 * np.pi * 2 * np.arange(M) / M)
    Q = np.sin(2 * np.pi * 2 * np.arange(M) / M)
    modulated_I = I[data]
    modulated_Q = Q[data]
    qam_signal = modulated_I * np.cos(2 * np.pi * Fc * t) - modulated_Q * np.sin(2 * np.pi * Fc * t)
    return np.column_stack([qam_signal, np.zeros_like(qam_signal)])