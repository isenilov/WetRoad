from scipy.io import wavfile
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn import preprocessing

nfft = 64
window_length = 0.03

rate, frames = wavfile.read("dataset/wet/train_wet.wav")
window = round(window_length * rate)

i = 50000
f, t, Sxx, im = plt.specgram(frames[i:i + window - 1], Fs=rate, NFFT=nfft, noverlap=round(nfft/2))
# f, t, Sxx = signal.spectrogram(frames[i:i + window - 1], rate, nperseg=nfft, noverlap=round(nfft/2), scaling='spectrum')
# pxx, freqs, bins, im = plt.specgram(frames[i:i + window - 1], NFFT=nfft, Fs=rate, noverlap=nfft / 2)
S = preprocessing.scale(Sxx)

# plt.pcolor(t, f, S)
plt.xlabel("Time, s")
plt.ylabel("Frequency, Hz")
print("Number of frequencies:", len(f))
print("Number of time bins:", len(t))
plt.show()
