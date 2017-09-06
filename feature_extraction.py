from scipy.io import wavfile
import numpy as np
from scipy import signal
from librosa import feature


def extract(wav_file, nfft=64, window_length=0.03, mel=False):
    rate, frames = wavfile.read(wav_file)
    window = round(window_length * rate)
    feat = []

    for i in range(0, len(frames)-window, window):
        if mel:
            pxx = feature.mfcc(frames[i:i + window - 1],
                               sr=rate,
                               n_fft=nfft,
                               hop_length=round(nfft / 2))
        else:
            _, _, pxx = signal.spectrogram(frames[i:i + window - 1],
                                           rate,
                                           nperseg=nfft,
                                           noverlap=round(nfft / 2))

        feat.append(pxx.flatten())

    return np.stack(feat)
