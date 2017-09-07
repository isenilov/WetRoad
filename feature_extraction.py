from scipy.io import wavfile
import numpy as np
from scipy import signal
from librosa import feature


def extract(wav_file, nfft=64, window_length=0.03, mel=False, flatten=True):
    rate, frames = wavfile.read(wav_file)
    window = round(window_length * rate)
    feat = []

    for i in range(0, len(frames)-window, window):
        if mel:
            pxx = feature.mfcc(frames[i:i + window - 1],
                               sr=rate,
                               n_fft=nfft,
                               hop_length=round(nfft / 2),
                               fmax=8000)
        else:
            _, _, pxx = signal.spectrogram(frames[i:i + window - 1],
                                           rate,
                                           nperseg=nfft,
                                           noverlap=round(nfft / 2))

        if flatten:
            feat.append(pxx.flatten())
        else:
            feat.append(pxx)
    return np.stack(feat)


def extract_features(mel=True, flatten=True):
    features_wet = extract("dataset/wet/train_wet.wav", mel=mel, flatten=flatten)
    labels_wet = np.ones(features_wet.shape[0])
    features_dry = extract("dataset/dry/train_dry.wav", mel=mel, flatten=flatten)
    labels_dry = np.zeros(features_dry.shape[0])
    return np.concatenate((features_wet, features_dry)), np.concatenate((labels_wet, labels_dry))
