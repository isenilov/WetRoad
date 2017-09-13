from scipy.io import wavfile
import numpy as np
from scipy import signal
from librosa import feature
from sklearn.preprocessing import scale



def extract(wav_file, nfft=64, window_length=0.03, mel=True, flatten=True):
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


def extract_features(file_wet, file_dry, mel=True, flatten=True, scaling=False, categorical=True):
    from keras.utils import to_categorical
    features_wet = extract(file_wet, mel=mel, flatten=flatten)
    features_dry = extract(file_dry, mel=mel, flatten=flatten)
    labels_wet = np.ones(features_wet.shape[0])
    labels_dry = np.zeros(features_dry.shape[0])
    features = np.concatenate((features_wet, features_dry))
    labels = np.concatenate((labels_wet, labels_dry))
    if categorical:
        labels = to_categorical(labels, 2)
    if scaling and flatten:
        features = scale(features)
    return features, labels


def get_last(path, type):
    import glob
    if type == "weights":
        list = sorted(glob.glob(path + "*.h5"))
    if type == "model":
        list = sorted(glob.glob(path + "*.yaml"))
    if len(list) > 0:
        return max(list)
    return None


if __name__ == "__main__":
    import plotly

    X_train, y_train = extract_features("dataset/wet/test_wet.wav",
                                        "dataset/dry/test_dry.wav", flatten=False, scaling=False, categorical=False)

    plotly.offline.plot([dict(z=X_train[0], type='surface')], filename='feature_vector.html')



