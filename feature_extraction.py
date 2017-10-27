from scipy.io import wavfile
import numpy as np
from scipy import signal
from librosa import feature, effects
from sklearn.preprocessing import minmax_scale
import os.path
import pickle


def extract(wav_file, nfft=64, window_length=0.03, mel=True, flatten=True):
    rate, frames = wavfile.read(wav_file)
    window = round(window_length * rate)
    feat = []

    for i in range(0, len(frames)-window, int(window/2)):
        if mel:
            pxx = feature.mfcc(frames[i:i + window - 1],
                               sr=rate,
                               n_fft=nfft,
                               hop_length=round(nfft / 2),
                               fmax=8000)
        else:
            pxx = frames[i:i + window]
        if flatten:
            feat.append(pxx.flatten())
        else:
            feat.append(pxx)
            '''TODO: experiments with augmentation'''
            feat.append(effects.pitch_shift(pxx, rate, n_steps=4.0))
            feat.append(effects.pitch_shift(pxx, rate, n_steps=-4.0))
    return np.stack(feat)


def extract_features(file_wet, file_dry, mel=True, flatten=True, scaling=False, categorical=True):
    to_replace ="\\/"
    for char in to_replace:
        fw = file_wet.replace(char, "_")
        fd = file_dry.replace(char, "_")
    pickle_file = fw + "-" + fd + ".pkl"
    if os.path.exists(pickle_file):
        print("Using pickle file", pickle_file)
        with open(pickle_file, "rb") as f:
            features, labels = pickle.load(f)
        return features, labels
    features_wet = extract(file_wet, mel=mel, flatten=flatten)
    features_dry = extract(file_dry, mel=mel, flatten=flatten)
    labels_wet = np.ones(features_wet.shape[0])
    labels_dry = np.zeros(features_dry.shape[0])
    features = np.concatenate((features_wet, features_dry))
    labels = np.concatenate((labels_wet, labels_dry))
    if categorical:
        from keras.utils import to_categorical
        labels = to_categorical(labels, 2)
    if scaling and flatten:
        features = minmax_scale(features)
    with open(pickle_file, "wb") as f:
        pickle.dump((features, labels), f)
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



