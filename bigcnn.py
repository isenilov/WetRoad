from cnn import def_model_cnn_blstm, TestCallback
from feature_extraction import extract_features, get_last
import numpy as np
from time import time
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from datetime import datetime
import soundfile as sf


dt = datetime.now().strftime("%d-%m-%Y.%H-%M")
N = 4096  # length of feature vector
B = 256  # batch size
S = 1000  # steps per epoch


def generator(w, d, batch_size=128):
    i = 0
    while 1:
        data = []
        labels = []
        wet = sf.blocks(w, blocksize=N, overlap=N//2, start=i)
        dry = sf.blocks(d, blocksize=N, overlap=N//2, start=i)
        for n in range(batch_size):
            data.append(next(wet))
            labels.append(1)
            data.append(next(dry))
            labels.append(0)
            i += N
        data = np.array(data)
        data = data[:, :, 0]
        data = np.expand_dims(data, axis=1)
        data = data.reshape((data.shape[0], 1, data.shape[2]))
        data = np.expand_dims(data, axis=3)
        # print("\n{:,}".format(i))
        yield data, np.array(to_categorical(labels))
        if i + batch_size * N > 1000000000:
            i = 0


def train():
    start = time()
    print("\nTraining model...")
    model = def_model_cnn_blstm((1, N, 1))
    weights = get_last("models/cnn/", "weights")
    if weights is not None:
        model.load_weights(weights)
    print("Using weights:", weights)

    X_1, y_1 = extract_features("dataset/wet1/audio_mono.wav", "dataset/dry1/audio_mono.wav",
                                mel=False, flatten=False, scaling=True, categorical=True)
    X_2, y_2 = extract_features("dataset/wet2/audio_mono.wav", "dataset/dry2/audio_mono.wav",
                                mel=False, flatten=False, scaling=True, categorical=True)
    X_3, y_3 = extract_features("dataset/wet3/audio_mono.wav", "dataset/dry3/audio_mono.wav",
                                mel=False, flatten=False, scaling=True, categorical=True)
    X_1 = np.expand_dims(X_1, axis=1)
    X_2 = np.expand_dims(X_2, axis=1)
    X_3 = np.expand_dims(X_3, axis=1)
    X_1 = X_1.reshape((X_1.shape[0], 1, int(X_1.shape[2])))
    X_2 = X_2.reshape((X_2.shape[0], 1, int(X_2.shape[2])))
    X_3 = X_3.reshape((X_3.shape[0], 1, int(X_3.shape[2])))
    X_1 = np.expand_dims(X_1, axis=3)
    X_2 = np.expand_dims(X_2, axis=3)
    X_3 = np.expand_dims(X_3, axis=3)

    mcCallback = ModelCheckpoint("models/cnn/weights.{epoch:02d}-{val_acc:.4f}.h5", monitor='val_acc', verbose=0,
                                 save_best_only=False, save_weights_only=True,
                                 mode='auto', period=1)  # saving weights every epoch
    testCallback1 = TestCallback((X_1, y_1), 1)
    testCallback2 = TestCallback((X_2, y_2), 2)
    testCallback3 = TestCallback((X_3, y_3), 3)

    # dt = datetime.now().strftime("%d-%m-%Y.%H-%M")
    model_filename = "models/cnn/model." + dt + ".yaml"
    with open(model_filename, "w") as model_yaml:
        model_yaml.write(model.to_yaml())

    model.fit_generator(generator("dataset/wet/yt_wet_10hrs.wav", "dataset/dry/yt_dry_8hrs.wav", batch_size=B),
                        steps_per_epoch=S, epochs=75, verbose=1,
                        callbacks=[mcCallback, testCallback1, testCallback2, testCallback3])

    weights_filename = "models/cnn/" + dt + ".h5"
    model.save_weights(weights_filename)
    end = time()
    training_time = end - start
    print("\nTook %.3f sec." % training_time)

if __name__ == '__main__':
    np.random.seed(1)
    train()
