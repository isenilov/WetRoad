from feature_extraction import extract_features, get_last
from sklearn.metrics import recall_score, accuracy_score
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
from time import time
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense, Flatten, LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, Callback
from keras import optimizers, regularizers
from keras.utils import to_categorical
from datetime import datetime
import os

dt = datetime.now().strftime("%d-%m-%Y.%H-%M")

class TestCallback(Callback):
    def __init__(self, test_data, number):
        self.test_data = test_data
        self.number = number

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        log_filename = "models/cnn/log." + dt + ".csv"
        with open(log_filename, "a") as log:
            log.write("{},{},{},{},{}\n".format(self.number, epoch, loss, acc, logs["acc"]))

def def_model_cnn_blstm(input_shape):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=64, strides=1, padding="same", activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Dropout(0.4)))
    # model.add(TimeDistributed(Conv1D(32, 32, activation='relu')))
    # model.add(TimeDistributed(MaxPooling1D(4)))
    # model.add(TimeDistributed(Dropout(0.3)))
    model.add(TimeDistributed(Conv1D(64, 64, padding="same", activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Dropout(0.4)))
    model.add(TimeDistributed(Conv1D(64, 64, padding="same", activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(2)))
    # model.add(TimeDistributed(Dropout(0.4)))
    # model.add(TimeDistributed(Conv1D(64, 16, activation='relu')))
    # model.add(TimeDistributed(MaxPooling1D(4)))
    # model.add(TimeDistributed(Dropout(0.4)))
    model.add(TimeDistributed(Conv1D(128, 64, padding="same", activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(2)))
    # model.add(TimeDistributed(Dropout(0.4)))
    model.add(TimeDistributed(Conv1D(128, 64, padding="same", activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(2)))
    # model.add(TimeDistributed(Dropout(0.4)))
    model.add(TimeDistributed(Conv1D(128, 64, padding="same", activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(2)))
    # model.add(TimeDistributed(Dropout(0.4)))
    model.add(TimeDistributed(Conv1D(256, 64, padding="same", activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(2)))
    # model.add(TimeDistributed(Dropout(0.4)))
    model.add(TimeDistributed(Conv1D(256, 64, padding="same", activation='relu')))
    model.add(TimeDistributed(GlobalAveragePooling1D()))

    # model.add(Conv1D(filters=8, kernel_size=64, strides=2, activation='relu', input_shape=input_shape))
    # model.add(Conv1D(64, 32, activation='relu'))
    # model.add(MaxPooling1D(4))
    # model.add(Dropout(0.5))
    # model.add(Conv1D(64, 32, activation='relu'))
    # model.add(MaxPooling1D(4))
    # model.add(Dropout(0.5))
    # model.add(Conv1D(128, 8, activation='relu'))
    # model.add(MaxPooling1D(4))
    # model.add(Dropout(0.5))
    # model.add(Conv1D(256, 2, activation='relu'))
    # model.add(MaxPooling1D(4))
    # model.add(Dropout(0.5))
    # model.add(Conv1D(256, 1, activation='relu'))
    # model.add(GlobalAveragePooling1D())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))

    # model.add(TimeDistributed(Dense(128, activation='relu')))
    # model.add(TimeDistributed(Dropout(0.5)))
    # model.add(Bidirectional(LSTM(256, return_sequences=True)))
    # model.add(TimeDistributed(Dropout(0.5)))
    # model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def def_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=8, strides=2, padding="same",
                     input_shape=input_shape, kernel_initializer='uniform',
                     activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Conv1D(32, 16, padding="same", activation='relu'))
    model.add(MaxPooling1D(4))
    model.add(Dropout(0.5))
    model.add(Conv1D(32, 16, padding="same", activation='relu'))
    model.add(MaxPooling1D(4))
    model.add(Dropout(0.5))
    model.add(Conv1D(32, 16, padding="same", activation='relu'))
    # model.add(MaxPooling1D(4))
    # model.add(Flatten())
    model.add(GlobalAveragePooling1D())
    # model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def train():
    X_train, X_1, X_2, y_train, y_1, y_2, = ex_feat()
    start = time()
    print("\nTraining model...")
    model = def_model_cnn_blstm(X_train.shape[1:])
    weights = get_last("models/cnn/", "weights")
    if weights is not None:
        model.load_weights(weights)
    print("Using weights:", weights)
    print("Dataset shape:", X_train.shape)

    # tbCallback = TensorBoard(histogram_freq=1, write_grads=True, write_graph=False)  # Tensorboard callback
    # esCallback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=5, verbose=1)  # early stopping callback
    mcCallback = ModelCheckpoint("models/cnn/weights.{epoch:02d}-{val_acc:.4f}.h5", monitor='val_acc', verbose=0,
                                 save_best_only=False, save_weights_only=True,
                                 mode='auto', period=1)  # saving weights every epoch
    testCallback0 = TestCallback((X_train, y_train), 3)
    testCallback1 = TestCallback((X_1, y_1), 1)
    testCallback2 = TestCallback((X_2, y_2), 2)
    # testCallback3 = TestCallback((X_3, y_3), 3)


    # dt = datetime.now().strftime("%d-%m-%Y.%H-%M")
    model_filename = "models/cnn/model." + dt + ".yaml"
    with open(model_filename, "w") as model_yaml:
        model_yaml.write(model.to_yaml())

    model.fit(X_train, y_train, validation_data=(X_1, y_1),
              batch_size=128, epochs=75, verbose=1,
              callbacks=[mcCallback, testCallback0, testCallback1, testCallback2]) #, esCallback])

    weights_filename = "models/cnn/" + dt + ".h5"
    model.save_weights(weights_filename)
    end = time()
    training_time = end - start
    print("\nTook %.3f sec." % training_time)
    # start = time()
    # print("\nEvaluating...")
    # y_pred = to_categorical(model.predict_classes(X_test, verbose=1))
    # print(y_pred, y_test)
    # acc = accuracy_score(y_test, y_pred)
    # print("\nAccuracy:", acc)
    # rec = recall_score(y_test, y_pred, average="macro")
    # print("Recall (wet):", rec)
    # end = time()
    # print("Took %.3f sec." % (end - start))
    # with open('results.txt', 'a') as f:
    #     f.write("CNN " + dt + " Input shape: " + str(X_train.shape) + " Accuracy: " + str(acc) +
    #             " Reacall: " + str(rec) + " Training time: " + str(training_time) + " s\n")
    #     model.summary(print_fn=lambda x: f.write(x + '\n'))


def ex_feat():
    start = time()
    print("\nExtracting features...")
    # X_1, y_1 = extract_features("dataset/wet1/audio_mono.wav",
    #                             "dataset/dry1/audio_mono.wav", flatten=False, scaling=False)
    # X_2, y_2 = extract_features("dataset/wet2/audio_mono.wav",
    #                             "dataset/dry2/audio_mono.wav", flatten=False, scaling=False)
    # X_3, y_3 = extract_features("dataset/wet3/audio_mono.wav",
    #                             "dataset/dry3/audio_mono.wav", flatten=False, scaling=False)
    #
    # X_train = np.concatenate((X_1, X_2, X_3))
    # y_train = np.concatenate((y_1, y_2, y_3))
    #
    # X_test, y_test = extract_features("dataset/wet/chevy_wet.wav",
    #                                   "dataset/dry/chevy_dry.wav", flatten=False, scaling=False)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # X_train, y_train = extract_features("dataset/wet/test_wet.wav", "dataset/dry/test_dry.wav",
    #                                     mel=False, flatten=False, scaling=True, categorical=True, augment=True)
    # X_test, y_test = extract_features("dataset/wet/test_wet.wav", "dataset/dry/test_dry.wav",
    #                                   mel=False, flatten=False, scaling=True, categorical=True)
    # X_val, y_val = extract_features("dataset/wet/test_wet.wav", "dataset/dry/test_dry.wav",
    #                                 mel=False, flatten=False, scaling=True, categorical=True)
    X_train, y_train = extract_features("dataset/wet/test_wet.wav", "dataset/dry/test_dry.wav",
                                        mel=False, flatten=False, scaling=True, categorical=True, augment=True, noise=True)
    X_1, y_1 = extract_features("dataset/wet1/audio_mono.wav", "dataset/dry1/audio_mono.wav",
                                      mel=False, flatten=False, scaling=True, categorical=True)
    X_2, y_2 = extract_features("dataset/wet2/audio_mono.wav", "dataset/dry2/audio_mono.wav",
                                    mel=False, flatten=False, scaling=True, categorical=True)
    # X_3, y_3 = extract_features("dataset/wet3/audio_mono.wav", "dataset/dry3/audio_mono.wav",
    #                            mel=False, flatten=False, scaling=True, categorical=True)

    X_train = np.expand_dims(X_train, axis=1)
    X_1 = np.expand_dims(X_1, axis=1)
    X_2 = np.expand_dims(X_2, axis=1)
    # X_3 = np.expand_dims(X_3, axis=1)

    X_train = X_train.reshape((X_train.shape[0], 1, int(X_train.shape[2])))
    X_1 = X_1.reshape((X_1.shape[0], 1, int(X_1.shape[2])))
    X_2 = X_2.reshape((X_2.shape[0], 1, int(X_2.shape[2])))
    # X_3 = X_3.reshape((X_3.shape[0], 1, int(X_3.shape[2])))

    X_train = np.expand_dims(X_train, axis=3)
    X_1 = np.expand_dims(X_1, axis=3)
    X_2 = np.expand_dims(X_2, axis=3)
    # X_3 = np.expand_dims(X_3, axis=3)

    end = time()
    print("Took %.3f sec." % (end - start))
    # start = time()
    # print("\nSplitting dataset...")
    # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # print(y_train, y_test)
    # end = time()
    # print("Took %.3f sec." % (end - start))

    return X_train, X_1, X_2, y_train, y_1, y_2


if __name__ == '__main__':
    np.random.seed(1)
    train()
'''
    try:
        train()

    except Exception as e:
        dt = datetime.now().strftime("%d-%m-%Y_%H-%M")
        with open(dt + ".log", "w") as f:
            f.write(str(e))
            print(str(e))
        # os.system("sudo poweroff")  # Shut down virtual machine in case of error

    else:
        pass
        # os.system("sudo poweroff")  # Shut down virtual machine (for training in the cloud)
'''