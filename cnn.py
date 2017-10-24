from feature_extraction import extract_features, get_last
from sklearn.metrics import recall_score, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from time import time
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense, Flatten, LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import optimizers, regularizers
from keras.utils import to_categorical
from datetime import datetime
import os


def def_model_cnn_blstm(input_shape):
    model = Sequential()
    # model.add(TimeDistributed(Dense(8), input_shape=input_shape))
    model.add(TimeDistributed(Conv1D(filters=8, kernel_size=64, strides=2, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(Conv1D(32, 32, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(4)))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Conv1D(64, 16, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(4)))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Conv1D(128, 8, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(4)))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Conv1D(256, 2, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(4)))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Conv1D(256, 1, activation='relu')))
    model.add(TimeDistributed(GlobalAveragePooling1D()))
    # model.add(Dropout(0.5))
    # model.add(TimeDistributed(Dense(128, activation='relu')))
    # model.add(TimeDistributed(Dropout(0.5)))
    # model.add(Bidirectional(LSTM(216, return_sequences=True, activation="relu")))
    # model.add(TimeDistributed(Dropout(0.5)))
    # model.add(Bidirectional(LSTM(216, return_sequences=True, activation="relu")))
    model.add(Bidirectional(LSTM(216, activation="relu")))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def def_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=8, strides=2,
                     input_shape=input_shape, kernel_initializer='uniform',
                     activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Conv1D(64, 32, activation='relu'))
    model.add(MaxPooling1D(4))
    model.add(Dropout(0.5))
    model.add(Conv1D(128, 16, activation='relu'))
    model.add(MaxPooling1D(4))
    model.add(Dropout(0.5))
    model.add(Conv1D(256, 8, activation='relu'))
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
    X_test, X_train, X_val, y_test, y_train, y_val = ex_feat()
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

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              batch_size=128, epochs=50, verbose=1,
              callbacks=[mcCallback]) #, esCallback])
    dt = datetime.now().strftime("%d-%m-%Y %H-%M")
    weights_filename = "models/cnn/" + dt + ".h5"
    model.save_weights(weights_filename)
    model_filename = "models/cnn/model " + dt + ".yaml"
    with open(model_filename, "w") as model_yaml:
        model_yaml.write(model.to_yaml())
    end = time()
    training_time = end - start
    print("\nTook %.3f sec." % training_time)
    start = time()
    print("\nEvaluating...")
    y_pred = to_categorical(model.predict_classes(X_test, verbose=1))
    print(y_pred, y_test)
    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", acc)
    rec = recall_score(y_test, y_pred, average="macro")
    print("Recall (wet):", rec)
    end = time()
    print("Took %.3f sec." % (end - start))
    with open('results.txt', 'a') as f:
        f.write("CNN " + dt + " Input shape: " + str(X_train.shape) + " Accuracy: " + str(acc) +
                " Reacall: " + str(rec) + " Training time: " + str(training_time) + " s\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))


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
    # X_train, y_train = extract_features("dataset/wet/test_wet.wav",
    #                                     "dataset/dry/test_dry.wav", mel=False, flatten=False, scaling=True, categorical=True)
    # X_test, y_test = extract_features("dataset/wet/test_wet.wav",
    #                                   "dataset/dry/test_dry.wav", mel=False, flatten=False, scaling=True, categorical=True)
    # X_val, y_val = extract_features("dataset/wet/test_wet.wav", "dataset/dry/test_dry.wav",
    #                                 mel=False, flatten=False, scaling=True, categorical=True)
    X_train, y_train = extract_features("dataset/wet1/audio_mono.wav", "dataset/dry1/audio_mono.wav",
                                        mel=False, flatten=False, scaling=True, categorical=True)
    X_test, y_test = extract_features("dataset/wet2/audio_mono.wav", "dataset/dry2/audio_mono.wav",
                                      mel=False, flatten=False, scaling=True, categorical=True)
    X_val, y_val = extract_features("dataset/wet3/audio_mono.wav", "dataset/dry3/audio_mono.wav",
                                    mel=False, flatten=False, scaling=True, categorical=True)

    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    X_val = np.expand_dims(X_val, axis=1)

    X_train = X_train.reshape((X_train.shape[0], 1, int(X_train.shape[2])))
    X_test = X_test.reshape((X_test.shape[0], 1, int(X_test.shape[2])))
    X_val = X_val.reshape((X_val.shape[0], 1, int(X_val.shape[2])))

    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    X_val = np.expand_dims(X_val, axis=3)
    # X_val, y_val = extract_features("dataset/wet/test_wet.wav",
    #                                 "dataset/dry/test_dry.wav", mel=False, flatten=False, scaling=True)
    end = time()
    print("Took %.3f sec." % (end - start))
    # start = time()
    # print("\nSplitting dataset...")
    # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # print(y_train, y_test)
    # end = time()
    # print("Took %.3f sec." % (end - start))
    return X_test, X_train, X_val, y_test, y_train, y_val


if __name__ == '__main__':
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