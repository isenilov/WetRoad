from feature_extraction import extract_features, get_last
from sklearn.metrics import recall_score, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from time import time
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense
from keras.layers.wrappers import Bidirectional
from keras.callbacks import TensorBoard
from keras import optimizers, regularizers
from keras.utils import to_categorical
from datetime import datetime
import os

try:
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

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    X_val = np.expand_dims(X_val, axis=2)

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

    start = time()
    print("\nTraining model...")
    tbCallback = TensorBoard()

    model = Sequential()
    model.add(Conv1D(filters=1, kernel_size=3, strides=2,
                     input_shape=X_train.shape[1:], kernel_initializer='uniform',
                     activation='tanh', kernel_regularizer='l2', activity_regularizer='l2'))
    model.add(Conv1D(64, 3, activation='tanh', kernel_regularizer='l2', activity_regularizer='l2'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='tanh', kernel_regularizer='l2', activity_regularizer='l2'))
    model.add(Conv1D(256, 3, activation='tanh', kernel_regularizer='l2', activity_regularizer='l2'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'])

    weights = get_last("models/cnn/", "weights")
    if weights is not None:
        model.load_weights(weights)
    print("Using weights:", weights)
    print("Dataset shape:", X_train.shape)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=128, epochs=10, verbose=1)

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

except Exception as e:
    dt = datetime.now().strftime("%d-%m-%Y_%H-%M")
    with open(dt + ".log", "w") as f:
        f.write(str(e))
    os.system("sudo poweroff")  # Shut down virtual machine in case of error

else:
    os.system("sudo poweroff")  # Shut down virtual machine (for training in the cloud)
