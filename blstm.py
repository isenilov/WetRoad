from sklearn.utils import shuffle
from feature_extraction import extract_features, get_last
from sklearn.metrics import recall_score, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from time import time
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers.wrappers import Bidirectional
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras import optimizers, regularizers
from keras.utils import to_categorical
from datetime import datetime
import os


# Writing evaluation of the model on dataset to file
class TestCallback(Callback):
    def __init__(self, test_data, number):
        self.test_data = test_data
        self.number = number

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        log_filename = "models/log." + dt + ".csv"
        with open(log_filename, "w") as log:
            log.write("{},{},{},{}\n".format(self.number, epoch, loss, acc))


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
#                                     "dataset/dry/test_dry.wav", flatten=False, scaling=False)
# X_test, y_test = extract_features("dataset/wet/test_wet.wav",
#                                   "dataset/dry/test_dry.wav", flatten=False, scaling=False)
# X_val, y_val = extract_features("dataset/wet/test_wet.wav",
#                                 "dataset/dry/test_dry.wav", flatten=False, scaling=False)


X_train, y_train = extract_features("dataset/wet/chevy_wet.wav", "dataset/dry/chevy_dry.wav",
                                    mel=True, flatten=False, scaling=True, categorical=True)
X_1, y_1 = extract_features("dataset/wet1/audio_mono.wav", "dataset/dry1/audio_mono.wav",
                                  mel=True, flatten=False, scaling=True, categorical=True)
X_2, y_2 = extract_features("dataset/wet2/audio_mono.wav", "dataset/dry2/audio_mono.wav",
                                  mel=True, flatten=False, scaling=True, categorical=True)
X_3, y_3 = extract_features("dataset/wet3/audio_mono.wav", "dataset/dry3/audio_mono.wav",
                                  mel=True, flatten=False, scaling=True, categorical=True)

# X_train = np.expand_dims(X_train, axis=1)
# X_test = np.expand_dims(X_test, axis=1)
# X_val = np.expand_dims(X_val, axis=1)
end = time()
print("Took %.3f sec." % (end - start))

# start = time()
# print("\nSplitting dataset...")
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
# print(y_train, y_test)
# end = time()
# print("Took %.3f sec." % (end - start))

start = time()
dt = datetime.now().strftime("%d-%m-%Y.%H-%M")
print("\nTraining model...")
tbCallback = TensorBoard()
mcCallback = ModelCheckpoint("models/weights.{epoch:02d}-{val_acc:.4f}.h5", monitor='val_acc', verbose=0,
                                 save_best_only=False, save_weights_only=True,
                                 mode='auto', period=1)  # saving weights every epoch
testCallback0 = TestCallback((X_train, y_train), 0)
testCallback1 = TestCallback((X_1, y_1), 1)
testCallback2 = TestCallback((X_2, y_2), 2)
testCallback3 = TestCallback((X_3, y_3), 3)

# architecture of the network is adopted from https://arxiv.org/pdf/1511.07035.pdf
model = Sequential()
model.add(Bidirectional(LSTM(216, return_sequences=True, activation="tanh",
                         ),
                    input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Bidirectional(LSTM(216, return_sequences=True, activation="relu")))
model.add(Bidirectional(LSTM(216, activation="relu")))
model.add(Dense(2, activation='softmax'))
model.summary()
optimizer = optimizers.Adam(lr=1e-5)
model.compile(loss='categorical_crossentropy',
          optimizer=optimizer,
          metrics=['accuracy'])
weights = get_last("models/", "weights")
if weights is not None:
    model.load_weights(weights)
print("Using weights:", weights)
hist = model.fit(X_train, y_train,
                 callbacks=[tbCallback, mcCallback, testCallback0, testCallback1, testCallback2, testCallback3],
                 validation_data=(X_1, y_1),
                 epochs=2,
                 batch_size=128,
                 verbose=1)

# weights_filename = "models/weights " + dt + ".h5"
# model.save_weights(weights_filename)
model_filename = "models/model " + dt + ".yaml"
with open(model_filename, "w") as model_yaml:
    model_yaml.write(model.to_yaml())
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
# f.write("BLSTM " + dt + " Input shape: " + str(X_train.shape) + " Accuracy: " + str(acc) +
#         " Reacall: " + str(rec) + " Training time: " + str(training_time) + " s\n")
# model.summary(print_fn=lambda x: f.write(x + '\n'))

# except Exception as e:
#     dt = datetime.now().strftime("%d-%m-%Y_%H-%M")
#     with open(dt + ".log", "w") as f:
#         f.write(str(e))
#     os.system("sudo poweroff")  # Shut down virtual machine in case of error
#
# else:
#     pass
#     os.system("sudo poweroff")  # Shut down virtual machine (for training in the cloud)
