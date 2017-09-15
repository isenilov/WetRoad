from feature_extraction import extract_features, get_last
from sklearn.metrics import recall_score, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from time import time
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers.wrappers import Bidirectional
from keras.callbacks import TensorBoard
from keras import optimizers, regularizers
from keras.utils import to_categorical
from datetime import datetime


start = time()
print("\nExtracting features...")
X_1, y_1 = extract_features("dataset/wet/chevy_wet.wav",
                            "dataset/dry/chevy_dry.wav", flatten=False, scaling=False)
X_2, y_2 = extract_features("dataset/wet1/audio_mono.wav",
                            "dataset/dry1/audio_mono.wav", flatten=False, scaling=False)
X = np.concatenate((X_1, X_2))
y = np.concatenate((y_1, y_2))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# X_train, y_train = extract_features("dataset/wet/test_wet.wav",
#                                     "dataset/dry/test_dry.wav", flatten=False, scaling=False)
# X_test, y_test = extract_features("dataset/wet/test_wet.wav",
#                                   "dataset/dry/test_dry.wav", flatten=False, scaling=False)
# X_val, y_val = extract_features("dataset/wet/test_wet.wav",
#                                 "dataset/dry/test_dry.wav", flatten=False, scaling=False)


# X_train, y_train = extract_features("dataset/wet1/audio_mono.wav",
#                                     "dataset/dry1/audio_mono.wav", flatten=False, scaling=False)
# X_test, y_test = extract_features("dataset/wet2/audio_mono.wav",
#                                   "dataset/dry2/audio_mono.wav", flatten=False, scaling=False)
# X_val, y_val = extract_features("dataset/wet3/audio_mono.wav",
#                                 "dataset/dry3/audio_mono.wav", flatten=False, scaling=False)
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

# architecture of the network is adopted from https://arxiv.org/pdf/1511.07035.pdf
model = Sequential()
model.add(Bidirectional(LSTM(216, return_sequences=True, activation="tanh",
                             ),
                        input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Bidirectional(LSTM(216, return_sequences=True, activation="tanh")))
model.add(Bidirectional(LSTM(216, activation="tanh")))
model.add(Dense(2, activation='softmax'))
optimizer = optimizers.Adam(lr=1e-5)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
weights = get_last("models/", "weights")
model.load_weights(weights)
print("Using weights:", weights)
model.fit(X_train, y_train,
#          validation_data=(X_val, y_val),
          callbacks=[tbCallback],
          epochs=25,
          batch_size=128,
          verbose=1)

dt = datetime.now().strftime("%d-%m-%Y %H:%M")
model.save_weights("models/models " + dt + ".h5")
with open("models/model " + dt + ".yaml", "w") as model_yaml:
    model_yaml.write(model.to_yaml())
end = time()
print("\nTook %.3f sec." % (end - start))

start = time()
print("\nEvaluating...")
y_pred = to_categorical(model.predict_classes(X_test, verbose=1))
print(y_pred, y_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Recall (wet):", recall_score(y_test, y_pred, average="macro"))
end = time()
print("Took %.3f sec." % (end - start))
