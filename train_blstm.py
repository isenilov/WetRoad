from feature_extraction import extract_features
from sklearn.metrics import recall_score, accuracy_score
from time import time
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers.wrappers import Bidirectional
from keras.callbacks import TensorBoard
from keras import optimizers
from keras.utils import to_categorical


start = time()
print("\nExtracting features...")
X_train, y_train = extract_features("dataset/wet1/audio_mono.wav",
                                    "dataset/dry1/audio_mono.wav", flatten=False, scaling=True)
X_test, y_test = extract_features("dataset/wet2/audio_mono.wav",
                                  "dataset/dry2/audio_mono.wav", flatten=False, scaling=True)
X_val, y_val = extract_features("dataset/wet3/audio_mono.wav",
                                "dataset/dry3/audio_mono.wav", flatten=False, scaling=True)
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

model.add(Bidirectional(LSTM(216, return_sequences=True, activation="tanh"),
                        input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Bidirectional(LSTM(216, return_sequences=True, activation="tanh")))
model.add(Bidirectional(LSTM(216, activation="tanh")))
model.add(Dense(2, activation='softmax'))

optimizer = optimizers.Adam(lr=1e-5)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          callbacks=[tbCallback],
          epochs=50,
          batch_size=128,
          verbose=1)

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
