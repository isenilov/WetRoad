from feature_extraction import extract_features, get_last
from sklearn.metrics import recall_score, accuracy_score
from time import time
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers.wrappers import Bidirectional
from keras.callbacks import TensorBoard
from keras import optimizers
from keras.utils import to_categorical
from keras.models import model_from_yaml


start = time()
print("\nExtracting features...")

# X_test, y_test = extract_features("dataset/wet3/audio_mono.wav",
#                                   "dataset/dry3/audio_mono.wav", flatten=False, scaling=False)
X_test, y_test = extract_features("dataset/wet/test_wet.wav",
                                  "dataset/dry/test_dry.wav", flatten=False, scaling=False)

end = time()
print("Took %.3f sec." % (end - start))

start = time()
print("\nTraining model...")
tbCallback = TensorBoard()

# architecture of the network is adopted from https://arxiv.org/pdf/1511.07035.pdf
file = get_last("models/", "model")
if file is None:
    model = Sequential()
    model.add(Bidirectional(LSTM(216, return_sequences=True, activation="tanh"),
                            input_shape=(X_test.shape[1], X_test.shape[2])))
    model.add(Bidirectional(LSTM(216, return_sequences=True, activation="tanh")))
    model.add(Bidirectional(LSTM(216, activation="tanh")))
    model.add(Dense(2, activation='softmax'))
else:
    with open(file, "r") as model_file:
        model_yaml = model_file.read()
    model = model_from_yaml(model_yaml)

optimizer = optimizers.Adam(lr=1e-5)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
weights = get_last("models/", "weights")
model.load_weights(weights)
print("Using wights:", weights)
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
