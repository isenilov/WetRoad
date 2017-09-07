from feature_extraction import extract_features
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score
import sklearn.preprocessing
import time
from keras.models import Sequential
import keras.layers
from keras.callbacks import TensorBoard
from keras import optimizers


start = time.time()
print("\nExtracting features...")
features, labels = extract_features(flatten=False)
print(features.shape, labels.shape)
end = time.time()
print("Took %.3f sec." % (end - start))

start = time.time()
print("\nNormalizing features...")
#features = sklearn.preprocessing.scale(features)
end = time.time()
print("Took %.3f sec." % (end - start))

start = time.time()
print("\nSplitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
end = time.time()
print("Took %.3f sec." % (end - start))

start = time.time()
print("\nTraining model...")
tbCallback = TensorBoard()

model = Sequential()

model.add(keras.layers.Conv1D(256, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_size=5, activation='tanh'))
model.add(keras.layers.Conv1D(256, kernel_size=3, activation='tanh'))
model.add(keras.layers.MaxPooling1D(pool_size=3))
model.add(keras.layers.LSTM(256, return_sequences=True, activation='tanh'))
model.add(keras.layers.LSTM(256, return_sequences=True, activation='tanh'))
model.add(keras.layers.LSTM(256, activation='tanh'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# model.add(keras.layers.LSTM(16, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, activation="sigmoid"))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.LSTM(32, return_sequences=False, activation="sigmoid"))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(1, activation="sigmoid"))
#
# model.add(keras.layers.Dense(256, input_dim=X_train.shape[1], activation='relu'))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(256, activation='relu'))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(256, activation='relu'))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(1, activation="sigmoid"))

optimizer = optimizers.SGD(lr=1e-5)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(X_train, y_train,
          callbacks=[tbCallback],
          epochs=200,
          batch_size=512,
          verbose=1)

end = time.time()
print("\nTook %.3f sec." % (end - start))

start = time.time()
print("\nEvaluating...")
y_pred = model.predict_classes(X_test, verbose=1)
print(y_pred, y_test)
print("\nAccuracy: %s" % accuracy_score(y_test, y_pred))
print("Recall (wet): %s" % recall_score(y_test, y_pred))
end = time.time()
print("Took %.3f sec." % (end - start))
