from cnn import def_model_cnn_blstm
from feature_extraction import get_last
from keras.utils import plot_model
from keras.models import model_from_yaml, Sequential

from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D, LSTM, Dense
'''
model = get_last("", "model")
if model is not None:
    yaml_file = open(model, "r")
    loaded_model_file = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_file)
    plot_model(loaded_model, to_file='model.png', show_shapes=True, show_layer_names=False)
# model = def_model_cnn_blstm((1, 16384, 1))
'''
model = Sequential()
model.add(Bidirectional(LSTM(216, return_sequences=True, activation="tanh", dropout=0.5),
                        input_shape=(20, 40)))

model.add(Bidirectional(LSTM(216, return_sequences=True, activation="tanh", dropout=0.4)))
model.add(Bidirectional(LSTM(216, activation="tanh", dropout=0.3)))
model.add(Dense(3, activation='softmax'))
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)