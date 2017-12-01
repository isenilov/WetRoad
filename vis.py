from cnn import def_model_cnn_blstm
from keras.utils import plot_model

model = def_model_cnn_blstm((1, 16384, 1))
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)
