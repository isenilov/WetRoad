from cnn import def_model_cnn_blstm
from feature_extraction import get_last
from keras.utils import plot_model
from keras.models import model_from_yaml

model = get_last("", "model")
if model is not None:
    yaml_file = open(model, "r")
    loaded_model_file = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_file)
    plot_model(loaded_model, to_file='model.png', show_shapes=True, show_layer_names=False)
# model = def_model_cnn_blstm((1, 16384, 1))

