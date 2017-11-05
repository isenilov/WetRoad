from keras.models import model_from_yaml
import numpy as np
from feature_extraction import extract_features, get_last

model = get_last("", "model")
if model is not None:
    yaml_file = open(model, "r")
    loaded_model_file = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_file)
    loaded_model.load_weights(get_last("", "weights"))
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_test, y_test = extract_features("dataset/wet2/audio_mono.wav", "dataset/dry2/audio_mono.wav",
                                      mel=False, flatten=False, scaling=True, categorical=True)
    X_test = np.expand_dims(X_test, axis=1)
    X_test = X_test.reshape((X_test.shape[0], 1, int(X_test.shape[2])))
    X_test = np.expand_dims(X_test, axis=3)

    score = loaded_model.evaluate(X_test, y_test, verbose=1)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))


