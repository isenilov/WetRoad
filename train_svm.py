from datetime import datetime
from feature_extraction import extract_features
from sklearn.metrics import recall_score, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import time
import os

try:
    start = time.time()
    print("\nExtracting features...")
    X_1, y_1 = extract_features("dataset/wet/chevy_wet.wav",
                                "dataset/dry/chevy_dry.wav", flatten=True, scaling=True, categorical=False)
    X_2, y_2 = extract_features("dataset/wet1/audio_mono.wav",
                                "dataset/dry1/audio_mono.wav", flatten=True, scaling=True, categorical=False)
    X = np.concatenate((X_1, X_2))
    y = np.concatenate((y_1, y_2))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # X_train, y_train = extract_features("dataset/wet/test_wet.wav",
    #                                     "dataset/dry/test_dry.wav", flatten=True, scaling=True, categorical=False)
    # X_test, y_test = extract_features("dataset/wet/test_wet.wav",
    #                                   "dataset/dry/test_dry.wav", flatten=True, scaling=True, categorical=False)
    end = time.time()
    print("Took %.3f sec." % (end - start))

    start = time.time()
    print("\nTraining model...")
    clf = SVC(verbose=2)
    clf.fit(X_train, y_train)
    end = time.time()
    print("\nTook %.3f sec." % (end - start))

    start = time.time()
    print("\nEvaluating...")
    y_pred = clf.predict(X_test)

    dt = datetime.now().strftime("%d-%m-%Y_%H-%M")
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    end = time.time()
    with open('results.txt','a') as f:
        f.write("SVM" + dt + " Accuracy: " + str(acc) + " Reacall: " + str(rec) + " Time: " + str(end-start) + " s\n")
    print("Accuracy: %s" % acc)
    print("Recall (wet): %s" % rec)
    print("Took %.3f sec." % (end - start))

except Exception as e:
    dt = datetime.now().strftime("%d-%m-%Y_%H-%M")
    with open(dt + ".log", "w") as f:
        f.write(str(e))
    os.system("sudo poweroff")

else:
    os.system("sudo poweroff")
