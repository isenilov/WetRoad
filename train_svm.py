from feature_extraction import extract_features
from sklearn.metrics import recall_score, accuracy_score
from sklearn.svm import SVC
import time

start = time.time()
print("\nExtracting features...")
X_train, y_train = extract_features("dataset/wet1/audio_mono.wav",
                                    "dataset/dry1/audio_mono.wav", flatten=True, scaling=True, categorical=False)
X_test, y_test = extract_features("dataset/wet2/audio_mono.wav",
                                  "dataset/dry2/audio_mono.wav", flatten=True, scaling=True, categorical=False)
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
print("Accuracy: %s" % accuracy_score(y_test, y_pred))
print("Recall (wet): %s" % recall_score(y_test, y_pred))
end = time.time()
print("Took %.3f sec." % (end - start))
