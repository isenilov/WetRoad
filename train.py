from feature_extraction import extract_features
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score
import sklearn.preprocessing
from sklearn.svm import SVC
import time

start = time.time()
print("\nExtracting features...")
features, labels = extract_features()
print(features.shape, labels.shape)
end = time.time()
print("Took %.3f sec." % (end - start))

start = time.time()
print("\nNormalizing features...")
features = sklearn.preprocessing.scale(features)
end = time.time()
print("Took %.3f sec." % (end - start))

start = time.time()
print("\nSplitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
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
