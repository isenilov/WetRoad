from feature_extraction import extract
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC


print("Extracting features...")
features_wet = extract("dataset/wet/train_wet.wav")
labels_wet = np.ones(features_wet.shape[0])
features_dry = extract("dataset/dry/train_dry.wav")
labels_dry = np.zeros(features_dry.shape[0])

features = np.concatenate((features_wet, features_dry))
labels = np.concatenate((labels_wet, labels_dry))
print(features_wet.shape, features_dry.shape)


print("Normalizing features...")
features_norm = scale(features)
# print(features)
# print(features_norm)


print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(features_norm, labels, test_size=0.33, random_state=42)

print("Training model...")
clf = SVC(verbose=2)
clf.fit(X_train, y_train)

print("Evaluating...")
print(clf.score(X_test, y_test))


# plt.xlabel("Time, s")
# plt.ylabel("Frequency, Hz")
# print("Number of frequencies:", len(freqs))
# print("Number of time bins:", len(bins))

# plt.show()

