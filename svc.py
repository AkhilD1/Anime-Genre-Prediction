import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

# Load data for training
with open('tfidf/tfidf_document_matrix.obj', 'rb') as f:
    x = pickle.load(f)
with open('tfidf/labels.obj', 'rb') as f:
    y = pickle.load(f)

categories = y.columns

# Split data into training and testing
x_train, x_test = train_test_split(
    x, random_state=0, test_size=0.3, shuffle=True)
y_train, y_test = train_test_split(
    y, random_state=0, test_size=0.3, shuffle=True)
print('Shape of training data', x_train.shape)
print('Shape of training labels', y_train.shape)
print('Shape of test data', x_test.shape)
print('Shape of test labels', y_test.shape)

clf = OneVsRestClassifier(LinearSVC(), n_jobs=1)
clf.fit(x_train, y_train)
predicted_matrix = clf.predict(x_test)

# Find the count of cases where at least one of the genres matches the original
df = pd.DataFrame(predicted_matrix, index=y_test.index)
new = pd.DataFrame(
    y_test.values*df.values, columns=y_test.columns, index=y_test.index)
count = np.count_nonzero(new.sum(axis=1))

custom_score = count/len(predicted_matrix)
print('score: {}'.format(custom_score))
# Logistic Regression score: 0.6450824679291387
# LinearSVC score: 0.6450824679291387
