import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
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

clf = OneVsRestClassifier(RandomForestClassifier(), n_jobs=1)
clf.fit(x_train, y_train)
prob_matrix = clf.predict_proba(x_test)

# For this problem, we will predict only one genre with max probability
# and verify if the entry belongs to that genre
predicted_labels = np.argmax(prob_matrix, axis=1)

tmp = y_test
tmp['predicted'] = predicted_labels

count = 0
for i, row in tmp.iterrows():
    if row[categories[row.predicted]] == 1:
        count += 1

custom_score = count/len(predicted_labels)
print('score: {}'.format(custom_score))
# Logistic Regression score: 0.6450824679291387
# LinearSVC score:           0.6450824679291387
# RandomForest score:        0.6463042150274894
