import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# Load data for training
with open('tfidf/tfidf_document_matrix.obj', 'rb') as f:
    x = pickle.load(f)
with open('tfidf/labels.obj', 'rb') as f:
    y = pickle.load(f)

categories = y.columns
scores = []

# Custom k-fold
for i in range(10):
    # Split data into training and testing
    x_train, x_test = train_test_split(
        x, random_state=i, test_size=0.3, shuffle=True)
    y_train, y_test = train_test_split(
        y, random_state=i, test_size=0.3, shuffle=True)

    clf = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)
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
    scores.append(custom_score)

print('scores: {}\navg: {}, max: {}, min: {}'.format(
    scores, np.mean(scores), max(scores), min(scores)))
