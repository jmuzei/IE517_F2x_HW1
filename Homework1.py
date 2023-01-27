# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn import datasets
iris = datasets.load_iris()
x_iris, y_iris = iris.data, iris.target
print(x_iris.shape, y_iris.shape)
print(x_iris[0], y_iris[0])
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Get datasets with only the first two attributes
x, y = x_iris[:, 0:2], y_iris

# split the dataset into a training and a testing set
# Test set will be the 25% taken randomly
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.25, random_state=33)
print(x_train.shape, y_train.shape)

# Standardise the features
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

import matplotlib.pyplot as plt

colors = ['red', 'greenyellow', 'blue']

for i in range(len(colors)):
    xs = x_train[:, 0] [y_train == i]
    ys = x_train[:, 1] [y_train == i]
    plt.scatter(xs, ys, c=colors[i])
    plt.legend(iris.target_names)
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(x_train, y_train)
print(clf.coef_)

print(clf.intercept_)
x_train[:, 0].min()-0.5
x_min, x_max = x_train[:, 0].min()-0.5, x_train[:, 0].max()+0.5
y_min, y_max = x_train[:, 1].min()-0.5, x_train[:, 1].max()+0.5
xs = np.arange(x_min, x_max, 0.5)

fig, axes = plt.subplots(1,3)
fig.set_size_inches(10, 6)
for i in [0, 1, 2]:
    axes[i].set_aspect('equal')
    axes[i].set_title('class' + str(i) + 'versus the rest')
    axes[i].set_xlabel('sepal length')
    axes[i].set_ylabel('sepal width')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    plt.sca(axes[i])
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.prism)
    ys = (-clf.intercept_[i] - xs * clf.coef_[i, 0])/ clf.coef_[i, 1] 
    plt.plot(xs, ys) #hold=True)
print(clf.predict(scaler.transform([[4.7, 3.1]])))
print(clf.decision_function(scaler.transform([[4.7, 3.1]])))

from sklearn import metrics

y_train_pred = clf.predict(x_train)
print(metrics.accuracy_score(y_train, y_train_pred))

y_pred = clf.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))

#Modern Metrics
print(metrics.classification_report(y_test, y_pred, target_names = iris.target_names))

print(metrics.confusion_matrix(y_test, y_pred)) 


