'''
Created on Apr 29, 2013

@author: tdomhan
'''

import numpy as np


"""Assuming the data is already loaded """


from sklearn.cross_validation import KFold
from sklearn.feature_selection import RFECV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import zero_one_loss

#http://scikit-learn.org/dev/auto_examples/plot_rfe_with_cross_validation.html#example-plot-rfe-with-cross-validation-py

n_folds = 3


rfecv = RFECV(estimator=LogisticRegression(penalty), step=5, loss_func=zero_one_loss, cv=KFold(len(X_all), n_folds=n_folds, shuffle=True, random_state=1))

rfecv.fit(X_all, y_all)




