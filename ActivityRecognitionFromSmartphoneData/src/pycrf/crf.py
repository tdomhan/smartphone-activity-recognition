'''
Created on Apr 12, 2013

@author: tdomhan
'''

import numpy as np

from scipy.optimize import minimize

import sklearn.base.BaseEstimator

class LinearCRF(sklearn.base.BaseEstimator):
    
    def __init__(self):
        self.label_names = np.array([])
        self.labels = np.array([])
    
    def set_params(self, **parameters):
        #TODO: implement!
        pass
    
    def fit(self, X, y):
        """Fit the CRF model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values (class labels)

        Returns
        -------
        self : object
            Returns self.

        Notes
        ------
        Nothing to note here ;)
        
        """
        
        n_samples, n_features = X.shape
        
        
        self.label_names, self.labels = np.unique(y, return_inverse=True)
        n_labels = len(self.label_names)
        
        self.fweights = np.zeros((n_labels,n_features))
        self.fweights = np.zeros((n_labels,n_labels))
        
        
        return self
    
    def predict(self, X):
        """Perform inference on samples in X.


        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        y_pred : array, shape = [n_samples]
        """
        pass
