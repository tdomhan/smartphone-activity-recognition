'''
Created on Apr 27, 2013

@author: tdomhan

python interface to the SVM-HMM classifier.

'''

from sklearn.base import BaseEstimator

from sklearn.datasets import dump_svmlight_file

import numpy as np

from subprocess import call

class SVMHMMCRF(BaseEstimator):
    def __init__(self, C=1):
        """
            C: regularization parameter.
        """
        self.model_file = "svmhmm.model"
        self.C = C
        self.path = "./svm_hmm/"
    
    def fit(self, X, y):
        Xs = [X]
        ys = [y]
        self.batch_fit(Xs, ys)
        
        return self
    
    def batch_fit(self, Xs, ys):
        qids = [np.array([i] * len(ys[i])) for i in range(len(ys))]
        print "dumping data to Xtrain.data"
        dump_svmlight_file(np.concatenate(Xs),np.concatenate(ys),"Xtrain.data",zero_based=False,query_id=np.concatenate(qids))
        
        print "now learning"
        
        print call([self.path + "svm_hmm_learn",  "-c", "%d" % self.C, "Xtrain.data", "svmhmm-model.dat"])
        
        return self
    
    def predict(self, X, viterbi=False):
        Xs = [X]
        return self.batch_predict(Xs)
    
    def batch_predict(self, Xs):
        qids = [np.array([i] * len(Xs[i])) for i in range(len(Xs))]
        ys = [np.array([0] * len(Xs[i])) for i in range(len(Xs))]
        print "dumping data to Xest.data"
        dump_svmlight_file(np.concatenate(Xs),np.concatenate(ys),"Xtest.data",zero_based=False,query_id=np.concatenate(qids))

        print "now classifying"

        print call([self.path + "svm_hmm_classify", "Xtest.data", "svmhmm-model.dat", "svmhmm-classified.tag"])
        y_predict = np.loadtxt("svmhmm-classified.tag")
        return y_predict
    
    
    
    

