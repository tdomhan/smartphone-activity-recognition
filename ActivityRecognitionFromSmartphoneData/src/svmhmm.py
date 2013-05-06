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
    def __init__(self, C=1, t=0, e=0):
        """
            C: regularization parameter.
            e: [0..1]  -> Order of dependencies of emissions in HMM (default 0)
            t [0..3]  -> Order of dependencies of transitions in HMM (default 1)
        """
        self.model_file = "svmhmm.model"
        self.C = C
        self.e = e
        self.t = t
        self.path = "./svm_hmm/"
    
    def fit(self, X, y):
        Xs = [X]
        ys = [y]
        self.batch_fit(Xs, ys)
        
        return self
    
    def batch_fit(self, Xs, ys, dump = True):
        qids = [np.array([i] * len(ys[i])) for i in range(len(ys))]
        print "dumping data to Xtrain.data"
        if dump:
            dump_svmlight_file(np.concatenate(Xs),np.concatenate(ys),"Xtrain.data",zero_based=False,query_id=np.concatenate(qids))
        
        print "now learning"
        
        print call([self.path + "svm_hmm_learn",  "-c", "%d" % self.C, "--t" , "%d" %self.t, "--e" , "%d" %self.e, "Xtrain.data", "svmhmm-model.dat"])
        
        return self
    
    def predict(self, X, viterbi=False):
        Xs = [X]
        return self.batch_predict(Xs)
    
    def batch_predict(self, Xs, dump = True):
        qids = [np.array([i] * len(Xs[i])) for i in range(len(Xs))]
        ys = [np.array([0] * len(Xs[i])) for i in range(len(Xs))]
        print "dumping data to Xest.data"
        if dump:
            dump_svmlight_file(np.concatenate(Xs),np.concatenate(ys),"Xtest.data",zero_based=False,query_id=np.concatenate(qids))

        print "now classifying"

        print call([self.path + "svm_hmm_classify", "Xtest.data", "svmhmm-model.dat", "svmhmm-classified.tag"])
        y_predict = np.loadtxt("svmhmm-classified.tag")
        
        idx = 0
        ys = []
        for i in range(len(Xs)):
            num = len(Xs[i])
            ys.append(y_predict[idx:idx+num])
            idx += num
        
        return ys
    
    
    
    

