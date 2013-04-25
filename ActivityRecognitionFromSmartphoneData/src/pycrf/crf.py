'''
Created on Apr 12, 2013

@author: tdomhan
'''

from collections import defaultdict

import numpy as np

from scipy.optimize import minimize
from scipy.misc import logsumexp

from sklearn.base import BaseEstimator

from profilehooks import *

import time

def trans_weight_function(X):
    """
    Calculates (x(t)-x(t+1))^2
    """
    #X_diff = np.square(np.diff(X, n=1, axis=0))
    X_diff = np.abs(np.diff(X, n=1, axis=0))
    Xnew = np.zeros(X.shape)
    Xnew[1:,:] = X_diff
    return X_diff

def predict_test_words(test_imgs,test_words,fweights,tweights,transition_weighting=False):
    
    num_characters = 0
    num_characters_correct = 0
    for i in range(0,len(test_words)):     
        beta = process_labels_mp(test_imgs[i], fweights, tweights, transition_weighting)
        pword = predict_labels(beta)

        for j,c in enumerate(pword):
            num_characters+=1
            if c == test_words[i][j]:
                num_characters_correct+=1
                
    accuracy = num_characters_correct/float(num_characters)
    print "accuracy: %.4f" % accuracy
    
    return accuracy

def get_conditioned_weights(x, fweights):
    """
    Get the weights after conditioning on the observed data
    """
    #implement as a 2d np array
    phi_ij = []
    for i,l in enumerate(x):
        phi = fweights*l
        phi = phi.sum(axis=1)
        phi_ij.append(phi)
    return phi_ij


def get_conditioned_t_weights(X, tweights):
    """
    Get the TRANSITION weights after conditioning on the observed data
    """
    #implement as a 2d np array
    phi_ij = []
    X_diff = trans_weight_function(X)
    for i,l in enumerate(X_diff):
        phi = tweights*l
        phi = phi.sum(axis=2)
        phi_ij.append(phi)
    return phi_ij

def get_neg_label_energy(labels, phi_ij):
    neg_engergy = 0
    for j,label in enumerate(labels):
        neg_engergy += phi_ij[j][label]
    return neg_engergy

def get_neg_transition_energy(labels, phi_trans):
    neg_engergy = 0
    for j,phi_t in enumerate(phi_trans[:-1]):
        neg_engergy += phi_t[labels[j]][labels[j+1]]
    return neg_engergy


def process_labels_mp(X,fweights,tweights, transition_weighting=False):
    """
    Sum-Product Message Passing
    """
    phi_ij = []
    #condition on the observed image sequence:
    for i,l in enumerate(X):
        phi = fweights*l
        phi = phi.sum(axis=1)
        #make phi a column vector
        phi.shape  = (phi.shape[0],1)
        phi_ij.append(phi)
        
    if transition_weighting:
        phi_t = get_conditioned_t_weights(X, tweights)
    else:
        phi_t = [tweights for _ in range(len(X))]
        
    #calculate the clique potentials
    psi = []
    for j in range(0,len(phi_ij)-1):
        #we need to add, because we are in log-space
        p = phi_t[j] + phi_ij[j]
        if j == len(phi_ij)-2:
            #the last entry gets two node potentials
            p = p + phi_ij[j+1].transpose()
        psi.append(p)
    
    #compute the messages
    
    #forward
    deltaf = []
    for i in range(0, len(psi)-1):
        if len(deltaf) > 0:
            w = psi[i] + deltaf[i-1]
        else:
            w = psi[i]
        d = logsumexp(w, axis=0)
        d.shape  = (d.shape[0],1)
        deltaf.append(d)
        
    #and back..
    deltab = []
    for i in range(len(psi)-1, 0, -1):
        #transpose so that the summation is over the correct 
        #axis as well as the broadcast works correctly
        if len(deltab) > 0:
            w = psi[i] + deltab[-1]
        else:
            w = psi[i]
        d = logsumexp(w, axis=1)
        #let's make it a row vector so that the broadcasting is done right
        d.shape  = (1,d.shape[0])
        deltab.append(d)
    
    
    #belief read-out
    beta = [x.view() for x in psi]
    #forward
    for i,d in enumerate(deltaf):
        beta[i+1] = beta[i+1] + d
    #and back
    deltab.reverse()
    for i,d in enumerate(deltab):
        beta[i] = beta[i] + d
    
    return beta


def predict_labels(beta):
    """
    Predict a labels sequence using the betas
    calculate in from the message passing algorithm.
    """
    pword = []
    for b in beta:
        pword.append(np.argmax(logsumexp(b, axis=1)))
    pword.append(np.argmax(logsumexp(beta[-1], axis=0)))
    
    return pword

def get_neg_energ(labels,phi_ij,phi_trans):
    """
    Get the negative energy given the labels and the transition weights
    """
    return get_neg_label_energy(labels, phi_ij) + get_neg_transition_energy(labels, phi_trans)

   

class CRFTrainer():
    def __init__(self, Xs, ys_labels, Xs_test, ys_test_labels, n_labels, n_features, sigma, transition_weighting):
        self.Xs = Xs
        self.ys_labels = ys_labels
        self.Xs_test = Xs_test
        self.ys_test_labels = ys_test_labels
        self.n_labels = n_labels
        self.n_features = n_features
        self.n_fweights = self.n_labels*self.n_features
        self.n_tweights = self.n_labels*self.n_labels
        self.sigma_square = sigma**2 if sigma else None
        self.transition_weighting = transition_weighting
        if transition_weighting:
            self.n_tweights *= self.n_features
        
    #@profile(immediate=True)
    def crf_log_lik(self, d, train_imgs, train_words):
        """
        lok-likelihood with respect to all model parameters W^T and W^F
        
        The derivates:
        dL/dWccn = 1/N sum_N (sum_{j=1}^{L} [y_{ij} = c][y_{ij+1} = c] - P_{W}(y_{ij} = c,y_{ij+1}=c|x))
        """
        
        tick = time.clock()
        
        lfweights = d[0:self.n_fweights].reshape(self.n_labels,self.n_features)
        if not self.transition_weighting:
            ltweights = d[self.n_fweights:].reshape(self.n_labels,self.n_labels)
        else:
            ltweights = d[self.n_fweights:].reshape((self.n_labels,self.n_labels,self.n_features))
        logprob = 0.
        derivatives = np.zeros(self.n_fweights + self.n_tweights)
        fderivatives = derivatives[0:self.n_fweights].reshape(self.n_labels,self.n_features)
        if not self.transition_weighting:
            tderivatives = derivatives[self.n_fweights:].reshape(self.n_labels,self.n_labels)
        else:
            tderivatives = derivatives[self.n_fweights:].reshape((self.n_labels,self.n_labels,self.n_features))
        
        #print "INITIALIZTAION: %f" % (time.clock() - tick)
        tick = time.clock()
        
        for X, word in zip(train_imgs, train_words):
            #print "BEGIN WORD: %f" % (time.clock() - tick)
            tick = time.clock()
            beta = process_labels_mp(X, lfweights, ltweights, self.transition_weighting)
            Z = logsumexp(beta[0])
            phi_ij = get_conditioned_weights(X, lfweights)
            
            if self.transition_weighting:
                phi_t = get_conditioned_t_weights(X, ltweights)
            else:
                phi_t = [ltweights for _ in range(len(X))]
            
            #log likelihood
            neg_erg = get_neg_energ(word, phi_ij, phi_t)
            logprob += neg_erg - Z
            
            #print "LOG-LIK: %f" % (time.clock() - tick)
            tick = time.clock()
            
            #derivatives:
            P_yij = [0] * len(word)
            for j in range(0,len(word)):
                if j < len(beta):
                    b = beta[j]
                    P_yij[j] = np.exp(logsumexp(b, axis=1)-Z)
                else:
                    b = beta[-1]
                    P_yij[j] = np.exp(logsumexp(b, axis=0)-Z)
                    
            #print "DER1: %f" % (time.clock() - tick)
            tick = time.clock()
            
            # feature derivatives:
            for j, y_ij in enumerate(word):
                P_c = P_yij[j] # vector: P[c]
                fderivatives += np.outer(-P_c, X[j][:])
                fderivatives[y_ij,:] += X[j][:]
                
#                #unvectorized version for reference:
#                for c in range(0,self.n_labels):
#                    P_c = P_yij[j][c]  
#                    if y_ij == c:
#                        #TODO: vectorize!
#                        fderivatives[c,:] += (1-P_c) * X[j][:]
#                        #unvectorized code for reference
#                        #for f in range(0,self.n_features):
#                        #    fderivatives[c,f] += (1-P_c) * X[j][f]
#                    else:
#                        #TODO: vectorize!
#                        fderivatives[c,:] += (0-P_c) * X[j][:]
#                        #unvectorized code for reference
#                        #for f in range(0,self.n_features):
#                        #    fderivatives[c,f] += (0-P_c) * X[j][f]
                        
            #print "DER2: %f" % (time.clock() - tick)
            tick = time.clock()
                            
            #transition derivatives:
            X_diff = trans_weight_function(X)
            for j, (y_ij, y_ij_n, x_diff) in enumerate(zip(word, word[1:], X_diff)):
                if j < len(beta):
                    b = beta[j]
                    P_ccn = np.exp(b-Z)
                   
                if not self.transition_weighting:
                    tderivatives[y_ij][y_ij_n] += 1
                    tderivatives += -P_ccn
                else:
                    tderivatives[y_ij][y_ij_n] += x_diff
                    tderivatives += np.outer(-P_ccn, x_diff).reshape((self.n_labels,self.n_labels,self.n_features))
                
                #TODO: put here the formula for better understanding
                #unvectorized code for reference
#                for c in range(0,self.n_labels):
#                    for cn in range(0,self.n_labels):
#                        P = P_ccn[c,cn]
#                        if c == y_ij and cn == y_ij_n:
#                            tderivatives[c][cn] += 1 - P
#                        else:
#                            tderivatives[c][cn] += 0 - P
            #print "DER3: %f" % (time.clock() - tick)
            tick = time.clock()
        
        derivatives *= 1./float(len(train_imgs))
        
        #L2 regularization of derivatives
        derivatives += self.l2_regularization_der(d)
                
        logprob = logprob / float(len(train_imgs))
        
        logprob += self.l2_regularization(d)
        
        print logprob
        print derivatives[0:10]
        
        print "max derivative: %f" % derivatives.max()
    
        #return the negative loglik and derivatives, because we are MINIMIZING
        return (-logprob, -derivatives) 
    
    def l2_regularization(self,d):
        if not self.sigma_square:
            return 0
        reg = np.square(d).sum()
        reg /= 2. * self.sigma_square
        return -1.0 * reg
    
    def l2_regularization_der(self,d):
        if not self.sigma_square:
            return np.zeros(len(d))
        reg = -1.0 * d / (self.sigma_square)
        return reg
        
    def train(self):
        try:
            x0 = np.zeros((self.n_fweights+self.n_tweights,1))
            res = minimize(self.crf_log_lik, x0, args = (self.Xs, self.ys_labels), method='L-BFGS-B', jac=True, options={'disp': True, 'maxiter': 100}, callback=self.test_accuracy)
        
            self.fweights = res.x[0:self.n_fweights].reshape(self.n_labels,self.n_features)
            if not self.transition_weighting:
                self.tweights = res.x[self.n_fweights:].reshape(self.n_labels,self.n_labels)
            else:
                self.tweights = res.x[self.n_fweights:].reshape((self.n_labels,self.n_labels,self.n_features))
        except KeyboardInterrupt:
            print "minimizing interrupted... using latest values"
        
    def get_weights(self):
        return (self.fweights, self.tweights)

    def test_accuracy(self, xk):
        learnedfweights = xk[0:self.n_fweights].reshape(self.n_labels,self.n_features)
        if not self.transition_weighting:
            learnedtweights = xk[self.n_fweights:].reshape(self.n_labels,self.n_labels)
        else:
            learnedtweights = xk[self.n_fweights:].reshape((self.n_labels,self.n_labels,self.n_features))
        
        #TODO: don't store the temporary learned weights
        self.fweights = learnedfweights
        self.tweights = learnedtweights
        
        if self.Xs_test == None or self.ys_test_labels == None:
            return
        
        predict_test_words(self.Xs_test,self.ys_test_labels,learnedfweights,learnedtweights, self.transition_weighting)

def add_const_feature(X):
    """
    add a constant feature to all samples
    """
    Xshape = X.shape
    Xnew = np.ones((Xshape[0],Xshape[1]+1))
    Xnew[:,:-1] = X
    return Xnew
    

class LinearCRF(BaseEstimator):
    
    def __init__(self, label_names=None, feature_names=None, addone=False, sigma=None, transition_weighting=False):
        """
            label_names: array of str objects that represent the labels
            feature_names: array of str objects that represent the features
            addone: add a feature that's always on to capture prior probabilities.
            sigma: L2 regularization parameter
            transition_weighting: if true the transition weights will be 
            multiplies by f(x), which is a F dimensional vector of the data.
            Hence the transition weights C x C x F dimenions
        """
        self.labels_orig = np.array([])
        self.labels = np.array([])
        self.label_names = label_names
        self.feature_names = feature_names
        self.sigma = sigma
        self.addone = addone
        self.transition_weighting = transition_weighting
    
    def set_params(self, **parameters):
        #TODO: implement!
        pass
    
    def fit(self, X, y, X_test=None, y_test=None):
        """Fit the CRF model (for a single chain) according to the given training data.

        Parameters
        ----------
        X : iterable of {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : iterable of array-like, shape = [n_samples]
            Target values (class labels)
            
        sigma: L2 regularization parameter
        
        Xs_test : iterable of {array-like, sparse matrix}, shape = [n_samples, n_features]
            Test vectors, where n_samples is the number of samples
            and n_features is the number of features.

        ys_test : iterable of array-like, shape = [n_samples]
            Test values (class labels)

        Returns
        -------
        self : object
            Returns self.

        Notes
        ------
        Nothing to note here ;)
        
        """
        
        Xs = [X]
        ys = [y]
        Xs_test = [X_test] if X_test != None else None
        ys_test = [y_test] if y_test != None else None
        self.batch_fit(Xs, ys, Xs_test, ys_test)
        
        return self
    
    def batch_fit(self, Xs, ys, Xs_test=None, ys_test=None):
        """Fit the CRF model according to the given training data.

        Parameters
        ----------
        X : iterable of {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : iterable of array-like, shape = [n_samples]
            Target values (class labels)
        
        Xs_test : iterable of {array-like, sparse matrix}, shape = [n_samples, n_features]
            Test vectors, where n_samples is the number of samples
            and n_features is the number of features.

        ys_test : iterable of array-like, shape = [n_samples]
            Test values (class labels)

        Returns
        -------
        self : object
            Returns self.

        Notes
        ------
        Nothing to note here ;)
        
        """
        #TODO: check that all Xs have the same shape
        n_features = Xs[0].shape[1]
        
        self.labels_orig, _ = np.unique(np.concatenate(ys), return_inverse=True)
        self.label_mapper = defaultdict(lambda :-1)
        self.label_mapper.update({label:i for i,label in enumerate(self.labels_orig)})
        self.inv_label_mapper = defaultdict(lambda :-1)
        self.inv_label_mapper.update({i:label for i,label in enumerate(self.labels_orig)})
        
        ys_labels = [np.array([self.label_mapper[i] for i in y]) for y in ys]
        ys_test_labels = None
        if ys_test:
            ys_test_labels = [np.array([self.label_mapper[i] for i in y]) for y in ys_test]

        n_labels = len(self.labels_orig)
        
        # add constant feature
        if self.addone:
            Xsnew = []
            for X in Xs:
                Xsnew.append(add_const_feature(X))
            n_features += 1
            Xs = Xsnew
            if Xs_test:
                Xs_test_new = []
                for X in Xs_test:
                    Xs_test_new.append(add_const_feature(X))
                Xs_test = Xs_test_new
                
        
        self.trainer = CRFTrainer(Xs, ys_labels, Xs_test, ys_test_labels, n_labels, n_features, self.sigma, self.transition_weighting)
        
        self.trainer.train()
        
        self.fweights, self.tweights = self.trainer.get_weights()
        del self.trainer
        
        return self
    
    def predict(self, X):
        if self.addone:
            X = add_const_feature(X)

        beta = process_labels_mp(X, self.fweights, self.tweights, transition_weighting=self.transition_weighting)
        
        labels = np.array(predict_labels(beta))
        labels = np.array([self.inv_label_mapper[l] for l in labels])
        return labels
    
    def batch_predict(self, Xs):
        """Perform inference on samples in X.


        Parameters
        ----------
        X : iteratble of {array-like, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        y_pred : array, shape = [n_samples]
        """
        results = []
        
        
        for X in Xs:
            results.append(self.predict(X))

        return results
    
    def plot_important_features(self, n=10, best=True, absolut=True):
        """
            Plot the most/least important features.
            
            best: if True it'll plot the most important features, otherwise the least
                  important ones.
        """
        
        label_names = self.label_names
        if not label_names:
            label_names = ["label %d" % d for d in range(self.fweights.shape[0])] 
        
        feature_names = self.feature_names
        if not feature_names:
            feature_names = ["feature %d" % d for d in range(self.fweights.shape[1])]
        
        print "feature weights:"
        if absolut:
            ranked_features = np.argsort(np.abs(self.fweights), axis=None)
        else:
            ranked_features = np.argsort(self.fweights, axis=None)
            
        if best:
            ranked_features = ranked_features[::-1] #inverse to get the best first

        for i, fweights_idx in enumerate(ranked_features[:n]):
            label_idx,feature_idx = np.unravel_index(fweights_idx, self.fweights.shape)
            print "%d. f: %s\t\t c: %s\t value: %f" % (i, feature_names[feature_idx], label_names[label_idx], self.fweights[(label_idx,feature_idx)])
        
        print ""
        print "transition weights:"
        if absolut:
            ranked_transitions = np.argsort(np.abs(self.tweights), axis=None)
        else:
            ranked_transitions = np.argsort(self.tweights, axis=None)
            
        if best:
            ranked_transitions = ranked_transitions[::-1] #inverse to get the best first
        
        for i, tweights_idx in enumerate(ranked_transitions[:n]):
            if not self.transition_weighting:
                label_t0_idx,label_t1_idx = np.unravel_index(tweights_idx, self.tweights.shape)
                print "%d. c(t): %s\t c(t+1): %\t value: %f" % (i, feature_names[feature_idx], label_names[label_idx], self.fweights[fweights_idx])
            else:
                label_t0_idx,label_t1_idx,feature_idx = np.unravel_index(tweights_idx, self.tweights.shape)
                print "%d. f(x): %s\t c(t): %s\t c(t+1): %s\t value: %f" % (i, feature_names[feature_idx], label_names[label_t0_idx], label_names[label_t1_idx] , self.tweights[(label_t0_idx,label_t1_idx,feature_idx)])
            
        
    def plot_most_important_features(self, n=10):
        self.plot_important_features(n,best=True)
        
    def plot_least_important_features(self, n=10):
        self.plot_important_features(n,best=False)
    
    def plot_weights(self):
        """
        Plot the weight matrices using matplotlib.
        """
        
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        n_labels = len(self.labels_orig)
                
        plt.figure(1)
        for i in range(n_labels):
            #TODO: generalize
            plt.subplot(2,5,i+1);
            plt.imshow(np.reshape(self.fweights[i,1:],(16,20)).T,interpolation='nearest',cmap = cm.Greys_r);
            #plt.title(labels[i]);
            plt.colorbar(shrink=0.6)
            plt.xticks([])
            plt.yticks([])
        
        plt.figure(2)
        plt.imshow(self.tweights,interpolation='nearest',cmap = cm.Greys_r);
        #plt.xticks(range(10),labels)
        #plt.yticks(range(10),labels)
        plt.colorbar();
        
        plt.show()