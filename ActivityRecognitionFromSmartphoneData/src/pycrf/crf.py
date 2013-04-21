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


def predict_test_words(test_imgs,test_words,fweights,tweights):
    #print process_labels_mp(test_imgs[i], fweights, tweights, debug = True)

    #for i in range(0,5):     
    #    print "case %d" % (i+1)  
    #    print "         word: %s" % id2char(test_words[i])
    #    pword, _ = process_labels_mp(test_imgs[i], fweights, tweights, debug = False)
    #    print "predicted word %s" % id2char(pword)
    
    num_characters = 0
    num_characters_correct = 0
    for i in range(0,len(test_words)):     
        #print "case %d" % (i+1)  
        #print "         word: %s" % id2char(test_words[i])
        beta = process_labels_mp(test_imgs[i], fweights, tweights)
        pword = predict_labels(beta)
        #print pword[0:200]
        
        #print "predicted word %s" % id2char(pword)
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

def get_neg_label_energy(labels, phi_ij):
    neg_engergy = 0
    for j,label in enumerate(labels):
        neg_engergy += phi_ij[j][label]
    return neg_engergy

def get_neg_transition_energy(labels, phi_trans):
    neg_engergy = 0
    for j in range(0,len(labels)-1):
        neg_engergy += phi_trans[labels[j]][labels[j+1]]
    return neg_engergy


def process_labels_mp(img,fweights,tweights):
    """
    Sum-Product Message Passing
    """
    phi_ij = []
    #condition on the observed image sequence:
    for i,l in enumerate(img):
        phi = fweights*l
        phi = phi.sum(axis=1)
        #make phi a column vector
        phi.shape  = (phi.shape[0],1)
        phi_ij.append(phi)
    #calculate the clique potentials
    psi = []
    for j in range(0,len(phi_ij)-1):
        #we need to add, because we are in log-space
        p = tweights + phi_ij[j]
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
    def __init__(self, Xs, ys_labels, Xs_test, ys_test_labels, n_labels, n_features, sigma):
        self.Xs = Xs
        self.ys_labels = ys_labels
        self.Xs_test = Xs_test
        self.ys_test_labels = ys_test_labels
        self.n_labels = n_labels
        self.n_features = n_features
        self.n_fweights = self.n_labels*self.n_features
        self.n_tweights = self.n_labels*self.n_labels
        self.sigma_square = sigma**2 if sigma else None
        
    #@profile(immediate=True)
    def crf_log_lik(self, d, train_imgs, train_words):
        """
        lok-likelihood with respect to all model parameters W^T and W^F
        
        The derivates:
        dL/dWccn = 1/N sum_N (sum_{j=1}^{L} [y_{ij} = c][y_{ij+1} = c] - P_{W}(y_{ij} = c,y_{ij+1}=c|x))
        """
        
        tick = time.clock()
        
        lfweights = d[0:self.n_fweights].reshape(self.n_labels,self.n_features)
        ltweights = d[self.n_fweights:].reshape(self.n_labels,self.n_labels)
        logprob = 0.
        derivatives = np.zeros(self.n_fweights + self.n_tweights)
        fderivatives = derivatives[0:self.n_fweights].reshape(self.n_labels,self.n_features)
        tderivatives = derivatives[self.n_fweights:].reshape(self.n_labels,self.n_labels)
        
        #print "INITIALIZTAION: %f" % (time.clock() - tick)
        tick = time.clock()
        
        for img, word in zip(train_imgs, train_words):
            #print "BEGIN WORD: %f" % (time.clock() - tick)
            tick = time.clock()
            beta = process_labels_mp(img, lfweights, ltweights)
            Z = logsumexp(beta[0])
            phi_ij = get_conditioned_weights(img, lfweights)
            
            #log likelihood
            neg_erg = get_neg_energ(word, phi_ij, ltweights)
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
                for c in range(0,self.n_labels):
                    P_c = P_yij[j][c]  
                    if y_ij == c:
                        #TODO: vectorize!
                        fderivatives[c,:] += (1-P_c) * img[j][:]
                        #unvectorized code for reference
                        #for f in range(0,self.n_features):
                        #    fderivatives[c,f] += (1-P_c) * img[j][f]
                    else:
                        #TODO: vectorize!
                        fderivatives[c,:] += (0-P_c) * img[j][:]
                        #unvectorized code for reference
                        #for f in range(0,self.n_features):
                        #    fderivatives[c,f] += (0-P_c) * img[j][f]
                            
            #print "DER2: %f" % (time.clock() - tick)
            tick = time.clock()
                            
            #transition derivatives:
            for j, (y_ij, y_ij_n) in enumerate(zip(word, word[1:])):
                if j < len(beta):
                    b = beta[j]
                    P_ccn = np.exp(b-Z)
                    
                tderivatives[y_ij][y_ij_n] += 1
                tderivatives += -P_ccn
                
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
            res = minimize(self.crf_log_lik, np.zeros((self.n_fweights+self.n_tweights,1)), args = (self.Xs, self.ys_labels), method='BFGS', jac=True, options={'disp': True, 'maxiter': 100}, callback=self.test_accuracy)
        
            self.fweights = res.x[0:self.n_fweights].reshape(self.n_labels,self.n_features)
            self.tweights = res.x[self.n_fweights:].reshape(self.n_labels,self.n_labels)
        except KeyboardInterrupt:
            print "minimizing interrupted... using latest values"
        
    def get_weights(self):
        return (self.fweights, self.tweights)

    def test_accuracy(self, xk):
        learnedfweights = xk[0:self.n_fweights].reshape(self.n_labels,self.n_features)
        learnedtweights = xk[self.n_fweights:].reshape(self.n_labels,self.n_labels)
        
        #TODO: don't store the temporary learned weights
        self.fweights = learnedfweights
        self.tweights = learnedtweights
        
        if self.Xs_test == None or self.ys_test_labels == None:
            return
        
        predict_test_words(self.Xs_test,self.ys_test_labels,learnedfweights,learnedtweights)

class LinearCRF(BaseEstimator):
    
    def __init__(self, sigma=None):
        """
            sigma: L2 regularization parameter
        """
        self.label_names = np.array([])
        self.labels = np.array([])
        self.sigma = sigma
    
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
        
        self.label_names, _ = np.unique(np.concatenate(ys), return_inverse=True)
        self.label_mapper = defaultdict(lambda :-1)
        self.label_mapper.update({label:i for i,label in enumerate(self.label_names)})
        self.inv_label_mapper = defaultdict(lambda :-1)
        self.inv_label_mapper.update({i:label for i,label in enumerate(self.label_names)})
        
        ys_labels = [np.array([self.label_mapper[i] for i in y]) for y in ys]
        ys_test_labels = None
        if ys_test:
            ys_test_labels = [np.array([self.label_mapper[i] for i in y]) for y in ys_test]

        n_labels = len(self.label_names)
        
        self.trainer = CRFTrainer(Xs, ys_labels, Xs_test, ys_test_labels, n_labels, n_features, self.sigma)
        
        self.trainer.train()
        
        self.fweights, self.tweights = self.trainer.get_weights()
        del self.trainer
        
        return self
    
    def predict(self, X):
        beta = process_labels_mp(X, self.fweights, self.tweights)
        labels = np.array(predict_labels(beta))
        labels = np.array([self.inv_label_mapper[l] for l in labels])
        return labels
    
    def predict_batch(self, Xs):
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
    
    
    def plot_weights(self):
        """
        Plot the weight matrices using matplotlib.
        """
        
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        n_labels = len(self.label_names)
                
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