'''
Created on Apr 12, 2013

@author: tdomhan
'''

from collections import defaultdict

import numpy as np

from scipy.optimize import minimize
from scipy.misc import logsumexp

from sklearn.base import BaseEstimator


def predict_test_words(test_imgs,test_words,fweights,tweights):
    #print process_test_word_mp(test_imgs[i], test_words[i], fweights, tweights, debug = True)

    #for i in range(0,5):     
    #    print "case %d" % (i+1)  
    #    print "         word: %s" % id2char(test_words[i])
    #    pword, _ = process_test_word_mp(test_imgs[i], test_words[i], fweights, tweights, debug = False)
    #    print "predicted word %s" % id2char(pword)
    
    num_characters = 0
    num_characters_correct = 0
    for i in range(0,len(test_words)):     
        #print "case %d" % (i+1)  
        #print "         word: %s" % id2char(test_words[i])
        pword, _ = process_test_word_mp(test_imgs[i], test_words[i], fweights, tweights)
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


def process_test_word_mp(img,txt,fweights,tweights):
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
        
    pword = []
    for b in beta:
        pword.append(np.argmax(np.exp(logsumexp(b, axis=1))))
    pword.append(np.argmax(np.exp(logsumexp(beta[-1], axis=0))))
    
    return pword, beta

def crf_log_lik(d, train_imgs, train_words):
    #lok-likelihood with respect to all model parameters W^T and W^F
    lfweights = d[0:3210].reshape(10,321)
    ltweights = d[3210:].reshape(10,10)
    logprob = 0.
    for img, word in zip(train_imgs, train_words):
        _, beta = process_test_word_mp(img, word, lfweights, ltweights)
        Z = logsumexp(beta[0])
        phi_ij = get_conditioned_weights(img, lfweights)
        
        #log likelihood
        neg_erg = get_neg_energ(word, phi_ij, ltweights)
        logprob += neg_erg - Z
        
        #derivatives:
            
    logprob = logprob / float(len(train_imgs))
    
    #L2 regularization
    sigma = 10
    reg = np.square(d)
    reg /= 2. * sigma * sigma
    logprob += -1.0 * reg.sum()

    #return the negative, because we are MINIMIZING
    return (-logprob,crf_log_lik_der(d, train_imgs, train_words)) 

def get_neg_energ(labels,phi_ij,phi_trans):
    """
    Get the negative energy given the labels and the transition weights
    """
    return get_neg_label_energy(labels, phi_ij) + get_neg_transition_energy(labels, phi_trans)


def crf_log_lik_der(d, train_imgs, train_words):
    """
    The actual derivates:
    dL/dWccn = 1/N sum_N (sum_{j=1}^{L} [y_{ij} = c][y_{ij+1} = c] - P_{W}(y_{ij} = c,y_{ij+1}=c|x))
    """
    #lok-likelihood with respect to all model parameters W^T and W^F
    lfweights = d[0:3210].reshape(10,321)
    ltweights = d[3210:].reshape(10,10)
    fderivatives = np.zeros((10,321))
    tderivatives = np.zeros((10,10))
    
    for img, word in zip(train_imgs, train_words):
        _, beta = process_test_word_mp(img, word, lfweights, ltweights)
        Z = logsumexp(beta[0])
        
        P_yij = [0] * len(word)
        for j in range(0,len(word)):
            if j < len(beta):
                b = beta[j]
                P_yij[j] = np.exp(logsumexp(b, axis=1)-Z)
            else:
                b = beta[-1]
                P_yij[j] = np.exp(logsumexp(b, axis=0)-Z)
        
        # feature derivatives:
        for j, y_ij in enumerate(word):
            for c in range(0,10):
                P_c = P_yij[j][c]    
                if y_ij == c:
                    for f in range(0,321):
                        fderivatives[c,f] += (1-P_c) * img[j][f]
                else:
                    for f in range(0,321):
                        fderivatives[c,f] += (0-P_c) * img[j][f]
                        
        #transition derivatives:
        
        for j, (y_ij, y_ij_n) in enumerate(zip(word, word[1:])):
            if j < len(beta):
                b = beta[j]
                P_ccn = np.exp(b-Z)
            for c in range(0,10):
                for cn in range(0,10):
                    P = P_ccn[c,cn]
                    if c == y_ij and cn == y_ij_n:
                        tderivatives[c][cn] += 1 - P
                    else:
                        tderivatives[c][cn] += 0 - P
    
    derivatives = np.concatenate((fderivatives.reshape((3210)), tderivatives.reshape((100))))
    #return the negative, because we are MINIMIZING
    derivatives = derivatives * -1./float(len(train_imgs))
    
    #L2 regularization
    sigma = 10
    derivatives += -1.0 * d / (2. * sigma * sigma)
    
    return derivatives     

class CRFTrainer():
    def __init__(self, Xs, ys_labels, Xs_test, ys_test_labels, n_labels, n_features):
        self.Xs = Xs
        self.ys_labels = ys_labels
        self.Xs_test = Xs_test
        self.ys_test_labels = ys_test_labels
        self.n_labels = n_labels
        self.n_features = n_features
        self.n_fweights = self.n_labels*self.n_features
        self.n_tweights = self.n_labels*self.n_labels
        
    def train(self):
        
        
        res = minimize(crf_log_lik, np.zeros((self.n_fweights+self.n_tweights,1)), args = (self.Xs, self.ys_labels), method='BFGS', jac=True, options={'disp': True}, callback=self.test_accuracy)
        
        self.fweights = res.x[0:self.n_fweights].reshape(self.n_labels,self.n_fweights)
        self.tweights = res.x[self.n_fweights:].reshape(self.n_labels,self.n_labels)
        
    def get_weights(self):
        return (self.fweights, self.tweights)

    def test_accuracy(self, xk):
        if self.Xs_test == None or self.ys_test_labels == None:
            return
        learnedfweights = xk[0:self.n_fweights].reshape(self.n_labels,self.n_features)
        learnedtweights = xk[self.n_fweights:].reshape(self.n_labels,self.n_labels)
        
        predict_test_words(self.Xs_test,self.ys_test_labels,learnedfweights,learnedtweights)

class LinearCRF(BaseEstimator):
    
    def __init__(self):
        self.label_names = np.array([])
        self.labels = np.array([])
    
    def set_params(self, **parameters):
        #TODO: implement!
        pass
    
    def batch_fit(self, Xs, ys, sigma=None, Xs_test=None, ys_test=None):
        """Fit the CRF model according to the given training data.

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
        #TODO: check that all Xs have the same shape
        n_features = Xs[0].shape[1]
        
        self.label_names, _ = np.unique(np.concatenate(ys), return_inverse=True)
        label_mapper = defaultdict(lambda :-1)
        label_mapper.update({label:i for i,label in enumerate(self.label_names)})
        
        ys_labels = [np.array([label_mapper[i] for i in y]) for y in ys]
        ys_test_labels = None
        if ys_test:
            ys_test_labels = [np.array([label_mapper[i] for i in y]) for y in ys_test]

        n_labels = len(self.label_names)
        
        trainer = CRFTrainer(Xs, ys_labels, Xs_test, ys_test_labels, n_labels, n_features)
        
        trainer.train()
        
        self.fweights, self.tweights = trainer.get_weights()
        
        
        
        return self
    
    def predict_batch(self, Xs):
        """Perform inference on samples in X.


        Parameters
        ----------
        X : iteratble of {array-like, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        y_pred : array, shape = [n_samples]
        """
        pass
