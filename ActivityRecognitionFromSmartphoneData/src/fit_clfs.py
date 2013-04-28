'''
Created on Mar 29, 2013

Let's do some prilimary test


@author: tdomhan
'''

import numpy as np

import pickle

#algorithms
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import Scaler

from sklearn.preprocessing import OneHotEncoder


from pycrf.crf import LinearCRF, LinearCRFEnsemble
from svmhmm import SVMHMMCRF
from sklearn.metrics import accuracy_score

from utils import *

#metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm.classes import LinearSVC


def SVM_feature_extraction(X_train, y_train, X_test):
    """
    Learn a standard SVM classifier first, then transform 
    X_test by mapping from feature space to the space of 
    the scores of the SVM classifier for each of the classes.
    The resulting data can then further be used by a classifier
    specialized in sequential labelling.
    """
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    X_train_t = clf.decision_function(X_train)
    X_test_t = clf.decision_function(X_test)
    return (X_train_t,X_test_t)


def get_diff_features(X):
    X_diff = np.diff(X, n=1, axis=0)
    Xnew = np.zeros(X.shape)
    Xnew[1:,:] = X_diff
    return Xnew

def get_last_action_feature(X,ys):
    """
    Convert the labels in ys into a one-hot representation.
    The result will be shifted so that each row represents the last action.
    The first row everything will be 0.
    """
    onehot = OneHotEncoder()
    onehot.fit([[y] for y in ys])
    
    actions = onehot.transform([[y] for y in ys])
    actions = np.asarray(actions.todense())
    #the first row is all zeros, because there is no prior action:
    last_action = np.zeros(actions.shape)
    last_action[1:,:] = actions[:-1,:]

    return onehot,np.concatenate([X_train,last_action],axis=1)


def predict_with_last_action(clf, X, onehot):
    """
        Predict one at a time and always add last action to 
        the next prediction using the onehot encoder.
    """
    lasty = np.zeros(6) #TODO: remove harding of 6
    y_predict = []
    for x in X:
        x_last_action = np.concatenate([x,lasty])
        y = clf.predict([x_last_action])
        y_predict.append(y[0])
        lasty = np.array(onehot.transform([y]).todense()).flatten()
        
    return y_predict

def run_clfs_on_data(classifiers, X_pers_all, y_pers_all):
    results = {}
    
    for name, clf in classifiers.iteritems():
        print "running %s" % name
        clf_results = fit_clf_kfold(clf['clf'], X_pers_all, y_pers_all,flatten=not clf['structured'])
        # with feature selection:
        #clf_results = fit_clf_kfold(clf['clf'], [X[:,select_features] for X in X_pers_all], y_pers_all,flatten=not clf['structured'])
        results[name] = clf_results
        
    return results
    

if __name__ == '__main__':
    print "loading data"
    X_train = np.loadtxt('../../UCI HAR Dataset/train/X_train.txt')
    y_train = np.loadtxt('../../UCI HAR Dataset/train/y_train.txt', dtype=np.int)
    persons_train = np.loadtxt('../../UCI HAR Dataset/train/subject_train.txt', dtype=np.int)
    X_test = np.loadtxt('../../UCI HAR Dataset/test/X_test.txt')
    y_test = np.loadtxt('../../UCI HAR Dataset/test/y_test.txt', dtype=np.int)
    persons_test = np.loadtxt('../../UCI HAR Dataset/test/subject_test.txt', dtype=np.int)
    
    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    
    
    features_selected = pickle.load(open('selected_features.pickle'))
    
    #SVM-HMM dumping
    #dump_svmlight_file(X_test,y_test,"/Users/tdomhan/Downloads/svm_hmm/activity-data/Xtest.data",zero_based=False,query_id=persons_test)
    #dump_svmlight_file(X_train,y_train,"/Users/tdomhan/Downloads/svm_hmm/activity-data/Xtrain.data",zero_based=False,query_id=persons_train)
    #laod the SVM-HMM predictions
    #y_predict = np.loadtxt("/Users/tdomhan/Downloads/svm_hmm/classfy.tag", dtype=int)
    
    feature_names = [x.split(' ')[1] for x in open('../../UCI HAR Dataset/features.txt').read().split('\n') if len(x) > 0]
    
    """ Split by person: """
    X_train_pers, y_train_pers = unflatten_per_person(X_train, y_train, persons_train)
    X_test_pers, y_test_pers = unflatten_per_person(X_test, y_test, persons_test)
    
    X_pers_all = []
    X_pers_all.extend(X_train_pers)
    X_pers_all.extend(X_test_pers)
    y_pers_all = []
    y_pers_all.extend(y_train_pers)
    y_pers_all.extend(y_test_pers)
    
    print "training classifier"
    
    ensemble_classifiers = {
                                "linear Support Vector Classifier": {'clf': LinearSVC(), 'structured': False},
                                "Logistic Regression": {'clf': LogisticRegression(), 'structured': False},
                                "KNN (weights: uniform, neighbors=5)": {'clf': KNeighborsClassifier(), 'structured': False},
                                "Decision Tree": {'clf': DecisionTreeClassifier(), 'structured': False},
                                "RandomForest": {'clf': RandomForestClassifier(), 'structured': False}
                                }
    crf_ensemble = LinearCRFEnsemble(ensemble_classifiers, addone=True, regularization=None, lmbd=0.01, sigma=100, transition_weighting=False)
    
    classifiers = {
                   "Logistic Regression": {'clf': LogisticRegression(), 'structured': False},
                   "Logistic Regression (l1 regularized)": {'clf': LogisticRegression(penalty='l1',C=100), 'structured': False},
                   "linear Support Vector Classifier": {'clf': LinearSVC(), 'structured': False},
                   "Gaussian Naive Bayes": {'clf': GaussianNB(), 'structured': False},
                   "SVMHMM": {'clf': SVMHMMCRF(C=1), 'structured': True},
                   "KNN (weights: uniform, neighbors=5)": {'clf': KNeighborsClassifier(), 'structured': False},
                   "Decision Tree": {'clf': DecisionTreeClassifier(), 'structured': False},
                   "RandomForest": {'clf': RandomForestClassifier(), 'structured': False},
                   }
    
    results = run_clfs_on_data(classifiers, X_pers_all, y_pers_all)
    
    for clf_name in results:
        clf_results = results[clf_name]
        accuracies = np.array([accuracy_score(gold, predict) for gold, predict in clf_results])
        print "%s accuracy: %f +- %f" % (clf_name, accuracies.mean(), accuracies.std())
        
        y_all_gold = np.concatenate(zip(*clf_results)[0])
        y_all_predict = np.concatenate(zip(*clf_results)[1])
        
    
    crf_classifiers_l2_best = {
                   "CRF (sigma=1)": {'clf': LinearCRF(feature_names=feature_names, label_names=labels, addone=True, regularization="l2", lmbd=0.01, sigma=1, transition_weighting=False),
                            'structured': True},
                   "CRF (sigma=10)": {'clf': LinearCRF(feature_names=feature_names, label_names=labels, addone=True, regularization="l2", lmbd=0.01, sigma=10, transition_weighting=False),
                            'structured': True},
                    "CRF (sigma=100)": {'clf': LinearCRF(feature_names=feature_names, label_names=labels, addone=True, regularization="l2", lmbd=0.01, sigma=100, transition_weighting=False),
                            'structured': True},
                    "CRF (sigma=1000)": {'clf': LinearCRF(feature_names=feature_names, label_names=labels, addone=True, regularization="l2", lmbd=0.01, sigma=1000, transition_weighting=False),
                            'structured': True},
                    "CRF (sigma=.1)": {'clf': LinearCRF(feature_names=feature_names, label_names=labels, addone=True, regularization="l2", lmbd=0.01, sigma=0.1, transition_weighting=False),
                            'structured': True},
                   }
    
    #results_feature_selection = run_clfs_on_data(classifiers, [X[:,features_selected] for X in X_pers_all], y_pers_all)
    
    
    #clf_results = fit_clf_kfold(clf, X_pers_all, y_pers_all,flatten=False)
    
    
    
    #sklearn
    #clf = svm.SVC()
    #clf = LogisticRegression()
    #clf = LogisticRegression(penalty='l1',C=100)
    #clf = SGDClassifier()
    #clf = GaussianNB()
    #clf = DecisionTreeClassifier()
    #clf = GradientBoostingClassifier()
    
    #clf = RandomForestClassifier()
    
    
    # diff features
#    diff_scaler = Scaler()
#    X_train_diff = get_diff_features(X_train)
#    
#    diff_scaler.fit(X_train_diff)
#    X_train_diff = diff_scaler.transform(X_train_diff)
#    X_train_diff = np.concatenate([X_train, X_train_diff],axis=1)
#    
#    X_test_diff  = get_diff_features(X_test)
#    X_test_diff = diff_scaler.transform(X_test_diff)
#    X_test_diff  = np.concatenate([X_test, X_test_diff],axis=1)


    #last action features
#    onehot,X_train_last_action = get_last_action_feature(X_train, y_train)
#    #clf = LinearSVC() # things get worse when adding the last action to a linear SVM classifier
#    #clf = svm.SVC(kernel='poly') # same is true for the poly kernel svm
#    #clf = svm.SVC()#and also with a rbf kernel
#    clf = RandomForestClassifier()
#    clf.fit(X_train_last_action, y_train)
#    
#    y_predict = predict_with_last_action(clf, X_test, onehot)
    
    
    #clf = LinearCRF(feature_names=feature_names, label_names=labels, addone=True, regularization=None, lmbd=0.01, sigma=100, transition_weighting=False)
    
    #a single chain:
    #clf.fit(X_train, y_train, X_test, y_test)
    #y_predict = clf.predict(X_test)
    #one chain per person
    
    #clf.batch_fit(X_train_pers, y_train_pers, X_test_pers, y_test_pers)
    #y_predict = np.concatenate(clf.batch_predict(X_test_pers))
    
#    #clf = LinearCRF(sigma=10)
#    X_train_svm, X_test_svm = SVM_feature_extraction(X_train, y_train, X_test)
#    clf = LinearCRF(addone=True,sigma=100)
#    #clf.fit(X_train, y_train, X_test, y_test)
#    
#    clf.fit(X_train_svm, y_train, X_test_svm, y_test)
#    y_predict = clf.predict(X_test_svm)
#    
    #clf.fit(X_train, y_train)  
    
    #print "predicting test data"
    
    #y_predict = clf.predict(X_test)
    
    #print classification_report(y_test, y_predict, target_names = labels)
    
    #print confusion_matrix_report(y_test, y_predict, labels)
    
    # measure the transitions we get right:
    
    
    print "done"
    
    
    