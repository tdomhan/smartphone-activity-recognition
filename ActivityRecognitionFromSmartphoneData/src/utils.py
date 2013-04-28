'''
Created on Apr 22, 2013

@author: tdomhan
'''

from sklearn.metrics import confusion_matrix

from sklearn import cross_validation

import numpy as np

labels = ["Walking", "Walking_upstairs", "Walking_downstairs", "Sitting", "Standing", "Laying"]

def draw_table(d,labels):
    import texttable
    table = texttable.Texttable()
    header = [""] + labels
    table.header(header)
    d = [[labels[i]]+list(row) for i,row in enumerate(d)]
    #first row:
    d.insert(0, header)
    table.add_rows(d)
    print table.draw()

def confusion_matrix_report(y_test, y_predict, labels):
    conf_matrix = confusion_matrix(y_test, y_predict)
    draw_table(conf_matrix,labels)
    
    
def num_label_changes(y):
    """
        For a label sequence this function calculates the number of times the label changes.
        e.g. num_label_changes([1,1,1,2,2,2,3,3]) = 2
    """
    num_changes = 0
    for y, y_next in zip(y, y[1:]):
        if y != y_next:
            num_changes += 1
    return num_changes
    
def label_smoothness(y_predict):
    n = len(y_predict)
    num_changes_predict = num_label_changes(y_predict)
    return num_changes_predict / float(n)
    
    
def unflatten_per_person(X_all,y_all,persons_all):
    """
        X: n_samples, n_features
            The full feature matrix.
        y: label for each row in X
        person: person label for each row in X
        
        returns: (X_person, y_person) 
            X_person: n_persons array of X and y that apply to this person.
    """
    Xtotal = []
    y_total = []
    
    Xperson = []
    y_person = []
    last_person = persons_all[0]
    for row,y,person in zip(X_all,y_all,persons_all):
        if person != last_person:
            Xtotal.append(Xperson)
            y_total.append(y_person)
            Xperson = []
            y_person = []
            
        Xperson.append(row)
        y_person.append(y)
        
        last_person = person
        
    Xtotal.append(Xperson)
    y_total.append(y_person)
    
    return ([np.array(x) for x in Xtotal], [np.array(y) for y in y_total])

def flatten_data(X,y):
    return np.concatenate(X), np.concatenate(y)

def fit_clf_kfold(clf,Xs,ys,flatten=True,n_folds=5):
    """
    X: an array of X, one for each person
    y: an array of array of labels, one for each person
    flatten: set to True for classifiers that don't take the structure into account.
    
    return:
    (y_gold, y_predict) for each fold
    """
    
    result = []
    
    kf = cross_validation.KFold(len(Xs), n_folds=n_folds, shuffle=True, random_state=1)
    for i,(train_index, test_index) in enumerate(kf):
        print "fold %d" % i
        X_train = [Xs[u] for u in train_index]
        y_train = [ys[u] for u in train_index]
        X_test = [Xs[u] for u in test_index]
        y_test = [ys[u] for u in test_index]
        if flatten:
            X_train, y_train = flatten_data(X_train, y_train)
            X_test, y_test = flatten_data(X_test, y_test)
            clf.fit(X_train,y_train)
            y_predict = clf.predict(X_test)
            y_gold = y_test
        else:
            clf.batch_fit(X_train,y_train)
            y_predict = clf.batch_predict(X_test)
            y_predict = np.concatenate(y_predict)
            y_gold = np.concatenate(y_test)
        
        result.append((y_gold,y_predict))
    return result
    
    
    