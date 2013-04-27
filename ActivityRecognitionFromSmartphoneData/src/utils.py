'''
Created on Apr 22, 2013

@author: tdomhan
'''

from sklearn.metrics import confusion_matrix

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
    
    
    
    