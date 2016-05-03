# -*- coding:utf-8 -*-

from sklearn.metrics import confusion_matrix

def precision_recall(true_label, predict_label):

    [[TP,FP],[FN,TN]] = confusion_matrix(true_label, predict_label)

    accuracy = float(TP + TN) / float(TP + FP + FN + TN)
    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f = 2.0 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f

