from sklearn.metrics import confusion_matrix
import numpy as np

def confusionMatrix(y_test, y_pred):
    """
    Returns confusion matrix accoring to standard format
    """
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    return np.array([[TP, FN],
                     [FP, TN]])