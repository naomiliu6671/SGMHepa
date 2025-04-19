from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
import numpy as np


def evaluation(pred, pred_, label):
    # pred, pred_ = pred.tolist()[0], pred_.tolist()[0]
    macro_precision = precision_score(label, pred, zero_division=0)
    macro_recall = recall_score(label, pred, zero_division=0)
    f1 = f1_score(label, pred, average='binary', zero_division=0)
    accuracy = accuracy_score(label, pred)
    fpr, tpr, _ = roc_curve(label, pred_[:, 1], pos_label=1)
    auc_score = auc(fpr, tpr)
    cm = confusion_matrix(label, pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    mcc = matthews_corrcoef(label, pred)
    return macro_precision, macro_recall, f1, accuracy, auc_score, specificity, mcc


def mean_std(data_list):
    return str(f'{np.mean(data_list):.3f}') + '±' + str(f'{np.std(data_list):.3f}')
