import anndata as ad
import scanpy as sc
import pandas as pd
import SpatialDE
import numpy as np
from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt
from sklearn import preprocessing
from allendigger.accessdata import get_data
from numpy import array
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier


def find_overlap_gene(spatial, scrna, time):
    '''
    find the overlap gene between spatial and scrna data
    :param spatial: spatial adata
    :param scrna: scrna adata
    :param time:the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult']
    :return:a list of overlap gene
    '''
    if time == 'Adult':
        overlap_gene = []
        for gene in scrna.var['gene'].tolist():
            if gene in spatial.var['gene'].tolist():
                overlap_gene.append(gene)
    else:
        overlap_gene = []
        for gene in scrna.var['gene'].tolist():
            gene = gene.upper()
            if gene in spatial.var['gene'].tolist():
                overlap_gene.append(gene)
    return overlap_gene


def find_highly_variable_gene(
        adata: ad.AnnData,
        section: str,
        section_id: int):
    '''
    Return
    ------
    spatial variable genes

    '''

    if not type(section_id) == int:
        raise TypeError('section id should be an integer')
    time_list = ['E11pt5', 'E13pt5', 'E15pt5', 'E18pt5', 'P4', 'P14', 'P28', 'P56']
    section_list = ['sagittal', 'horizontal', 'coronal']

    if not (section in section_list):
        raise ValueError("inputed section_is not avalible, avalible sections are" + '\n' +
                         "'sagittal','horizontal','coronal'")
    section_dict = {'sagittal': 'z', 'horizontal': 'y', 'coronal': 'x'}
    mat = adata.X.astype(np.float32)
    if section == 'sagittal':
        frame = {'X': adata.obs['x'], 'Y': adata.obs['y']}
        df = pd.DataFrame(frame)
        mat_df = pd.DataFrame(mat)
        mat_df.index = adata.obs_names
        mat_df.columns = adata.var
    if section == 'horizontal':
        frame = {'X': adata.obs['x'], 'Z': adata.obs['z']}
        df = pd.DataFrame(frame)
        mat_df = pd.DataFrame(mat)
        mat_df.index = adata.obs_names
        mat_df.columns = adata.var
    if section == 'coronal':
        frame = {'Y': adata.obs['y'], 'Z': adata.obs['z']}
        df = pd.DataFrame(frame)
        mat_df = pd.DataFrame(mat)
        mat_df.index = adata.obs_names
        mat_df.columns = adata.var

    spatial_var_gene = SpatialDE.run(df, pd.DataFrame(mat_df))
    # spatial_var_gene = spatial_var_gene.sort_values('qval')['g'][0:200]

    return spatial_var_gene



def accuracy(y_true, y_pred):
    # Intitializing variable to store count of correctly predicted classes
    correct_predictions = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == yp:
            correct_predictions += 1

    # returns accuracy
    return correct_predictions / len(y_true)


def true_positive(y_true, y_pred):
    tp = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 1 and yp == 1:
            tp += 1

    return tp


def true_negative(y_true, y_pred):
    tn = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 0 and yp == 0:
            tn += 1

    return tn


def false_positive(y_true, y_pred):
    fp = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 0 and yp == 1:
            fp += 1

    return fp


def false_negative(y_true, y_pred):
    fn = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 1 and yp == 0:
            fn += 1

    return fn


def macro_precision(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize precision to 0
    precision = 0

    # loop over all classes
    for class_ in list(y_true.unique()):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)

        # compute false positive for current class
        fp = false_positive(temp_true, temp_pred)

        # compute precision for current class
        temp_precision = tp / (tp + fp + 1e-6)
        # keep adding precision for all classes
        precision += temp_precision

    # calculate and return average precision over all classes
    precision /= num_classes

    return precision


def micro_precision(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize tp and fp to 0
    tp = 0
    fp = 0

    # loop over all classes
    for class_ in y_true.unique():
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        # and update overall tp
        tp += true_positive(temp_true, temp_pred)

        # calculate false positive for current class
        # and update overall tp
        fp += false_positive(temp_true, temp_pred)

    # calculate and return overall precision
    precision = tp / (tp + fp)
    return precision


def macro_recall(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize recall to 0
    recall = 0

    # loop over all classes
    for class_ in list(y_true.unique()):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)

        # compute false negative for current class
        fn = false_negative(temp_true, temp_pred)

        # compute recall for current class
        temp_recall = tp / (tp + fn + 1e-6)

        # keep adding recall for all classes
        recall += temp_recall

    # calculate and return average recall over all classes
    recall /= num_classes

    return recall


def micro_recall(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize tp and fp to 0
    tp = 0
    fn = 0

    # loop over all classes
    for class_ in y_true.unique():
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        # and update overall tp
        tp += true_positive(temp_true, temp_pred)

        # calculate false negative for current class
        # and update overall tp
        fn += false_negative(temp_true, temp_pred)

    # calculate and return overall recall
    recall = tp / (tp + fn)
    return recall


def macro_f1(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize f1 to 0
    f1 = 0

    # loop over all classes
    for class_ in list(y_true.unique()):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)

        # compute false negative for current class
        fn = false_negative(temp_true, temp_pred)

        # compute false positive for current class
        fp = false_positive(temp_true, temp_pred)

        # compute recall for current class
        temp_recall = tp / (tp + fn + 1e-6)

        # compute precision for current class
        temp_precision = tp / (tp + fp + 1e-6)

        temp_f1 = 2 * temp_precision * temp_recall / (temp_precision + temp_recall + 1e-6)

        # keep adding f1 score for all classes
        f1 += temp_f1

    # calculate and return average f1 score over all classes
    f1 /= num_classes

    return f1


def micro_f1(y_true, y_pred):


    #micro-averaged precision score
    P = micro_precision(y_true, y_pred)

    #micro-averaged recall score
    R = micro_recall(y_true, y_pred)

    #micro averaged f1 score
    f1 = 2*P*R / (P + R)

    return f1


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict

