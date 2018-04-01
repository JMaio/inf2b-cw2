import numpy as np
from scipy import stats

from collections import Counter
import time


def my_sq_dist(Xtrn, Xtst):
    # optimise by expressing as matrix operarions
    #     (Xtrn-Xtst)^2 = Xtrn^2 + Xtst^2 - 2(Xtrn * Xtst)
    #     Xtrn^2 = np.sum(Xtrn ** 2, axis=1)[:, np.newaxis] # lengths of all training vectors
    #     Xtst^2 = np.sum(Xtst ** 2, axis=1)                # lengths of all test vectors
    #     Xtrn * Xtst = np.dot(Xtrn, Xtst.T)                # matrix product of sums of products

    return np.sum(np.square(Xtrn), axis=1)[:, np.newaxis] + np.sum(np.square(Xtst), axis=1) - 2 * np.dot(Xtrn, Xtst.T)


def my_knn_classify(Xtrn, Ctrn, Xtst, Ks):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of labels for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data (dtype=np.float_)
    #   Ks   : List of the numbers of nearest neighbours in Xtrn
    # Output:
    #  Cpreds : N-by-L ndarray of predicted labels for Xtst (dtype=np.int_)


    Cpreds = np.empty((0, Xtst.shape[0]), dtype=np.int_)   # set shape of prediction matrix

    d = np.argsort(my_sq_dist(Xtrn, Xtst).T, axis=1, kind='quicksort').T     # get minimum value indices as columns

    for (i, k) in enumerate(Ks):
        inds = d[:k]      # grab k nearest from columns

        preds = stats.mode(Ctrn[inds].reshape(k, Xtst.shape[0]))[0]

#              \         /                   \
#              /         \                   /
#      grab letter    transpose      transpose the result,
#      number from    and shed       reshape to allow for
#    closest vector    a layer     concatenation with output

        Cpreds = np.concatenate((Cpreds[:i], preds, Cpreds[i:]))

    return Cpreds
