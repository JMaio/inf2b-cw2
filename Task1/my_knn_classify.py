import numpy as np
from scipy import stats
import time


def my_sq_dist(Xtrn, Xtst):
    # optimise by expressing as matrix operarions
    #     (Xtrn-Xtst)^2 = Xtrn^2 + Xtst^2 - 2(Xtrn * Xtst)
    #     Xtrn^2 = np.sum(Xtrn ** 2, axis=1)[:, np.newaxis] # lengths of all training vectors
    #     Xtst^2 = np.sum(Xtst ** 2, axis=1)                # lengths of all test vectors
    #     Xtrn * Xtst = np.dot(Xtrn, Xtst.T)                # matrix product of sums of products

    t = time.clock()
    m = np.sum(Xtst**2, axis=1)[:, np.newaxis] \
      + np.sum(Xtrn**2, axis=1) \
      - 2 * np.dot(Xtst, Xtrn.T)
    print("distances: %.2fs" % (time.clock() - t))
    return m


def my_knn_classify(Xtrn, Ctrn, Xtst, Ks):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of labels for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data (dtype=np.float_)
    #   Ks   : List of the numbers of nearest neighbours in Xtrn
    # Output:
    #  Cpreds : N-by-L ndarray of predicted labels for Xtst (dtype=np.int_)

    # set shape of prediction matrix
    Cpreds = np.empty((len(Ks), Xtst.shape[0]), dtype=np.int_)

    # get minimum value indices as columns
    d = my_sq_dist(Xtrn, Xtst)
    t = time.clock()
    # d = np.argpartition(d, max(Ks), axis=0)
    d = np.argsort(d, kind='quicksort')
    print("sort: %.2fs" % (time.clock() - t))

    # foreach in Ks, calculate k nearest neighbour classification
    for (i, k) in enumerate(Ks):
        # grab k nearest from columns
        inds = d[:,:k].T
        # calculate most likely class from mode of closest k
        preds = stats.mode(Ctrn[inds].reshape(k, Xtst.shape[0]))[0]
        # concatenate to output
        Cpreds[i] = preds

    return Cpreds
