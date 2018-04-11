import numpy as np
import time
# import logdet function from coursework
from logdet import *


def my_mean(Xtrn_c):
    # number of features
    N = Xtrn_c.shape[0]
    # sum over all components of each feature
    sum_N = np.sum(Xtrn_c, axis=0)
    # divide totals by number of features
    return sum_N / N


def my_cov(Xtrn_c, mu):
    # subtract mean from training data
    m = (Xtrn_c - mu)
    # matrix multiply (m.T x m) and divide by occurrences to calculate covariance matrix
    cov = np.dot(m.T, m) / Xtrn_c.shape[0]
    # (by transposing, D-by-D dimensions are maintained for this array)
    return cov


def my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of label vector for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data matrix (dtype=np.float_)
    #   epsilon   : A scalar parameter for regularisation (type=float)
    # Output:
    #  Cpreds : N-by-L ndarray of predicted labels for Xtst (dtype=int_)
    #  Ms    : D-by-K ndarray of mean vectors (dtype=np.float_)
    #  Covs  : D-by-D-by-K ndarray of covariance matrices (dtype=np.float_)

    # Bayes classification with multivariate Gaussian distributions.
    # convert training classes into 1D array
    Ctrn_1d = Ctrn.ravel()
    # define number of features / dimensions
    d = Xtrn.shape[1]
    # create empty array to hold each class mean
    Ms = np.empty((26, d))
    # create empty array to hold each class covariance matrix
    Covs = np.empty((d, d, 26))

    # foreach class
    for c in range(26):
        # get mean (mu) for this class
        Xtrn_c = Xtrn[Ctrn_1d == c]
        # write mu to Ms
        Ms[c] = my_mean(Xtrn_c)
        # calculate covariance on class basis, regularise with epsilon
        Covs[:, :, c] = my_cov(Xtrn_c, Ms[c]) + np.identity(d) * epsilon

    print("covariance matrices: %.2fs" % time.clock())

    # define log posterior probabilities
    log_pps = np.empty((Xtst.shape[0], 26))

    # foreach class
    for c in range(26):
        # calculate log determinant of covariance matrix
        cov_logdet = logdet(Covs[:, :, c])
        # calculate inverse of covariance matrix
        cov_inv = np.linalg.inv(Covs[:, :, c])
        # subtract mean from test vectors
        m = (Xtst - Ms[c])
        for (i, v) in enumerate(m):
            # ignoring "+ ln P(C)" assuming uniform prior distribution
            log_pps[i, c] = - 0.5 * (v.dot(cov_inv.dot(v.T)) + cov_logdet)
        # give feedback on class calculation (pretty printing included)
        s = ["calculating classes: # %d",
             "                     # %d"]
        print(s[int(bool(c))] % c)

    Cpreds = log_pps.argmax(axis=1)

    return (Cpreds, Ms, Covs)
