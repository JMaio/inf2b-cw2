import numpy as np
import time


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
    # by transposing, D-by-D dimensions are maintained for this array
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
    # create empty array to hold each class covariance matrix
    d = Xtrn.shape[1]
    Ms = np.empty((26, d))
    Covs = np.empty((d, d, 26))

    # start independent timer for covariance calculation
    t = time.clock()
    for c in range(26):
        Covs[:, :, c] = my_cov(Xtrn[Ctrn_1d == c])

    print("covariance matrices: %.2fs" % (time.clock() - t))

    return (Cpreds, Ms, Covs)
