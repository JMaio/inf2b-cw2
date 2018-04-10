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
    # define number of dimensions
    d = Xtrn.shape[1]
    # create empty array to hold each test vector prediction
    Cpreds = np.empty((Xtst.shape[0]), dtype=np.int_)
    # create empty array to hold each class mean
    Ms = np.empty((26, d))
    # create empty array to hold each class covariance matrix
    Covs = np.empty((d, d, 26))
    # create empty array to hold each class covariance matrix inverse
    Cov_invs = np.empty((d, d, 26))
    # create empty array to hold each class covariance matrix log determinant
    Cov_logdets = np.empty((26))

    # start independent timer for covariance calculation
    t = time.clock()
    for c in range(26):
        # get mean (mu) for this class
        Xtrn_c = Xtrn[Ctrn_1d == c]
        # write mu to Ms
        Ms[c] = my_mean(Xtrn_c)
        # calculate covariance on class basis, regularise with epsilon
        Covs[:, :, c] = my_cov(Xtrn_c, Ms[c]) + np.identity(d) * epsilon

    print("covariance matrices: %.2fs" % (time.clock() - t))

    # define log posterior probabilities
    log_pps = np.empty((Xtst.shape[0], 26))

    for c in range(26):
        # calculate covariance matrix log determinant
        cov_logdet = logdet(Covs[:, :, c])
        # calculate covariance matrix inverse
        cov_inv = np.linalg.inv(Covs[:, :, c])
        # subtract mean from test vectors
        m = (Xtst - Ms[c])
        # ignoring "+ ln P(C)" assuming uniform prior distribution
        log_pp = - 0.5 * (m.dot(cov_inv.dot(m.T)) - cov_logdet)
        # for (i, v) in enumerate(Xtst):
        #     log_pps[i, c] =
        log_pps[:, c] = log_pp.diagonal().ravel()
        # print(log_pps[:, c])
        print("calculated class #%2d" % c)
    Cpreds = log_pps.argsort(axis=1)[:, -1]
    return (Cpreds, Ms, Covs)
