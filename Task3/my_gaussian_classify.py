import numpy as np
import time


def my_mean(Xtrn):
    # number of features
    N = Xtrn.shape[0]
    # sum over all components of each feature
    sum_N = np.sum(Xtrn, axis=0)
    # divide totals by number of features
    return sum_N / N


def my_cov(Xtrn_c):
    # get mean (mu) for this class
    mu = my_mean(Xtrn_c)
    # subtract mean from training data
    m = (Xtrn_c - mu)
    # matrix multiply (m x m.T) to calculate covariance matrix
    return np.dot(m, m.T)


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
    Covs = np.empty((Xtrn.shape[0] / 26, 26))

    for c in range(26):
        pass



    # print(Covs)
    # print(Covs.shape)



    return (Cpreds, Ms, Covs)
