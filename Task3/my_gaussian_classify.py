import numpy as np


def my_mean(Xtrn):
    # number of features
    N = Xtrn.shape[0]
    # sum over all components of each feature
    sum_N = np.sum(Xtrn, axis=0)
    # divide totals by number of features
    return sum_N / N


def my_cov(Xtrn):
    mu = my_mean(Xtrn)
    # m = (Xtrn - mu)
    # print(m.shape)
    # n = m[:10]
    # # return np.dot(m, m.T)
    # return np.dot(n, n.T)


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

    #YourCode - Bayes classification with multivariate Gaussian distributions.
    # convert training classes into 1D array
    Ctrn_1d = Ctrn.ravel()
    # create empty array to hold each class covariance matrix
    Covs = np.empty((Xtrn.shape[0] / 26, 26))

    for c in range(26):
        pass

    cov = my_cov(Xtrn)
    print(cov)
    print(cov.shape)



    return (Cpreds, Ms, Covs)
