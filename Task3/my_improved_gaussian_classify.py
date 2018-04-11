import numpy as np
# import functions from classifier
from my_gaussian_classify import *
# import logdet function from coursework
from logdet import *


def my_improved_gaussian_classify(Xtrn, Ctrn, Xtst):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of labels for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data (dtype=np.float_)
    # Output:
    #  Cpreds : N-by-L ndarray of predicted labels for Xtst (dtype=np.int_)
# --------------------- similar to my_gaussian_classify --------------------- #
    ## Bayes classification with multivariate Gaussian distributions
    # define number of classes
    c_n = Ctrn.max() + 1
    # convert training classes into 1D array
    Ctrn_1d = Ctrn.ravel()
    # define number of features / dimensions
    d = Xtrn.shape[1]
    # create empty array to hold each class mean
    Ms = np.empty((c_n, d))
    # create empty array to hold each class covariance matrix
    Covs = np.empty((d, d, c_n))

    # start timer
    time.clock()

    # foreach class
    for c in range(c_n):
        # get mean (mu) for this class
        Xtrn_c = Xtrn[Ctrn_1d == c]
        # write mu to Ms
        Ms[c] = my_mean(Xtrn_c)
        # calculate covariance on class basis, (no regularisation)
        Covs[:, :, c] = my_cov(Xtrn_c, Ms[c])

    t_cov = time.clock()
    print("covariance matrices: %.2fs" % t_cov)

# ------------------------ begin improved classifier ------------------------ #
    eig_vals = np.empty((c_n, d), dtype=np.complex_)
    eig_vecs = np.empty((d, d, c_n), dtype=np.complex_)
    # apply PCA
    for c in range(c_n):
        eig_val, eig_vec = np.linalg.eig(Covs[:, :, c])
        # print(eig_val)
        # print(eig_vec.astype(np.float_))
        eig_vals[c] = eig_val
        eig_vecs[:, :, c] = eig_vec

    # print(np.argsort(eig_vals.max(axis=1)))
    # print(np.sort(eig_vals.max(axis=1)))

    # returns max eigenvalue position per class
    # print(np.abs(eig_vals).argmax(axis=1))
    # print(np.abs(eig_vals[0, :3]))
    # class max eigenvalues
    c_eigvec = np.empty((c_n, d), dtype=np.complex_)
    # get column / row of 2 max eigenvectors
    max_row_i = np.abs(eig_vals).argmax(axis=1)
    max_col_i = eig_vals[max_row_i, np.arange(eig_vals.shape[0])].argsort().ravel()[:-3:-1]
    print(eig_vals[max_row_i, np.arange(eig_vals.shape[0])])
    print(max_row_i)
    print(max_col_i)

    print(max_row_i[max_col_i])

    # (ax1, ax2) = eig_vecs[np.argsort(eig_vals.max(axis=1))[-2:].ravel()]
    print("axes:")
    print(e1)
    print(e2)

    return Cpreds
