import numpy as np
# import functions from classifier
from my_gaussian_classify import *
# import logdet function from coursework
from logdet import *


def my_improved_gaussian_classify(Xtrn, Ctrn, Xtst, dims=None, epsilon=1e-10,
                                  epsilon_pca=1e-10):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of labels for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data (dtype=np.float_)
    # Output:
    #  Cpreds : N-by-L ndarray of predicted labels for Xtst (dtype=np.int_)

# ________________ custom logic for handling dimensionality _________________ #
    # define number of classes
    c_n = Ctrn.max() + 1

    if not dims:
        print("""
dims undefined: setting to max (dims=%d)
         """ % (c_n))
        dims = c_n
    elif dims > c_n:
        print("""
dims=%d: cannot go over %d dimensions!
         using dims=%d instead
         """ % (dims, c_n, c_n))
        dims = c_n
# ________________ section identical to my_gaussian_classify ________________ #
    ## Bayes classification with multivariate Gaussian distributions
    # convert training classes into 1D array
    Ctrn_1d = Ctrn.ravel()
    # define number of features / dimensions
    d = Xtrn.shape[1]
    # create empty array to hold each class mean
    Ms = np.empty((c_n, d))
    # create empty array to hold each class covariance matrix
    Covs = np.empty((d, d, c_n))

    # define epsilon as matrix
    epsil = np.identity(d) * epsilon

    # start timer
    time.clock()

    # foreach class
    for c in range(c_n):
        # get mean (mu) for this class
        Xtrn_c = Xtrn[Ctrn_1d == c]
        # write mu to Ms
        Ms[c] = my_mean(Xtrn_c)
        # calculate covariance on class basis, regularise with epsilon
        Covs[:, :, c] = my_cov(Xtrn_c, Ms[c]) + epsil

    t_cov = time.clock()
    print("covariance matrices (%dx%d)x%d: %.2fs" % (d, d, c_n, t_cov))

# ________________________ begin improved classifier ________________________ #

### ___________________ find eigenvalues & eigenvectors ___________________ ###
    eig_vals = np.empty((c_n, d), dtype=np.complex_)
    eig_vecs = np.empty((d, d, c_n), dtype=np.complex_)
    # apply PCA
    for c in range(c_n):
        # find class eigenvalues, eigenvectors
        eig_val, eig_vec = np.linalg.eig(Covs[:, :, c])
        eig_vals[c] = eig_val
        eig_vecs[:, :, c] = eig_vec

    # class max eigenvalues
    c_eigvec = np.empty((c_n, d), dtype=np.complex_)
    # get column & row of 2 max eigenvectors
    max_row_i = np.abs(eig_vals).argmax(axis=1)
    max_col_i = eig_vals[max_row_i, np.arange(eig_vals.shape[0])].argsort().ravel()[:-(dims + 1):-1]

    # set e1, e2 to eigenvectors with 1st and 2nd largest associated eigenvalues
    eig_vecs_pca = eig_vecs[:, max_row_i[max_col_i], max_col_i].T

    t_eig = time.clock()
    print("eigenvectors: %.2fs" % (t_eig - t_cov))

### ________________________ apply transformations ________________________ ###
    Xtrn_pca = Xtrn.dot(eig_vecs_pca.T)
    Xtst_pca = Xtst.dot(eig_vecs_pca.T)

    t_trans = time.clock()
    print("PCA transforms: %.2fs" % (t_trans - t_eig))

### __________________ continue with gaussian classifier __________________ ###
    # define epsilon_pca as matrix
    epsil_pca = np.identity(dims) * epsilon_pca
    # create dedicated array to hold each pca class mean
    Ms_pca = np.empty((c_n, dims), dtype=np.complex_)
    # create dedicated array to hold each pca class covariance matrix
    Covs_pca = np.empty((dims, dims, c_n), dtype=np.complex_)
    # define log posterior probabilities
    log_pps_pca = np.empty((Xtst_pca.shape[0], c_n), dtype=np.complex_)

    # foreach class
    for c in range(c_n):
        ### recompute class mean
        # get mean (mu) for this class
        Xtrn_pca_c = Xtrn_pca[Ctrn_1d == c]
        # write mu to Ms_pca
        Ms_pca[c] = my_mean(Xtrn_pca_c)
        # recalculate covariance matrix
        Covs_pca[:, :, c] = my_cov(Xtrn_pca_c, Ms_pca[c]) + epsil_pca

    t_cov_pca = time.clock()
    print("covariance matrices (%dx%d)x%d: %.2fs" % (dims, dims, c_n, t_cov_pca - t_trans))

    # foreach class
    for c in range(c_n):
        # calculate log determinant of covariance matrix
        cov_logdet_pca = logdet(Covs_pca[:, :, c])
        # calculate inverse of covariance matrix
        cov_inv_pca = np.linalg.inv(Covs_pca[:, :, c])
        # subtract mean from test vectors
        m_pca = (Xtst_pca - Ms_pca[c])
        # foreach testing vector from above (for loop uses less memory and
        # does not need to calculate unnecessary off-diagonal values --> faster)
        for (i, v) in enumerate(m_pca):
            # ignoring "+ ln P(C)" assuming uniform prior distribution
            log_pps_pca[i, c] = - 0.5 * (v.dot(cov_inv_pca.dot(v.T)) + cov_logdet_pca)
        # give feedback on class calculation (pretty printing included)
        s = ["calculating classes: # %d",
             "                     # %d"]
        print(s[int(bool(c))] % c)
    t_classes_pca = time.clock()

    print("classes: %.2fs" % (t_classes_pca - t_cov_pca))

    Cpreds = log_pps_pca.argmax(axis=1)

    # Cpreds = np.abs(Xtst_pca).argmax(axis=1).ravel()

    return Cpreds
