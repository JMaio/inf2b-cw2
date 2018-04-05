import numpy as np

def my_bnb_classify(Xtrn, Ctrn, Xtst, threshold):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of label vector for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data matrix (dtype=np.float_)
    #   threshold   : A scalar threshold (type=float)
    # Output:
    #  Cpreds : N-by-L ndarray of predicted labels for Xtst (dtype=np.int_)

    # binarisation of Xtrn and Xtst.
    Xtrn_b = np.zeros(Xtrn.shape, dtype=np.int8) #
    Xtst_b = np.zeros(Xtst.shape, dtype=np.int8) # store as byte to conserve memory

    Xtrn_b[Xtrn > threshold] = 1
    Xtst_b[Xtst > threshold] = 1

    # naive Bayes classification with multivariate Bernoulli distributions
    # print(Ctrn.shape)

    # ck_mat = np.tile(Ctrn, Xtrn.shape[1])     # repeat class matrix to match Xtrn shape
    # print(ck_mat.shape)
    # print(ck_mat)
    # print Xtrn_b.shape
    total_occurs = Xtrn_b.sum(axis=0)   # define total occurreces of each feature
    class_prob = np.empty((26, Xtrn_b.shape[1])) # feature-based class probability
    probs = np.empty((Xtst_b.shape[0], 26)) # final class probability of each test vector
    prior = 1.0 / 26                      # assume uniform prior distribution
    # print total_occurs.shape

    for k in range(26):
        # find occurrences of features for this class
        occurs = np.where(Ctrn == k, Xtrn_b, 0).sum(axis=0) # mask occurreces
        # set class probability to (class occurreces / total occurreces)
        class_prob[k] = np.true_divide(occurs, np.where(total_occurs > 0, total_occurs, 1))
        # -------------------------------- (prevent division by zero by substituting by ones)

    return Cpreds
    x = Xtst_b[:, :, np.newaxis]        # add third dimension
    c = class_prob.T[np.newaxis, :, :]  # prepend axis to create 3d array
    p = x * c
    p = (1 - c)**(1 - x) * (c)**x   # 7800x784x26 array of class likelihoods using given formula
    Cpreds = np.argmax(p.prod(axis=1), axis=1) # take product of probabilities, find which class has max probability
