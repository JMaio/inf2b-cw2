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

    ## naive Bayes classification with multivariate Bernoulli distributions
    # feature-based class probability
    class_prob = np.zeros((26, Xtrn_b.shape[1]), dtype=np.float_)

    # convert training classes into 1D array
    Ctrn_1d = Ctrn.ravel()

    for c in range(26):
        # find occurrences of this class
        class_prob[c] = np.true_divide(Xtrn_b[Ctrn_1d == c, :].sum(axis=0),
                                       Ctrn[Ctrn_1d == c].shape[0])

    Cpreds = np.zeros((Xtst_b.shape[0]), dtype=np.int_)

    for (i, v) in enumerate(Xtst_b):



    print(Cpreds)

    return (log_class_prob, Cpreds)
