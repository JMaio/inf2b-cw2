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
    Xtrn_b = np.empty(Xtrn.shape, dtype=np.int8) #
    Xtst_b = np.empty(Xtst.shape, dtype=np.int8) # store as byte to conserve memory

    Xtrn_b[Xtrn <= threshold], Xtrn_b[Xtrn > threshold] = 0, 1
    Xtst_b[Xtst <= threshold], Xtst_b[Xtst > threshold] = 0, 1

    # naive Bayes classification with multivariate Bernoulli distributions

    return Cpreds
