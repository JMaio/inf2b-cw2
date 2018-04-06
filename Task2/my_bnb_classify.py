import numpy as np
# ignore numpy division errors
# np.seterr(divide='ignore', invalid='ignore')

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
    # define total occurreces of each feature, use add-one smoothing
    total_occurs = Xtrn_b.sum(axis=0) + 1.0

    # feature-based class probability
    class_prob = np.empty((26, Xtrn_b.shape[1]))

    for c in range(26):
        # find occurrences of features for this class
        occurs = Xtrn_b[np.ravel(Ctrn == c), :].sum(axis=0)
        # set class probability to (class occurreces / total occurreces)
        class_prob[c] = occurs / total_occurs

    Cpreds = np.empty((Xtst_b.shape[0]), dtype=np.int_)

    for (i, v) in enumerate(Xtst_b):
        # print(v[:120])
        # print (1 - v) * np.log((1+1e-10) - class_prob)
        # print v * np.log(1e-10 + class_prob)
        # print("-----------------------------------------------")
        p = (1 - v) * np.log((1+1e-10) - class_prob) + v * np.log(1e-10 + class_prob)
        # print(p.shape)
        # print(p)
        m = p.sum(axis=1)
        # print(m.shape)
        # print(m)
        # print(m.argmax())
        Cpreds[i] = m.argmax()

    print(Cpreds)

    return Cpreds
