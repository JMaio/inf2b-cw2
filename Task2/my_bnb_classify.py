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
    total_occurs = Xtrn_b.sum(axis=0)

    prior = 1.0 / 26

    # feature-based class count
    class_count = np.zeros((26, Xtrn_b.shape[1]), dtype=np.float_)
    # feature-based class probability
    log_class_prob = np.zeros((26, Xtrn_b.shape[1]), dtype=np.float_)

    # convert training classes into 1D array
    Ctrn_1d = Ctrn.ravel()

    for c in range(26):
        # find occurrences of features for this class
        class_count[c] = Xtrn_b[Ctrn_1d == c, :].sum(axis=0)

    # set class probability to class occurrences / total occurrences)
    log_class_prob = np.log(class_count + 1.0) - np.log(total_occurs + 26.0)
    # (here, class probability is Laplace smoothed)

    print(log_class_prob)

    # class_prob *= prior
    Cpreds = np.zeros((Xtst_b.shape[0]), dtype=np.int_)
    # Cpreds = np.zeros((Xtst_b.shape[0], 26), dtype=np.int_)

    # log_class_prob = np.log(class_prob)
    # print(log_class_prob[0,:8])
    # print((1 - class_prob[0,:8]))

    for (i, v) in enumerate(Xtst_b):

        neg_p = np.log(1 - class_prob)

        p0 = np.dot(v, (log_class_prob -  neg_p).T)
        p1 = neg_p.sum(axis=1)

        p = p0 + p1 + np.log(prior)

        Cpreds[i] = p.argmax()

    print(Cpreds)

    return (log_class_prob, Cpreds)
