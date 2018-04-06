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
    # print(total_occurs.shape)
    class_prob = np.empty((26, Xtrn_b.shape[1])) # feature-based class probability
    # probs = np.empty((Xtst_b.shape[0], 26)) # final class probability of each test vector
    # class_likelihood = np.empty((26, 1))
    # assume uniform prior distribution
    # print total_occurs.shape

    # print Ctrn.shape
    # print (Ctrn == 4).flatten()
    #
    # print(Xtrn_b[np.ravel(Ctrn == 4), :])

    for k in range(26):
        # find occurrences of features for this class
        occurs = Xtrn_b[np.ravel(Ctrn == k), :].sum(axis=0) # mask occurreces
        # set class probability to (class occurreces / total occurreces)
        # print(np.true_divide(0,1))
        # cond = np.logical_or(, occurs == 0)
        # print(occurs[:6])
        # print(total_occurs[:6])

        class_prob[k] = occurs / total_occurs
        # feature_prob = np.where(
        #     total_occurs > 0,
        #     occurs / total_occurs,
        #     1e-10)

        # print(feature_prob)
        # print("-----------------------------------------------")

        # feature_prob[feature_prob == 0] = 1e-10

        # class_prob[k] = feature_prob
        # print class_prob[k]


    Cpreds = np.empty((1, Xtst_b.shape[0]), dtype=np.int_)

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
        # print(m.argmax())
        Cpreds[0, i] = m.argmax()

    print(np.ravel(Cpreds))
    # x = Xtst_b[:, :, np.newaxis]        # add third dimension
    # c = class_prob.T[np.newaxis, :, :]  # prepend axis to create 3d array
    #
    # p0 = (1 - c)**(1 - x)
    # p1 = (c**x)
    # # p0 = (1 - x) * np.log(1 - c)
    # # p1 = x * np.log(c)
    #
    # # replace zero probabilities with "small" number
    # p0 = np.where(p0 <= 1e-10, 1e-10, p0)
    # p1 = np.where(p1 <= 1e-10, 1e-10, p1)
    #
    # p0_log = np.log(p0) # np.where(p0 < 1e-10, np.log(1e-10), np.log(p0))
    # p1_log = np.log(p1) # np.where(p1 < 1e-10, np.log(1e-10), np.log(p1))
    # #
    # # print(np.log(1e-10))
    #
    # # p = p0 + p1 # 7800x784x26 log array of class likelihoods using given formula
    # p = p0_log + p1_log # 7800x784x26 log array of class likelihoods using given formula
    #
    # s = p.sum(axis=1)
    #
    # Cpreds = np.argmax(s, axis=1) # take product of probabilities, find which class has max probability

    return np.ravel(Cpreds)
