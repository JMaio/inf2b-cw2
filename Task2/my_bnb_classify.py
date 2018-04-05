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
    # class_likelihood = np.empty((26, 1))
    # assume uniform prior distribution
    # print total_occurs.shape

    for k in range(26):
        # find occurrences of features for this class
        occurs = np.where(Ctrn == k, Xtrn_b, 0).sum(axis=0) # mask occurreces
        # set class probability to (class occurreces / total occurreces)
        class_prob[k] = np.true_divide(occurs, np.where(total_occurs > 0, total_occurs, 1))
        # -------------------------------- (prevent division by zero by substituting by ones)
        # mask multiplication form P(bi=0|Ck)^(1-bi) * P(bi=1|Ck)^(bi)
        # ps = np.where(class_prob[k] > 0, class_prob[k], 1)  # workaround `div 0`
        # # take product of resultant individual probabilities
        # print(ps)
        # print(np.product(ps))

    # replace zero probabilities with "small" number
    # class_prob[class_prob == 0] = 1e-10

    # print(np.min(class_prob))

    x = Xtst_b[:, :, np.newaxis]        # add third dimension
    c = class_prob.T[np.newaxis, :, :]  # prepend axis to create 3d array
    # print(np.where(x[] == 0, c, 1 - c))
    # p = x * c  # 7800x784x26 array of class likelihoods
    # t1 = (1 - c)**(1 - x)
    # print(p.min(), p.max())
    # print(t1[:, 197, :].shape)
    # print(t1[:, 197, :].sum())
    # print(c[:, 200:204, :].shape)
    # print(c[:, 200:204, :].sum(axis=1))
    #
    # print(x[:, 200:204, :].shape)
    # print(x[:, 200:204, :])
    #
    # t2 = c[:, 200:204, :]**x[:, 200:204, :] * (1 - c[:, 200:204, :])**(1 - x[:, 200:204, :])
    # print(t2.shape)
    # print(t2)

    # Cpreds = np.empty((Xtst.shape[0]), dtype=np.int8)
    #
    # print class_prob
    #
    # for (i, v) in enumerate(Xtst_b):
    #     # print np.min((1 - class_prob)**(1 - v))
    #     # print class_prob**v
    #     ps = (1 - class_prob)**(1 - v) * class_prob**v
    #     # print ps.min(axis=1)
    #     # print(ps.prod(axis=1))
    #     Cpreds[i] = np.argmax(ps.prod(axis=0))
    #
    #
    # print Xtst_b
    #
    # print Cpreds.shape
    # print Cpreds

    p0 = (1 - c)**(1 - x)
    p1 = (c**x)

    # replace zero probabilities with "small" number
    p0 = np.where(p0 <= 1e-10, 1e-10, p0)
    p1 = np.where(p1 <= 1e-10, 1e-10, p1)

    p0_log = np.log(p0) # np.where(p0 < 1e-10, np.log(1e-10), np.log(p0))
    p1_log = np.log(p1) # np.where(p1 < 1e-10, np.log(1e-10), np.log(p1))
    #
    # print(np.log(1e-10))

    p = p0_log + p1_log # 7800x784x26 log array of class likelihoods using given formula
    # replace zero probabilities with "small" number
    # p = np.where(p == 0, np.log(1e-10), np.log(p))

    # print(np.sum(p[0][17120/26]))
    Cpreds = np.argmax(p.prod(axis=1), axis=1) # take product of probabilities, find which class has max probability
    # print(np.min(p))
    # pp = p.sum(axis=1)[:4]
    # print(p.shape)
    # print(np.min(pp))
    # print(pp)

    # pp = p.prod(axis=1)
    s = p.sum(axis=1)
    # print(s.shape)
    # print(s)

    Cpreds = np.argmax(s, axis=1) # take product of probabilities, find which class has max probability
    # print(np.max(p.prod(axis=1), axis=1))
    # print(Xtst_b[:,].shape, class_prob.shape)

    # print(p)
    # print(p.shape)
    # print(p.sum(axis=0))
    # print(p.sum(axis=1))
    # print(p.sum(axis=2))
    # probs[:,] = np.product(, axis=0)
    # print(probs)

    # print(np.sum(class_prob, axis=0).reshape((28,28)))

        # print probs[k][300:310]
        # probs
    # print class_prob[12][300:310]

    # print(Xtrn[Xtrn == 1].shape)
    # print(Xtrn[Xtrn == 1])
    # print(Xtrn.shape)
    # print(Xtrn)

    return Cpreds
