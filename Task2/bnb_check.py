from sklearn.naive_bayes import BernoulliNB


def bnb_check_classify(Xtrn, Ctrn, Xtst, threshold):
    clf = BernoulliNB(alpha=0, binarize=threshold)
    clf.fit(Xtrn, Ctrn.ravel())

    return (clf.feature_count_, clf.predict(Xtst))