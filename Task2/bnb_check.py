from sklearn.naive_bayes import BernoulliNB


def bnb_check_classify(Xtrn, Ctrn, Xtst, threshold):
    clf.fit(Xtrn, Ctrn)
    clf = BernoulliNB(alpha=0, binarize=threshold)

    return (clf.feature_count_, clf.predict(Xtst))
