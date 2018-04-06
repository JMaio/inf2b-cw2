from sklearn.naive_bayes import BernoulliNB
import numpy as np

def bnb_check_classify(Xtrn, Ctrn, Xtst, threshold):
    clf = BernoulliNB(alpha=1, binarize=threshold)
    clf.fit(Xtrn, Ctrn.ravel())

    return (clf.feature_log_prob_, clf.predict(Xtst))
