#! /usr/bin/env python
# A sample template for my_improved_gaussian_system.py

import numpy as np
import scipy.io
import time
from my_improved_gaussian_classify import *
from my_confusion import *

# ________________ parse arguments to facilitate experiments ________________ #
import argparse
parser = argparse.ArgumentParser(description='run improved variant of my_gaussian_classify.')
parser.add_argument('-e', default=-2, type=int,
                    help='select which experiment number to run')
args = parser.parse_args()
# ___________________________ begin actual system ___________________________ #
# Load the data set
filename = "/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1621503/data.mat";
# use local data set while not connected to afs
try:
    data = scipy.io.loadmat(filename)
    print("loaded data from afs!")
except Exception:
    data = scipy.io.loadmat("../data.mat")
    print("couldn't load data from afs! loading local data...")

# Feature vectors: Convert uint8 to double, and divide by 255
Xtrn = data['dataset']['train'][0, 0]['images'][0, 0].astype(dtype=np.float_) / 255.0
Xtst = data['dataset']['test'][0, 0]['images'][0, 0].astype(dtype=np.float_) / 255.0
# Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1
Ctrn = data['dataset']['train'][0, 0]['labels'][0, 0].astype(dtype=np.int_) - 1
Ctst = data['dataset']['test'][0, 0]['labels'][0, 0].astype(dtype=np.int_) - 1

# Prepare measuring time
print("starting timer...")
time.clock()

## Using PCA to improve the gaussian classifier
# Cpreds = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon)
# my_improved_gaussian_classify has keyword arguments:
#  - dims=None : specify dimensions to reduce to (max 26)
#  - epsilon=1e-10 : epsilon to be used in covariance matrices
#  - epsilon_pca=1e-10 : epsilon to be used (after pca) in covariance matrices
print("running experiment #%2d: dims=%s, ε=%s"
                % (args.e, str(dims), str(epsilon)))
(dims, Cpreds) = my_improved_gaussian_classify(Xtrn, Ctrn, Xtst, dims=dims,
                                       epsilon=epsilon, epsilon_pca=epsilon_pca)

# Measure the user time taken, and display it
print("done! - time elapsed: %.2f seconds" % time.clock())

# Get a confusion matrix and accuracy
CM, acc = my_confusion(Ctst, Cpreds)

#YourCode - Save the confusion matrix as "Task3/cm_improved.mat"
scipy.io.savemat("cm_improved.mat", {'cm': CM}, oned_as='row')

#YourCode - Display information if any
N = Xtst.shape[0]
print("N = %d, Nerrs = %4d, acc = %.2f%%" \
    % (N, N * (1 - acc), acc*100))
