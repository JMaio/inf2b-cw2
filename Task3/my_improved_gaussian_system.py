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
parser.add_argument('-e', default=None, type=int,
                    help='select which experiment number to run')
parser.add_argument('-d', default=None, type=int,
                    help='number of dimensions to reduce to')
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
# define custom experiments for replicating statistics from report
experiments = [
    (1, 0.01, 0.01),
    (1, 1e-10, 1e-10),
    (2, 0.01, 0.01),
    (2, 1e-10, 1e-10),
    (4, 0.01, 0.01),
    (4, 1e-10, 1e-10),
    (8, 1e-10, 1e-10),
    (16, 1e-10, 1e-10),
    (21, 1e-10, 1e-10),
    (None, 0.02, 0.02),
    (None, 0.01, 0.01),
    (26, 1e-10, 1e-10),
    (None, 0, 0),
]
le = len(experiments)
e = (args.e - 1)
if e not in range(le):
    e = 11
    print("experiment number not valid! running default: #%d" % (e + 1))
dims, epsilon, epsilon_pca = experiments[e]

print("running experiment #%2d: dims=%s, ε=%s"
                % (args.e, str(dims), str(epsilon)))
(dims, Cpreds) = my_improved_gaussian_classify(Xtrn, Ctrn, Xtst, dims=dims,
                                       epsilon=epsilon, epsilon_pca=epsilon_pca)

# Measure the user time taken, and display it
print("experiment #%2d done! - time elapsed: %.2f seconds" % (e, time.clock()))

# Get a confusion matrix and accuracy
CM, acc = my_confusion(Ctst, Cpreds)

#YourCode - Save the confusion matrix as "Task3/cm_improved.mat"
scipy.io.savemat("cm_improved.mat", {'cm': CM}, oned_as='row')

#YourCode - Display information if any
N = Xtst.shape[0]
print("dims = %2d, ε = %.0e, N = %d, Nerrs = %4d, acc = %.2f%%" \
    % (dims, epsilon, N, N * (1 - acc), acc*100))
