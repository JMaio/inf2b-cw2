#! /usr/bin/env python
# A sample template for my_gaussian_system.py

import numpy as np
import scipy.io
import time
from my_gaussian_classify import *
from my_confusion import *

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

epsilon = 0.01

# Prepare measuring time
print("starting timer...")
time.clock()

# Run Gaussian classification
print("running my_gaussian_classify...")
(Cpreds, Ms, Covs) = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon)

# Measure the user time taken, and display it
print("done! - time elapsed: %.2f seconds" % time.clock())

# Get a confusion matrix and accuracy
CM, acc = my_confusion(Ctst, Cpreds)

# Save the confusion matrix as "Task3/cm.mat"
scipy.io.savemat("cm.mat", {'cm': CM}, oned_as='row')
# Save the mean vector matrix...
scipy.io.savemat("m26.mat", {'m26': Ms[25]}, oned_as='row')
# ...and covariance matrix for class 26
scipy.io.savemat("cov26.mat", {'cov26': Covs[:, :, 25]}, oned_as='row')

#YourCode - Display the required information - N, Nerrs, acc
N = Xtst.shape[0]
print("N = %d, Nerrs = %4d, acc = %.2f%%" \
    % (N, N * (1 - acc), acc*100))
