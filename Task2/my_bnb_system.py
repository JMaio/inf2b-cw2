#! /usr/bin/env python
# A sample template for my_bnb_system.py

import numpy as np
import scipy.io
import time
from my_bnb_classify import *
from my_confusion import *

# Load the data set
filename = "/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1621503/data.mat"
# use local data set while not connected to afs
try:
    data = scipy.io.loadmat(filename)
    print("loaded data from afs")
except Exception:
    data = scipy.io.loadmat("../data.mat")
    print("loaded data from local")

# Feature vectors: Convert uint8 to double (but do not divide by 255)
Xtrn = data['dataset']['train'][0, 0]['images'][0, 0].astype(dtype=np.float_)
Xtst = data['dataset']['test'][0, 0]['images'][0, 0].astype(dtype=np.float_)
# Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1.
Ctrn = data['dataset']['train'][0, 0]['labels'][0, 0].astype(dtype=np.int_) - 1
Ctst = data['dataset']['test'][0, 0]['labels'][0, 0].astype(dtype=np.int_) - 1

# Prepare measuring time
print("starting timer...")
t0 = time.clock()

# Run classification
threshold = 1.0

print("running my_bnb_classify...")
Cpreds = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold)

# Measure the user time taken, and display it.
print("done! - time elapsed: %.2f seconds" % (time.clock() - t0))

# Get a confusion matrix and accuracy
CM, acc = my_confusion(Ctst, Cpreds)

# Save the confusion matrix as "Task2/cm.mat".
scipy.io.savemat("cm.mat", {'cm': CM}, oned_as='row')

# Display the required information - N, Nerrs, acc
N = Xtst.shape[0]
print("N = %d, Nerrs = %4d, acc = %.2f%%" \
    % (N, N * (1 - acc), acc*100))
