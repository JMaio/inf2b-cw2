
# coding: utf-8

import numpy as np
import scipy.io
import time
from my_knn_classify import *
from my_confusion import *

# Load the data set
#   NB: replace <UUN> with your actual UUN.
filename = "/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1621503/data.mat"
# filename = "../data.mat"
data = scipy.io.loadmat(filename)

# Feature vectors: Convert uint8 to double, and divide by 255.
Xtrn = data['dataset']['train'][0, 0]['images'][0, 0].astype(dtype=np.float_) / 255.0
Xtst = data['dataset']['test'][0, 0]['images'][0, 0].astype(dtype=np.float_) / 255.0
# Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1.
Ctrn = data['dataset']['train'][0, 0]['labels'][0, 0].astype(dtype=np.int_)-1
Ctst = data['dataset']['test'][0, 0]['labels'][0, 0].astype(dtype=np.int_)-1

#Prepare measuring time
print("starting timer...")
t0 = time.clock()

# Run K-NN classification
kb = [1, 3, 5, 10, 20]
print("running my_knn_classify...")
# Cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, kb)
print(my_knn_classify(Xtrn, Ctrn, Xtst, kb))

# Measure the user time taken, and display it.
print("running my_knn_classify...")
# elapsed = timeit.default_timer() - start_time
print("time elapsed: %s" % str(time.clock()))

# YourCode - Get confusion matrix and accuracy for each k in kb.

# YourCode - Save each confusion matrix.

# YourCode - Display the required information - k, N, Nerrs, acc for each element of kb
