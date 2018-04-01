import numpy as np
import scipy.io
import time
from my_knn_classify import *
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

# Feature vectors: Convert uint8 to double, and divide by 255.
Xtrn = data['dataset']['train'][0, 0]['images'][0, 0].astype(dtype=np.float_) / 255.0
Xtst = data['dataset']['test'][0, 0]['images'][0, 0].astype(dtype=np.float_) / 255.0
# Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1.
Ctrn = data['dataset']['train'][0, 0]['labels'][0, 0].astype(dtype=np.int_) - 1
Ctst = data['dataset']['test'][0, 0]['labels'][0, 0].astype(dtype=np.int_) - 1

#Prepare measuring time
print("starting timer...")
t0 = time.clock()

# Run K-NN classification
kb = [1, 3, 5, 10, 20]

print("running my_knn_classify...")
Cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, kb)

print Cpreds

# Measure the user time taken, and display it.
print("done! - time elapsed: %.2f seconds" % (time.clock() - t0))

#YourCode - Get confusion matrix and accuracy for each k in kb.

#YourCode - Save each confusion matrix.

#YourCode - Display the required information - k, N, Nerrs, acc for each element of kb
