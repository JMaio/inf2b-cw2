# ------------------------------------------------------------
# dispImage
# Heru Praptono,
# this tool is for displaying the images, from the dataset
# data.mat
# release (v1.0): 2018-03-14
# update:
#    v1.1: 2018-03-15
#      change histogram (per pixel) colour
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np


#example of use? see the bottom area of this file


def dispImage(ds,trts,i):
    # Input(s):
    #     ds(.mat)
    #         name of dataset for the experiment
    #     trts(1 or 2)
    #         1 show train data, 2 shows test data
    #     i
    #         contains integer between 1 to the number of the data
    # Output(s):
    #         image 's display
    #         image 's histogram
    #     Example:
    #         dispImage('data.mat', 1, 2)

    data = sio.loadmat(ds)
    dst = data['dataset']
    dst_tr = dst['train']
    X = dst_tr[0][0]['images'][0][0][:].reshape(46800, 784)
    Y = dst_tr[0][0]['labels'][0][0][:].reshape(46800, 1)
    dst_ts = dst['test']
    X_t = dst_ts[0][0]['images'][0][0][:].reshape(7800, 784)
    Y_t = dst_ts[0][0]['labels'][0][0][:].reshape(7800, 1)

    s_train = np.shape(X)
    n_train = s_train[0]
    s_train = np.shape(X_t)
    n_test = s_train[0]

    if trts == 1:
        ll = 'TRAIN DATA'
    else:
        ll = 'TEST DATA'
        X = X_t
        Y = X_t

    X_img = np.asarray(X[i].reshape(28, 28)).T
    Y_lbl = int(Y[i][0])

    plt.figure(1)
    plt.tight_layout()

    plt.subplot(221)
    plt.imshow(X_img, cmap='gray')
    plt.axis('off')
    plt.title(ll + ' image: ' + str(i) + ' - class: ' + str(Y_lbl), fontsize=7)

    plt.subplot(222)
    plt.hist(X_img.ravel(), bins=range(0, 255), color='red')
    plt.xticks(range(0, 300, 50), fontsize=7)
    plt.yticks(fontsize=7)
    plt.title('Image Histogram (per pixel map)', fontsize=7)
    plt.xlabel('Gray level (from 0 to 255)', fontsize=7)
    plt.ylabel('Pixel count', fontsize=7)

    plt.subplot(223)
    plt.hist(X_img.ravel(), bins=range(0, 255, 50), color='black')
    plt.xticks(range(0, 300, 50), fontsize=7)
    plt.yticks(fontsize=7)
    plt.title('Image Histogram (per pixel map)', fontsize=7)
    plt.xlabel('Gray level (bin size = 50)', fontsize=7)
    plt.ylabel('Pixel count', fontsize=7)

    plt.subplot(224)
    plt.bar([1, 2], [n_train, n_test])
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.title('NUMBER OVERALL DATA data.mat', fontsize=7)
    plt.xlabel('1: training data, 2: testing data', fontsize=7)
    plt.ylabel('num of instances', fontsize=7)

    plt.show()

#input: vary these variables
ds = 'data.mat'
trts = 1
i = 200

dispImage(ds,trts,i)