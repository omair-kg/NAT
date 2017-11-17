import numpy as np
import torch
import torch.nn as nn

from scipy import misc
from scipy.optimize import linear_sum_assignment
#https://github.com/cromulen/noise-as-target/blob/master/src/utils.py
def rand_unit_sphere(n, z=100):
    '''
    Generates "npoints" number of vectors of size "ndim"
    such that each vectors is a point on an "ndim" dimensional sphere
    that is, so that each vector is of distance 1 from the center
    npoints -- number of feature vectors to generate
    ndim -- how many features per vector
    returns -- np array of shape (npoints, ndim), dtype=float64

    vec = np.random.randn(npoints, ndim)
    vec = np.divide(vec, np.expand_dims(np.linalg.norm(vec, axis=1), axis=1))
    return vec
    '''
    samples = np.random.normal(0, 1, [n, z]).astype(np.float32)
    radiuses = np.expand_dims(np.sqrt(np.sum(np.square(samples),axis=1)),1)
    reps = samples/radiuses
    return reps



def calc_optimal_target_permutation(reps, targets):
    # Compute cost matrix
    cost_matrix = np.zeros([reps.shape[0], targets.shape[0]])
    #cost_matrix = np.dot(reps, np.transpose(targets))
    for i in range(reps.shape[0]):
       cost_matrix[:, i] = np.sum(np.square(reps-targets[i, :]), axis=1)

    _, col_ind = linear_sum_assignment(cost_matrix)
    # Permute
    targets[range(reps.shape[0])] = targets[col_ind]
    return targets

class convert_grayScale(object):

    def __call__(self,im):
        ch, h, w = im.size()
        im_ = im.clone()
        im_ = im_.numpy()
        gray_im = torch.from_numpy(np.mean(im_,axis=0))
        gray_im = torch.unsqueeze(gray_im, 0)
        return gray_im

