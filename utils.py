import numpy as np
from scipy import misc
from scipy.optimize import linear_sum_assignment
#https://github.com/cromulen/noise-as-target/blob/master/src/utils.py
def rand_unit_sphere(npoints, ndim=100):
    '''
    Generates "npoints" number of vectors of size "ndim"
    such that each vectors is a point on an "ndim" dimensional sphere
    that is, so that each vector is of distance 1 from the center
    npoints -- number of feature vectors to generate
    ndim -- how many features per vector
    returns -- np array of shape (npoints, ndim), dtype=float64
    '''
    vec = np.random.randn(npoints, ndim)
    vec = np.divide(vec, np.expand_dims(np.linalg.norm(vec, axis=1), axis=1))
    return vec


def shuffle_assigned_noises(noises):
    '''
    shuffles all of the noises assigned to images
    done every N epoch to avoid plateaus
    '''
    keys = noises.keys()
    values = list(noises.values())
    np.random.shuffle(values)

    for k, v in zip(keys, values):
        noises[k] = v

def calc_optimal_target_permutation(reps, targets):
    # Compute cost matrix
    cost_matrix = np.zeros([reps.shape[0], targets.shape[0]])
    for i in range(reps.shape[0]):
        cost_matrix[:, i] = np.sum(np.square(reps-targets[i, :]), axis=1)

    _, col_ind = linear_sum_assignment(cost_matrix)
    # Permute
    targets[range(reps.shape[0])] = targets[col_ind]
    return targets