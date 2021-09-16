# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:39:09 2018

@author: gawe
"""

import numpy as _np
import numpy.linalg as _la
import matplotlib.pyplot as _plt


def cov(data):
    """
    Covariance matrix
    note: specifically for mean-centered data
    note: numpy's `cov` uses N-1 as normalization
    """
    return _np.dot(data.T, data) / data.shape[0]


def basic_pca(data, pc_count = None):
    """

   Principal component analysis using eigenvalues
    note: this mean-centers and auto-scales the data (in-place)
    """
    # data = data.copy()
    # data = data/_np.mean(data, 0)   # normalize the data

    data -= _np.mean(data, 0)
    data /= _np.std(data, 0)
    # C = _np.cov(data.T)   # numpy's covaraince calculation
    C = cov(data)       # simple one
    E, V = _la.eigh(C)
    key = _np.argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    # U = _np.dot(data, V)  # numpy's cov ....used to be _np.dot(V.T, data.T).T
    U = _np.dot(data, V)  # simple cov .... two PC's?
    return U, E, V


def test(data=None):
    if data is None:
        data = test_data()
    # end if

    """ visualize """
    trans = basic_pca(data, 3)[0]
    fig, (ax1, ax2) = _plt.subplots(1, 2)

    ax1.scatter(data[:50, 0], data[:50, 1], c = 'r')
    ax2.scatter(trans[:50, 0], trans[:50, 1], c = 'r')

    ax1.scatter(data[50:, 0], data[50:, 1], c = 'b')
    ax2.scatter(trans[50:, 0], trans[50:, 1], c = 'b')
    _plt.show()
# end def test()

# =========================================================================== #


def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = _np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = _la.eigh(R)
    # sort eigenvalue in decreasing order
    idx = _np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return _np.dot(evecs.T, data.T).T, evals, evecs

def test_PCA(data=None, dims_rescaled_data=2):
    '''
    test by attempting to recover original data array from
    the eigenvectors of its covariance matrix & comparing that
    'recovered' array with the original data
    '''
    if data is None:
        data = test_data()
    # end if

    m, n = data.shape
    _ , _ , eigenvectors = PCA(data.copy(), dims_rescaled_data=dims_rescaled_data)
    # data_recovered = _np.dot(eigenvectors, m).T
    # data_recovered += data_recovered.mean(axis=0)
    # data_recovered = _np.dot(eigenvectors, _np.ones((n,m))).T
    data_recovered = _np.dot(eigenvectors.T, data.T).T
    # data_recovered += data.mean(axis=0)
    # data_recovered += data_recovered.mean(axis=0)

    _plt.figure()
    _plt.plot(data, '-')
    _plt.plot(data_recovered, '.')
    # assert _np.allclose(data, data_recovered)

    plot_pca(data)
# end def

def plot_pca(data, pcindices = [0,1]):
    clr1 =  '#2026B2'
    fig = _plt.figure()
    ax1 = fig.add_subplot(111)
    data_resc, eigenval, eivenvec = PCA(data.copy())

    # _plt.figure()
    # ax1.plot(data_resc[:, 0], data_resc[:, 1], '.', mfc=clr1, mec=clr1)

    _plt.figure()
    _ax1 = _plt.subplot(3, 1, 1)
    _ax1.plot(data, '-', data_resc, '.')

    _ax2 = _plt.subplot(3, 1, 2)
    _ax2.plot(1+_np.arange(0, len(eigenval)), eigenval, 's-')
    _ax2.set_ylabel('eigval')

    _ax3 = _plt.subplot(3, 1, 3)
    _ax3.plot(data_resc[:,pcindices[0]], data_resc[:,pcindices[1]], '.', mfc=clr1, mec=clr1)
    _ax3.set_xlabel('PC%i'%(pcindices[0],))
    _ax3.set_ylabel('PC%i'%(pcindices[1],))
    _plt.show()

def test_data():
    """ test data """
    data = _np.array([_np.random.randn(8) for k in range(150)])
    data[:50, 2:4] += 5
    data[50:, 2:5] += 5
    return data
# end def

# =========================================================================== #


if __name__=="__main__":

    test()
    test_PCA()
# end if