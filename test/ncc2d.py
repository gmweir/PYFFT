#!/usr/bin/python

import numpy as _np

def find_max(A):
    i1, i2 = _np.unravel_index(A.argmax(), A.shape)
    maximum = A[i1,i2]
    j1, j2 = _np.unravel_index(A.argmin(), A.shape)
    minimum = A[j1,j2]
    return maximum, minimum, i1+1, i2+1


def template_functions(A1, kernel, N1, Q1, M1, P1, N2, Q2, M2, P2):
    fft_A1 = fft2(A1)
    squ_A1 = _np.square(_np.absolute(A1))
    fft_squ_A1 = fft2(squ_A1)

    pg = _np.zeros((N2,N1),dtype=_np.int8)
    pg[0:M2,0:M1] = A1[Q2-1:Q2+M2-1,Q1-1:Q1+M1-1]

    IFTpg = ifft2(pg)*((N1*N2)/(M1*M2))

    tmp = ifft2(_np.multiply(fft_A1,kernel))
    gc = tmp[0:P2,0:P1]

    tmp = ifft2(_np.multiply(fft_squ_A1,kernel))
    gg = _np.real(tmp[0:P2,0:P1])

    return gc, gg, IFTpg

# ======================================== #


def complex_ccor(A2, gc, gg, kernel, IFTpg,
                 N1, Q1, M1, P1, N2, Q2, M2, P2):
    fft_A2 = fft2(A2)
    squ_A2 = _np.square(_np.absolute(A2))
    fft_squ_A2 = fft2(squ_A2)

    tmp = ifft2(_np.multiply(fft_A2,kernel))
    fc = tmp[0:P2,0:P1]

    tmp = ifft2(_np.multiply(kernel,fft_squ_A2))
    ff = _np.real(tmp[0:P2,0:P1])

    tmp = ifft2(_np.multiply(fft_A2,IFTpg))
    fgc = tmp[0:P2,0:P1]

    gcq = gc[Q2-1,Q1-1]
    ggq = gg[Q2-1,Q1-1]

    numerator = _np.real(fgc - _np.conjugate(fc)*gcq)

    denominator = (ff-_np.square(_np.absolute(fc)))* \
                  (ggq-_np.square(_np.absolute(gcq)))

    # denominator should be non-negative from the definition
    # of variances. It turns out that it takes negative values
    # in the background where there is no tissue and the signal
    # is dominated by noise. If this is the case we give it a
    # large arbitrary value, therefore rendering the CC
    # effectively zero at these points.

    denominator[denominator <= 0] = 1e14
    denominator = _np.sqrt(denominator)

    return numerator/denominator


def test_ccf2d():
    import os as _os

    tx1 = 308
    tx2 = 355

    n1 = 512
    q1 = tx1
    m1 = tx2-tx1+1
    p1 = n1-m1+1

    ty1 = 250
    ty2 = 303

    n2 = 512
    q2 = ty1
    m2 = ty2-ty1+1
    p2 = n2-m2+1

    A1 = _np.fromfile(_os.path.join(_os.path.abspath(_os.path.curdir), "test", "image1.dat")
                      ,sep=" ").reshape(n2,n1)
    A2 = _np.fromfile(_os.path.join(_os.path.abspath(_os.path.curdir), "test", "image2.dat")
                      ,sep=" ").reshape(n2,n1)

    k1 = _np.arange(1,n1)
    kernel1 = (1.0/m1)*((_np.exp(1j*2*_np.pi*m1*k1/n1) - 1)/(_np.exp(1j*2*_np.pi*k1/n1) - 1))
    kernel1 = _np.concatenate(([1+1j*0.0], kernel1))

    k2 = _np.arange(1,n2)
    kernel2 = (1.0/m2)*((_np.exp(1j*2*_np.pi*m2*k2/n2) - 1)/(_np.exp(1j*2*_np.pi*k2/n2) - 1))
    kernel2 = _np.concatenate(([1+1j*0.0], kernel2))

    kernel = _np.zeros((n2,n1),dtype=_np.complex_)
    for i1 in range(n1):
        for i2 in range(n2):
            kernel[i1][i2] = kernel2[i1]*kernel1[i2]

    gc, gg, IFTpg = \
        template_functions(A1, kernel, n1, q1, m1, p1, n2, q2, m2, p2)

    cc = \
        complex_ccor(A2, gc, gg, kernel, IFTpg,
                     n1, q1, m1, p1, n2, q2, m2, p2)

    cc_max, cc_min, i2, i1 = find_max(cc)

    print(cc_max, i1, i2)
# end if def test_ccf2d