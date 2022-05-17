# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:15:38 2019

@author: gawe
"""
# =========================================================================== #
# =========================================================================== #


from __future__ import absolute_import, with_statement, absolute_import, \
                       division, print_function, unicode_literals
import numpy as _np
import matplotlib.pyplot as _plt
import pybaseutils.utils as _ut
__metaclass__ = type

# You can replace these brute-force ancient algorithms with results from
# fft_analysis, np or scipy for better results. They are all slow in correlation land.
if 1:
    try:
        from FFT import fft as fftmod
        from FFT.fft import fft, ifft, fft2, ifft2
        # from FFT.dft import fft, ifft
    except:
        from . import fft as fftmod
        from .fft import fft, ifft, fft2, ifft2
        # from .dft import fft, ifft
    # end try
else:
    import numpy.fft as fftmod   # analysis:ignore
    from numpy.fft import fft, ifft, fft2, ifft2
# end if


import scipy.signal as _dsp  # only used for convolve_fft etc. Very slow and unnecessary

# =========================================================================== #
# =========================================================================== #


def align_signals(a, b):
    """Finds optimal delay to align two 1D signals
    maximizes hstack((zeros(shift), b)) = a

    Parameters
    ----------
    a : _np.array, shape(n)
    b : _np.array, shape(m)

    Output
    ------
    shift : int
        Integer that maximizes hstack((zeros(shift), b)) - a = 0
    """
    # check inputs
    a = _np.asarray(a)
    b = _np.asarray(b)
    if _np.prod(a.ndim) > 1 or _np.prod(b.ndim) > 1:
        raise ValueError('Can only vectorize vectors')
    # longest first
    sign = 1
    if len(b) > len(a):
        sign = -1
        a, b = b, a
    r = cross_correlation_fft(a, b)
    shift = _np.argmax(r) - len(a) + len(a) / 2
    # deal with odd / even lengths (b doubles in size by cross_correlation_fft)
    if len(a) % 2 and len(b) % 2:
        shift += 1
    if len(a) > len(b) and len(a) % 2 and not(len(b) % 2):
        shift += 1
    return sign * shift


# =========================================================================== #
# =========================================================================== #


def ccf(x1,x2,fs):
    """
    Return the cross-correlation function and lags between two signals (x1, x2)
    - a little slower than cross_correlation_fft, but also returns time-lags
    """
    npts=len(x1)
    lags=_np.arange(-npts+1,npts)
    tau=-lags/float(fs)         # time-lags in input scales
    ccov = _np.correlate(x1-x1.mean(), x2-x2.mean(), mode='full')
    # cross-covariance (by substracting mean from data first before correlating)
    co = ccov / (npts * x1.std() * x2.std())  # normalized cross-covariance
    return tau, co


def ccf_sh(x1, x2, fs, nav):
    """
    Returns the average cross-correlation within a sliding window
    inputs:
        x1 - data-series 1
        x2 - data-series 2
        fs - sampling frequency for scaling index-lags to time-lags
        nav - window length for each time-window (in samples)
    outputs:
        tau - time-lags
        csh - average cross-correlatin between x1 and x2
    """
    _, xs1, _ =_ut.sliding_window_1d(x1,x1,nav,ss=None)
    _, xs2, _ =_ut.sliding_window_1d(x1,x2,nav,ss=None)

    # Calculate the cross-correlation and time-lag between the time-series
    # within each window along the time-series data
    co=_np.zeros((2*len(xs1)-1,nav))
    for ii in range(0, nav):
        tau, co[:,ii]=ccf(xs1[:,ii],xs2[:,ii],fs)
    # end for

    # The average cross-correlation within each sliding window
    csh=_np.mean(co,1)
    return tau, csh

# =========================================================================== #
# =========================================================================== #

#def ccf_test():
#    fs=1e5
#    N=2048
#    f=1e3
#    phi=50*_np.pi/180        #phase lag phi>0 : x2 lags behind, phi<0 : x2 is ahead
#    t=_np.arange(0,N)*1.0/fs
#
#    ff = _np.asarray(_np.arange(-N//2, N//2)*fs, dtype=_np.float64)
#    x1 = fftmod.ifft(2.0*_np.exp(-2*_np.abs(ff)/(100e3)), len(ff))
#    x2 = fftmod.ifft(2.0*_np.exp(-1.0*_np.abs(ff)/(30e3)), len(ff))   \
#    x2 += _np.random.random.rarandn(len(ff))
#
#
#    _plt.figure()
#    _plt.plot(ff, x1, 'b-', ff, x2, 'r-')
#
#    x1=_np.sin(2*_np.pi*f*t)+_np.random.normal(0,1,N)
#    x2=_np.sin(2*_np.pi*f*t+phi)+_np.random.normal(0,1,N)
#    tau,co=ccf(x1,x2,fs)
#    print('expect max at t=%2.3f us' % (-phi/(2*_np.pi*f)*1e6))
#    _plt.figure(1)
#    _plt.clf()
#    _plt.subplot(2,1,1)
#    _plt.plot(t,x1,t,x2)
#    _plt.legend(['x1','x2'])
#    _plt.subplot(2,1,2)
#    _plt.plot(tau*1e6,co)
#    _plt.show()


def ccf_test():
    fs=1e5
    N=2048
    f=1e3
    phi=50*_np.pi/180        #phase lag phi>0 : x2 lags behind, phi<0 : x2 is ahead
    t=_np.arange(0,N)*1./fs
    x1=_np.sin(2*_np.pi*f*t)+_np.random.normal(0,1,N)
    x2=_np.sin(2*_np.pi*f*t+phi)+_np.random.normal(0,1,N)
    tau,co=ccf(x1,x2,fs)
    print('expect max at t=%2.3f us' % (-phi/(2*_np.pi*f)*1e6))
    _plt.figure()
    _plt.clf()
    _plt.subplot(2,1,1)
    _plt.plot(t,x1,t,x2)
    _plt.legend(['x1','x2'])
    _plt.subplot(2,1,2)
    _plt.plot(tau*1e6,co)
    _plt.show()

def ccf_sh_test():
    fs=1e5
    N=2048
    f=1e3
    phi=50*_np.pi/180        #phase lag phi>0 : x2 lags behind, phi<0 : x2 is ahead
    t=_np.arange(0,N)*1./fs
    x1=_np.sin(2*_np.pi*f*t)+_np.random.normal(0,1,N)
    x2=_np.sin(2*_np.pi*f*t+phi)+_np.random.normal(0,1,N)
    tau,co=ccf_sh(x1,x2,fs, nav=64)
    print('expect max at t=%2.3f us' % (-phi/(2*_np.pi*f)*1e6))
    _plt.figure()
    _plt.clf()
    _plt.subplot(2,1,1)
    _plt.plot(t,x1,t,x2)
    _plt.legend(['x1','x2'])
    _plt.subplot(2,1,2)
    _plt.plot(tau*1e6,co)
    _plt.show()

# =========================================================================== #
# =========================================================================== #

def conv(x, y):
    """
    Convolution of 2 causal signals, x(t<0) = y(t<0) = 0, using discrete
    summation.
        x*y(t) = \int_{u=0}^t x(u) y(t-u) du = y*x(t)
    where the size of x[], y[], x*y[] are P, Q, N=P+Q-1 respectively.
    """
    P, Q, N = len(x), len(y), len(x)+len(y)-1
    z = []
    for k in range(N):
        lower, upper = max(0, k-(Q-1)), min(P-1, k)
        z.append(sum(x[i] * y[k-i]
                      for i in range(lower, upper+1)))
    return z
# end def conv


def corr(x, y):
    """
    Correlation of 2 causal signals, x(t<0) = y(t<0) = 0, using discrete
    summation.
        Rxy(t) = \int_{u=0}^{\infty} x(u) y(t+u) du = Ryx(-t)
    where the size of x[], y[], Rxy[] are P, Q, N=P+Q-1 respectively.

    The Rxy[i] data is not shifted, so relationship with the continuous
    Rxy(t) is preserved.  For example, Rxy(0) = Rxy[0], Rxy(t) = Rxy[i],
    and Rxy(-t) = Rxy[-i].  The data are ordered as follows:
        t:  -(P-1),  -(P-2),  ..., -3,  -2,  -1,  0, 1, 2, 3, ..., Q-2, Q-1
        i:  N-(P-1), N-(P-2), ..., N-3, N-2, N-1, 0, 1, 2, 3, ..., Q-2, Q-1
    """
#    P, Q, N = len(x), len(y), len(x)+len(y)-1
    P, Q = len(x), len(y)
    z1=[]
    for k in range(Q):
        lower, upper = 0, min(P-1, Q-1-k)
        z1.append(sum(x[i] * y[i+k]
                       for i in range(lower, upper+1))) # 0, 1, 2, ..., Q-1
    z2=[]
    for k in range(1,P):
        lower, upper = k, min(P-1, Q-1+k)
        z2.append(sum(x[i] * y[i-k]
                       for i in range(lower, upper+1))) # N-1, N-2, ..., N-(P-2), N-(P-1)
    z2.reverse()
    return z1 + z2
# end def corr


def fftconv(x, y):
    """
    FFT convolution using relation
        x*y <==> XY
    where x[] and y[] have been zero-padded to length N, such that N >=
    P+Q-1 and N = 2^n.
    """
    X, Y = fft(x), fft(y)
    return ifft([a * b for a,b in zip(X,Y)])
# end def fftconv


def fftcorr(x, y):
    """
    FFT correlation using relation
        Rxy <==> X'Y
    where x[] and y[] have been zero-padded to length N, such that N >=
    P+Q-1 and N = 2^n.
    """
    X, Y = len(x), fft(x), fft(y)
    return ifft([a.conjugate() * b for a,b in zip(X,Y)])
# end def fftcorr


# =========================================================================== #


def lewis_ccor(navigt, templt, N, Q, M, P):
    cc = _np.zeros(P)       # normalized cross-correlation
    ns = _np.zeros(N+1)     # navigator sum

    ns2 = _np.zeros(N+1)    # navigator sum of squares

    for i in range(N):
        a = navigt [i]
        ns [i + 1] = a + ns[i]
        ns2 [i + 1] = a*a + ns2[i]
    # end for
    q = Q-1

    template = templt[q:q+M]
    ts = sum(template)          # template sum
    ts2 = sum(_np.square(template))  # template sum of squares
    tm = ts/M       # template mean

    tv = ts2 - _np.square(ts)/M     # template variance
    v1 = template - tm
    for i in range(P):
        k = i+M
        A = ns[k] - ns[i]
        C = ns2[k] - ns2[i]

        nm = A/M
        nv = C - A*A/M
        v2 = navigt[i:k] - nm

        numerator = sum(v1*v2)
        denominator = _np.sqrt(tv*nv)
        cc[i] = numerator/denominator
    # end for
    return cc


def luo_ccor(navigt, templt, N, Q, M, P):
    cc = _np.zeros(P)    # normalized cross-correlation
    ns = _np.zeros(N+1)  # navigator sum

    ns2 = _np.zeros(N+1)  # navigator sum of squares
    tns = _np.zeros((N+1,N))  # template-navigator cross terms

    for i in range(N):
        a = navigt[i]
        ns[i + 1] = a + ns[i]
        ns2[i + 1] = a*a + ns2[i]
        for d in range(N):
            k = (i+d)%N
            tns[i + 1][d] = tns[i][d] + templt[i]*navigt[k]
        # end for
    # end for
    q = Q-1
    template = templt [q:q+M]

    ts = sum(template)     # template sum
    ts2 = sum(_np.square(template))  # template sum of squares
    tm = ts/M   # template mean

    tv = ts2 - _np.square(ts)/M  # template variance
    for i in range(P):
        k = i+M
        A = ns[k] - ns[i]
        C = ns2[k] - ns2[i]

        nv = C - A*A/M
        d = (i-q)%N

        numerator = (tns[q+M,d] - tns[q,d]) - A*tm
        denominator = _np.sqrt(tv*nv)
        cc [i] = numerator/denominator
    # end for
    return cc


def template_functions(templt, kernel, N, Q, M, P):
    templt2 = _np.square(_np.absolute(templt))
    tmp = ifft(fft(templt)*kernel)

    gc = tmp [range(P)]
    tmp = ifft(fft(templt2)*kernel)

    gg = _np.real(tmp[range(P)])
    templt_padded = _np.concatenate((templt [Q-1:Q+M-1],_np.zeros(N-M)))

    FTpg = fft(templt_padded)/M
    return gc, gg, FTpg


def complex_ccor(navigt, gc, gg, kernel, FTpg, N, Q, M, P):

    navigt2 = _np.square(_np.absolute(navigt))
    tmp = ifft(fft(navigt)*kernel)
    fc = tmp[range(P)]

    tmp = ifft(fft(navigt2)*kernel)
    ff = _np.real(tmp [range(P)])
    FTnv = fft(navigt)

    tmp = fft(_np.conjugate(FTnv)*FTpg)/N
    fgc = tmp [range(P)]

    q = Q-1
    gcq = gc[q]
    ggq = gg[q]

    numerator = _np.real(fgc - _np.conjugate(fc)*gcq)
    denominator = _np.sqrt((ff - _np.square(_np.absolute(fc)))*(ggq - _np.square(_np.absolute(gcq))))
    return numerator/denominator
# end def complex_ccor


def test_ccf_funcs():
    """
    test functions for 1D cross-correlation from

    Computation of the normalized cross-correlation by fast Fourier transform
    https://doi.org/10.1371/journal.pone.0203434


    """
    import os as _os
    import matplotlib.pyplot as _plt

    tx1 = 80
    tx2 = 106

    n = 128
    q = tx1
    m = tx2-tx1+1
    p = n-m+1

    A = _np.fromfile(_os.path.join(_os.path.abspath(_os.path.curdir), "test", "navigators.dat")
                     , sep="\t").reshape(n,3)
    template = []
    navigator = []

    for i in range(n):
        template = template + [A[i][1]]
        navigator = navigator + [A[i][2]]
    # end for

    k = _np.arange(1,n)
    kernel = (1.0/m) * ((_np.exp(1j*2*_np.pi*m*k/n) - 1)/(_np.exp(1j*2*_np.pi*k/n) - 1))
    kernel = _np.concatenate(([1 + 1j*0.0], kernel))

    gc, gg, FTpg = template_functions(template, kernel, n, q, m, p)
    cc = complex_ccor(navigator, gc, gg, kernel, FTpg, n, q, m, p)

    lewis_cc = lewis_ccor(navigator, template, n, q, m, p)
    luo_cc = luo_ccor(navigator, template, n, q, m, p)

    for i in range(n-m+1):
        print("%3d % 16.14f % 16.14f %16.14f" % \
            (i+1, lewis_cc[i], cc[i], abs(lewis_cc[i]-cc[i])))
    # end for

    for i in range(n-m+1):
        print("%3d % 16.14f % 16.14f %16.14f" % \
            (i+1, luo_cc[i], cc[i], abs(luo_cc[i]-cc[i])))
    # end for
    lags = _np.asarray(range(n-m+1))

    _plt.figure()
    _plt.subplot(211)
    _plt.plot(lags, _np.abs(cc), 'k-', lags, _np.abs(lewis_cc), 'b.')
    _plt.subplot(212)
    _plt.plot(lags, _np.abs(cc), 'k-', lags, _np.abs(luo_cc), 'g.')

# end test_ccf_funcs


# =========================================================================== #
# =========================================================================== #


def find_max2D(A):
    i1, i2 = _np.unravel_index(A.argmax(), A.shape)
    maximum = A[i1,i2]
    j1, j2 = _np.unravel_index(A.argmin(), A.shape)
    minimum = A[j1,j2]
    return maximum, minimum, i1+1, i2+1


def template_functions2(A1, kernel, N1, Q1, M1, P1, N2, Q2, M2, P2):
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


def complex_ccor2(A2, gc, gg, kernel, IFTpg,
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
    """
    test functions for 2D cross-correlation from

    Computation of the normalized cross-correlation by fast Fourier transform
    https://doi.org/10.1371/journal.pone.0203434


    """
    import os as _os
    import matplotlib.pyplot as _plt
    import matplotlib.cm as cm

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
        # end for
    # end for
    gc, gg, IFTpg = \
        template_functions2(A1, kernel, n1, q1, m1, p1, n2, q2, m2, p2)

    cc = complex_ccor2(A2, gc, gg, kernel, IFTpg,
                     n1, q1, m1, p1, n2, q2, m2, p2)

    cc_max, cc_min, i2, i1 = find_max2D(cc)

    print(cc_max, i1, i2)
    print(_np.shape(A1), _np.shape(A2))

    _plt.figure()
    _plt.subplot(311)
    nrows, ncols = A1.shape
    _plt.imshow(A1, extent=(0, nrows, ncols, 0),
                interpolation='nearest', cmap=cm.gist_rainbow)

    _plt.subplot(312)
    nrows, ncols = A2.shape
    _plt.imshow(A2, extent=(0, nrows, ncols, 0),
                interpolation='nearest', cmap=cm.gist_rainbow)

    _plt.subplot(313)
    _plt.imshow(_np.abs(cc), extent=(0, nrows, ncols, 0),
                interpolation='nearest', cmap=cm.gist_rainbow)
    _plt.show()
# end if def test_ccf2d

# =========================================================================== #
# =========================================================================== #


def _preconvolve_fft(a, b):
    a = _np.asarray(a)
    b = _np.asarray(b)
    if _np.prod(a.ndim) > 1 or _np.prod(b.ndim) > 1:
        raise ValueError('Can only vectorize vectors')
    if len(b) > len(a):
        a, b = b, a
    n = len(a)
    # Pad vector
    c = _np.hstack((_np.zeros(n/2), b, _np.zeros(n/2 + len(a) - len(b) + 1)))
    return c

def convolve_fft(a, b, mode='valid'):
    """
    Convolution between two 1D signals.

    Parameters
    ----------
    a : _np.array, shape(n)
    b : _np.array, shape(m)
        If len(b) > len(a), a, b = b, a

    Output
    ------
    r : _np.array
    """
    c = _preconvolve_fft(a, b)
    # Convolution of signal:
    return _dsp.fftconvolve(c, a, mode=mode)

def cross_correlation_fft(a, b, mode='valid'):
    """
    Cross correlation between two 1D signals. Similar to _np.correlate, but
    faster.

    Parameters
    ----------
    a : _np.array, shape(n)
    b : _np.array, shape(m)
        If len(b) > len(a), a, b = b, a

    Output
    ------
    r : _np.array
        Correlation coefficients. Shape depends on mode.
    """
    c = _preconvolve_fft(a, b)
    # Convolution of reverse signal:
    return _dsp.fftconvolve(c, a[::-1], mode=mode)


# =========================================================================== #
# =========================================================================== #

if __name__ == "__main__":
    # ccf_test()

    # ccf_sh_test()

    test_ccf_funcs()
    test_ccf2d()

# def
#     y1=y1-np.mean(y1)
#     y2=y2-np.mean(y2)

# siy1=len(y1);
# siy2=len(y2);

# pad=65536;
# number=siy1;

# data1=np.zeros(pad);
# data2=np.zeros(pad);

# data1[:siy2]=y1;
# data2[:siy2]=y2;

# nn=np.floor(number/2);
# findgen1=_np.arange(0,nn)
# findgen2=_np.arange(0,nn);
# norm1=number-nn+findgen1;
# norm2=number-findgen2;

# corr=np.zeros(2*nn);

# fft1=np.fft.fft(data1);
# fft2=np.fft.fft(data2);

# pwrspc=np.conjugate(fft1)*fft2;

# pwrspc=fft1*np.conjugate(fft2);

# ztmp = _np.real(np.fft.ifft(pwrspc));
# norm=(np.sqrt(np.mean(y1*y1)*np.mean(y2*y2)))

# corr[:nn]=ztmp[(pad-nn):(pad)]/norm/norm1;
# corr[nn:(2*nn)]=ztmp[:nn]/norm/norm2;

# tau=(findgen1+1)/fs;

# tau2=np.flipud(-tau)
# tau2=np.append(tau2,0)
# tau2=np.append(tau2,tau)

# return tau,corr