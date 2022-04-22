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
        from FFT.fft import fft, ifft
        # from FFT.dft import fft, ifft        
    except:
        from . import fft as fftmod
        from .fft import fft, ifft
        # from .dft import fft, ifft
    # end try
else:
    import numpy.fft as fftmod   # analysis:ignore
    from numpy.fft import fft, ifft
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
    ccf_test()

    ccf_sh_test()

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
# findgen1=np.arange(0,nn)
# findgen2=np.arange(0,nn);
# norm1=number-nn+findgen1;
# norm2=number-findgen2;

# corr=np.zeros(2*nn);

# fft1=np.fft.fft(data1);
# fft2=np.fft.fft(data2);

# pwrspc=np.conjugate(fft1)*fft2;

# pwrspc=fft1*np.conjugate(fft2);

# ztmp=np.real(np.fft.ifft(pwrspc));
# norm=(np.sqrt(np.mean(y1*y1)*np.mean(y2*y2)))

# corr[:nn]=ztmp[(pad-nn):(pad)]/norm/norm1;
# corr[nn:(2*nn)]=ztmp[:nn]/norm/norm2;

# tau=(findgen1+1)/fs;

# tau2=np.flipud(-tau)
# tau2=np.append(tau2,0)
# tau2=np.append(tau2,tau)

# return tau,corr