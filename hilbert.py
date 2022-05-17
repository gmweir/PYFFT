# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 17:22:19 2021

@author: gawe
"""
# ========================================================================== #
# ========================================================================== #

# This section is to improve python compataibilty between py27 and py3
from __future__ import absolute_import, with_statement, absolute_import, division, print_function, unicode_literals
__metaclass__ = type

# ========================================================================== #

import numpy as _np
import matplotlib.pyplot as _plt

if 1:
    try:
        from FFT import fft as fftmod
    except:
        from . import fft as fftmod
    # end try
else:
    import numpy.fft as fftmod
# end if

# ========================================================================== #
# ========================================================================== #

def hilbert(uin, nfft=None, axes=-1):
    """
    returns the analytic signal
       z = x + j*y
          x -- original signal
          y -- Hilbert transofrm of input H[u]

    inputs and parameters
        nfft - fft length
        mfft - Number of elements to zero out in the negative frequency space
        Ufft - discrete fourier transform of u
        axes - axis to transform along
    Returns:
        analytic signal - u + j*H(u) - the inverse discrete fourier transform
                                       of H(Ufft)

    Note:
        - doesn't work with arbitrary axis yet. Use roll-axis to get this working
        - only works on 1d arrays while testing due to line 3579, etc.
    """
    if nfft is None:
        uin = _np.atleast_1d(uin)
        nfft = _np.shape(uin)[axes]
    # end if

    nyq = nfft//2       # Even
    if nfft % 2:        # Odd
#        nyq = nfft//2 +1
        nyq = (nfft+1)//2
     # end if

    # Forward fourier transform:
    Ufft = fftmod.fft(uin, n=nfft, axis=axes) # defaults to last axis
    # mfft = nfft - nfft//2 - 1

    # zero out the negative frequency components and double
    # the power in the positive frequency components
#        # this is what we are doing:
#        Ufft[_ut.fast_slice(Ufft, axis=axes, start=nfft//2+1, end=None, step=1)] = 0.0
#        Ufft[_ut.fast_slice(Ufft, axis=axes, start=1, end=nfft//2+1, step=1)] *= 2.0
    # this is much faster in general for large arrays:
    Ufft[(slice(None),) * (axes % Ufft.ndim) + (slice(nyq+1, None),)] = 0.0
    Ufft[(slice(None),) * (axes % Ufft.ndim) + (slice(1, nyq),)] *= 2.0

    # Inverse Fourier transform is the analytic signal
    return fftmod.ifft(Ufft, n=nfft, axis=axes).squeeze()


def hilbert_1d(uin, nfft=None):
    """
    returns the analytic signal
       z = x + j*y
          x -- original signal
          y -- Hilbert transform of input H[u]

    inputs and parameters
        nfft - fft length
        mfft - Number of elements to zero out in the negative frequency space
        Ufft - discrete fourier transform of u
    Returns:
        analytic signal - u + j*H(u) - the inverse discrete fourier transform
                                       of H(Ufft)

    Note:
        -  It looks like scipy's result is multiplied by 1j to make it real
            Scipy's hilbert transform returns 1j*H[u]
    """
    if nfft is None:
        uin = _np.atleast_1d(uin)
        nfft = len(uin)
    # end if

    nyq = nfft//2       # Even
    if nfft % 2:        # Odd
#        nyq = nfft//2 +1
        nyq = (nfft+1)//2
     # end if

    # Forward fourier transform:
    Ufft = fftmod.fft(uin, n=nfft, axis=-1) # defaults to last axis

    # Create a mask to zero out the negative frequency components and double
    # the power in the positive frequency components
    h = _np.zeros(nfft)
    h[0] = 1.0        # don't change the DC value
    h[1:nyq] = 2.0*_np.ones(nyq-1) # double positive frequencies
#    h[1:nfft//2] = 2.0*_np.ones(nfft//2-1) # double positive frequencies
    h[nyq] = 1.0  # don't forget about the last point in the spectrum

    # Inverse Fourier transform is the analytic signal
    return fftmod.ifft(Ufft*h, n=nfft, axis=-1)


def test_hilbert(plotit=True, verbose=True):
    from scipy.fftpack import hilbert as scipyHilbert
    N = 32
    f = 1
    dt = 1.0/N
    t = []
    y = []
    z3 = []
    for n in range(N):
        x = 2*_np.pi*f*dt*n
        y.append(_np.sin(x))
        z3.append(-1.0*_np.cos(x))  # hilbert transform of a sine
        t.append(x)
    # end for
    z1 = hilbert(y)
#    z1 = hilbert_1d(y)

    # It looks like scipy's result is multiplied by 1j to make it real
    z2 = scipyHilbert(y)

#    # remove that weirdness in the residual and make it complex (conjugate)
#    res = (_np.asarray(z1)-(_np.asarray(y)-1j*_np.asarray(z2)) ).tolist()

    # or just compare to the mathematically accurate hilbert transform
    res1 = (_np.asarray(z1) - (_np.asarray(y) + 1j*_np.asarray(z3)) ).tolist()
    _np.allclose(_np.asarray(z1), (_np.asarray(y) + 1j*_np.asarray(z3)) )

    # for the verbose case, compare the scipy version to hilbert transform
    res2 = (_np.asarray(z2) - _np.asarray(z3) ).tolist()

    if verbose:
        print(" n      y       H[y]        scipy    my anal. sig.  residual of the analytic signal  ")
        for n in range(N):
            print('{:2d}    {:+5.2f}    {:+5.2f}   {:+10.2f}    {:+5.2f}    {:+10.4f}'.format(n, y[n], z3[n], z2[n], z1[n], res1[n]))
    #        print('{:2d}    {:+5.2f}    {:+5.2f}   {:+10.2f}    {:+5.2f}    {:+10.4f}'.format(n, y[n], z3[n], -1j*z2[n], z1[n], res1[n]))
        # end for
    # end if

    if plotit:
        _plt.figure()
        ax1 = _plt.subplot(3,1,1)
        ax1.plot(t, y, 'g-')
        ax1.plot(t, _np.abs(z1), 'b*')  # mathematically correct |analytic signal|
        ax1.plot(t, _np.abs(_np.asarray(y)+1j*_np.asarray(z2)), 'r-')
        ax1.plot(t, _np.abs(_np.asarray(y)+1j*_np.asarray(z3)), 'k--')
    #    ax1.plot(t, _np.abs(_np.imag(z1)), 'b*') # matches abs|z3|
    #    ax1.plot(t, _np.abs(z2), 'r-')
    #    ax1.plot(t, _np.abs(z3), 'k--')
        ax1.set_xlabel('t [s]')
        ax1.set_ylabel('y(t), a(t)')

        ax2 = _plt.subplot(3,1,2)
        ax2.plot(t, _np.imag(_np.asarray(z1)), 'b*')
        ax2.plot(t, z2, 'r-')
        ax2.plot(t, z3, 'k--')
        ax2.set_xlabel('t [s]')
        ax2.set_ylabel('|H[y]|')

        ax3 = _plt.subplot(3,1,3)
        ax3.plot(t, _np.abs(res1), 'b*')
        ax3.plot(t, _np.abs(res2), 'r-')
        ax3.set_xlabel('t [s]')
        ax3.set_ylabel('|Residual|')
    # end if
# end def

# ========================================================================== #
# ========================================================================== #
if __name__ == "__main__":
    test_hilbert()
# end if


#
# ========================================================================== #
# ========================================================================== #