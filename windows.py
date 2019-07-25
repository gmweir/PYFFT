"""The suite of window functions."""

from __future__ import division, print_function, absolute_import

#import operator
#import warnings

import numpy as _np

# check for version differences
try:
    from scipy import fftpack, linalg, special
    # from scipy._lib.six import string_types
    hasscipy = True

    __all__ = ['boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall',
               'blackmanharris', 'flattop', 'bartlett', 'hanning', 'barthann',
               'hamming', 'kaiser', 'gaussian', 'general_cosine','general_gaussian',
               'general_hamming', 'cosine', 'hann', 'chebwin', 'slepian', 'dpss',
               'exponential', 'tukey', 'get_window']
except:
    hasscipy = False
    __all__ = ['boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall',
               'blackmanharris', 'flattop', 'bartlett', 'hanning', 'barthann',
               'hamming', 'kaiser', 'gaussian', 'general_cosine','general_gaussian',
               'general_hamming', 'cosine', 'hann',
               'exponential', 'tukey', 'get_window']
# end try



def _len_guards(M):
    """Handle small or incorrect window lengths"""
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    return M <= 1


def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
    if not sym:
        return M + 1, True
    else:
        return M, False


def _truncate(w, needed):
    """Truncate window by 1 sample if needed for DFT-even symmetry"""
    if needed:
        return w[:-1]
    else:
        return w

# ========================================================================= #


def windows(windowfunction, **kwargs):
    verbose = kwargs.setdefault('verbose', True)
    periodic = kwargs.setdefault('periodic', True)  # periodic = True for FFT-analysis, False for filters
    msgout = kwargs.setdefault('msgout', False)

    # Define windowing function for apodization
    if windowfunction.lower().find('hann')>-1:
        # Hanning in reference, pg 54
        # highest sidelobe at +_2.36 bins
        # first zero at +-2.00 bins
        str1 = 'Hanning'
        str2 = '(SLDR~f**-3, PSLL=-31.5dB, ROV=50.0%, AF=1.000, PF=0.707, OC=0.167)'
        func = _np.hanning
        ROV = 0.50

    elif windowfunction.lower().find('hamm')>-1:
        # Hamming in reference, pg 55
        # first sidelobe is nearly minimal
        # discontinuity of 0.08 at the boundary
        # highest sidelobe is -42.7 dB at +-4.50 bins
        #
        str1 = 'Hamming'
        str2 = '(SLDR~f**-1, PSLL=-42.7dB, ROV=50.0%, AF=1.000, PF=0.761, OC=0.234)'
        func = _np.hamming
        ROV = 0.50

    elif windowfunction.lower().find('black')>-1:
        # Blackman-harris window in reference, pg 56
        # very low-sidelobes
        # first zero at +-4.00 bins
        # highest sidelobe at +-4.52 bins
        str1 = 'Blackman-Harris type'
        str2 = '(SLDR~f**-1, PSLL=-92dB, ROV=66.1%, AF=0.926, PF=0.718, OC=0.235)'
        # func = _np.blackman    # not exactly a Nutall3b

        def func(nwins):
            z = 2.0*_np.pi*_np.asarray(range(nwins), dtype=int)/nwins
            win = 0.35875 - 0.48829*_np.cos(z) + 0.14128*_np.cos(2*z) - 0.01168*_np.cos(3*z)
            return win
        ROV = 0.661

    elif windowfunction.lower().find('nut')>-1 or \
        windowfunction.lower().find('flat')>-1 or windowfunction.lower().find('sft')>-1:

        if windowfunction.lower().find('3f')>-1:
            # fast-decaying flat-top window, pg 67
            # first zero at +-3.00 bins, highest sidelobe at -31.7 dB, +-3.37 bins
            str1 = 'Fast-decaying Flattop'
            str2 = '(SLDR~f**-3, PSLL=-31.7dB, ROV=66.7%, AF=0.998, PF=0.558, OC=-0.029)'
            cc = (0.26526, -0.5, 0.23474)
            ROV = 0.667

        elif windowfunction.lower().find('4f')>-1:
            # fast-decaying flat-top window, pg 68
            # first zero at +-4.00 bins, highest sidelobe is -44.7 dB at +-4.33 bins
            str1 = 'Fast-decaying Flattop'
            str2 = '(SLDR~f**-5, PSLL=-44.7dB, ROV=75.0%, AF=1.000, PF=0.647, OC=0.039)'
            cc = (0.21706, -0.42103, 0.28294, -0.07897)
            ROV = 0.75

        elif windowfunction.lower().find('5f')>-1:
            # fast-decaying flat-top window, pg 69
            # first zero at +-5.00 bins, highest sidelobe is -57.3 dB at +-5.31 bins
            str1 = 'Fast-decaying Flattop'
            str2 = '(SLDR~f**-7, PSLL=-57.3dB, ROV=78.5%, AF=0.969, PF=0.648, OC=0.052)'
            cc = (0.1881, -0.36923, 0.28702, -0.13077, 0.02488)
            ROV = 0.785

        elif windowfunction.lower().find('3m')>-1:
            # minimum sidelobe flat-top window,
            # first zero at +-3.00 bins, highest sidelobe is -44.2 dB at +-5.50 bins
            str1 = 'Minimum sidelobe Flattop'
            str2 = '(SLDR~f**-1, PSLL=-44.2dB, ROV=65.5%, AF=0.949, PF=0.584, OC=-0.005)'
            cc = (0.28235, -0.52105, 0.19659)
            ROV = 0.655

        elif windowfunction.lower().find('4m')>-1:
            # minimum sidelobe flat-top window,
            # first zero at +-4.00 bins, highest sidelobe is -66.5 dB at +-10.50 bins
            str1 = 'Minimum sidelobe Flattop'
            str2 = '(SLDR~f**-1, PSLL=-66.5dB, ROV=72.1%, AF=0.964, PF=0.641, OC=0.044)'
            cc = (0.241906, -0.460841, 0.255381, -0.041872)
            ROV = 0.721

        elif windowfunction.lower().find('5m')>-1:
            # minimum sidelobe flat-top window,
            # first-zero at +-5.00 bins, highest at -89.9 dB, +-5.12 bins
            str1 = 'Minimum sidelobe Flattop'
            str2 = '(SLDR~f**-1, PSLL=-89.9dB, ROV=76.0%, AF=0.953, PF=0.645, OC=0.053)'
            cc = (0.209671, -0.407331, 0.281225, -0.092669, 0.0091036)
            ROV = 0.760

        elif windowfunction.lower().find('3a')>-1:
            # Nutall3a in reference, pg 58
            # once differentiable, low-spectral leakage
            # first zero at +-3.00 bins
            # highest sidelobe at -64.2dB at +-4.49 bins
            str1 = '3-term Blackman-Harris type'
            str2 = '(SLDR~f**-3, PSLL=-64.2dB, ROV=61.2%, AF=0.943, PF=0.723, OC=0.227)'
            cc = (0.40897, -0.5, 0.09103)
            ROV = 0.612

        elif windowfunction.lower().find('3b')>-1:
            # Nutall3b in reference, pg 59
            # once differentiable, low-spectral leakage
            # first zero at +-3.00 bins
            # highest sidelobe at -71.5 dB, +-3.64 bins
            str1 = '3-term Blackman-Harris type'
            str2 = '(SLDR~f**-1, PSLL=-71.5dB, ROV=59.8%, AF=0.939, PF=0.721, OC=0.229)'
            cc = (0.4243801,-0.4973406,0.0782793)
            ROV = 0.598

        elif windowfunction.lower().find('3')>-1:
            # Nutall3 in reference, pg 57
            # once differentiable, low-spectral leakage
            # first zero at +-3.00 bins
            # highest sidelobe at +-3.33 bins
            # maximally differentiable with 3 terms
            # identical to cos(pi*(j/N-0.5))**4.0
            str1 = '3-term Blackman-Harris type'
            str2 = '(SLDR~f**-5, PSLL=-46.7dB, ROV=64.7%, AF=0.969, PF=0.738, OC=0.228)'
            cc = (0.375, -0.5, 0.125)
            ROV = 0.647

        elif windowfunction.lower().find('4a')>-1:
            # Nutall4a in reference, pg 61
            # first zero at +-4.00 bins
            # highest sidelobe is -82.6 dB at +-5.45 bins

            str1 = '4-term Blackman-Harris type'
            str2 = '(SLDR~f**-5, PSLL=-82.6dB, ROV=68.0%, AF=0.931, PF=0.721, OC=0.234)'
            cc = (0.338946, -0.481973, 0.161054, -0.018027)
            ROV = 0.68

        elif windowfunction.lower().find('4b')>-1:
            # Nutall4b in reference, pg 62
            # first zero at +-4.00 bins, highest sidelobe is -93.3 dB at +-4.57 bins
            str1 = '4-term Blackman-Harris type'
            str2 = '(SLDR~f**-3, PSLL=-93.3dB, ROV=66.3%, AF=0.924, PF=0.715, OC=0.233)'
            cc = (0.355768, -0.487396, 0.144232, -0.012604)
            ROV = 0.663

        elif windowfunction.lower().find('4c')>-1:
            # Nutall4c in reference, pg 63
            # first zero is at +-4.00 bins, highest sidelobe is -98.1 dB at +-6.48 bins
            str1 = '4-term Blackman-Harris type'
            str2 = '(SLDR~f**-1, PSLL=-98.1dB, ROV=65.6%, AF=0.923, PF=0.716, OC=0.235)'
            cc = (0.3635819, -0.4891775, 0.1365995, -0.0106411)
            ROV = 0.656

        elif windowfunction.lower().find('4')>-1:
            # Nutall4 in reference, pg 60
            # maximally differentiable with 4 terms
            # identical to cos(pi*(j/N-0.5))**6.0
            # first zero at +-4.00 bins
            # highest sidelobe is -60.9 dB at +-4.3 bins

            str1 = '4-term Blackman-Harris type'
            str2 = '(SLDR~f**-7, PSLL=-60.9dB, ROV=70.5%, AF=0.937, PF=0.723, OC=0.233)'
            cc = (0.3125, -0.46875, 0.1875, -0.03125)
            ROV = 0.705

        # end if

        # nutall class of functions
        def func(nwins):
            z = 2.0*_np.pi*_np.asarray(range(nwins), dtype=int)/nwins
            win = _np.zeros((len(z),), dtype=_np.float64)
            for ii in range(len(cc)):
                if ii == 0:
                    win += cc[ii]
                else:
                    win += cc[ii]*_np.cos(ii*z)
                # end if
            # end for
            return win
        # end def

    elif windowfunction.lower().find('kaiser')>-1:
        beta = kwargs['beta']
        str1 = 'Kaiser type'
        str2 = '(parameters dependent on input shaping parameter %4.3f)'%(beta,)
        func = lambda nwins: _np.kaiser(nwins, beta)
        ROV = 2.0/3.0  # this is variable and depends on beta!  see table on page 37 of reference

    elif windowfunction.lower().find('welch')>-1:
        # parabolic function
        # first zero at +-1.43 bins
        # highest sidelobe is at -21.3 dB at 1.83 bins
        # low overlap correlation
        str1 = 'Welch'
        str2 = '(SLDR~f**-2, PSLL=-21.3dB, ROV=29.3%, AF=0.828, PF=0.707, OC=0.091)'
        def func(nwins):
            z = 2.0*_np.asarray(range(nwins), dtype=int)/nwins
            win = 1.0-(z-1.0)*(z-1.0)
            return win
        ROV = 0.293

    elif windowfunction.lower().find('bart')>-1:
        # page 52 of reference
        # first zero at +-2.00 bins,
        # highest sidelobe is -26.5 dB at +-2.86 bins
        str1 = 'Bartlett'
        str2 = '(SLDR~f**-2, PSLL=-26.5dB, ROV=50.0%, AF=1.000, PF=0.707, OC=0.250)'
        func = _np.bartlett
        ROV = 0.50

    else:
        # No window function (actually a uniform-window)
        # first zero at +- 1.00 bins
        # highest sidelobe is -13.3 dB at +-1.43 bins
        str1 = 'Rectangular'
        str2 = '(SLDR~f**-1, PSLL=-13.3dB, ROV=0.0%, AF=0, PF=1, OC=0)'
        func = lambda nwins: _np.ones( (nwins,), dtype=_np.float64)
        ROV = 0.0
    # endif windowfunction.lower()

    if 'nwins' in kwargs:
        nwins = kwargs['nwins']
        if periodic:
            str3 = 'periodic'
            win = func(nwins+1)  # periodic window
            win = win[:-1]  # truncate last point to make it periodic
        else:
            str3 = 'aperiodic'
            win = func(nwins)  # aperiodic window
        # end if
        val = win
        msg = 'Using a %s %s window function\n%s'%(str3,str1,str2)
    else:
        val = ROV
#        msg = 'Getting recommended overlap for a %s window function\n%s'%(str1,str2)
        msg = 'Getting recommended overlap for a %s window function'%(str1,)
    # end if

    if verbose:
        print(msg)
    if msgout:
        return val, (str1, str2)
    return val
# end def


# ========================================================================= #
def general_cosine(M, a, sym=True):
    r"""
    Generic weighted sum of cosine terms window

    Parameters
    ----------
    M : int
        Number of points in the output window
    a : array_like
        Sequence of weighting coefficients. This uses the convention of being
        centered on the origin, so these will typically all be positive
        numbers, not alternating sign.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    References
    ----------
    .. [1] A. Nuttall, "Some windows with very good sidelobe behavior," IEEE
           Transactions on Acoustics, Speech, and Signal Processing, vol. 29,
           no. 1, pp. 84-91, Feb 1981. :doi:`10.1109/TASSP.1981.1163506`.
    .. [2] Heinzel G. et al., "Spectrum and spectral density estimation by the
           Discrete Fourier transform (DFT), including a comprehensive list of
           window functions and some new flat-top windows", February 15, 2002
           https://holometer.fnal.gov/GH_FFT.pdf

    Examples
    --------
    Heinzel describes a flat-top window named "HFT90D" with formula: [2]_

    .. math::  w_j = 1 - 1.942604 \cos(z) + 1.340318 \cos(2z)
               - 0.440811 \cos(3z) + 0.043097 \cos(4z)

    where

    .. math::  z = \frac{2 \pi j}{N}, j = 0...N - 1

    Since this uses the convention of starting at the origin, to reproduce the
    window, we need to convert every other coefficient to a positive number:

    >>> HFT90D = [1, 1.942604, 1.340318, 0.440811, 0.043097]

    The paper states that the highest sidelobe is at -90.2 dB.  Reproduce
    Figure 42 by plotting the window and its frequency response, and confirm
    the sidelobe level in red:

    >>> from scipy.signal.windows import general_cosine
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = general_cosine(1000, HFT90D, sym=False)
    >>> plt.plot(window)
    >>> plt.title("HFT90D window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 10000) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = _np.abs(fftshift(A / abs(A).max()))
    >>> response = 20 * _np.log10(_np.maximum(response, 1e-10))
    >>> plt.plot(freq, response)
    >>> plt.axis([-50/1000, 50/1000, -140, 0])
    >>> plt.title("Frequency response of the HFT90D window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    >>> plt.axhline(-90.2, color='red')
    >>> plt.show()
    """
    if _len_guards(M):
        return _np.ones(M)
    M, needs_trunc = _extend(M, sym)

    fac = _np.linspace(-_np.pi, _np.pi, M)
    w = _np.zeros(M)
    for k in range(len(a)):
        w += a[k] * _np.cos(k * fac)

    return _truncate(w, needs_trunc)


def boxcar(M, sym=True):
    """Return a boxcar or rectangular window.

    Also known as a rectangular window or Dirichlet window, this is equivalent
    to no window at all.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        Whether the window is symmetric. (Has no effect for boxcar.)

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1.

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.boxcar(51)
    >>> plt.plot(window)
    >>> plt.title("Boxcar window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the boxcar window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return _np.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = _np.ones(M, float)

    return _truncate(w, needs_trunc)


def triang(M, sym=True):
    """Return a triangular window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    See Also
    --------
    bartlett : A triangular window that touches zero

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.triang(51)
    >>> plt.plot(window)
    >>> plt.title("Triangular window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = _np.abs(fftshift(A / abs(A).max()))
    >>> response = 20 * _np.log10(_np.maximum(response, 1e-10))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the triangular window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return _np.ones(M)
    M, needs_trunc = _extend(M, sym)

    n = _np.arange(1, (M + 1) // 2 + 1)
    if M % 2 == 0:
        w = (2 * n - 1.0) / M
        w = _np.r_[w, w[::-1]]
    else:
        w = 2 * n / (M + 1.0)
        w = _np.r_[w, w[-2::-1]]

    return _truncate(w, needs_trunc)


def parzen(M, sym=True):
    """Return a Parzen window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    References
    ----------
    .. [1] E. Parzen, "Mathematical Considerations in the Estimation of
           Spectra", Technometrics,  Vol. 3, No. 2 (May, 1961), pp. 167-190

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.parzen(51)
    >>> plt.plot(window)
    >>> plt.title("Parzen window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Parzen window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return _np.ones(M)
    M, needs_trunc = _extend(M, sym)

    n = _np.arange(-(M - 1) / 2.0, (M - 1) / 2.0 + 0.5, 1.0)
    na = _np.extract(n < -(M - 1) / 4.0, n)
    nb = _np.extract(abs(n) <= (M - 1) / 4.0, n)
    wa = 2 * (1 - _np.abs(na) / (M / 2.0)) ** 3.0
    wb = (1 - 6 * (_np.abs(nb) / (M / 2.0)) ** 2.0 +
          6 * (_np.abs(nb) / (M / 2.0)) ** 3.0)
    w = _np.r_[wa, wb, wa[::-1]]

    return _truncate(w, needs_trunc)


def bohman(M, sym=True):
    """Return a Bohman window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.bohman(51)
    >>> plt.plot(window)
    >>> plt.title("Bohman window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Bohman window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return _np.ones(M)
    M, needs_trunc = _extend(M, sym)

    fac = _np.abs(_np.linspace(-1, 1, M)[1:-1])
    w = (1 - fac) * _np.cos(_np.pi * fac) + 1.0 / _np.pi * _np.sin(_np.pi * fac)
    w = _np.r_[0, w, 0]

    return _truncate(w, needs_trunc)


def blackman(M, sym=True):
    r"""
    Return a Blackman window.

    The Blackman window is a taper formed by using the first three terms of
    a summation of cosines. It was designed to have close to the minimal
    leakage possible.  It is close to optimal, only slightly worse than a
    Kaiser window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Blackman window is defined as

    .. math::  w(n) = 0.42 - 0.5 \cos(2\pi n/M) + 0.08 \cos(4\pi n/M)

    The "exact Blackman" window was designed to null out the third and fourth
    sidelobes, but has discontinuities at the boundaries, resulting in a
    6 dB/oct fall-off.  This window is an approximation of the "exact" window,
    which does not null the sidelobes as well, but is smooth at the edges,
    improving the fall-off rate to 18 dB/oct. [3]_

    Most references to the Blackman window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function. It is known as a
    "near optimal" tapering function, almost as good (by some measures)
    as the Kaiser window.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] Oppenheim, A.V., and R.W. Schafer. Discrete-Time Signal Processing.
           Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.
    .. [3] Harris, Fredric J. (Jan 1978). "On the use of Windows for Harmonic
           Analysis with the Discrete Fourier Transform". Proceedings of the
           IEEE 66 (1): 51-83. :doi:`10.1109/PROC.1978.10837`.

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.blackman(51)
    >>> plt.plot(window)
    >>> plt.title("Blackman window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = _np.abs(fftshift(A / abs(A).max()))
    >>> response = 20 * _np.log10(_np.maximum(response, 1e-10))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Blackman window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # Docstring adapted from NumPy's blackman function
    return general_cosine(M, [0.42, 0.50, 0.08], sym)


def nuttall(M, sym=True):
    """Return a minimum 4-term Blackman-Harris window according to Nuttall.

    This variation is called "Nuttall4c" by Heinzel. [2]_

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    References
    ----------
    .. [1] A. Nuttall, "Some windows with very good sidelobe behavior," IEEE
           Transactions on Acoustics, Speech, and Signal Processing, vol. 29,
           no. 1, pp. 84-91, Feb 1981. :doi:`10.1109/TASSP.1981.1163506`.
    .. [2] Heinzel G. et al., "Spectrum and spectral density estimation by the
           Discrete Fourier transform (DFT), including a comprehensive list of
           window functions and some new flat-top windows", February 15, 2002
           https://holometer.fnal.gov/GH_FFT.pdf

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.nuttall(51)
    >>> plt.plot(window)
    >>> plt.title("Nuttall window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Nuttall window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    return general_cosine(M, [0.3635819, 0.4891775, 0.1365995, 0.0106411], sym)


def blackmanharris(M, sym=True):
    """Return a minimum 4-term Blackman-Harris window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.blackmanharris(51)
    >>> plt.plot(window)
    >>> plt.title("Blackman-Harris window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Blackman-Harris window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    return general_cosine(M, [0.35875, 0.48829, 0.14128, 0.01168], sym)


def flattop(M, sym=True):
    """Return a flat top window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    Flat top windows are used for taking accurate measurements of signal
    amplitude in the frequency domain, with minimal scalloping error from the
    center of a frequency bin to its edges, compared to others.  This is a
    5th-order cosine window, with the 5 terms optimized to make the main lobe
    maximally flat. [1]_

    References
    ----------
    .. [1] D'Antona, Gabriele, and A. Ferrero, "Digital Signal Processing for
           Measurement Systems", Springer Media, 2006, p. 70
           :doi:`10.1007/0-387-28666-7`.

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.flattop(51)
    >>> plt.plot(window)
    >>> plt.title("Flat top window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the flat top window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
    return general_cosine(M, a, sym)


def bartlett(M, sym=True):
    r"""
    Return a Bartlett window.

    The Bartlett window is very similar to a triangular window, except
    that the end points are at zero.  It is often used in signal
    processing for tapering a signal, without generating too much
    ripple in the frequency domain.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The triangular window, with the first and last samples equal to zero
        and the maximum value normalized to 1 (though the value 1 does not
        appear if `M` is even and `sym` is True).

    See Also
    --------
    triang : A triangular window that does not touch zero at the ends

    Notes
    -----
    The Bartlett window is defined as

    .. math:: w(n) = \frac{2}{M-1} \left(
              \frac{M-1}{2} - \left|n - \frac{M-1}{2}\right|
              \right)

    Most references to the Bartlett window come from the signal
    processing literature, where it is used as one of many windowing
    functions for smoothing values.  Note that convolution with this
    window produces linear interpolation.  It is also known as an
    apodization (which means"removing the foot", i.e. smoothing
    discontinuities at the beginning and end of the sampled signal) or
    tapering function. The Fourier transform of the Bartlett is the product
    of two sinc functions.
    Note the excellent discussion in Kanasewich. [2]_

    References
    ----------
    .. [1] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika 37, 1-16, 1950.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 109-110.
    .. [3] A.V. Oppenheim and R.W. Schafer, "Discrete-Time Signal
           Processing", Prentice-Hall, 1999, pp. 468-471.
    .. [4] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [5] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 429.

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.bartlett(51)
    >>> plt.plot(window)
    >>> plt.title("Bartlett window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Bartlett window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # Docstring adapted from NumPy's bartlett function
    if _len_guards(M):
        return _np.ones(M)
    M, needs_trunc = _extend(M, sym)

    n = _np.arange(0, M)
    w = _np.where(_np.less_equal(n, (M - 1) / 2.0),
                 2.0 * n / (M - 1), 2.0 - 2.0 * n / (M - 1))

    return _truncate(w, needs_trunc)


def hann(M, sym=True):
    r"""
    Return a Hann window.

    The Hann window is a taper formed by using a raised cosine or sine-squared
    with ends that touch zero.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Hann window is defined as

    .. math::  w(n) = 0.5 - 0.5 \cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    The window was named for Julius von Hann, an Austrian meteorologist. It is
    also known as the Cosine Bell. It is sometimes erroneously referred to as
    the "Hanning" window, from the use of "hann" as a verb in the original
    paper and confusion with the very similar Hamming window.

    Most references to the Hann window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 106-108.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.hann(51)
    >>> plt.plot(window)
    >>> plt.title("Hann window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = _np.abs(fftshift(A / abs(A).max()))
    >>> response = 20 * _np.log10(_np.maximum(response, 1e-10))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Hann window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # Docstring adapted from NumPy's hanning function
    return general_hamming(M, 0.5, sym)


@_np.deprecate(new_name='scipy.signal.windows.hann')
def hanning(*args, **kwargs):
    return hann(*args, **kwargs)


def tukey(M, alpha=0.5, sym=True):
    r"""Return a Tukey window, also known as a tapered cosine window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    alpha : float, optional
        Shape parameter of the Tukey window, representing the fraction of the
        window inside the cosine tapered region.
        If zero, the Tukey window is equivalent to a rectangular window.
        If one, the Tukey window is equivalent to a Hann window.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    References
    ----------
    .. [1] Harris, Fredric J. (Jan 1978). "On the use of Windows for Harmonic
           Analysis with the Discrete Fourier Transform". Proceedings of the
           IEEE 66 (1): 51-83. :doi:`10.1109/PROC.1978.10837`
    .. [2] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function#Tukey_window

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.tukey(51)
    >>> plt.plot(window)
    >>> plt.title("Tukey window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    >>> plt.ylim([0, 1.1])

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Tukey window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return _np.ones(M)

    if alpha <= 0:
        return _np.ones(M, 'd')
    elif alpha >= 1.0:
        return hann(M, sym=sym)

    M, needs_trunc = _extend(M, sym)

    n = _np.arange(0, M)
    width = int(_np.floor(alpha*(M-1)/2.0))
    n1 = n[0:width+1]
    n2 = n[width+1:M-width-1]
    n3 = n[M-width-1:]

    w1 = 0.5 * (1 + _np.cos(_np.pi * (-1 + 2.0*n1/alpha/(M-1))))
    w2 = _np.ones(n2.shape)
    w3 = 0.5 * (1 + _np.cos(_np.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))

    w = _np.concatenate((w1, w2, w3))

    return _truncate(w, needs_trunc)


def barthann(M, sym=True):
    """Return a modified Bartlett-Hann window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.barthann(51)
    >>> plt.plot(window)
    >>> plt.title("Bartlett-Hann window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Bartlett-Hann window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return _np.ones(M)
    M, needs_trunc = _extend(M, sym)

    n = _np.arange(0, M)
    fac = _np.abs(n / (M - 1.0) - 0.5)
    w = 0.62 - 0.48 * fac + 0.38 * _np.cos(2 * _np.pi * fac)

    return _truncate(w, needs_trunc)


def general_hamming(M, alpha, sym=True):
    r"""Return a generalized Hamming window.

    The generalized Hamming window is constructed by multiplying a rectangular
    window by one period of a cosine function [1]_.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    alpha : float
        The window coefficient, :math:`\alpha`
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The generalized Hamming window is defined as

    .. math:: w(n) = \alpha - \left(1 - \alpha\right) \cos\left(\frac{2\pi{n}}{M-1}\right)
              \qquad 0 \leq n \leq M-1

    Both the common Hamming window and Hann window are special cases of the
    generalized Hamming window with :math:`\alpha` = 0.54 and :math:`\alpha` =
    0.5, respectively [2]_.

    See Also
    --------
    hamming, hann

    Examples
    --------
    The Sentinel-1A/B Instrument Processing Facility uses generalized Hamming
    windows in the processing of spaceborne Synthetic Aperture Radar (SAR)
    data [3]_. The facility uses various values for the :math:`\alpha`
    parameter based on operating mode of the SAR instrument. Some common
    :math:`\alpha` values include 0.75, 0.7 and 0.52 [4]_. As an example, we
    plot these different windows.

    >>> from scipy.signal.windows import general_hamming
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> fig1, spatial_plot = plt.subplots()
    >>> spatial_plot.set_title("Generalized Hamming Windows")
    >>> spatial_plot.set_ylabel("Amplitude")
    >>> spatial_plot.set_xlabel("Sample")

    >>> fig2, freq_plot = plt.subplots()
    >>> freq_plot.set_title("Frequency Responses")
    >>> freq_plot.set_ylabel("Normalized magnitude [dB]")
    >>> freq_plot.set_xlabel("Normalized frequency [cycles per sample]")

    >>> for alpha in [0.75, 0.7, 0.52]:
    ...     window = general_hamming(41, alpha)
    ...     spatial_plot.plot(window, label="{:.2f}".format(alpha))
    ...     A = fft(window, 2048) / (len(window)/2.0)
    ...     freq = _np.linspace(-0.5, 0.5, len(A))
    ...     response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
    ...     freq_plot.plot(freq, response, label="{:.2f}".format(alpha))
    >>> freq_plot.legend(loc="upper right")
    >>> spatial_plot.legend(loc="upper right")

    References
    ----------
    .. [1] DSPRelated, "Generalized Hamming Window Family",
           https://www.dsprelated.com/freebooks/sasp/Generalized_Hamming_Window_Family.html
    .. [2] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [3] Riccardo Piantanida ESA, "Sentinel-1 Level 1 Detailed Algorithm
           Definition",
           https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Level-1-Detailed-Algorithm-Definition
    .. [4] Matthieu Bourbigot ESA, "Sentinel-1 Product Definition",
           https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Product-Definition
    """
    return general_cosine(M, [alpha, 1. - alpha], sym)


def hamming(M, sym=True):
    r"""Return a Hamming window.

    The Hamming window is a taper formed by using a raised cosine with
    non-zero endpoints, optimized to minimize the nearest side lobe.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Hamming window is defined as

    .. math::  w(n) = 0.54 - 0.46 \cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    The Hamming was named for R. W. Hamming, an associate of J. W. Tukey and
    is described in Blackman and Tukey. It was recommended for smoothing the
    truncated autocovariance function in the time domain.
    Most references to the Hamming window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 109-110.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.hamming(51)
    >>> plt.plot(window)
    >>> plt.title("Hamming window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Hamming window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # Docstring adapted from NumPy's hamming function
    return general_hamming(M, 0.54, sym)


def gaussian(M, std, sym=True):
    r"""Return a Gaussian window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    std : float
        The standard deviation, sigma.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Gaussian window is defined as

    .. math::  w(n) = e^{ -\frac{1}{2}\left(\frac{n}{\sigma}\right)^2 }

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.gaussian(51, std=7)
    >>> plt.plot(window)
    >>> plt.title(r"Gaussian window ($\sigma$=7)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title(r"Frequency response of the Gaussian window ($\sigma$=7)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return _np.ones(M)
    M, needs_trunc = _extend(M, sym)

    n = _np.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = _np.exp(-n ** 2 / sig2)

    return _truncate(w, needs_trunc)


def general_gaussian(M, p, sig, sym=True):
    r"""Return a window with a generalized Gaussian shape.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    p : float
        Shape parameter.  p = 1 is identical to `gaussian`, p = 0.5 is
        the same shape as the Laplace distribution.
    sig : float
        The standard deviation, sigma.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The generalized Gaussian window is defined as

    .. math::  w(n) = e^{ -\frac{1}{2}\left|\frac{n}{\sigma}\right|^{2p} }

    the half-power point is at

    .. math::  (2 \log(2))^{1/(2 p)} \sigma

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.general_gaussian(51, p=1.5, sig=7)
    >>> plt.plot(window)
    >>> plt.title(r"Generalized Gaussian window (p=1.5, $\sigma$=7)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title(r"Freq. resp. of the gen. Gaussian "
    ...           r"window (p=1.5, $\sigma$=7)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return _np.ones(M)
    M, needs_trunc = _extend(M, sym)

    n = _np.arange(0, M) - (M - 1.0) / 2.0
    w = _np.exp(-0.5 * _np.abs(n / sig) ** (2 * p))

    return _truncate(w, needs_trunc)


def cosine(M, sym=True):
    """Return a window with a simple cosine shape.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----

    .. versionadded:: 0.13.0

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.cosine(51)
    >>> plt.plot(window)
    >>> plt.title("Cosine window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the cosine window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    >>> plt.show()

    """
    if _len_guards(M):
        return _np.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = _np.sin(_np.pi / M * (_np.arange(0, M) + .5))

    return _truncate(w, needs_trunc)


def exponential(M, center=None, tau=1., sym=True):
    r"""Return an exponential (or Poisson) window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    center : float, optional
        Parameter defining the center location of the window function.
        The default value if not given is ``center = (M-1) / 2``.  This
        parameter must take its default value for symmetric windows.
    tau : float, optional
        Parameter defining the decay.  For ``center = 0`` use
        ``tau = -(M-1) / ln(x)`` if ``x`` is the fraction of the window
        remaining at the end.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Exponential window is defined as

    .. math::  w(n) = e^{-|n-center| / \tau}

    References
    ----------
    S. Gade and H. Herlufsen, "Windows to FFT analysis (Part I)",
    Technical Review 3, Bruel & Kjaer, 1987.

    Examples
    --------
    Plot the symmetric window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> M = 51
    >>> tau = 3.0
    >>> window = signal.exponential(M, tau=tau)
    >>> plt.plot(window)
    >>> plt.title("Exponential Window (tau=3.0)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = _np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -35, 0])
    >>> plt.title("Frequency response of the Exponential window (tau=3.0)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    This function can also generate non-symmetric windows:

    >>> tau2 = -(M-1) / _np.log(0.01)
    >>> window2 = signal.exponential(M, 0, tau2, False)
    >>> plt.figure()
    >>> plt.plot(window2)
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    """
    if sym and center is not None:
        raise ValueError("If sym==True, center must be None.")
    if _len_guards(M):
        return _np.ones(M)
    M, needs_trunc = _extend(M, sym)

    if center is None:
        center = (M-1) / 2

    n = _np.arange(0, M)
    w = _np.exp(-_np.abs(n-center) / tau)

    return _truncate(w, needs_trunc)



# ========================================================================= #
#
#
# Check if scipy is playing well today or not (version differences!)

if hasscipy:
    def kaiser(M, beta, sym=True):
        r"""Return a Kaiser window.

        The Kaiser window is a taper formed by using a Bessel function.

        Parameters
        ----------
        M : int
            Number of points in the output window. If zero or less, an empty
            array is returned.
        beta : float
            Shape parameter, determines trade-off between main-lobe width and
            side lobe level. As beta gets large, the window narrows.
        sym : bool, optional
            When True (default), generates a symmetric window, for use in filter
            design.
            When False, generates a periodic window, for use in spectral analysis.

        Returns
        -------
        w : ndarray
            The window, with the maximum value normalized to 1 (though the value 1
            does not appear if `M` is even and `sym` is True).

        Notes
        -----
        The Kaiser window is defined as

        .. math::  w(n) = I_0\left( \beta \sqrt{1-\frac{4n^2}{(M-1)^2}}
                   \right)/I_0(\beta)

        with

        .. math:: \quad -\frac{M-1}{2} \leq n \leq \frac{M-1}{2},

        where :math:`I_0` is the modified zeroth-order Bessel function.

        The Kaiser was named for Jim Kaiser, who discovered a simple approximation
        to the DPSS window based on Bessel functions.
        The Kaiser window is a very good approximation to the Digital Prolate
        Spheroidal Sequence, or Slepian window, which is the transform which
        maximizes the energy in the main lobe of the window relative to total
        energy.

        The Kaiser can approximate other windows by varying the beta parameter.
        (Some literature uses alpha = beta/pi.) [4]_

        ====  =======================
        beta  Window shape
        ====  =======================
        0     Rectangular
        5     Similar to a Hamming
        6     Similar to a Hann
        8.6   Similar to a Blackman
        ====  =======================

        A beta value of 14 is probably a good starting point. Note that as beta
        gets large, the window narrows, and so the number of samples needs to be
        large enough to sample the increasingly narrow spike, otherwise NaNs will
        be returned.

        Most references to the Kaiser window come from the signal processing
        literature, where it is used as one of many windowing functions for
        smoothing values.  It is also known as an apodization (which means
        "removing the foot", i.e. smoothing discontinuities at the beginning
        and end of the sampled signal) or tapering function.

        References
        ----------
        .. [1] J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by
               digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.
               John Wiley and Sons, New York, (1966).
        .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
               University of Alberta Press, 1975, pp. 177-178.
        .. [3] Wikipedia, "Window function",
               https://en.wikipedia.org/wiki/Window_function
        .. [4] F. J. Harris, "On the use of windows for harmonic analysis with the
               discrete Fourier transform," Proceedings of the IEEE, vol. 66,
               no. 1, pp. 51-83, Jan. 1978. :doi:`10.1109/PROC.1978.10837`.

        Examples
        --------
        Plot the window and its frequency response:

        >>> from scipy import signal
        >>> from scipy.fftpack import fft, fftshift
        >>> import matplotlib.pyplot as plt

        >>> window = signal.kaiser(51, beta=14)
        >>> plt.plot(window)
        >>> plt.title(r"Kaiser window ($\beta$=14)")
        >>> plt.ylabel("Amplitude")
        >>> plt.xlabel("Sample")

        >>> plt.figure()
        >>> A = fft(window, 2048) / (len(window)/2.0)
        >>> freq = _np.linspace(-0.5, 0.5, len(A))
        >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
        >>> plt.plot(freq, response)
        >>> plt.axis([-0.5, 0.5, -120, 0])
        >>> plt.title(r"Frequency response of the Kaiser window ($\beta$=14)")
        >>> plt.ylabel("Normalized magnitude [dB]")
        >>> plt.xlabel("Normalized frequency [cycles per sample]")

        """
        # Docstring adapted from NumPy's kaiser function
        if _len_guards(M):
            return _np.ones(M)
        M, needs_trunc = _extend(M, sym)

        n = _np.arange(0, M)
        alpha = (M - 1) / 2.0
        w = (special.i0(beta * _np.sqrt(1 - ((n - alpha) / alpha) ** 2.0)) /
             special.i0(beta))

        return _truncate(w, needs_trunc)

    # `chebwin` contributed by Kumar Appaiah.
    def chebwin(M, at, sym=True):
        r"""Return a Dolph-Chebyshev window.

        Parameters
        ----------
        M : int
            Number of points in the output window. If zero or less, an empty
            array is returned.
        at : float
            Attenuation (in dB).
        sym : bool, optional
            When True (default), generates a symmetric window, for use in filter
            design.
            When False, generates a periodic window, for use in spectral analysis.

        Returns
        -------
        w : ndarray
            The window, with the maximum value always normalized to 1

        Notes
        -----
        This window optimizes for the narrowest main lobe width for a given order
        `M` and sidelobe equiripple attenuation `at`, using Chebyshev
        polynomials.  It was originally developed by Dolph to optimize the
        directionality of radio antenna arrays.

        Unlike most windows, the Dolph-Chebyshev is defined in terms of its
        frequency response:

        .. math:: W(k) = \frac
                  {\cos\{M \cos^{-1}[\beta \cos(\frac{\pi k}{M})]\}}
                  {\cosh[M \cosh^{-1}(\beta)]}

        where

        .. math:: \beta = \cosh \left [\frac{1}{M}
                  \cosh^{-1}(10^\frac{A}{20}) \right ]

        and 0 <= abs(k) <= M-1. A is the attenuation in decibels (`at`).

        The time domain window is then generated using the IFFT, so
        power-of-two `M` are the fastest to generate, and prime number `M` are
        the slowest.

        The equiripple condition in the frequency domain creates impulses in the
        time domain, which appear at the ends of the window.

        References
        ----------
        .. [1] C. Dolph, "A current distribution for broadside arrays which
               optimizes the relationship between beam width and side-lobe level",
               Proceedings of the IEEE, Vol. 34, Issue 6
        .. [2] Peter Lynch, "The Dolph-Chebyshev Window: A Simple Optimal Filter",
               American Meteorological Society (April 1997)
               http://mathsci.ucd.ie/~plynch/Publications/Dolph.pdf
        .. [3] F. J. Harris, "On the use of windows for harmonic analysis with the
               discrete Fourier transforms", Proceedings of the IEEE, Vol. 66,
               No. 1, January 1978

        Examples
        --------
        Plot the window and its frequency response:

        >>> from scipy import signal
        >>> from scipy.fftpack import fft, fftshift
        >>> import matplotlib.pyplot as plt

        >>> window = signal.chebwin(51, at=100)
        >>> plt.plot(window)
        >>> plt.title("Dolph-Chebyshev window (100 dB)")
        >>> plt.ylabel("Amplitude")
        >>> plt.xlabel("Sample")

        >>> plt.figure()
        >>> A = fft(window, 2048) / (len(window)/2.0)
        >>> freq = _np.linspace(-0.5, 0.5, len(A))
        >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
        >>> plt.plot(freq, response)
        >>> plt.axis([-0.5, 0.5, -120, 0])
        >>> plt.title("Frequency response of the Dolph-Chebyshev window (100 dB)")
        >>> plt.ylabel("Normalized magnitude [dB]")
        >>> plt.xlabel("Normalized frequency [cycles per sample]")

        """
        if _np.abs(at) < 45:
#            warnings.warn("This window is not suitable for spectral analysis "
#                          "for attenuation values lower than about 45dB because "
#                          "the equivalent noise bandwidth of a Chebyshev window "
#                          "does not grow monotonically with increasing sidelobe "
#                          "attenuation when the attenuation is smaller than "
#                          "about 45 dB.")
            print("This window is not suitable for spectral analysis "
                  "for attenuation values lower than about 45dB because "
                  "the equivalent noise bandwidth of a Chebyshev window "
                  "does not grow monotonically with increasing sidelobe "
                  "attenuation when the attenuation is smaller than "
                  "about 45 dB.")
        if _len_guards(M):
            return _np.ones(M)
        M, needs_trunc = _extend(M, sym)

        # compute the parameter beta
        order = M - 1.0
        beta = _np.cosh(1.0 / order * _np.arccosh(10 ** (_np.abs(at) / 20.)))
        k = _np.r_[0:M] * 1.0
        x = beta * _np.cos(_np.pi * k / M)
        # Find the window's DFT coefficients
        # Use analytic definition of Chebyshev polynomial instead of expansion
        # from scipy.special. Using the expansion in scipy.special leads to errors.
        p = _np.zeros(x.shape)
        p[x > 1] = _np.cosh(order * _np.arccosh(x[x > 1]))
        p[x < -1] = (2 * (M % 2) - 1) * _np.cosh(order * _np.arccosh(-x[x < -1]))
        p[_np.abs(x) <= 1] = _np.cos(order * _np.arccos(x[_np.abs(x) <= 1]))

        # Appropriate IDFT and filling up
        # depending on even/odd M
        if M % 2:
            w = _np.real(fftpack.fft(p))
            n = (M + 1) // 2
            w = w[:n]
            w = _np.concatenate((w[n - 1:0:-1], w))
        else:
            p = p * _np.exp(1.j * _np.pi / M * _np.r_[0:M])
            w = _np.real(fftpack.fft(p))
            n = M // 2 + 1
            w = _np.concatenate((w[n - 1:0:-1], w[1:n]))
        w = w / max(w)

        return _truncate(w, needs_trunc)


    def slepian(M, width, sym=True):
        """Return a digital Slepian (DPSS) window.

        Used to maximize the energy concentration in the main lobe.  Also called
        the digital prolate spheroidal sequence (DPSS).

        .. note:: Deprecated in SciPy 1.1.
                  `slepian` will be removed in a future version of SciPy, it is
                  replaced by `dpss`, which uses the standard definition of a
                  digital Slepian window.

        Parameters
        ----------
        M : int
            Number of points in the output window. If zero or less, an empty
            array is returned.
        width : float
            Bandwidth
        sym : bool, optional
            When True (default), generates a symmetric window, for use in filter
            design.
            When False, generates a periodic window, for use in spectral analysis.

        Returns
        -------
        w : ndarray
            The window, with the maximum value always normalized to 1

        See Also
        --------
        dpss

        References
        ----------
        .. [1] D. Slepian & H. O. Pollak: "Prolate spheroidal wave functions,
               Fourier analysis and uncertainty-I," Bell Syst. Tech. J., vol.40,
               pp.43-63, 1961. https://archive.org/details/bstj40-1-43
        .. [2] H. J. Landau & H. O. Pollak: "Prolate spheroidal wave functions,
               Fourier analysis and uncertainty-II," Bell Syst. Tech. J. , vol.40,
               pp.65-83, 1961. https://archive.org/details/bstj40-1-65

        Examples
        --------
        Plot the window and its frequency response:

        >>> from scipy import signal
        >>> from scipy.fftpack import fft, fftshift
        >>> import matplotlib.pyplot as plt

        >>> window = signal.slepian(51, width=0.3)
        >>> plt.plot(window)
        >>> plt.title("Slepian (DPSS) window (BW=0.3)")
        >>> plt.ylabel("Amplitude")
        >>> plt.xlabel("Sample")

        >>> plt.figure()
        >>> A = fft(window, 2048) / (len(window)/2.0)
        >>> freq = _np.linspace(-0.5, 0.5, len(A))
        >>> response = 20 * _np.log10(_np.abs(fftshift(A / abs(A).max())))
        >>> plt.plot(freq, response)
        >>> plt.axis([-0.5, 0.5, -120, 0])
        >>> plt.title("Frequency response of the Slepian window (BW=0.3)")
        >>> plt.ylabel("Normalized magnitude [dB]")
        >>> plt.xlabel("Normalized frequency [cycles per sample]")

        """
#        warnings.warn('slepian is deprecated and will be removed in a future '
#                      'version, use dpss instead', DeprecationWarning)
        print('slepian is deprecated and will be removed in a future '
              'version, use dpss instead', DeprecationWarning)
        if _len_guards(M):
            return _np.ones(M)
        M, needs_trunc = _extend(M, sym)

        # our width is the full bandwidth
        width = width / 2
        # to match the old version
        width = width / 2
        m = _np.arange(M, dtype='d')
        H = _np.zeros((2, M))
        H[0, 1:] = m[1:] * (M - m[1:]) / 2
        H[1, :] = ((M - 1 - 2 * m) / 2)**2 * _np.cos(2 * _np.pi * width)

        _, win = linalg.eig_banded(H, select='i', select_range=(M-1, M-1))
        win = win.ravel() / win.max()

        return _truncate(win, needs_trunc)


    def dpss(M, NW, Kmax=None, sym=True, norm=None, return_ratios=False):
        """
        Compute the Discrete Prolate Spheroidal Sequences (DPSS).

        DPSS (or Slepian sequences) are often used in multitaper power spectral
        density estimation (see [1]_). The first window in the sequence can be
        used to maximize the energy concentration in the main lobe, and is also
        called the Slepian window.

        Parameters
        ----------
        M : int
            Window length.
        NW : float
            Standardized half bandwidth corresponding to ``2*NW = BW/f0 = BW*N*dt``
            where ``dt`` is taken as 1.
        Kmax : int | None, optional
            Number of DPSS windows to return (orders ``0`` through ``Kmax-1``).
            If None (default), return only a single window of shape ``(M,)``
            instead of an array of windows of shape ``(Kmax, M)``.
        sym : bool, optional
            When True (default), generates a symmetric window, for use in filter
            design.
            When False, generates a periodic window, for use in spectral analysis.
        norm : {2, 'approximate', 'subsample'} | None, optional
            If 'approximate' or 'subsample', then the windows are normalized by the
            maximum, and a correction scale-factor for even-length windows
            is applied either using ``M**2/(M**2+NW)`` ("approximate") or
            a FFT-based subsample shift ("subsample"), see Notes for details.
            If None, then "approximate" is used when ``Kmax=None`` and 2 otherwise
            (which uses the l2 norm).
        return_ratios : bool, optional
            If True, also return the concentration ratios in addition to the
            windows.

        Returns
        -------
        v : ndarray, shape (Kmax, N) or (N,)
            The DPSS windows. Will be 1D if `Kmax` is None.
        r : ndarray, shape (Kmax,) or float, optional
            The concentration ratios for the windows. Only returned if
            `return_ratios` evaluates to True. Will be 0D if `Kmax` is None.

        Notes
        -----
        This computation uses the tridiagonal eigenvector formulation given
        in [2]_.

        The default normalization for ``Kmax=None``, i.e. window-generation mode,
        simply using the l-infinity norm would create a window with two unity
        values, which creates slight normalization differences between even and odd
        orders. The approximate correction of ``M**2/float(M**2+NW)`` for even
        sample numbers is used to counteract this effect (see Examples below).

        For very long signals (e.g., 1e6 elements), it can be useful to compute
        windows orders of magnitude shorter and use interpolation (e.g.,
        `scipy.interpolate.interp1d`) to obtain tapers of length `M`,
        but this in general will not preserve orthogonality between the tapers.

        .. versionadded:: 1.1

        References
        ----------
        .. [1] Percival DB, Walden WT. Spectral Analysis for Physical Applications:
           Multitaper and Conventional Univariate Techniques.
           Cambridge University Press; 1993.
        .. [2] Slepian, D. Prolate spheroidal wave functions, Fourier analysis, and
           uncertainty V: The discrete case. Bell System Technical Journal,
           Volume 57 (1978), 1371430.
        .. [3] Kaiser, JF, Schafer RW. On the Use of the I0-Sinh Window for
           Spectrum Analysis. IEEE Transactions on Acoustics, Speech and
           Signal Processing. ASSP-28 (1): 105-107; 1980.

        Examples
        --------
        We can compare the window to `kaiser`, which was invented as an alternative
        that was easier to calculate [3]_ (example adapted from
        `here <https://ccrma.stanford.edu/~jos/sasp/Kaiser_DPSS_Windows_Compared.html>`_):

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from scipy.signal import windows, freqz
        >>> N = 51
        >>> fig, axes = plt.subplots(3, 2, figsize=(5, 7))
        >>> for ai, alpha in enumerate((1, 3, 5)):
        ...     win_dpss = windows.dpss(N, alpha)
        ...     beta = alpha*_np.pi
        ...     win_kaiser = windows.kaiser(N, beta)
        ...     for win, c in ((win_dpss, 'k'), (win_kaiser, 'r')):
        ...         win /= win.sum()
        ...         axes[ai, 0].plot(win, color=c, lw=1.)
        ...         axes[ai, 0].set(xlim=[0, N-1], title=r'$\\alpha$ = %s' % alpha,
        ...                         ylabel='Amplitude')
        ...         w, h = freqz(win)
        ...         axes[ai, 1].plot(w, 20 * _np.log10(_np.abs(h)), color=c, lw=1.)
        ...         axes[ai, 1].set(xlim=[0, _np.pi],
        ...                         title=r'$\\beta$ = %0.2f' % beta,
        ...                         ylabel='Magnitude (dB)')
        >>> for ax in axes.ravel():
        ...     ax.grid(True)
        >>> axes[2, 1].legend(['DPSS', 'Kaiser'])
        >>> fig.tight_layout()
        >>> plt.show()

        And here are examples of the first four windows, along with their
        concentration ratios:

        >>> M = 512
        >>> NW = 2.5
        >>> win, eigvals = windows.dpss(M, NW, 4, return_ratios=True)
        >>> fig, ax = plt.subplots(1)
        >>> ax.plot(win.T, linewidth=1.)
        >>> ax.set(xlim=[0, M-1], ylim=[-0.1, 0.1], xlabel='Samples',
        ...        title='DPSS, M=%d, NW=%0.1f' % (M, NW))
        >>> ax.legend(['win[%d] (%0.4f)' % (ii, ratio)
        ...            for ii, ratio in enumerate(eigvals)])
        >>> fig.tight_layout()
        >>> plt.show()

        Using a standard :math:`l_{\\infty}` norm would produce two unity values
        for even `M`, but only one unity value for odd `M`. This produces uneven
        window power that can be counteracted by the approximate correction
        ``M**2/float(M**2+NW)``, which can be selected by using
        ``norm='approximate'`` (which is the same as ``norm=None`` when
        ``Kmax=None``, as is the case here). Alternatively, the slower
        ``norm='subsample'`` can be used, which uses subsample shifting in the
        frequency domain (FFT) to compute the correction:

        >>> Ms = _np.arange(1, 41)
        >>> factors = (50, 20, 10, 5, 2.0001)
        >>> energy = _np.empty((3, len(Ms), len(factors)))
        >>> for mi, M in enumerate(Ms):
        ...     for fi, factor in enumerate(factors):
        ...         NW = M / float(factor)
        ...         # Corrected using empirical approximation (default)
        ...         win = windows.dpss(M, NW)
        ...         energy[0, mi, fi] = _np.sum(win ** 2) / _np.sqrt(M)
        ...         # Corrected using subsample shifting
        ...         win = windows.dpss(M, NW, norm='subsample')
        ...         energy[1, mi, fi] = _np.sum(win ** 2) / _np.sqrt(M)
        ...         # Uncorrected (using l-infinity norm)
        ...         win /= win.max()
        ...         energy[2, mi, fi] = _np.sum(win ** 2) / _np.sqrt(M)
        >>> fig, ax = plt.subplots(1)
        >>> hs = ax.plot(Ms, energy[2], '-o', markersize=4,
        ...              markeredgecolor='none')
        >>> leg = [hs[-1]]
        >>> for hi, hh in enumerate(hs):
        ...     h1 = ax.plot(Ms, energy[0, :, hi], '-o', markersize=4,
        ...                  color=hh.get_color(), markeredgecolor='none',
        ...                  alpha=0.66)
        ...     h2 = ax.plot(Ms, energy[1, :, hi], '-o', markersize=4,
        ...                  color=hh.get_color(), markeredgecolor='none',
        ...                  alpha=0.33)
        ...     if hi == len(hs) - 1:
        ...         leg.insert(0, h1[0])
        ...         leg.insert(0, h2[0])
        >>> ax.set(xlabel='M (samples)', ylabel=r'Power / $\\sqrt{M}$')
        >>> ax.legend(leg, ['Uncorrected', r'Corrected: $\\frac{M^2}{M^2+NW}$',
        ...                 'Corrected (subsample)'])
        >>> fig.tight_layout()

        """  # noqa: E501
        if _len_guards(M):
            return _np.ones(M)
        if norm is None:
            norm = 'approximate' if Kmax is None else 2
        known_norms = (2, 'approximate', 'subsample')
        if norm not in known_norms:
            raise ValueError('norm must be one of %s, got %s'
                             % (known_norms, norm))
        if Kmax is None:
            singleton = True
            Kmax = 1
        else:
            singleton = False
#        Kmax = operator.index(Kmax)
        Kmax = int(Kmax)
        if not 0 < Kmax <= M:
            raise ValueError('Kmax must be greater than 0 and less than M')
        if NW >= M/2.:
            raise ValueError('NW must be less than M/2.')
        if NW <= 0:
            raise ValueError('NW must be positive')
        M, needs_trunc = _extend(M, sym)
        W = float(NW) / M
        nidx = _np.arange(M)

        # Here we want to set up an optimization problem to find a sequence
        # whose energy is maximally concentrated within band [-W,W].
        # Thus, the measure lambda(T,W) is the ratio between the energy within
        # that band, and the total energy. This leads to the eigen-system
        # (A - (l1)I)v = 0, where the eigenvector corresponding to the largest
        # eigenvalue is the sequence with maximally concentrated energy. The
        # collection of eigenvectors of this system are called Slepian
        # sequences, or discrete prolate spheroidal sequences (DPSS). Only the
        # first K, K = 2NW/dt orders of DPSS will exhibit good spectral
        # concentration
        # [see https://en.wikipedia.org/wiki/Spectral_concentration_problem]

        # Here we set up an alternative symmetric tri-diagonal eigenvalue
        # problem such that
        # (B - (l2)I)v = 0, and v are our DPSS (but eigenvalues l2 != l1)
        # the main diagonal = ([N-1-2*t]/2)**2 cos(2PIW), t=[0,1,2,...,N-1]
        # and the first off-diagonal = t(N-t)/2, t=[1,2,...,N-1]
        # [see Percival and Walden, 1993]
        d = ((M - 1 - 2 * nidx) / 2.) ** 2 * _np.cos(2 * _np.pi * W)
        e = nidx[1:] * (M - nidx[1:]) / 2.

        # only calculate the highest Kmax eigenvalues
        w, windows = linalg.eigh_tridiagonal(
            d, e, select='i', select_range=(M - Kmax, M - 1))
        w = w[::-1]
        windows = windows[:, ::-1].T

        # By convention (Percival and Walden, 1993 pg 379)
        # * symmetric tapers (k=0,2,4,...) should have a positive average.
        fix_even = (windows[::2].sum(axis=1) < 0)
        for i, f in enumerate(fix_even):
            if f:
                windows[2 * i] *= -1
        # * antisymmetric tapers should begin with a positive lobe
        #   (this depends on the definition of "lobe", here we'll take the first
        #   point above the numerical noise, which should be good enough for
        #   sufficiently smooth functions, and more robust than relying on an
        #   algorithm that uses max(abs(w)), which is susceptible to numerical
        #   noise problems)
        thresh = max(1e-7, 1. / M)
        for i, w in enumerate(windows[1::2]):
            if w[w * w > thresh][0] < 0:
                windows[2 * i + 1] *= -1

        # Now find the eigenvalues of the original spectral concentration problem
        # Use the autocorr sequence technique from Percival and Walden, 1993 pg 390
        if return_ratios:
            dpss_rxx = _fftautocorr(windows)
            r = 4 * W * _np.sinc(2 * W * nidx)
            r[0] = 2 * W
            ratios = _np.dot(dpss_rxx, r)
            if singleton:
                ratios = ratios[0]
        # Deal with sym and Kmax=None
        if norm != 2:
            windows /= windows.max()
            if M % 2 == 0:
                if norm == 'approximate':
                    correction = M**2 / float(M**2 + NW)
                else:
                    s = _np.fft.rfft(windows[0])
                    shift = -(1 - 1./M) * _np.arange(1, M//2 + 1)
                    s[1:] *= 2 * _np.exp(-1j * _np.pi * shift)
                    correction = M / s.real.sum()
                windows *= correction
        # else we're already l2 normed, so do nothing
        if needs_trunc:
            windows = windows[:, :-1]
        if singleton:
            windows = windows[0]
        return (windows, ratios) if return_ratios else windows


    def _fftautocorr(x):
        """Compute the autocorrelation of a real array and crop the result."""
        N = x.shape[-1]
        use_N = fftpack.next_fast_len(2*N-1)
        x_fft = _np.fft.rfft(x, use_N, axis=-1)
        cxy = _np.fft.irfft(x_fft * x_fft.conj(), n=use_N)[:, :N]
        return cxy

else:
    def kaiser(*args):
        return _np.kaiser(*args)

    def chebwin(*args):
        raise NotImplementedError

    def slepian(*args):
        raise NotImplementedError

    def _fftautocorr(x):
        """Compute the autocorrelation of a real array and crop the result."""
        N = x.shape[-1]
        # Or equivalently (but in most cases slower):
        cxy = _np.array([_np.convolve(xx, yy[::-1], mode='full')
                        for xx, yy in zip(x, x)])[:, N-1:2*N-1]
        return cxy
# end if

# ========================================================================= #
# ========================================================================= #

_win_equiv_raw = {
    ('barthann', 'brthan', 'bth'): (barthann, False),
    ('bartlett', 'bart', 'brt'): (bartlett, False),
    ('blackman', 'black', 'blk'): (blackman, False),
    ('blackmanharris', 'blackharr', 'bkh'): (blackmanharris, False),
    ('bohman', 'bman', 'bmn'): (bohman, False),
    ('boxcar', 'box', 'ones',
        'rect', 'rectangular'): (boxcar, False),
    ('chebwin', 'cheb'): (chebwin, True),
    ('cosine', 'halfcosine'): (cosine, False),
    ('exponential', 'poisson'): (exponential, True),
    ('flattop', 'flat', 'flt'): (flattop, False),
    ('gaussian', 'gauss', 'gss'): (gaussian, True),
    ('general gaussian', 'general_gaussian',
        'general gauss', 'general_gauss', 'ggs'): (general_gaussian, True),
    ('hamming', 'hamm', 'ham'): (hamming, False),
    ('hanning', 'hann', 'han'): (hann, False),
    ('kaiser', 'ksr'): (kaiser, True),
    ('nuttall', 'nutl', 'nut'): (nuttall, False),
    ('parzen', 'parz', 'par'): (parzen, False),
    ('slepian', 'slep', 'optimal', 'dpss', 'dss'): (slepian, True),
    ('triangle', 'triang', 'tri'): (triang, False),
    ('tukey', 'tuk'): (tukey, True),
}

# Fill dict with all valid window name strings
_win_equiv = {}
for k, v in _win_equiv_raw.items():
    for key in k:
        _win_equiv[key] = v[0]

# Keep track of which windows need additional parameters
_needs_param = set()
for k, v in _win_equiv_raw.items():
    if v[1]:
        _needs_param.update(k)


def get_window(window, Nx, fftbins=True):
    """
    Return a window of a given length and type.

    Parameters
    ----------
    window : string, float, or tuple
        The type of window to create. See below for more details.
    Nx : int
        The number of samples in the window.
    fftbins : bool, optional
        If True (default), create a "periodic" window, ready to use with
        `ifftshift` and be multiplied by the result of an FFT (see also
        `fftpack.fftfreq`).
        If False, create a "symmetric" window, for use in filter design.

    Returns
    -------
    get_window : ndarray
        Returns a window of length `Nx` and type `window`

    Notes
    -----
    Window types:

    - `~scipy.signal.windows.boxcar`
    - `~scipy.signal.windows.triang`
    - `~scipy.signal.windows.blackman`
    - `~scipy.signal.windows.hamming`
    - `~scipy.signal.windows.hann`
    - `~scipy.signal.windows.bartlett`
    - `~scipy.signal.windows.flattop`
    - `~scipy.signal.windows.parzen`
    - `~scipy.signal.windows.bohman`
    - `~scipy.signal.windows.blackmanharris`
    - `~scipy.signal.windows.nuttall`
    - `~scipy.signal.windows.barthann`
    - `~scipy.signal.windows.kaiser` (needs beta)
    - `~scipy.signal.windows.gaussian` (needs standard deviation)
    - `~scipy.signal.windows.general_gaussian` (needs power, width)
    - `~scipy.signal.windows.slepian` (needs width)
    - `~scipy.signal.windows.dpss` (needs normalized half-bandwidth)
    - `~scipy.signal.windows.chebwin` (needs attenuation)
    - `~scipy.signal.windows.exponential` (needs decay scale)
    - `~scipy.signal.windows.tukey` (needs taper fraction)

    If the window requires no parameters, then `window` can be a string.

    If the window requires parameters, then `window` must be a tuple
    with the first argument the string name of the window, and the next
    arguments the needed parameters.

    If `window` is a floating point number, it is interpreted as the beta
    parameter of the `~scipy.signal.windows.kaiser` window.

    Each of the window types listed above is also the name of
    a function that can be called directly to create a window of
    that type.

    Examples
    --------
    >>> from scipy import signal
    >>> signal.get_window('triang', 7)
    array([ 0.125,  0.375,  0.625,  0.875,  0.875,  0.625,  0.375])
    >>> signal.get_window(('kaiser', 4.0), 9)
    array([ 0.08848053,  0.29425961,  0.56437221,  0.82160913,  0.97885093,
            0.97885093,  0.82160913,  0.56437221,  0.29425961])
    >>> signal.get_window(4.0, 9)
    array([ 0.08848053,  0.29425961,  0.56437221,  0.82160913,  0.97885093,
            0.97885093,  0.82160913,  0.56437221,  0.29425961])

    """
    sym = not fftbins
    try:
        beta = float(window)
    except (TypeError, ValueError):
        args = ()
        if isinstance(window, tuple):
            winstr = window[0]
            if len(window) > 1:
                args = window[1:]
        elif type(window) == type(''):  # isinstance(window, string_types):
            if window in _needs_param:
                raise ValueError("The '" + window + "' window needs one or "
                                 "more parameters -- pass a tuple.")
            else:
                winstr = window
        else:
            raise ValueError("%s as window type is not supported." %
                             str(type(window)))

        try:
            winfunc = _win_equiv[winstr]
        except KeyError:
            raise ValueError("Unknown window type.")

        params = (Nx,) + args + (sym,)
    else:
        winfunc = kaiser
        params = (Nx, beta, sym)

    return winfunc(*params)