# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:35:40 2022

@author: gawe
"""

import sys
import numpy as _np
import importlib


__all__ = ['fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
           'hfft', 'ihfft', 'fftfreq', 'fftshift', 'ifftshift']


# ================================================================= #
# ================================================================= #

# First test which libraries are installed and set booleans
#
# Mark which FFT submodules are available...
fft_modules = {'numpy.fft': _np.fft}
for mod_name in ('scipy.fftpack', 'scipy.fft', 'pyfftw.interfaces.numpy_fft'):
    try:
        mod = importlib.import_module(mod_name)
    except ImportError:
        pass
    else:
        fft_modules[mod_name] = mod
    # end try
# end for

# ================================================================= #
# ================================================================= #

# Compare (numpy) version numbers with this nice utiliy
try:
    from numpy.lib import NumpyVersion as versiontuple
except:
    def versiontuple(v):
        return tuple(map(int, (v.split("."))))
# end try

# ================================================================= #

# Note that the default is to use numpy!  overwrite these later if possible
#
from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn
from numpy.fft import rfft, irfft, rfft2, irfft2, rfftn, irfftn
from numpy.fft import hfft, ihfft
from numpy.fft import fftfreq, fftshift, ifftshift


if versiontuple(_np.__version__)>versiontuple('1.8.0'):
    from numpy.fft import rfftfreq
else:
    # note this is copied over from numpy version 1.20.3
    def rfftfreq(n, d=1.0):
        """
        Return the Discrete Fourier Transform sample frequencies
        (for usage with rfft, irfft).

        The returned float array `f` contains the frequency bin centers in cycles
        per unit of the sample spacing (with zero at the start).  For instance, if
        the sample spacing is in seconds, then the frequency unit is cycles/second.

        Given a window length `n` and a sample spacing `d`::

          f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
          f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd

        Unlike `fftfreq` (but like `scipy.fftpack.rfftfreq`)
        the Nyquist frequency component is considered to be positive.

        Parameters
        ----------
        n : int
            Window length.
        d : scalar, optional
            Sample spacing (inverse of the sampling rate). Defaults to 1.

        Returns
        -------
        f : ndarray
            Array of length ``n//2 + 1`` containing the sample frequencies.

        Examples
        --------
        >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)
        >>> fourier = np.fft.rfft(signal)
        >>> n = signal.size
        >>> sample_rate = 100
        >>> freq = np.fft.fftfreq(n, d=1./sample_rate)
        >>> freq
        array([  0.,  10.,  20., ..., -30., -20., -10.])
        >>> freq = np.fft.rfftfreq(n, d=1./sample_rate)
        >>> freq
        array([  0.,  10.,  20.,  30.,  40.,  50.])
        """
        if not isinstance(n, (int, )):
            raise ValueError("n should be an integer")
        val = 1.0/(n*d)
        N = n//2 + 1
        results = _np.arange(0, N, dtype=int)
        return results * val
    # end def
# end if
__all__ += ['rfftfreq']

# try:
#     from numpy.fft import rfftfreq
# except ImportError:
#     pass
# # end try:


# ================================================================= #
# ================================================================= #

# now if we want, we could overwrite these with mkl_fft, scipy.fft, pyfftw, or dask fft.
# Let's start by overwriting with pyfftw
#
# # PYFFTW interfaces:
# # pyfftw.interfaces.numpy_fft
# # pyfftw.interfaces.scipy_fft
# # pyfftw.interfaces.scipy_fftpack
#
if 'pyfftw.interfaces.numpy_fft' in fft_modules:
    """
    watch out for normalizations here. The numpy-like interface to pyfftw should take care of it already.
    """
    from pyfftw.interfaces.numpy_fft import fft, ifft, fft2, ifft2, fftn, ifftn
    from pyfftw.interfaces.numpy_fft import rfft, irfft, rfft2, irfft2, rfftn, irfftn
    from pyfftw.interfaces.numpy_fft import hfft, ihfft
    from pyfftw.interfaces.numpy_fft import fftfreq, fftshift, ifftshift, rfftfreq
# end if


# ================================================================= #

# has_mkl = False
# has_dask = False
# has_fftw = False
# has_scipy = False

# def __pick_mod(kwargs):
#     fftmod = kwargs.pop('fftmod', _np.fft)
#     return kwargs, fftmod

# # alternatively we can define each function and switch
# def fft(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.fft(*args, **kwargs)

# def ifft(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.ifft(*args, **kwargs)

# def fft2(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.fft2(*args, **kwargs)

# def ifft2(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.ifft2(*args, **kwargs)

# def fftn(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.fftn(*args, **kwargs)

# def ifftn(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.ifftn(*args, **kwargs)

# # ==== #

# def rfft(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.rfft(*args, **kwargs)

# def irfft(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.irfft(*args, **kwargs)

# def rfft2(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.rfft2(*args, **kwargs)

# def irfft2(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.irfft2(*args, **kwargs)

# def rfftn(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.rfftn(*args, **kwargs)

# def irfftn(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.irfftn(*args, **kwargs)

# # ==== #

# def hfft(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.hfft(*args, **kwargs)

# def ihfft(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.ihfft(*args, **kwargs)

# # === #



# def fftfreq(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.fftfreq(*args, **kwargs)

# def fftshift(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.fftshift(*args, **kwargs)

# def ifftshift(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.ifftshift(*args, **kwargs)

# def rfftfreq(*args, **kwargs):
#     kwargs, fftmod = __pick_mod(kwargs)
#     return fftmod.rfftfreq(*args, **kwargs)

# # ================================================================= #
# # ================================================================= #
#
# # =============== MKL ============ #
# # mkl_fft started as a part of Intel (R) Distribution for Python*
# # optimizations to NumPy, and is now being released as a stand-alone
# # package.
# # see:  https://github.com/scipy-conference/scipy_proceedings/blob/2017/papers/oleksandr_pavlyk/fft/fft.rst
# #
# try:
#     import mkl_fft
#     has_mkl = True
# except:
#     pass
# # end try
#
#
# # =============== DASK ============ #
# #    dask is a wrapper for distributed (parallel) computing.
# #    It is a thin python wrapper around numpy, scipy, Pandas, etc.
# try:
#     import dask
#     has_dask = True
# except:
#     has_dask = False
# # end try
#
# #    pyfftw is a wrapper around the C-libraries for FFTW
# #       FFTW--> the fastest fft in the west
# #
# #    scipy has two implementations with different API's:
# #       "legacy" code under scipy.fftpack (deprecated at SciPy 1.4.0)
# #       "standard" code under scipy.fft   (implemented in SciPy 1.4.0)
# try:
#     # if the python bindings for fftw are in your python path set the flag
#     import pyfftw
#     has_pyfftw = True
#     has_scipy_fft = pyfftw.interfaces.has_scipy_fft
#
# except:
#     # Use the numpy fft
#     has_pyfftw = False
#
#     try:
#         import scipy
#         if versiontuple(scipy.__version__)>=versiontuple('1.4.0'):
#             has_scipy_fft = True    # scipy.fft is the current version
#         else:
#             has_scipy_fft = False   # scipy.fftpack is the legacy version
#         # end if
#     except:
#         has_scipy_fft = False       # no scipy
#     # end try
#
#     pass
# # end try

# ======================================================= #
# ======================================================= #

def test_pyfftw():
    """

    """
    from pyfftw.tests import test_pyfftw_numpy_interface as test_np

    test_np.run_test_suites(test_np.test_cases, test_np.test_set)

# end def


def compare_pyfftw_scipy():
    import pyfftw
    import multiprocessing
    import scipy.signal
    import scipy.fft
    import numpy
    from timeit import Timer

    # Configure PyFFTW to use all cores (the default is single-threaded)
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()

    # ==== #

    a = pyfftw.empty_aligned((128, 64), dtype='complex128')
    b = pyfftw.empty_aligned((128, 64), dtype='complex128')

    a[:] = numpy.random.randn(128, 64) + 1j*numpy.random.randn(128, 64)
    b[:] = numpy.random.randn(128, 64) + 1j*numpy.random.randn(128, 64)


    def fftconv(x, y, modfft):
        """
        FFT convolution using relation
            x*y <==> XY
        where x[] and y[] have been zero-padded to length N, such that N >=
        P+Q-1 and N = 2^n.
        """
        X, Y = modfft.fft(x), modfft.fft(y)
        return modfft.ifft([a * b for a,b in zip(X,Y)])
    # end def fftconv

    t = Timer(lambda: fftconv(a, b, modfft))

    # first test using my function
    modfft = scipy.fft
    print('Time with my fft convolver and the scipy.fft default backend: %1.3f seconds' %
          t.timeit(number=100))


    modfft = pyfftw.interfaces.scipy_fft
    print('Time with my fft convolver and the  pyfftw.scipy fft backend: %1.3f seconds' %
          t.timeit(number=100))

    # === #

    # now test using scipy's function
    t = Timer(lambda: scipy.signal.fftconvolve(a, b))

    print('Time with the scipy fft convolver and scipy.fft default backend: %1.3f seconds' %
          t.timeit(number=100))

    # Use the backend pyfftw.interfaces.scipy_fft
    with scipy.fft.set_backend(pyfftw.interfaces.scipy_fft):

          # We cheat a bit by doing the planning first
         scipy.signal.fftconvolve(a, b)

         print('Time with the scipy fft convolver and pyfftw backend installed: %1.3f seconds' %
                t.timeit(number=100))
    # end with
# end def

def compare_pyfftw_numpy():
    import pyfftw
    import multiprocessing
    import scipy.signal
    import scipy.fft
    import numpy
    from timeit import Timer

    def fftconv(x, y, modfft):
        """
        FFT convolution using relation
            x*y <==> XY
        where x[] and y[] have been zero-padded to length N, such that N >=
        P+Q-1 and N = 2^n.
        """
        X, Y = modfft.fft(x), modfft.fft(y)
        return modfft.ifft([a * b for a,b in zip(X,Y)])
    # end def fftconv

    a = pyfftw.empty_aligned((128, 64), dtype='complex128')
    b = pyfftw.empty_aligned((128, 64), dtype='complex128')

    a[:] = numpy.random.randn(128, 64) + 1j*numpy.random.randn(128, 64)
    b[:] = numpy.random.randn(128, 64) + 1j*numpy.random.randn(128, 64)

    modfft = _np.fft
    t = Timer(lambda: fftconv(a, b, modfft))

    print('Time with my fft convolvera nd numpy.fft default backend: %1.3f seconds' %
          t.timeit(number=100))

    # Configure PyFFTW to use all cores (the default is single-threaded)
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()

    modfft = pyfftw.interfaces.numpy_fft
    print('Time with my fft convolver and pyfftw backend installed: %1.3f seconds' %
           t.timeit(number=100))
    # end with
# end def


def test_simple_sine(plotit=True):
    import matplotlib.pyplot as _plt

    x, y, hfig = test_sine_data(plotit=plotit)

    Fs = (len(x)-1)/(x[-1]-x[0])
    nfft = len(x)

    f1 = _np.fft.fftfreq(nfft, 1/Fs)
    ff = fftfreq(nfft, 1/Fs)

    f1 = _np.fft.fftshift(f1)
    ff = fftshift(ff)

    yfft = _np.fft.fft(y, n=nfft)
    pyfft = fft(y, n=nfft)

    yfft = _np.fft.fftshift(yfft)
    pyfft = fftshift(pyfft)

    if plotit:
        _plt.figure()
        _plt.subplot(311)
        _plt.plot(f1, yfft.real, 'k-')
        _plt.plot(ff, pyfft.real, 'b--')
        _plt.title('FFT: Real part')

        _plt.subplot(312)
        _plt.plot(f1, yfft.imag, 'k-')
        _plt.plot(ff, pyfft.imag, 'b--')
        _plt.title('FFT: Imaginary part')

        _plt.subplot(313)
        _plt.plot(f1, _np.abs(yfft*yfft.conj()), 'k-')
        _plt.plot(ff, _np.abs(pyfft*pyfft.conj()), 'b--')
        _plt.title('Power Spectra')
    # end if
# end def

def test_simple_tones(plotit=True):
    import matplotlib.pyplot as _plt
    x, y, hfig = test_tone_data(plotit=plotit)

    Fs = (len(x)-1)/(x[-1]-x[0])
    nfft = len(x)

    f1 = _np.fft.fftfreq(nfft, 1/Fs)
    ff = fftfreq(nfft, 1/Fs)

    yfft = _np.fft.fft(y, n=nfft)
    pyfft = fft(y, n=nfft)

    f1 = _np.fft.fftshift(f1)
    ff = fftshift(ff)

    yfft = _np.fft.fftshift(yfft)
    pyfft = fftshift(pyfft)

    if plotit:
        _plt.figure()
        _plt.subplot(311)
        _plt.plot(f1, yfft.real, 'k-')
        _plt.plot(ff, pyfft.real, 'b--')
        _plt.title('FFT: Real part')

        _plt.subplot(312)
        _plt.plot(f1, yfft.imag, 'k-')
        _plt.plot(ff, pyfft.imag, 'b--')
        _plt.title('FFT: Imaginary part')

        _plt.subplot(313)
        _plt.plot(f1, _np.abs(yfft*yfft.conj()), 'k-')
        _plt.plot(ff, _np.abs(pyfft*pyfft.conj()), 'b--')
        _plt.title('Power Spectra')
    # end if
# end def


def test_sine_data(plotit=False):
    SAMPLE_RATE = 44100  # Hertz
    DURATION = 5  # Seconds

    def generate_sine_wave(freq, sample_rate, duration):
        x = _np.linspace(0, duration, sample_rate * duration, endpoint=False)
        frequencies = x * freq
        # 2pi because np.sin takes radians
        y = _np.sin((2 * _np.pi) * frequencies)
        return x, y

    # Generate a 2 hertz sine wave that lasts for 5 seconds
    x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)

    if plotit:
        from matplotlib import pyplot as _plt
        hfig = _plt.figure()
        _plt.plot(x, y)
        _plt.title('Test data: %i Hz'%(2,))
        _plt.show()
        return x, y, hfig
    # end if
    return x, y
# end def

def test_tone_data(plotit=False):
    SAMPLE_RATE = 44100  # Hertz
    DURATION = 1  # Seconds

    def generate_sine_wave(freq, sample_rate, duration):
        x = _np.linspace(0, duration, sample_rate * duration, endpoint=False)
        frequencies = x * freq
        # 2pi because np.sin takes radians
        y = _np.sin((2 * _np.pi) * frequencies)
        return x, y

    # Generate a two-tone signal with 400 and 4000 Hz
    x, nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
    _, noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
    noise_tone = noise_tone * 0.3

    mixed_tone = nice_tone + noise_tone

    if plotit:
        from matplotlib import pyplot as _plt
        hfig = _plt.figure()
        _plt.plot(x, mixed_tone)
        _plt.title('Test tone data: %i and %i Hz'%(400, 4000))
        _plt.show()
        return x, mixed_tone, hfig
    # end if
    return x, mixed_tone
# end def


if __name__ == "__main__":
    compare_pyfftw_scipy()
    compare_pyfftw_numpy()

    test_simple_sine()
    test_simple_tones()

    # test_pyfftw()
    pass
# end if
