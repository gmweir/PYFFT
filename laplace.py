# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 17:24:08 2021

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

"""
Laplace Transform:

Maps complex-valued signal x(t) with real-valued
independent variable t to its complex-valued Laplace transform
with complex-valued independent variable s
    Whether a Laplace transform X(s)=L{x(t)} exists depends on complex
    frequency s and the signal x(t) itself.

    All values s for which the Laplace transform converges form a
    region of convergence (ROC). The Laplace transform of two different
    signals may differe only wrt their ROCs. Conseqeuently, the ROC needs
    to be explicitly given for a unique inversion of the Laplace transform.

Laplace transforms are extensively used for signals/systems analysis and
filter analysis/design with linear, time-invariant (LTI) systems.

The bilateral (+- time) Laplace transform of a causal signal is identical
to the unilateral (t>0) Laplace transform of that signal.

A rational Laplace transform (i.e., of an LTI system) can always be
written as either the quotient of two polynomials in s:
    F(s) = P_n(s) / Q_m(s) where m>=n
or the same quotient with a constant damping factor
    F(s) = (P_n(s) / Q_m(s))*exp[-a*s] where m>=n, a>0

    Roots:  zeros of the Laplace transform (where P_n = 0)
    Poles:  discontinuities of the Laplace transform (where Q_m = 0)

Special case:
    The Laplace transform of the Dirac-delta function delta(t) is 1
        L{delta(t)} = 1 for s an element of all complex-numbers
            ROC covers the entire complex plane


-----

Laplace transform (unilateral):    s = alpha + j*omega
   F[u(t)] = U(s) = int_0^infty{  u(t)*exp[-(alpha+j*omega)*t]dt}

Laplace transform (bilateral):    s = alpha + j*omega
   F[u(t)] = U(s) = int_-infty^infty{  u(t)*exp[-(alpha+j*omega)*t]dt}

Fourier transform:
(just for this example we are normalizing on the inverse transform)
   F[u(t)] = U(omega) = int_-infty^infty{  u(t)*exp[-j*omega*t]dt}

By inspection:

The unilateral Laplace transform is the Fourier transform of a causal
function multiplied by a decaying exponential
    U(s) = F[ u(t)*H(t)*exp(-alpha*t) ]  - H() is the Heaviside function

The bilateral Laplace transform is the Fourier transform of a generally
non-causal function multiplied by a decaying exponential
    U(s) = F[ u(t)*exp(-alpha*t) ]
"""


def laplace():
    pass


def laplace_1d(uin, real_sigma_interval=_np.arange(-1, 1 + 0.001, 0.001), nfft=None):
    """
    Returns the Laplace transform of a signal using a brute-force method and
    Fourier transforms,
        first axis is the real range
        second axis the imaginary range.
    Complex numbers are returned.
    """
    uin = _np.copy(uin)

    if nfft is None:
        uin = _np.atleast_1d(uin)
        nfft = len(uin)
    # end if

    # The transform is from last timestep to first, so "x" is reversed
    uin = _np.array(uin)[::-1]

    d = []
    for sigma in real_sigma_interval:
        exp = _np.exp( sigma*_np.array(range(len(uin))) )
        exp /= _np.sum(exp)
        exponentiated_signal = exp * uin
        # print (max(exponentiated_signal), min(exponentiated_signal))
        d.append(exponentiated_signal[::-1])  # re-reverse for straight signal
    # end for

    # Now apply the imaginary part and "integrate" (sum)
    return _np.array([fftmod.rfft(k) for k in d])


def test_laplace():
    # sample_rate = 50  # 50 Hz resolution
    # signal_length = 10*sample_rate  # 10 seconds
    # # Generate a random x(t) signal with waves and noise.
    # t = _np.linspace(0, 10, signal_length)
    # g = 30*( _np.sin((t/10)**2) )
    # # x = g.copy()

    # x  = 0.30*_np.cos(2*_np.pi*0.25*t - 0.2)
    # x += 0.28*_np.sin(2*_np.pi*1.50*t + 1.0)
    # x += 0.10*_np.sin(2*_np.pi*5.85*g + 1.0)
    # x += 0.09*_np.cos(2*_np.pi*10.0*t)
    # x += 0.04*_np.sin(2*_np.pi*20.0*t)
    # x += 0.15*_np.cos(2*_np.pi*135.0*(t/5.0-1)**2)
    # x += 0.04*_np.random.randn(len(t))

    # # Normalize between -0.5 to 0.5:
    # x -= _np.min(x)
    # x /= _np.max(x)
    # x -= 0.5

    N = int(32)
    f = 1
    dt = 1.0/N
    t = []
    x = []
    # z3 = []
    for n in range(N):
        _x = 2*_np.pi*f*dt*n
        x.append(_np.sin(_x))
        # z3.append(-1.0*_np.cos(_x))  # hilbert transform of a sine
        t.append(_x)
    # end for
    # z1 = laplace(x)
    z1 = laplace_1d(x).transpose()

    norm_surface = _np.absolute(z1)
    angle_surface = _np.angle(z1)

    # Plotting the transform:

    _plt.figure(figsize=(11, 9))
    _plt.title("Norm of Laplace transform")
    _plt.imshow(norm_surface, aspect='auto', interpolation='none', cmap=_plt.cm.rainbow)
    _plt.ylabel('Imaginary: Frequency (Hz)')
    _plt.xlabel('Real (exponential multiplier)')
    _plt.xticks([0, 500, 1000, 1500, 2000], [-1, -0.5, 0.0, 0.5, 1.0])
    _plt.gca().invert_yaxis()
    _plt.colorbar()

    _plt.figure(figsize=(11, 9))
    _plt.title("Phase of Laplace transform")
    _plt.imshow(angle_surface, aspect='auto', interpolation='none', cmap=_plt.cm.hsv)
    _plt.ylabel('Imaginary (Frequency, Hz)')
    _plt.xlabel('Real (exponential multiplier)')
    _plt.xticks([0, 500, 1000, 1500, 2000], [-1, -0.5, 0.0, 0.5, 1.0])
    _plt.gca().invert_yaxis()
    _plt.colorbar()
    _plt.show()

    _plt.figure(figsize=(11, 9))
    _plt.title("Laplace transform, stacked phase and norm")
    _plt.imshow(angle_surface, aspect='auto', interpolation='none', cmap=_plt.cm.hsv)
    _plt.ylabel('Imaginary: Frequency (Hz)')
    _plt.xlabel('Real (exponential multiplier)')
    _plt.xticks([0, 500, 1000, 1500, 2000], [-1, -0.5, 0.0, 0.5, 1.0])
    _plt.colorbar()
    _plt.gca().invert_yaxis()
    # Rather than a simple alpha channel option, I would have preferred a better transfer mode such as "multiply".
    _plt.imshow(norm_surface, aspect='auto', interpolation='none', cmap=_plt.cm.gray, alpha=0.9)
    _plt.ylabel('Imaginary: Frequency (Hz)')
    _plt.xlabel('Real (exponential multiplier)')
    _plt.xticks([0, 500, 1000, 1500, 2000], [-1, -0.5, 0.0, 0.5, 1.0])
    _plt.gca().invert_yaxis()
    _plt.colorbar()
    _plt.show()

    _plt.figure(figsize=(11, 9))
    _plt.title("Log inverse norm of Laplace transform (poles visible)")
    _plt.imshow(-_np.log(norm_surface), aspect='auto', interpolation='none', cmap=_plt.cm.summer)
    _plt.ylabel('Imaginary: Frequency (Hz)')
    _plt.xlabel('Real (exponential multiplier)')
    _plt.xticks([0, 500, 1000, 1500, 2000], [-1, -0.5, 0.0, 0.5, 1.0])
    _plt.gca().invert_yaxis()
    _plt.colorbar()
    _plt.show()

# ========================================================================== #
# ========================================================================== #
if __name__ == "__main__":
    test_laplace()
# end if


#
# ========================================================================== #
# ========================================================================== #