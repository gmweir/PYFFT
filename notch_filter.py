# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:30:52 2016

@author: gawe
"""
# ========================================================================== #
# ========================================================================== #


# This section is to improve python compataibilty between py27 and py3
from __future__ import absolute_import, with_statement, absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as _np

# ========================================================================== #

def iirnotch(w0, Q):
    """
    Design second-order IIR notch digital filter.

    A notch filter is a band-stop filter with a narrow bandwidth
    (high quality factor). It rejects a narrow frequency band and
    leaves the rest of the spectrum little changed.

    Parameters
    ----------
    w0 : float
        Normalized frequency to remove from a signal. It is a
        scalar that must satisfy  ``0 < w0 < 1``, with ``w0 = 1``
        corresponding to half of the sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.

    See Also
    --------
    iirpeak

    Notes
    -----
    .. versionadded: 0.19.0

    References
    ----------
    .. [1] Sophocles J. Orfanidis, "Introduction To Signal Processing",
           Prentice-Hall, 1996

    Examples
    --------
    Design and plot filter to remove the 60Hz component from a
    signal sampled at 200Hz, using a quality factor Q = 30

    >>> from scipy import signal
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> fs = 200.0  # Sample frequency (Hz)
    >>> f0 = 60.0  # Frequency to be removed from signal (Hz)
    >>> Q = 30.0  # Quality factor
    >>> w0 = f0/(fs/2)  # Normalized Frequency
    >>> # Design notch filter
    >>> b, a = signal.iirnotch(w0, Q)

    >>> # Frequency response
    >>> w, h = signal.freqz(b, a)
    >>> # Generate frequency axis
    >>> freq = w*fs/(2*_np.pi)
    >>> # Plot
    >>> fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    >>> ax[0].plot(freq, 20*_np.log10(abs(h)), color='blue')
    >>> ax[0].set_title("Frequency Response")
    >>> ax[0].set_ylabel("Amplitude (dB)", color='blue')
    >>> ax[0].set_xlim([0, 100])
    >>> ax[0].set_ylim([-25, 10])
    >>> ax[0].grid()
    >>> ax[1].plot(freq, _np.unwrap(_np.angle(h))*180/_np.pi, color='green')
    >>> ax[1].set_ylabel("Angle (degrees)", color='green')
    >>> ax[1].set_xlabel("Frequency (Hz)")
    >>> ax[1].set_xlim([0, 100])
    >>> ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    >>> ax[1].set_ylim([-90, 90])
    >>> ax[1].grid()
    >>> plt.show()
    """

    return _design_notch_peak_filter(w0, Q, "notch")

def iirpeak(w0, Q):
    """
    Design second-order IIR peak (resonant) digital filter.

    A peak filter is a band-pass filter with a narrow bandwidth
    (high quality factor). It rejects components outside a narrow
    frequency band.

    Parameters
    ----------
    w0 : float
        Normalized frequency to be retained in a signal. It is a
        scalar that must satisfy  ``0 < w0 < 1``, with ``w0 = 1`` corresponding
        to half of the sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        peak filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.

    See Also
    --------
    iirnotch

    Notes
    -----
    .. versionadded: 0.19.0

    References
    ----------
    .. [1] Sophocles J. Orfanidis, "Introduction To Signal Processing",
           Prentice-Hall, 1996

    Examples
    --------
    Design and plot filter to remove the frequencies other than the 300Hz
    component from a signal sampled at 1000Hz, using a quality factor Q = 30

    >>> from scipy import signal
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> fs = 1000.0  # Sample frequency (Hz)
    >>> f0 = 300.0  # Frequency to be retained (Hz)
    >>> Q = 30.0  # Quality factor
    >>> w0 = f0/(fs/2)  # Normalized Frequency
    >>> # Design peak filter
    >>> b, a = signal.iirpeak(w0, Q)

    >>> # Frequency response
    >>> w, h = signal.freqz(b, a)
    >>> # Generate frequency axis
    >>> freq = w*fs/(2*_np.pi)
    >>> # Plot
    >>> fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    >>> ax[0].plot(freq, 20*_np.log10(abs(h)), color='blue')
    >>> ax[0].set_title("Frequency Response")
    >>> ax[0].set_ylabel("Amplitude (dB)", color='blue')
    >>> ax[0].set_xlim([0, 500])
    >>> ax[0].set_ylim([-50, 10])
    >>> ax[0].grid()
    >>> ax[1].plot(freq, _np.unwrap(_np.angle(h))*180/_np.pi, color='green')
    >>> ax[1].set_ylabel("Angle (degrees)", color='green')
    >>> ax[1].set_xlabel("Frequency (Hz)")
    >>> ax[1].set_xlim([0, 500])
    >>> ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    >>> ax[1].set_ylim([-90, 90])
    >>> ax[1].grid()
    >>> plt.show()
    """

    return _design_notch_peak_filter(w0, Q, "peak")

def _design_notch_peak_filter(w0, Q, ftype):
    """
    Design notch or peak digital filter.

    Parameters
    ----------
    w0 : float
        Normalized frequency to remove from a signal. It is a
        scalar that must satisfy  ``0 < w0 < 1``, with ``w0 = 1``
        corresponding to half of the sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    ftype : str
        The type of IIR filter to design:

            - notch filter : ``notch``
            - peak filter  : ``peak``

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.
    """

    # Guarantee that the inputs are floats
    w0 = float(w0)
    Q = float(Q)

    # Checks if w0 is within the range
    if w0 > 1.0 or w0 < 0.0:
        raise ValueError("w0 should be such that 0 < w0 < 1")

    # Get bandwidth
    bw = w0/Q

    # Normalize inputs
    bw = bw*_np.pi
    w0 = w0*_np.pi

    # Compute -3dB atenuation
    gb = 1/_np.sqrt(2)

    if ftype == "notch":
        # Compute beta: formula 11.3.4 (p.575) from reference [1]
        beta = (_np.sqrt(1.0-gb**2.0)/gb)*_np.tan(bw/2.0)
    elif ftype == "peak":
        # Compute beta: formula 11.3.19 (p.579) from reference [1]
        beta = (gb/_np.sqrt(1.0-gb**2.0))*_np.tan(bw/2.0)
    else:
        raise ValueError("Unknown ftype.")

    # Compute gain: formula 11.3.6 (p.575) from reference [1]
    gain = 1.0/(1.0+beta)

    # Compute numerator b and denominator a
    # formulas 11.3.7 (p.575) and 11.3.21 (p.579)
    # from reference [1]
    if ftype == "notch":
        b = gain*_np.array([1.0, -2.0*_np.cos(w0), 1.0])
    else:
        b = (1.0-gain)*_np.array([1.0, 0.0, -1.0])
    a = _np.array([1.0, -2.0*gain*_np.cos(w0), (2.0*gain-1.0)])

    return b, a
