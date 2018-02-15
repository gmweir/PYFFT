"""
In this module:
    spectrogram()

"""
# ========================================================================== #
# ========================================================================== #


# This section is to improve python compataibilty between py27 and py3
from __future__ import absolute_import, with_statement, absolute_import, division, print_function, unicode_literals
__metaclass__ = type

# ========================================================================== #

import numpy as _np

def spectrogram(t, s, wl = 512, hanning = True, overlap = True, windowAverage = None):
    """
    Annoyed with the lack of flexibility of the matplotlib specgram, I wrote this program to make my
    own spectrogram. It simply retruns the spectrogram array with time and frequency vectors

    INPUTS:
        t  - the time vector of the signal (same length as s)
        s  - the 1D time signal for which to make a spectrogram
        wl - the window length

    RETURNS:
        time        - time axis of the spectrogram
        frequency   - frequency axis of the spectrogram
        spectrogram - the spectrogram array

    """

    if windowAverage != None:
        overlap = False

    s = s.flatten()
    n  = len(s)
    dt = _np.abs(t[1]-t[0])


    if overlap:
        nWindows = 2*(n - (n % wl))//wl - 1
    else:
        nWindows = (n - (n % wl))//wl - 1

    spectrogram = _np.zeros((wl,nWindows))

    print('nWindows = ', nWindows)
    try:
        nProg = 100
        a = _np.floor_divide(nWindows/nProg)
        print('[',' '*(nWindows/a) , ']')
        progressBar = True
    except:
        progressBar = False

    for i in range(nWindows):

            if progressBar:
                if ((i % a) == 0):
                    print('.',end='')

            if overlap:
                idx1 = i*wl//2
                idx2 = idx1+wl
            else:
                idx1 = i*wl
                idx2 = idx1+wl

            # first Fourier transform each block within the data. Either with a Hanning window or otherwise
            if hanning:
                spectrogram[:,i] = _np.sqrt(8.0/3.0)*_np.abs(_np.fft.fft(_np.hanning(wl)*(s[idx1:idx2])))**2/wl

            else:
                spectrogram[:,i] = _np.abs(_np.fft.fft(s[idx1:idx2]))**2/wl

    if windowAverage != None:

        spectrogramAverage = _np.zeros((wl,nWindows/windowAverage))
        for i in range(nWindows/windowAverage):
            spectrogramAverage[:,i] = _np.mean(spectrogram[:,i*windowAverage:(i+1)*windowAverage],axis=1)

        fAxis = _np.fft.fftfreq(wl,dt)
        time  = _np.linspace(t[0]+wl*dt/2, t[0]+wl*dt*((nWindows-1) + 1/2), num = nWindows/windowAverage)

        return time, fAxis, spectrogramAverage

    else:

        fAxis = _np.fft.fftfreq(wl,dt)
        if overlap == False:
            time  = _np.linspace(t[0]+wl*dt/2, t[0]+wl*dt*((nWindows-1) + 1/2), num = nWindows)
        else:
            time  = _np.linspace(t[0]+wl*dt/2, t[0]+wl*dt*((nWindows/2-1) + 1/2), num = nWindows)

        return time, fAxis, spectrogram