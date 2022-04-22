# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 17:45:40 2021

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

from pybaseutils.Struct import Struct
from pybaseutils.utils import detrend_mean, detrend_none, detrend_linear
from pybaseutils import utils as _ut

try:
    from FFT.windows import windows
    from FFT.stft import stft
    from FFT.filters import downsample, downsample_efficient
    from FFT.fft_analysis import fftanal
except:
    from .windows import windows
    from .stft import stft
    from .filters import downsample, downsample_efficient
    from .fft_analysis import fftanal
# end try

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
    Center of gravity functions -- primary component in a Doppler spectrum etc
"""


def cog(x, fs, fmin=None, fmax=None):
    """
    Center of gravity of data from PSD of input data
    - power spectral density weighted average of frequency
    """
    if fmax is None: fmax = fs  # end if
    n=len(x)
    freq=fftmod.fftshift(fftmod.fftfreq(n,1/fs))
    spec=fftmod.fftshift(fftmod.fft(x))/_np.sqrt(n/2)
    if fmin is not None:
        freq = freq[_np.where((_np.abs(freq)>=fmin)*(_np.abs(freq)<=fmax))]
        spec = spec[_np.where((_np.abs(freq)>=fmin)*(_np.abs(freq)<=fmax))]
    if len(freq)>0:
        return _np.sum(_np.abs(_np.square(spec))*freq)/_np.sum(_np.abs(_np.square(spec)))
    else:
        return 0.0
    # end if


def cogspec(t, x, fs, fmin=100, fmax=500e3, n=256, win=512, ov=0.5, plotit=1):
    """
    Calculate the center of gravity for the spectra along a running window
    - power spectral density weighted average of frequency

    use n-point sliding window to calculate the cog
    """
    ind=_ut.sliding_window_1d(t, x, win, int(_np.floor((1.0-ov)*win)), ind_only=1)
    # start, stop indices of each window, 50% overlap, n-point window length
#    ind=_ut.sliding_window_1d(t, x, n, n/2, ind_only=1)

    N=ind.shape[0]
    coge=_np.zeros(N)
    tcog=_np.zeros(N)
    for ii in range(0,N):
        dw=x[ind[ii,0]:ind[ii,1]]   # data within each window
        tcog[ii]=_np.mean(t[ind[ii,0]:ind[ii,1]]) # avg time / window
        coge[ii] = cog(dw,fs)       # center of gravity of each window
    # end for

    # Break the COG along a sliding window with window/overlap given by input
    winstep=int(_np.floor(win*ov))
    tw, cogw, tcogw = _ut.sliding_window_1d(tcog,coge,win,winstep)
    cogfs=1/(tcog[1]-tcog[0])*1000   # KHz, new sampling frequency

    # Calculate the power spectral density for each window
    N=cogw.shape[0]  # number of windows
    print('using %i ensembles for cogspec()')%(N)
    PS, F = psd(cogw[0], cogfs, nfft=win, fmax=fmax) # PSD, freq
    for jj in range(1,N):
        print(jj)
        # for each 'time-point' in window, calculate PSD
        PS2, F2 = psd(cogw[jj],cogfs,nfft=win,fmax=fmax)
        PS = _np.vstack([PS,PS2])
    # end for

    if plotit:
        _fig = _plt.figure(figsize=(12, 6),facecolor='w') # analysis:ignore
        PS=PS/_np.max(PS)
        h=_plt.subplot(3,1,1)
        _plt.pcolormesh(tcogw, F/1e3, 10*_np.log10(_np.transpose(PS)), cmap='bwr')
        _plt.xlabel('time [ms]')
        _plt.ylabel('freq [kHz]')

        _plt.subplot(3,1,2)
        a=_np.sum(PS,axis=0)
        _plt.plot(F/1e3,10*_np.log10(a))
        _plt.xlabel('freq [kHz]')
        _plt.ylabel('COG')

        _plt.subplot(3,1,3,sharex=h)
        _plt.plot(tcog,coge)
        _plt.xlabel('time [ms]')
        _plt.ylabel('COG')
    # end if

    vardict={'cogfs':cogfs}
    vardict['cog']=coge
    vardict['tcog']=tcog
    vardict['cogtime']=tcog
    vardict['cogspectime']=tcogw
    vardict['cogspec']=PS
    vardict['cogspecf']=F

    return vardict
#
#
# n=512
#        freq=fftmod.fftshift(fftmod.fftfreq(n,1./self.fs))
#        N=ind.shape[0]
#        cog=_np.zeros(N)
#        tcog=_np.zeros(N)
#        print 'using %i ensembles' % (N)
#        for i in range(0,N):
#            dw=D[ind[i,0]:ind[i,1]]
#            tcog[i]=_np.mean(T[ind[i,0]:ind[i,1]])
#            spec=fftshift(fft(dw))/_np.sqrt(n/2)
#            spec2=spec[_np.where(_np.abs(freq)>=deltaf)]
#            freq2=freq[_np.where(_np.abs(freq)>=deltaf)]
#            cog[i] = _np.sum(_np.abs(_np.square(spec2))*freq2)/_np.sum(_np.abs(_np.square(spec2)))
#
#
#        winstep=int(_np.floor(win*ov))
#        tw,cogw,tscog=ol.sliding_window_1d(tcog,cog,win,winstep)
#        cogfs=1/(tcog[1]-tcog[0])*1000
#        PS,F=ol.psd(cogw[0],cogfs,nfft=nfft,fmax=200e3)
#
#        N=cogw.shape[0]
#
#        for j in range(1,N):
#
#            PS2,F2=ol.psd(cogw[j],cogfs,nfft=nfft,fmax=200e3)
#            PS=_np.vstack([PS,PS2])
##

# =========================================================================== #
# =========================================================================== #





def test_DopplerSignal(ampModulation=None):
    fs = 50e3
    fsig = 10e3
    psig = 0.25*_np.pi
    LO = 6e6
    IF = 0.3e6
    fmult = 2*LO+IF
    N = 2**21
    amp = 1.0
    time = _np.arange(N)/(3*fmult)

    mod = amp*_np.sin(2*_np.pi*fsig*time)  # plasma Doppler shift
    carrier = _np.sin(2*_np.pi*fmult*time)       # TX signal from reflectometer

    if ampModulation is None:
        # the plasma is a single-sideband mixer (blue or red doppler shift), remove the upper sideband
        sigz = amp*_np.sin(2*_np.pi*(fmult-fsig)*time-psig)  # plasma is already a single-sideband mixer
    elif ampModulation:
        # does not reproduce the correct phase from the modulation signal
        # RF, this is the signal returning from the plasma down the RX line of the transmission line
        # amplitude modulate the carrier with our reference signal
        sigz = 2*carrier*mod       # amplitude modulation
    else:
        # reproduces the correct phase from the modulation signal when I am really weird

        # RF, this is the signal returning from the plasma down the RX line of the transmission line
        sigz = _np.sin(2.0*_np.pi*(fmult*time+mod))  # frequency modulation
    # end if
    # 5*LO+IF + fsig = 500e6+10e3
    # 5*LO+IF - fsig = 500e6-10e3

#    # low pass filter the upper sideband from the mixing
#    lowpass_n, lowpass_d = _dsp.butter(3, fmult/(3*fmult), btype='low')
#    sigz = _dsp.filtfilt(lowpass_n, lowpass_d, sigz)   # this is the signal out of the plasma

    # demodulate the carrier signal by mixing with a local oscillator
    locosc  = _np.sin(2*_np.pi*(fmult-IF)*time)  # LO signal from reflectometer
    sigz = 2*locosc*sigz  # LO*RF: reflectometer mixer,        amplitude modulation
#    sigz = _np.sin(2.0*_np.pi*((fmult-IF)*time+sigz))  # frequency modulation
    # (5*LO)+(5*LO+IF+-fsig),= 10*LO+IF+-fsig = 501e6+500e6 - fsig
    # (5*LO)-(5*LO+IF+-fsig) = -IF+-fsig = -1e6+-10e3 = -990e3 (-1010e3)

    # The IF port expects a low-frequency signal (the SMA cables filter out the rest)
    # lower side-band selection again, filter out the upper sideband
    sigz = downsample(sigz, 3*fmult, 3*IF, plotit=False).flatten()
    time = _np.arange(time[0], time[-1], 1.0/(3*IF))

    # demodulate the in-phase and quadrature components of the signal by mixing with phase-shifted carrier reference
    # IF-fsig + IF = 2*IF-fsig
    # IF-fsig - IF = -fsig
    Isig = 2*sigz*_np.sin(2.0*_np.pi*IF*time)         # amplitude modulation
    Qsig = -2*sigz*_np.cos(2.0*_np.pi*IF*time)      # amplitude modulation
#    Isig = _np.sin(-2.0*_np.pi*(IF*time+sigz))         # frequency modulation
#    Qsig = _np.cos(-2.0*_np.pi*(IF*time+sigz))      # frequency modulation

    # filter out any high-order mixing products and resample at the requested video bandwidth
    Isig = downsample(Isig, 3*IF, fs, plotit=False).flatten()
    Qsig = downsample(Qsig, 3*IF, fs, plotit=False).flatten()
    time = _np.arange(time[0], time[-1], 1.0/fs)
#
    # form the complex signal
    sigz = Isig + 1j*Qsig

    # ============================================================= #

#    # full power spectra / correlation analysis
#    ft1 = fftanal(time, Isig, Qsig, tbounds = [time[0],time[-1]],
#            minFreq=0.1*fsig, windowfunction = 'SFT3F',
#            useMLAB=False, plotit=True, verbose=True,
#            detrend_style=1, onesided=False)
#    ft1.fftpwelch()

#    # single signal analysis
    ft = None
    ft = fftanal(tvec=time, sigx=sigz, minFreq=0.3*fsig)
    ft.pwelch()
    ft.convert2amplitudes()
    ipeak = _np.argmax(_np.abs(sigz))
    print(_np.angle(sigz[ipeak]), _np.abs(sigz[ipeak]))
#
    phi = _np.angle(ft.Xfft)
#    phi = _np.unwrap(phi)
##    phi -= phi[0]
#
    _plt.figure()
    _ax1 = _plt.subplot(2,1,1)
#    _plt.plot(ft.freq, _np.abs(ft.Xfft), 'b-')
    _plt.plot(ft.freq, _np.abs(ft.Lxx), 'b-')
    _ax2 = _plt.subplot(2,1,2, sharex=_ax1)
    _plt.plot(ft.freq, phi, 'r-')

#    _plt.figure()
#    rms = _np.sqrt(_np.mean(sigz*_np.conj(sigz)))
#    dfidt = _np.zeros_like(time)
#    dfidt[:-1] = _np.diff(_np.unwrap(_np.angle(sigz)))/_np.diff(time)
#    dfidt[-1] = dfidt[-2]
#    _plt.plot(time, rms*_np.ones_like(time), 'b-', time, dfidt, 'r-')
#
#    # simple analysis
#    nfft = len(sigz)
#    freq = fftmod.fftfreq(nfft, d=1.0/fs)
#    freq = fftmod.fftshift(freq)
#    sig = fftmod.fft(sigz, n=nfft)/_np.sqrt(nfft)
#    sig = fftmod.fftshift(sig)
#    phi = _np.angle(sig)
##    phi[phi<-2.7] += _np.pi
##    phi[phi>2.7] -= _np.pi
##    phi[phi<-2.0] += _np.pi
##    phi[phi>0] -= _np.pi
#    phi = _np.unwrap(phi)
#
#    _plt.figure()
#    _ax1 = _plt.subplot(2,1,1)
#    _plt.plot(freq, _np.abs(sig*_np.conj(sig)), 'b-')
#    _ax2 = _plt.subplot(2,1,2, sharex=_ax1)
#    _plt.plot(freq, phi, 'r-')

    return ft

# =========================================================================== #
# ===========================================================================


if __name__ == "__main__":
    test_DopplerSignal()
#    test_DopplerSignal(True)
#    test_DopplerSignal(False)

#    create_turb_spectra()
#    create_turb_spectra(True)

# =========================================================================== #
# =========================================================================== #