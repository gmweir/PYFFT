# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 16:56:15 2021

@author: gawe
"""
# ========================================================================= #
# ========================================================================= #

import numpy as _np
import matplotlib.pyplot as _plt
import scipy.signal as _dsp
import pybaseutils.utils as _ut


# ========================================================================= #
# ========================================================================= #


def upsample(u_t, Fs, Fs_new, plotit=False):
    # Upsample a signal to a higher sampling rate
    # Use cubic interpolation to increase the sampling rate

    nt = len(u_t)
    tt = _np.arange(0,nt,1)/Fs
    ti = _np.arange(tt[0],tt[-1],1/Fs_new)

    # _ut.interp(xi,yi,ei,xo)
    u_n = _ut.interp( tt, u_t, ei=None, xo=ti)   # TODO!:  Add quadratic interpolation
    # uinterp = interp1d(tt, u_t, kind='cubic', axis=0)
    # u_n = uinterp(ti)

    return u_n
# end def upsample

def downsample(u_t, Fs, Fs_new, plotit=False):
    """
     The proper way to downsample a signal.
       First low-pass filter the signal
       Interpolate / Decimate the signal down to the new sampling frequency
    """

    tau = 2/Fs_new
    nt  = len(u_t)
    tt  = _np.arange(0, nt, 1)/Fs
    # tt  = tt.reshape(nt, 1)
    # tt  = (_np.arange(0, 1/Fs, nt)).reshape(nt, 1)
    try:
        nch = _np.size(u_t, axis=1)
    except:
        nch = 1
        u_t = u_t.reshape(nt, nch)
    # end try

    # ----------- #

    #2nd order LPF gets converted to a 4th order LPF by filtfilt
    lowpass_n, lowpass_d = _dsp.butter(2, 2.0/(Fs*tau), btype='low')

    if plotit:

        # ------- #

        #Calculate the frequency response of the lowpass filter,
        w, h = _dsp.freqz(lowpass_n, lowpass_d, worN=12000) #

        #Convert to frequency vector from rad/sample
        w = (Fs/(2.0*_np.pi))*w

        # ------- #

        _plt.figure(num=3951)
        # _fig.clf()
        _ax1 = _plt.subplot(3, 1, 1)
        _ax1.plot( tt,u_t, 'k')
        _ax1.set_ylabel('Signal', color='k')
        _ax1.set_xlabel('t [s]')

        _ax2 = _plt.subplot(3, 1, 2)
        _ax2.plot(w, 20*_np.log10(abs(h)), 'b')
        _ax2.plot(1.0/tau, 0.5*_np.sqrt(2), 'ko')
        _ax2.set_ylabel('|LPF| [dB]', color='b')
        _ax2.set_xlabel('Frequency [Hz]')
        _ax2.set_title('Digital LPF frequency response (Stage 1)')
        _plt.xscale('log')
        _plt.grid(which='both', axis='both')
        _plt.axvline(1.0/tau, color='k')
        _plt.grid()
        _plt.axis('tight')
    # endif plotit

    # nskip = int(_np.round(Fs/Fs_new))
    # ti = tt[0:nt:nskip]
    ti = _np.arange(0, nt/Fs, 1/Fs_new)

    u_n = _np.zeros((len(ti), nch), dtype=_np.float64)
    for ii in range(nch):
        # (Non-Causal) LPF
        u_t[:, ii] = _dsp.filtfilt(lowpass_n, lowpass_d, u_t[:, ii])

        # _ut.interp(xi,yi,ei,xo)
        u_n[:, ii] = _ut.interp(tt, u_t[:, ii], ei=None, xo=ti)
#        uinterp = interp1d(tt, u_t[:, ii], kind='cubic', axis=0)
#        u_n[:, ii] = uinterp(ti)
    #endif

    if plotit:
        _ax1.plot(ti, u_n, 'b-')

        _ax3 = _plt.subplot(3, 1, 3, sharex=_ax1)
        _ax3.plot(tt, u_t, 'k')
        _ax3.set_ylabel('Filt. Signal', color='k')
        _ax3.set_xlabel('t [s]')
#        _plt.show(hfig, block=False)
        _plt.draw()
#        _plt.show()

    # endif plotit

    return u_n
# end def downsample

def downsample_efficient(u_t, Fs, Fs_new, plotit=False, halforder=2, lowpass=None):
    """
     The proper way to downsample a signal.
       First low-pass filter the signal
       Interpolate / Decimate the signal down to the new sampling frequency
    """
    if lowpass is None:     lowpass = 0.5*Fs_new       # end if
    tau = 1.0/lowpass
#    tau = 2/Fs_new
    nt  = len(u_t)
    try:
        nch = _np.size(u_t, axis=1)
    except:
        nch = 1
        u_t = u_t.reshape(nt, nch)
    # end try

    # ----------- #

    #2nd order LPF gets converted to a 4th order LPF by filtfilt
    lowpass_n, lowpass_d = _dsp.butter(halforder, 2.0*lowpass/Fs, btype='low')
#    lowpass_n, lowpass_d = _dsp.butter(halforder, 2.0/(Fs*tau), btype='low')

    if plotit:
        # ------- #

        #Calculate the frequency response of the lowpass filter,
        w, h = _dsp.freqz(lowpass_n, lowpass_d, worN=12000) #

        #Convert to frequency vector from rad/sample
        w = (Fs/(2.0*_np.pi))*w

        # ------- #

        _plt.figure(num=3951)
        # _fig.clf()
        _ax1 = _plt.subplot(3, 1, 1)
        _ax1.plot(_np.arange(0, nt, 1)/Fs, u_t, 'k')
        _ax1.set_ylabel('Signal', color='k')
        _ax1.set_xlabel('t [s]')

        _ax2 = _plt.subplot(3, 1, 2)
        _ax2.plot(w, 20*_np.log10(abs(h)), 'b')
        _ax2.plot(1.0/tau, 0.5*_np.sqrt(2), 'ko')
        _ax2.axvline(1.0/tau, color='k')
        _ax2.set_ylabel('|LPF| [dB]', color='b')
        _ax2.set_xlabel('Frequency [Hz]')
        _ax2.set_title('Digital LPF frequency response (Stage 1)')
        _ax2.set_xscale('log')
        ylims = _ax2.get_ylim()
        _ax2.set_ylim((ylims[0], max((3,ylims[1]))))
        _ax2.grid(which='both', axis='both')
        _ax2.grid()

        _ax3 = _plt.subplot(3, 1, 3, sharex=_ax1, sharey=_ax1)
        _ax3.plot(_np.arange(0, nt, 1)/Fs, u_t, 'k')
#        _plt.xscale('log')
#        _plt.grid(which='both', axis='both')
#        _plt.axvline(1.0/tau, color='k')
#        _plt.grid()
        _plt.axis('tight')
    # endif plotit

    # nskip = int(_np.round(Fs/Fs_new))
    # ti = tt[0:nt:nskip]
#    ti = _np.arange(0, nt/Fs, 1/Fs_new)

    u_t = _ut.interp(xi=_np.arange(0, nt, 1)/Fs,
                     yi=_dsp.filtfilt(lowpass_n, lowpass_d, u_t, axis=0),
                     ei=None,
                     xo=_np.arange(0, nt/Fs, 1/Fs_new))
#    u_n = _np.zeros((len(ti), nch), dtype=_np.float64)
#    for ii in range(nch):
#        # (Non-Causal) LPF
#        u_t[:, ii] = _dsp.filtfilt(lowpass_n, lowpass_d, u_t[:, ii])
#
#        # _ut.interp(xi,yi,ei,xo)
#        u_n[:, ii] = _ut.interp(tt, u_t[:, ii], ei=None, xo=ti)
##        uinterp = interp1d(tt, u_t[:, ii], kind='cubic', axis=0)
##        u_n[:, ii] = uinterp(ti)
#    #endif

    if plotit:
#        _ax1.plot(_np.arange(0, nt/Fs, 1/Fs_new), u_t, 'b-')

#        _ax3 = _plt.subplot(3, 1, 3, sharex=_ax1, sharey=_ax1)
        _ax3.plot(_np.arange(0, nt/Fs, 1/Fs_new), u_t, 'b-')
        _ax3.set_ylabel('Filt. Signal', color='k')
        _ax3.set_xlabel('t [s]')
#        _plt.show(hfig, block=False)
        _plt.draw()
#        _plt.show()
    # endif plotit

    return u_t
# end def downsample_efficient


# ========================================================================= #
# ========================================================================= #



def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")
    # end if
    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")
    # end if
    if window_len<3:
        return x
    # end if

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    # end if

    # Reflect the data in the first and last windows at the end-points
    s=_np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=_np.ones(window_len,'d')
    else:
        w=eval('_np.'+window+'(window_len)')
    # end if

    # Convolve the n-point window (normalized to its sum, with the reflected data
    y=_np.convolve(w/w.sum(),s,mode='valid')
#    return y
    # return the window weighted data (same length as input data)
    return  y[(window_len/2-1):-(window_len/2)]

def smooth_demo():
    t=_np.linspace(-4,4,100)
    x=_np.sin(t)
    xn=x+_np.randn(len(t))*0.1
    ws=31

    _plt.subplot(211)
    _plt.plot(_np.ones(ws))

    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    _plt.hold(True)
    for w in windows[1:]:
        eval('plot('+w+'(ws) )')

    _plt.axis([0,30,0,1.1])

    _plt.legend(windows)
    _plt.title("The smoothing windows")
    _plt.subplot(212)
    _plt.plot(x)
    _plt.plot(xn)
    for w in windows:
        _plt.plot(smooth(xn,10,w))
    l=['original signal', 'signal with noise']
    l.extend(windows)

    _plt.legend(l)
    _plt.title("Smoothing a noisy signal")
    _plt.show()

# ========================================================================= #
# ========================================================================= #



def butter_bandpass(x,fs=4e6,lf=1000,hf=500e3,order=3,disp=0):
    nyq=0.5*fs
    low=lf/nyq
    high=hf/nyq
    b,a = _dsp.butter(order,[low, high], btype='band', analog=False)
    y = _dsp.lfilter(b,a,x)
#    y = _dsp.filtfilt(b, a, data, axis=0)
    w,h=_dsp.freqz(b,a,worN=2000)
    #_plt.plot(fs*0.5/_np.pi*w,_np.abs(h))
    #_plt.show()
    return y

#This is a lowpass filter design subfunction
def butter_lowpass(cutoff, fnyq, order=5):
    normal_cutoff = cutoff / fnyq
    b, a = _dsp.butter(order, normal_cutoff, btype='low', analog=False)
#    b, a = _dsp.butter(order, normal_cutoff, btype='low')
    return b, a
#end butter_lowpass

#This is a subfunction for filtering the data
def butter_lowpass_filter(data, cutoff, fs, order=5, axis=0):
    b, a = butter_lowpass(cutoff, fs, order=order)
#    y = _dsp.lfilter(b, a, data)
    y = _dsp.filtfilt(b, a, data, axis=axis)
    return y
#end butter_lowpass_filter

def complex_filtfilt(filt_n,filt_d,data):
    # dRR = _np.mean(data.real)+_dsp.filtfilt(filt_n, filt_d, data.real-_np.mean(data.real) ) #LPF injected signal
    # dII = _np.mean(data.imag)+_dsp.filtfilt(filt_n, filt_d, data.imag-_np.mean(data.imag) ) #LPF injected signal
    dRR = _dsp.filtfilt(filt_n, filt_d, data.real ) #LPF injected signal
    dII = _dsp.filtfilt(filt_n, filt_d, data.imag ) #LPF injected signal
    data = dRR+1j*dII
    return data
#end complex_filtfilt

# ========================================================================= #
# ========================================================================= #
