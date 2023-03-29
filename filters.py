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

# ========================================================================= #
# ========================================================================= #


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
    # return  y[int(window_len/2-1):-int(window_len/2)]
    if _np.mod(window_len+1, 2):   # odd
        return y[(window_len//2-1):-(window_len//2)]
    else:   # even
        return y[((window_len-1)//2):-(window_len//2)]        
    # end if
# end def

def smooth_demo():
    t = _np.linspace(-4,4,100)
    x = _np.sin(t)
    xn = x+_np.random.randn(len(t))*0.1
    ws=31
    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    
    _, (_ax1, _ax2, _ax3) = _plt.subplots(3, 1, sharex=False)
    _ax1.plot(_np.ones(ws))
    for w in windows[1:]:
        _ax1.plot(eval('_np.'+w+'(ws)'))
    # end for
    _ax1.set_xlim([0, 30])
    _ax1.set_ylim([0, 1.1])
    _ax1.legend(windows)
    _ax1.set_title("The smoothing windows")

    _ax2.plot(t, x)
    _ax2.plot(t, xn)
    for w in windows:
        _ax2.plot(t, smooth(xn, window_len=ws, window=w))
    # end for
    _ax2.set_title("Smoothing a noisy signal: %i pnts"%(ws,))
    
    _ax3.plot(t, x)
    _ax3.plot(t, xn)
    for w in windows:
        _ax3.plot(t, smooth(xn, window_len=10, window=w))
    # end for    
    
    l=['original signal', 'signal with noise']
    l.extend(windows)
    _ax3.legend(l)
    _ax3.set_title("Smoothing a noisy signal: %i pnts"%(10,))
    _plt.show()
# end def smooth_demo

# ========================================================================= #
# ========================================================================= #


def __aply_abfilter(b, a, x, axis=-1, filtfilt=True):
    if filtfilt:
        # Pad the input signal before applying the filter
        # return _dsp.filtfilt(b, a, x, axis=axis, method='pad', padtype='odd', padlen=None)
        #
        # the length of the impulse response is ignored
        #  ---> irlen=None, none of the impulse response is ignored
        #  ---> irlen specifies the length of the impulse response (can improve performance) 
        return _dsp.filtfilt(b, a, x, axis=axis, method='gust', irlen=None)
    else:
        return _dsp.lfilter(b, a, x, axis=axis)
    # end if
# end def

def __aply_sosfilter(sos, x, axis=-1, filtfilt=True, zi=None):
    if filtfilt:
        return _dsp.sosfiltfilt(sos, x, axis=axis, padtype='odd', padlen=None)
    else:
        return _dsp.sosfilt(sos, x, axis=axis, zi=zi)
    # end if
# end def 

    
def butter_bandpass(x,fs=4e6,lf=1000,hf=500e3,order=3,disp=0, axis=-1, filtfilt=True):
    """
    wrapper around the scipy butterworth filter design function
    
    Applies a bandpass digital filter designed using the "butter" function to input data
    
    
    Parameters
    ----------
    x : 
        Signal to filter.
    fs : double [Hz], optional
        Sampling frequency of the data. The default is 4e6 (4 MHz)
    lf : double [Hz], optional
        lower -3dB frequency. The default is 1000 Hz.
    hf : double [Hz], optional
        upper -3dB frequency. The default is 500e3 Hz.
    order : int, optional
        filter order. The default is 3.
    disp : boolean, optional
        Plot the data? The default is 0.
    axis : integer, optional
        Axis along with to apply the filter. The default is -1 (the last axis).
    filtfilt : boolean flag, optional
        Apply the filtfilt zero-phase filtering method. The default is True.
        Note--> filtfilt applies the numerical filter twice. 
            Once in a forward direction, and once in a backwards direction. 
            The resulting phase-shift cancels, but this makes the effective
            filter order twice the input order. 
    
    Returns
    -------
    Bandpass Filtered signal

    """
    nyq=0.5*fs
    low=lf/nyq
    high=hf/nyq
    b,a = _dsp.butter(order,[low, high], btype='band', analog=False)

    y = __aply_abfilter(b, a, x, axis=axis, filtfilt=filtfilt)

    if disp:
        _, (_ax1, _ax2) = _plt.subplots(2, 1, sharex=False)
        
        # _plt.figure()
        w,h=_dsp.freqz(b, a, worN=2048, fs=fs)

        _ax1.plot(w, _np.abs(h))
        _ax1.set_xlabel('f [Hz]')
        _ax1.set_ylabel(r'|H($\omega$)|')
        _ax1.set_title('Digital bandpass filter response')        
        
        _tt = _np.linspace(0, len(x)/fs, num=len(x), endpoint=True)
        _ax2.plot(_tt, x)
        _ax2.plot(_tt, y)
        _ax2.set_xlabel('t [s]')
        _ax2.set_ylabel('Signals')
        _ax2.set_title('Input and Output Signal')                

        # _plt.plot((fs*0.5/_np.pi)*w, _np.abs(h))
        _plt.show()
        return y, (_ax1, _ax2)
    # end if
    return y


def butter_lowpass(cutoff, fnyq, order=5):
    """
    This is a lowpass filter design subfunction

    Parameters
    ----------
    cutoff : double
        lowpass filter frequency
    fnyq : double
        Nyquist frequency
    order : TYPE, optional
        Filter order. The default is 5.

    Returns
    -------
    Transfer function polynomials (numerator, denominator) for a lowpass filter 

    """   
    normal_cutoff = cutoff / fnyq
    b, a = _dsp.butter(order, normal_cutoff, btype='low', analog=False)
#    b, a = _dsp.butter(order, normal_cutoff, btype='low')
    return b, a
#end butter_lowpass


#This is a subfunction for filtering the data
def butter_lowpass_filter(data, cutoff, fs, order=5, axis=0, disp=False, filtfilt=True): 
    b, a = butter_lowpass(cutoff, fs, order=order)
#    y = _dsp.lfilter(b, a, data)
    # y = _dsp.filtfilt(b, a, data, axis=axis)
    
    y = __aply_abfilter(b, a, data, axis=axis, filtfilt=filtfilt)

    if disp:
        _, (_ax1, _ax2) = _plt.subplots(2, 1, sharex=False)
        
        # _plt.figure()
        w,h=_dsp.freqz(b, a, worN=2048, fs=fs)

        _ax1.plot(w, _np.abs(h))
        _ax1.set_xlabel('f [Hz]')
        _ax1.set_ylabel(r'|H($\omega$)|')
        _ax1.set_title('Digital lowpass filter response')        
        
        nlen = _np.shape(data)[axis]
        _tt = _np.linspace(0, nlen/fs, num=nlen, endpoint=True)
        _ax2.plot(_tt, _np.take(data, indices=_np.asarray(range(nlen), int), axis=axis))
        _ax2.plot(_tt, y)
        _ax2.set_xlabel('t [s]')
        _ax2.set_ylabel('Signals')
        _ax2.set_title('Input and Output Signal')                

        # _plt.plot((fs*0.5/_np.pi)*w, _np.abs(h))
        _plt.show()
        return y, (_ax1, _ax2)        
    # end if    
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

def test_data(A1=2.50, A2=1.70):
    # Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz
    t = _np.linspace(0, 1, 1000, False)  # 1 second
    Fs = (len(t)-1)/(t[-1]-t[0])   # Sampling frequency
    
    sig = A1*_np.sin(2*_np.pi*10*t) + A2*_np.sin(2*_np.pi*20*t)
    return t, sig, Fs    


def test_smooth():    
    """
    Smooth a noisy signal by convolution with a n-point stencil.
    """
    smooth_demo()
# end def


def test_butter_bandpass():
    """
    
    """
    # Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz
    A1=2.50
    A2=1.70
    t, sig, Fs = test_data(A1=A1, A2=A2)

    # Create a bandpass filter to isolate the 10 Hz signal
    x10_filtered, (_, _ax1) = butter_bandpass(sig, fs=Fs, lf=7, hf=13, order=3, disp=1)
    _ax1.axhline(y=A1, color='r', linestyle='--')
    
    # Create a bandpass filter to isolate the 20 Hz signal
    x20_filtered, (_, _ax2) = butter_bandpass(sig, fs=Fs, lf=17, hf=23, order=3, disp=1)    
    _ax2.axhline(y=A2, color='r', linestyle='--')
    return x10_filtered, x20_filtered
# end def


def test_butter_lowpass():
    """
    
    """
    # Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz
    A1=2.50
    A2=1.70
    t, sig, Fs = test_data(A1=A1, A2=A2)

    # Create a lowpass filter to isolate the 10 Hz signal
    x10_filtered, (_, _ax3) = butter_lowpass_filter(sig, cutoff=12, fs=Fs, order=3, axis=0, disp=True, filtfilt=True)
    _ax3.axhline(y=A1, color='r', linestyle='--')
    
    try:
        from FFT.fft import fft
    except:
        from .fft import fft
    # end try
    
    return x10_filtered
# end def


        
        
    # b, a = _dsp.butter(4, 100, 'low', analog=True)
    # w, h = _dsp.freqs(b, a)
    
    # _plt.semilogx(w, 20 * _np.log10(abs(h)))
    # _plt.title('Butterworth filter frequency response')
    # _plt.xlabel('Frequency [radians / second]')
    # _plt.ylabel('Amplitude [dB]')
    # _plt.margins(0, 0.1)
    # _plt.grid(which='both', axis='both')
    # _plt.axvline(100, color='green') # cutoff frequency
    # _plt.show()
    

    
    # fig, (_ax1, _ax2) = _plt.subplots(2, 1, sharex=True)
    # _ax1.plot(t, sig)
    # _ax1.set_title('10 Hz and 20 Hz sinusoids')
    # _ax1.axis([0, 1, -2, 2])
    
    # # Design a digital high-pass filter at 15 Hz to remove the 10 Hz tone, and
    # # apply it to the signal. (It's recommended to use second-order sections
    # # format when filtering, to avoid numerical error with transfer function
    # # (``ba``) format):
    # sos = _dsp.butter(10, 15, 'hp', fs=1000, output='sos')
    # filtered = _dsp.sosfilt(sos, sig)
    
    # _ax2.plot(t, filtered)
    # _ax2.set_title('After 15 Hz high-pass filter')
    # _ax2.axis([0, 1, -2, 2])
    # _ax2.set_xlabel('Time [seconds]')
    # _plt.tight_layout()
    # _plt.show()
    
# end def


# ========================================================================= #
# ========================================================================= #


if __name__=="__main__":
    
    # test_smooth()
    test_butter_bandpass()
    test_butter_lowpass()

# end if


# ========================================================================= #
# ========================================================================= #



