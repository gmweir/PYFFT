# General purpose utilities

import numpy as np
from numpy import log, arange, ones
import pylab as pl

import swan.gui
from swan import pycwt
swanrgb = swan.gui.swanrgb


def log2(x):
    """Base 2 logarithm"""
    return log(x)/log(2.)

def make_progress_str(i,L=100):
    """Make a progress string with a rotating stick"""
    return '\r%3d%s %s' % (100*i/L, '%', ('|','/','-','\\')[i%4])


def deriv(x,y):
    """Calculate 3-point derivative of a signal"""
    return  .5*((y[1:-1] - y[0:-2]) / (x[1:-1] - x[0:-2]) + \
                (y[2:] - y[1:-1]) / (x[2:] - x[1:-1]))

def get_extr_deriv(x,y,maxf=True,wth=4):
    """Derivative-based extrema detection"""
    der = deriv(x,y)
    a = der > 0
    if maxf:
        a = a[1:] < a[:-1]
    else:
        a = a[1:] > a[:-1]

    k = np.arange(len(a))[a]
    xe = np.ones(len(k),dtype='d')
    ye = np.ones(len(k),dtype='d')

    for j,i in enumerate(k):
        p = np.polyfit(x[i+1:i+3], der[i:i+2], 1)
        xe[j] = -p[1] / p[0] #???
        p = np.polyfit(x[i:i+3], y[i:i+3], 2)
        #ye[j] = polyval(p, x[i+1:i+2])[0]
        ye[j] = np.polyval(p, (xe[j],) )[0]
    return xe,ye

def detrend1(vect, n=1):
    j = arange(len(vect))
    return vect - polyval(polyfit(j, vect, n), j)

def deoutburst(vect, n = 4):
    d = np.std(vect)
    m = np.mean(vect)

    low = m - n*d
    high = m+ n*d

    res = vect.copy()
    for i in range(len(vect)):
        if vect[i] < low:
            res[i] = low
        elif vect[i] > high:
            res[i] = high
        else:
            res[i] = vect[i]
    return res

def default_freqs(Ns, f_s, num=100):
    """
    Return default frequencies vector
    -- Ns:  number of samples in data vector
    -- f_s: sampling frequency
    -- num: number of frequencies required
    """
    T = Ns/f_s
    return np.linspace(4/T, f_s/2, num=num)

def default_freqs_log(Ns, f_s, num=100):
    """
    Return default frequencies vector
    -- Ns:  number of samples in data vector
    -- f_s: sampling frequency
    -- num: number of frequencies required
    """
    T = Ns/f_s
    return np.logspace(np.log10(4/T), np.log10(f_s/2), num=num)


def alias_freq(f, fs):
    if f < 0.5*fs:
        return f
    elif 0.5*fs < f < fs:
        return fs - f
    else:
        return alias_freq(f%fs, fs)

def in_subplots(seq):
    pl.figure();
    N = len(seq)
    for k,s in enumerate(seq):
        subplot(N,1,k+1);
        plot(s)

def ifnot(a, b):
    "if a is not None, return a, else return b"
    if a is None: return b
    else: return a


def DFoSD(vec, normL=None, th = 1e-6):
    "Remove mean and normalize to S.D."
    normL = ifnot(normL, len(vec))
    m, x  = np.mean, vec[:normL]
    sdx = np.std(x,0)
    out =  np.zeros(vec.shape, vec.dtype)
    if sdx.shape is ():
        if np.abs(sdx) > th:
                out = (vec-m(x))/sdx
    else:
        zi = np.where(np.abs(sdx) < th)[0]
        sdx[zi] = 1.0
        out = (vec-m(x))/sdx
        out[zi]=0
    return out

def confidence_contour(esurf, extent, ax, L=3.0):
    # Show 95% confidence level (against white noise, v=3 \sigma^2)
    ax.contour(esurf, [L], extent=extent,
               cmap=mpl.cm.gray)

def cone_infl(freqs, extent, wavelet, ax):
    try:
        ax.fill_betweenx(freqs,
                         extent[0],
                         extent[0]+wavelet.cone(freqs),
                         alpha=0.5, color='black')
        ax.fill_betweenx(freqs,
                         extent[1]+wavelet.cone(-freqs),
                         extent[1],
                         alpha=0.5, color='black')
    except:
        print("Can't use fill_betweenx function: update\
        maptlotlib?")


def wavelet_specgram(signal, f_s, freqs,  ax,
                     wavelet = pycwt.Morlet(),
                     padding = 'constant',
                     cax = None,
                     vmin=None, vmax=None,
                     correct = None,
                     cwt_fn = pycwt.eds,
                     confidence_level = False,
                     title = "",
                     cmap=swanrgb):
    wcoefs = pycwt.cwt_f(signal, freqs, f_s, wavelet, padding)
    surf = cwt_fn(wcoefs, wavelet.f0)
    
    if correct == 'freq1':
        coefs = freqs*2.0/np.pi
        for i in range(surf.shape[1]):
            surf[:,i] *= coefs

    if vmax is None: vmax = np.percentile(surf, 99.5)
    if vmin is None: vmin = np.percentile(surf, 0.5)


    endtime = len(signal)/f_s
    extent=[0, endtime, freqs[0], freqs[-1]]
    im = ax.imshow(surf, extent = extent,
                   origin = 'low',
                   aspect='auto',
                   vmin = vmin, vmax = vmax,
                   cmap = cmap,
                   alpha = 0.95)
    if not cax:
        pl.colorbar(im, ax=ax)
    else:
        pl.colorbar(im, cax = cax)
    cone_infl(freqs, extent, wavelet, ax)
    if confidence_level:
        confidence_contour(surf, extent, ax, confidence_level)
    return surf

def setup_axes_for_spectrogram(figsize = (12,6)):
    "Set up axes for a plot with signal, spectrogram and a colorbar"
    fig = pl.figure(figsize = figsize)
    ax = [fig.add_axes((0.08, 0.4, 0.8, 0.5))]
    ax.append(fig.add_axes((0.08, 0.07, 0.8, 0.3), sharex=ax[0]))
    ax.append(fig.add_axes((0.9, 0.4, 0.02, 0.5), 
                           xticklabels=[], 
                           yticklabels=[]))
    return fig,ax


def plot_wavelet_coherence_with_ts(s1, s2, f_s, freqs=None,
                                   sc_scale='log',
                                   figsize=(12,6),
                                   cmap='spectral'):
    l1,l2 = list(map(len, (s1,s2)))
    if l1 != l2:
        raise ValueError("two input signals must have same length")
    Ns = len(s1)
    if freqs is None:
        if sc_scale == 'log':
            freqs = default_freqs_log(Ns, f_s,512)
        elif sc_scale == 'lin':
            freqs = default_freqs(Ns, f_s,512)

    tvec = np.arange(0, (Ns+2)/f_s, 1./f_s)[:Ns]
    fig,axlist = setup_axes_for_spectrogram(figsize)
    fig2,axlist2 = setup_axes_for_spectrogram(figsize)    

    axlist[1].plot(tvec, s1,'r-',lw=0.5)
    axlist[1].plot(tvec, s2,'b-',lw=0.5)

    axlist2[1].plot(tvec, s1,'r-',lw=0.5)
    axlist2[1].plot(tvec, s2,'b-',lw=0.5)


    surf,phase = pycwt.wtc_f(s1,s2,freqs,f_s, sc_scale=sc_scale)
    if sc_scale is 'log':
        extent = (tvec[0], tvec[-1], np.log(freqs[0]), np.log(freqs[-1]))
    else:
        extent = (tvec[0], tvec[-1], freqs[0], freqs[-1])

    im = axlist[0].imshow(surf, extent=extent,cmap=cmap,origin='lower')
    pl.colorbar(im, cax = axlist[2])
    axlist[0].axis(extent)
    
    im2 = axlist2[0].imshow(phase/np.pi, extent=extent,cmap='jet',origin='lower')    
    pl.colorbar(im2, cax = axlist2[2])
    axlist2[0].axis(extent)

    if sc_scale =='log':
        nticks = 5;
        yticks = np.logspace((np.round(np.log10(freqs[0]))),
                             (np.round(np.log10(freqs[-1]))),nticks)
        yticklabels = ['%2.2f'%i for i in yticks]
        axlist[1].set_xlabel("Time, s")    
        axlist2[1].set_xlabel("Time, s")

        pl.setp(axlist[0], yticks=np.log(yticks), yticklabels=yticklabels)
        pl.setp(axlist2[0], yticks=np.log(yticks), yticklabels=yticklabels)

    pl.setp(axlist[0], ylabel='Frequency, Hz')
    pl.setp(axlist2[0], ylabel='Frequency, Hz')

    fig.suptitle('Wavelet coherence')
    fig2.suptitle('Wavelet phase shift')
    

def plot_spectrogram_with_ts(signal, f_s, freqs=None,
                             figsize=(12,6),
                             lc = 'k', title_string = '',
                             **kwargs):
    "Create a figure of a signal, spectrogram and a colorbar"
    Ns = len(signal)
    freqs = ifnot(freqs, default_freqs(Ns, f_s, 512))
    tvec = np.arange(0, (Ns+2)/f_s, 1./f_s)[:Ns]

    fig,axlist = setup_axes_for_spectrogram(figsize)

    axlist[1].plot(tvec, signal,'-',color=lc)

    if 'cax' not in kwargs:
        kwargs['cax'] = axlist[2]
    surf = wavelet_specgram(signal, f_s, freqs,  axlist[0], **kwargs)
    axlist[0].set_title(title_string)
    axlist[0].axis((tvec[0],tvec[-1], freqs[0],freqs[-1]))
    #axlist[1].xlim((tvec[0],tvec[-1]))
    return fig, surf

