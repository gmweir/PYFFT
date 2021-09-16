# Continuous wavelet transfrom via Fourier transform
# Collection of routines for wavelet transform via FFT algorithm


#-- Some naming and other conventions --
# use f instead of omega wherever rational/possible
# *_ft means Fourier transform

#-- Some references --
# [1] Mallat, S.  A wavelet tour of signal processing
# [2] Addison, Paul S. The illustrated wavelet transform handbook
# [3] Torrence and Compo. A practical guide to wavelet
#     analysis. Bulletin of American Meteorological Society. 1998, 79(1):61-78

import numpy as np
from numpy.fft import fft, ifft, fftfreq



_maxshape_ = 1e8 ## for arrays larger than this number of values use
                 ## memory-mapped arrays

_maxexpanded_ = 5e6 ## only expand up to the next power of 2 signals shorter
                   ## than this

def memsafe_arr(shape, dtype):
    import tempfile as tmpf
    N = shape[0] * shape[1]
    if N < _maxshape_:
        return np.zeros(shape, dtype=dtype)
    else:
        print("Using memory-mapped arrays...")
        _tmpfile = tmpf.TemporaryFile('w+')
        return np.memmap(_tmpfile, dtype=dtype, shape=shape)
        
    

#try:
#    from scipy.special import gamma
#except:

pi = np.pi

try:
    from scipy.special import gamma
    class DOG:
        """Derivative of Gaussian, general form"""
        # Incomplete, as the general form of the mother wavelet
        # would require symbolic differentiation.
        # Should be enough for the CWT computation, though

        def __init__(self, m = 1.):
            self.order = m
            self.fc = (m+.5)**.5 / (2*pi)
            self.f0 = self.fc
    
        def psi_ft(self, f):
            c = 1j**self.order / np.sqrt(gamma(self.order + .5)) #normalization
            w = 2*pi*f
            return c * w**self.order * np.exp(-.5*w**2)
except:
    class DOG:
        def __init__(self,*args):
            print("Couldn't create DOG wavelet because scipy is unavailable")
            return None
        

class Mexican_hat:
    def __init__(self, sigma = 1.0):
        self.sigma = sigma
        self.fc = .5 * 2.5**.5 / pi
        self.f0 = self.fc
    def psi_ft(self, f):
        """Fourier transform of the Mexican hat wavelet"""
        c = np.sqrt(8./3.) * pi**.25 * self.sigma**2.5 
        wsq = (2. * pi * f)**2.
        return -c * wsq * np.exp(-.5 * wsq * self.sigma**2.)
    def psi(self, tau):
        """Mexian hat wavelet as described in [1]"""
        xsq = (tau / self.sigma)**2.
        c = 2 * pi**-.25 / np.sqrt(3 * self.sigma) # normalization constant from [1]
        out =  c * (1 - xsq) * np.exp(-.5 * xsq)
        out *= (tau < 6) * (tau > -6) # make it 'compact support'
        return out
    def cone(self, f):
        "e-folding time [TC98]. For frequencies"
        return self.f0*2.0**0.5 / f
    def cone_s(self, s):
        "e-folding time [TC98]. For scales"
        return 2.**0.5*s
    def set_f0(self, f0):
        pass

def heavyside(x):
    return 1.0*(x>0.0)

class Morlet:
    def __init__(self, f0 = 1.5):
        self.set_f0(f0)
    def psi(self, t):
        return pi**(-0.25) * np.exp(-t**2 / 2.) * np.exp(2j*pi*self.f0*t) #[3]
    def psi_ft(self, f):
        """Fourier transform of the approximate Morlet wavelet
            f0 should be more than 0.8 for this function to be
            correct."""
        # [3]
        #coef = (pi**-.25) * heavyside(f)
        coef = (pi**-.25) * 2**0.5 # (Addison)
        return  coef * np.exp(-.5 * (2 * pi * (f - self.f0))**2.)
    def sigma_t_s(self,s):
        """
        Heisenberg box width for scales [Addison]
        s/sqrt(2)
        """
        return s/np.sqrt(2)

    def sigma_f_s(self, s):
        """
        Heisenberg box height for scales [Addison]
        1/(s*pi*2*sqrt(2))
        // close to simulations of |W|^2 for sine waves
        // with fixed frequencies
        """
        return 1.0/(s*pi*np.sqrt(8.0))

    def sigma_f_f(self,f):
        """
        Heisenberg box height in frequencies
        s*sqrt(2)/4
        """
        return self.sigma_f_s(self.f0/f)

    def cone(self, f):
        "e-folding time [Addison]. For frequencies"
        return self.f0/(f*2.0**0.5)
    def cone_s(self, s):
        "e-folding time [Addison]. For scales"
        return self.sigma_t_s(s)
    def set_f0(self, f0):
        self.f0 = f0
        #self.fc = f0


def next_pow2(x):
    return 2.**np.ceil(np.log2(x))

def pad_func(ppd):
    func = None
    if ppd == 'zpd':
        func = lambda x,a:  x*0.0
    elif ppd == 'cpd':
        func = lambda x,a:  a*np.ones(np.size(x))
    elif ppd == 'mpd':
        func = lambda x,a: x[::-1]
    return func
    
def evenp(n):
    return not n%2

def cwt_a(signal, scales, sampling_scale = 1.0,
          wavelet=Mexican_hat(),
          ppd = 'constant',
          verbose = False,):
    """ Continuous wavelet transform via fft. Scales version."""
    import sys
    siglen = len(signal)

    if hasattr(wavelet, 'cone_s'):
        needpad = wavelet.cone_s(np.max(scales))/sampling_scale
    else:
        needpad = 1.2*np.max(scales)
    needpad = int(np.ceil(needpad))

    if siglen < _maxexpanded_: # only do expansion up to next ^2 for short signals
        ftlen = int(next_pow2(siglen + 2*needpad))
    else:
        #if not evenp(siglen):
        #    siglen -= 1
        #    signal = signal[:-1]
        ftlen = int(siglen + 2*needpad)

    padlen1 = int(round((ftlen - siglen)/2))
    padlen2 = evenp(siglen) and padlen1 or padlen1+1

    padded_signal = np.pad(signal, (padlen1,padlen2),mode=ppd)
    
    
    #padded_signal = np.ones(ftlen)
    #padfunc = pad_func(ppd)
    #padded_signal[:padlen1] = padfunc(padded_signal[:padlen1],signal[0])
    #padded_signal[-padlen2:] = padfunc(padded_signal[-padlen2:],signal[-1])
    #padded_signal[padlen1:-padlen2] = signal



    signal_ft = fft(padded_signal)     # FFT of the signal
    del padded_signal
    ftfreqs = fftfreq(ftlen, sampling_scale)  # FFT frequencies

    psi_ft = wavelet.psi_ft
    coef = np.sqrt(2*pi/sampling_scale)

    ### version with map is slower :(
    #def _ls(s):
    #    return ifft(signal_ft*coef*(s**0.5)*np.conjugate(psi_ft(s*ftfreqs)))[padlen:-padlen]
    #xx = map(_ls, scales)
    #W = np.array(xx)

    ### matrix version
    scales = scales.reshape(-1,1)
    #s_x_ftf = np.dot(scales, ftfreqs.reshape(1,-1))
    #psi_ft_bar = np.conjugate(psi_ft(s_x_ftf))
    #psi_ft_bar *= coef*np.sqrt(scales)
    #W = ifft( psi_ft_bar * signal_ft.reshape(1,-1))[:,padlen1:-padlen2]
    #return W

    ## create the result matrix beforehand
    #W = np.zeros((len(scales), siglen), 'complex')
    W = memsafe_arr((len(scales), siglen), 'complex')
    ## Now fill in the matrix
    for n,s in enumerate(scales):
        if verbose :
            sys.stderr.write('\r Processing scale %04d of %04d'%(n+1,len(scales)))
        psi_ft_bar = np.conjugate(psi_ft(s * ftfreqs))
        psi_ft_bar *= coef*np.sqrt(s) # Normalization from [3]
        W[n,:] = ifft(signal_ft * psi_ft_bar)[padlen1:-padlen2]
    return W


def cwt_f(signal, freqs, Fs=1.5, wavelet = Morlet(), ppd = 'constant', verbose=False):
    """Continuous wavelet transform -- frequencies version"""
    scales = wavelet.f0/freqs
    dt = 1./Fs
    return cwt_a(signal, scales, dt, wavelet, ppd, verbose=verbose)


def eds(x, f0=1.5):
    "Energy density surface [2,3]"
    ## Update, simulations with MK (as in [3]) suggest that I
    ## shouldn't divide by f0 to obtain correct normalisation,
    ## e.g. 1 s.d. is mean for white noise signal
    #return abs(x)**2/f0
    return abs(x)**2

def real(x, *args):
    return x.real

def cwt_phase(x, *args):
    return np.arctan2(x.imag, x.real)


def xwt_f(sig1, sig2, freqs, Fs=1.0, wavelet=Morlet()):
    "Cross-wavelet coeficients for 2 signals"
    cwtf = lambda x: cwt_f(x, freqs, Fs, wavelet)
    return xwt(cwtf(sig1),cwtf(sig2))

def absxwt_f(sig1, sig2, freqs, Fs=1.0, wavelet=Morlet()):
    "Cross-wavelet power for 2 signals"
    return abs(xwt_f(sig1,sig2, freqs, Fs, wavelet))/wavelet.f0**0.5

def absxwt_a(sig1, sig2, scales, dt=1.0, wavelet=Morlet()):
    "Cross-wavelet power for 2 signals"
    Fs = 1./dt
    freqs = wavelet.f0/scales
    return abs(xwt_f(sig1,sig2, freqs, Fs, wavelet))/wavelet.f0**0.5


def xwt(wcoefs1,wcoefs2):
    "Cross wavelet transform for 2 sets of coefficients"
    return wcoefs1*wcoefs2.conjugate()

def absxwt(wcoefs1,wcoefs2, f0=1.0):
    "Cross-wavelet power for 2 sets of coefficients"
    ## Why do I divide by f_0^2 here?
    return  abs(xwt(wcoefs1,wcoefs2))/f0**0.5

def wtc_a(sig1, sig2, scales, dt=1.0, wavelet=Morlet(1.0),sc_scale='log'):
    cwta = lambda x: cwt_a(x, scales, dt, wavelet)
    return coherence_a(cwta(sig1), cwta(sig2), scales, dt, wavelet.f0, sc_scale=sc_scale)


def wtc_f(sig1, sig2, freqs, Fs=1.0, wavelet=Morlet(f0=1.0),
          sc_scale='log'):
    cwtf = lambda x: cwt_f(x, freqs, Fs, wavelet)
    return coherence_f(cwtf(sig1), cwtf(sig2), freqs, Fs, wavelet.f0,sc_scale)

def coherence_f(x,y,freqs, Fs=1.0, f0=1.0, sc_scale = 'log'):
    return coherence_a(x,y,f0/freqs, 1.0/Fs, f0, sc_scale)

def coherence_a(x,y,scales,dt,f0=1.0,sc_scale = 'log'):
    # almost not useful 
    scv = np.reshape(scales, (-1,1))
    sx = wsmooth_a((abs(x)**2)/scv,scales,dt=dt,f0=f0,sc_scale=sc_scale)
    sy= wsmooth_a((abs(y)**2)/scv,scales,dt=dt,f0=f0,sc_scale=sc_scale)
    sxy = wsmooth_a((x*y.conjugate())/scv, scales, dt=dt,f0=f0,sc_scale=sc_scale)
    out_ph = np.arctan2(sxy.real, sxy.imag)
    out =  abs(sxy)**2/(sx.real*sy.real)
    out = np.ma.masked_invalid(out)
    
    return out, np.ma.masked_invalid(out_ph)


def cphase(x,y):
    #hard to interprete :(
    d = xwt(x,y)
    return np.arctan2(d.imag, d.real)

def wsmooth_a_conv(coefs, scales, dt=1.0, f0=1.0,
                   sc_scale ='log',
                   dj0=0.6):
    "Smoothing of wavelet coefs. Scales version, scales must be in log scale (octaves)"
    from scipy import signal
    G = lambda t,s: np.exp(-0.5*(t/s)**2) # Torrence and Webster 1998
    # G = lambda t,s : (1/s*(2*pi)**0.5)*exp(-(t/2)**2) ## Bloomfield et al 2004

    # "correct Ghat for G"
    #Ghat = lambda k,s: s*((2*pi)**0.5) * np.exp(-pi**2 * k**2 * 2 * s**2)
    Ghat = lambda k,s: s*np.exp(-0.5*(s*2*pi*k)**2)


    W = np.zeros(coefs.shape, 'complex')
    fftk = fftfreq(coefs.shape[1], dt)

    for n,s in enumerate(scales):
        kernhat = Ghat(fftk, s)
        W[n,:] = ifft(fft(coefs[n,:]) * kernhat)

    if sc_scale == 'log':
        dj0steps = dj0/(2*abs(np.mean(np.diff(np.log2(scales)))))
        kern = np.concatenate([[dj0steps%1], np.ones(2*np.round(dj0steps)-1),
                               [dj0steps%1]])
        kern /= np.sum(kern)

        for j in range(coefs.shape[1]):
            W[:,j] = np.convolve(W[:,j], kern,mode='same')

    else:
        print("not smoothing in scales, as scales are not in log stepping")
    return W

def coherence_a_diff(W1,W2,scales,dt,f0=1.0,h=0.1, tdiffuse=10,
                     acc_sp = 10,
                     *args,**kwargs):
    "wavelet coherence with smoothing via diffusion"
    scv = np.reshape(scales, (-1,1))
    sx_p = (abs(W1)**2)/scv
    sy_p = (abs(W2)**2)/scv
    sxy_p = (W1*W2.conjugate())/scv
    wcoher = lambda sx,sy,sxy: abs(sxy)**2/(sx.real*sy.real)
    acc = [wcoher(sx_p, sy_p, sxy_p)]
    #t = np.linspace(0,tdiffuse, acc_sp)
    #sx_n = _wsmooth_a_diffusion_integrate(sx_p, t, scales, dt, *args,**kwargs)
    #sy_n = _wsmooth_a_diffusion_integrate(sy_p, t, scales, dt, *args,**kwargs)
    #sxy_n = _wsmooth_a_diffusion_integrate(sxy_p, t, scales, dt, *args,**kwargs)
    #acc  = [wcoher(ax,ay,axy) for ax,ay,axy in zip(sx_n, sy_n, sxy_n)]
    for k,te in enumerate(arange(0,tdiffuse,h)):
        sx_n = sx_p + h*_wsmooth_a_diffusion_tderiv(sx_p, scales,dt,*args,**kwargs)
        sy_n = sy_p + h*_wsmooth_a_diffusion_tderiv(sy_p,scales,dt,*args,**kwargs)
        sxy_n  = sxy_p + h*_wsmooth_a_diffusion_tderiv(sxy_p,scales,dt, *args,**kwargs)
        if not k%acc_sp:
            acc.append(wcoher(sx_n, sy_n, sxy_n))
        sx_p, sy_p, sxy_p = sx_n, sy_n, sxy_n
        sys.stderr.write('\r diff time: %03.3f'%te)
    acc.append(wcoher(sx_n, sy_n, sxy_n))
    return acc


def _wsmooth_a_diffusion_integrate(I0, t, *args):
    from scipy import integrate
    sh = I0.shape
    def _step(y,t,*args):
        im = y.reshape(sh)
        return _wsmooth_a_diffusion_tderiv(im, *args).reshape(-1)
    y0 = I0.reshape(-1)
    I2 = integrate.odeint(_step,y0,t,args)
    return np.array([i2.reshape(sh) for i2 in I2])


def _wsmooth_a_diffusion_tderiv(I, scales, dt, psi=Morlet(),pow=1.):
    beta = 1/(2*pi*2**0.5)
    Dx  = np.reshape(psi.sigma_t_s(scales)**pow, (-1,1))
    Ds = np.reshape(psi.sigma_f_s(scales)**pow, (-1,1))
    kdiff = np.array([1,-2,1])
    Diff_x = np.zeros(I.shape, 'complex')
    Diff_s = np.zeros(I.shape, 'complex')
    Diff_x = signal.convolve2d(I, kdiff.reshape(1,-1), mode='same')
    Diff_s = signal.convolve2d(I, kdiff.reshape(-1,1), mode='same')    
    ## for n in xrange(len(scales)):
    ##     Diff_x[n,:] = np.convolve(I[n,:], kdiff,mode='same')
    ## for j in xrange(I.shape[1]):
    ##     Diff_s[:,j] = np.convolve(I[:,j], kdiff, mode='same')
    return Dx*Diff_x + Ds*Diff_s



def wsmooth_a_diffusion(coefs, scales, dt=0.1, f0=1.0, sc_scale='lin',
                        alpha=1.0e-2, niter=1600):
    from scipy import signal
    beta = 1/(2*pi*2**0.5)

    Dx = (scales/2**0.5).reshape(-1,1)
    Ds = (beta/scales).reshape(-1,1)

    Wprev = coefs
    kdiff = np.array([1,-2,1])
    lapl = np.array([(0,1,0), (1,-4,1), (0,1,0)])
    #acc = [coefs]
    for k in range(niter):
        if not k%10:
            sys.stderr.write('\r iteration: %0d'%k)
        Diff_x = np.zeros(coefs.shape, 'complex')
        Diff_s = np.zeros(coefs.shape, 'complex')
        for n in range(len(scales)):
            Diff_x[n,:] = np.convolve(Wprev[n,:], kdiff,mode='same')
        for j in range(coefs.shape[1]):
            Diff_s[:,j] = np.convolve(Wprev[:,j], kdiff, mode='same')
        Diff_x *= Dx
        Diff_s *= Ds
        Wnext = Wprev + alpha*(Diff_x + Diff_s)
        d = np.std(Wnext-Wprev)
        #acc.append(Wnext.real)
        Wprev = Wnext.copy()
    return Wnext

wsmooth_a = wsmooth_a_conv



def mirrorpd(k, L):
    if 0 <= k < L : return k
    else: return -(k)%L


def phase_coherence_other(sig1,sig2,freqs, Fs=1.0, wavelet=Morlet()):
    _w1 = cwt_phase(cwt_f(sig1, freqs, Fs, wavelet))
    _w2 = cwt_phase(cwt_f(sig2, freqs, Fs, wavelet))
    L = len(sig1)
    out = np.zeros(_w1.shape)
    phdiff = np.exp(1j*(_w1 - _w2))
    ns = 10
    for j in range(len(freqs)-ns):
        if hasattr(wavelet, 'cone_s'):
            needpad = wavelet.cone_s(wavelet.f0/freqs[j])*Fs
        else:
            needpad = 1.2*np.max(wavelet.f0/freqs)*Fs

        padlen = int(np.round(needpad))
        
        indices = [mirrorpd(i, L) for i in list(range(-padlen, 0)) + list(range(0,L)) + list(range(L, L+padlen))]
        s = phdiff[j:j+ns,indices]
        for i in range(L):
            out[j,i] = np.abs(np.mean(s[:,i:i+2*padlen]))
    return out, np.abs(phdiff.mean(axis=1))

import time 
def speed_test(sig_len, nscales = 512, N = 100, ppd='zpd',wavelet=Morlet()):
    tick = time.clock()
    s = np.random.randn(sig_len)
    scales = np.linspace(0.5, sig_len/4.0, nscales)
    for j in range(N):
        eds = cwt_a(s, scales, sampling_scale = 1.0, ppd = ppd,
                    wavelet=wavelet)
    tock =  time.clock() - tick
    print(tock)
    return tock

