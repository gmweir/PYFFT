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

# ========================================================================== #

import numpy as _np
from scipy import linalg as _la
import matplotlib.mlab as _mlab
import matplotlib.pyplot as _plt

from pybaseutils.Struct import Struct
from pybaseutils.utils import detrend_mean, detrend_none, detrend_linear
from pybaseutils import utils as _ut

try:
    from FFT.windows import windows
except:
    from .windows import windows

if 1:
    try:
        from FFT import fft as fftmod
    except:
        from . import fft as fftmod
else:
    fftmod = _np.fft

__all__ = ['fft_pwelch', 'integratespectra', 'bandpower', 'bandpower_multitaper',
           'fftinfosc', 'getNpeaks', 'coh2', 'psd', 'csd', 'varcoh', 'varphi',
           'rescale', 'unscale', 'fft_deriv', 'Cxy_Cxy2', 'fftanal']

# ========================================================================== #
# ========================================================================== #


def fft_pwelch(tvec, sigx, sigy, tbounds=None, Navr=None, windowoverlap=None,
               windowfunction=None, useMLAB=None, plotit=None, verbose=None,
               detrend_style=None, onesided=None, **kwargs):
    """
    function [freq, Pxy, Pxx, Pyy, Cxy, phi_xy, info] =
        fft_pwelch(tvec, sigx, sigy, tbounds, Navr, windowoverlap,
                   windowfunction, useMLAB, plotit, verbose, plottofig)

    This function computes the spectra from an input and reference signal.  It
    detrends the input, calculates the cross- and auto-power spectra,
    normalizes everything, and saves parameters necessary for future work
        this function includes a trend removal!

    # ------------------------------------------------------------------- #

    inputs
      tvec    - [s], time-series of signal
      sigx    - [V], signal X,   ntx x nsigs  (ntx can be shorter than nt)
      sigy    - [V], signal Y,   nt x nsigs
      tbounds - [s], time-window for analysis. [tstart,tend] def:[1,len(tvec)]
      Navr    - [#], number of segments to split the input signals into
      windowoverlap  - [#], 0 to 1, fraction of segment overlap, def: 0
      windowfunction - [string], type of window function, def: 'box'
                      The hamming, hanning, and kaiser window are also
                      supported, and feel free to add more
      useMLAB - [Boolean], use CPSD and mscohere for FFT analysis, def:homebrew!
      plotit    - [Boolean], plot the linear amplitude spectra / coherence?
      verbose   - [Boolean], print to the screen
      plottofig - [fig_handle], plot to this figure

    outputs
      freq      - [Hz], frequency vector
      Pxy       - [V], Linear cross-power spectra of signal X and Y
      Pxx       - [V], Linear auto-power spectra of signal X
      Pyy       - [V], Linear auto-power spectra of signal Y
      Cxy       - [-], Coherence between signal X and signal Y
      phi_xy    - [rad], cross-phase between signal X and signal Y
      info      - [structure], contains all the calculations and extras

      including -  fftinfo.ENBW  -> Convert back to power spectral densities by
                    Pxy = Pxy/ENBW [V^2/Hz]
                      + much more

    # ------------------------------------------------------------------- #

      Written by GMW - Feb 7th, 2013 in Matlab
      Revisions:
          Feb 10th, 2016 - GMW - Ported to Python for use at W7-X
          Mar 8th, 2016 - GMW - Extensive revisions:
              - Made inputs work with column vectors for speed
              - Added uncertainty analysis of the power spectra so that it
                  takes into account the complex part of the signal
              - Added the mean_angle definition for averaging the phase with
                  uncertainty propagation
              - Added the coh_var definition for propagating uncertainty to
                  the coherence
              - Converted the main module into a class that calls this function

      Major revisions compiled July, 29th, 2019:
          multi-channel support
          different length inputs signals: resampling and tiling windows to match length (nT-crossphase stuff)
          upgraded window input selection by automatically selecting recomended overlap percentage
          added option to input minimum resolvable frequency instead of number of windows.
          added normalized auto- and cross-correlation calculations (getting this right is a pain)
    """

    if (Navr is None) or ('minFreq' in kwargs):
        calcNavr = True
    if windowfunction is None:
#        windowfunction = 'SFT3F'    # very low overlap correlation, wider peak to get lower frequencies
#        windowfunction = 'SFT3M'   # very low overlap correlation, low sidebands
        windowfunction = 'Hanning'  # moderate overlap correlation, perfect amplitude flattness at optimum overlap
    if windowoverlap is None:
        # get recommended overlap by function name
        windowoverlap = windows(windowfunction, verbose=False)
    if useMLAB is None:
        useMLAB=False
    if plotit is None:
        plotit=True
    if verbose is None:
        verbose=False
    if detrend_style is None:
        detrend_style=1
    if tbounds is None:
        tbounds = [tvec[0], tvec[-1]]
    # end if

    if onesided is None:
        onesided = True
        if _np.iscomplexobj(sigx) or _np.iscomplexobj(sigy):
            onesided = False
        # end if
    # end if

    # Matlab returns the power spectral denstiy in [V^2/Hz], and doesn't
    # normalize it's FFT by the number of samples or the power in the windowing
    # function.  These can be handled by controlling the inputs and normalizing
    # the output:
#    dt   = 0.5*(tvec[2]-tvec[0])       #[s],  time-step in time-series
#    Fs   = 1.0/dt                      #[Hz], sampling frequency
    Fs = (len(tvec)-1)/(tvec[-1]-tvec[0])
#    dt = 1.0/Fs

    # ==================================================================== #
    # ==================================================================== #

    # Detrend the two signals to get FFT's
    i0 = int( _np.floor( Fs*(tbounds[0]-tvec[0] ) ) )
    i1 = int( _np.floor( 1+Fs*(tbounds[1]-tvec[0] ) ) )
#    i0 = _np.where(tvec>tbounds[0])[0]
#    if len(_np.atleast_1d(i0))==0:       i0 = 0   # end if
#    i1 = _np.where(tvec>tbounds[1])[0]
#    if len(_np.atleast_1d(i0))==0:       i0 = 0   # end if
    nsig = _np.size( tvec[i0:i1] )

    # Must know two of these inputs to determine third
    # k = # windows, M = Length of data to be segmented, L = length of segments,
    #      K = (M-NOVERLAP)/(L-NOVERLAP)
    #           Navr  = (nsig-noverlap)/(nwins-noverlap)
    #           nwins = (nsig - noverlap)/Navr + noverlap
    #           noverlap = (nsig - nwins*Navr)/(1-Navr)
    #   noverlap = windowoverlap*nwins
    #           nwins = nsig/(Navr-Navr*windowoverlap + windowoverlap)
    #

    # Integral number of periods in data set, need 2 to detect signal
#    sigx = _np.atleast_2d(sigx) # multi-channel input only supported for sigy
    sigy = _np.atleast_2d(sigy)
    if _np.shape(sigy)[1] == len(tvec):
        sigy = sigy.T
    # end if
    nch = _np.size(sigy, axis=1)

    # ====================================================================== #
    if _np.size(sigx, axis=0) != _np.size(sigy, axis=0):
        nTmodel = True
        if calcNavr:
            nwins = _np.size(sigx, axis=0)
        else:
            nwins = fftanal._getNwins(nsig, Navr, windowoverlap)
        # end if
    else:
        nTmodel = False
        # override Navr if someone requested a period for each window, or a minimum frequency to resolve
        if 'minFreq' in kwargs:
            kwargs['tper'] = 2.0/kwargs['minFreq']
        if 'tper' in kwargs :
            nwins = int(Fs*kwargs['tper'])
            calcNavr = True
        else:
            if Navr is None:
                Navr = 8
            # end if
            calcNavr = False
            nwins = fftanal._getNwins(nsig, Navr, windowoverlap)
        # end if
    # end if

    # get the number of points to overlap based on unique data
    noverlap = fftanal._getNoverlap(nwins, windowoverlap)

    # Reflect the data in the first and last windows at the end-points
    reflecting = False
    if i0 == 0 and i1 == len(tvec):
        reflecting = True
#        sigx=_np.r_['0', sigx[nwins-1:0:-1],sigx,sigx[-1:-nwins:-1]]
#        sigy=_np.r_['0',sigy[nwins-1:0:-1,:],sigy,sigy[-1:-nwins:-1,:]]  # concatenate along axis 0
        sigx = _np.concatenate((sigx[nwins-1:0:-1,...],sigx,sigx[-1:-nwins:-1,...]), axis=0)
        sigy = _np.concatenate((sigy[nwins-1:0:-1,...],sigy,sigy[-1:-nwins:-1,...]), axis=0)
        nsig = sigx.shape[0]
    # end if

    # if necessary get the number of averaging windows
    if calcNavr: # nTmodel or 'tper' in kwargs:
        Navr = fftanal._getNavr(nsig, nwins, noverlap)
#        Navr  = int( 1 + (nsig-noverlap)//(nwins-noverlap) )
    # end if

    if verbose:
        print(f'Calculating spectra from {tbounds[0]} to {tbounds[1]} s with {Navr} windows to yield a frequency resolution of {1e-3*Fs/nwins} KHz')
    
    # ====================================================================== #
    if nwins>=nsig:
        Navr = 1
        nwins = nsig
    # endif
    # nfft     = max(2^12,2^nextpow2(nwins))
    nfft = nwins

    Nnyquist = fftanal._getNnyquist(nfft)

    # Remember that since we are not dealing with infinite series, the lowest
    # frequency we actually resolve is determined by the period of the window
    # fhpf = 1.0/(nwins*dt)  # everything below this should be set to zero (when background subtraction is applied)

    # ==================================================================== #
    # =================================================================== #

    # Define windowing function for apodization
    win, winparams = windows(windowfunction, nwins=nwins, verbose=verbose, msgout=True)

    # Instantiate the information class that will be output
    fftinfo = fftinfosc()
    fftinfo.win = win
    fftinfo.winparams = winparams
    fftinfo.windowoverlap = windowoverlap
    fftinfo.ibnds = [i0, i1]    # time-segment

    # Define normalization constants
    fftinfo.S1 = fftanal._getS1(win)
    fftinfo.S2 = fftanal._getS2(win)

    # Normalized equivalent noise bandwidth
    fftinfo.NENBW = fftanal._getNENBW(Nnyquist, fftinfo.S1, fftinfo.S2)
    fftinfo.ENBW = fftanal._getENBW(Fs, fftinfo.S1, fftinfo.S2) # Effective noise bandwidth

    # ================================================================ #

    detrend = fftanal._detrend_func(detrend_style=detrend_style)

    # ================================================================ #

    if useMLAB:
        if onesided:  # True is boolean 1
            sides = 'onesided'
        else:
            sides = 'twosided'
        # end if
#        sides = 'twosided'

        # Use MLAB for the power spectral density calculations
        if verbose:
            print('using matlab built-ins for spectra/coherence calculations')
        # endif verbose

        tx = tvec
        if nTmodel:
#            # Does not work very well.  amplitude is all wrong, and coherence is very low
#            sigx = _np.hstack((sigx, _np.zeros((nsig-len(sigx)+1,), dtype=sigx.dtype)))
#            sigx = _np.tile(sigx, _np.size(sigy, axis=0)//len(sigx)+1)
#            sigx = sigx[:len(tvec)]
            while sigx.shape[0]<sigy.shape[0]:
                # Wrap the data periodically
#                sigx=_np.r_[sigx[nwins-1:0:-1],sigx,sigx[-1:-nwins:-1]]
                sigx=_np.r_[sigx, sigx[-1:-nwins:-1]]
            # end while
            if sigx.shape[0]>sigy.shape[0]:
                sigx = sigx[:sigy.shape[0]]
            # end if
        # end if
        x_in = sigx[i0:i1]
        y_in = sigy[i0:i1,:]

        # Power spectral density (auto-power spectral density), and
        # cross-power spectral density of signal 1 and signal 2,
        # Pyy = Yfft.*conj(Yfft), and the linear amplitude spectrum, Lxx:
        Pxx, freq = _mlab.csd(x_in, x_in, nfft, Fs=Fs, detrend=detrend,
                              window=win, noverlap=noverlap, sides=sides,
                              scale_by_freq=True)

        Pyy = _np.zeros((nch, len(freq)), dtype=_np.float64)
        Pxy = _np.zeros((nch, len(freq)), dtype=_np.complex128)

        for ii in range(nch):
            # [V^2/Hz], RMS power spectral density calculation
            Pyy[ii,:], freq = _mlab.csd(y_in[:,ii], y_in[:,ii], nfft, Fs=Fs, detrend=detrend,
                                  window=win, noverlap=noverlap, sides=sides,
                                  scale_by_freq=True)

            Pxy[ii,:], freq = _mlab.csd(x_in, y_in[:,ii], nfft, Fs=Fs, detrend=detrend,
                                  window=win, noverlap=noverlap, sides=sides,
                                  scale_by_freq=True)
        # end if
        # Get the coherence
#        if (Navr==1):
#            Cxy2 = _np.ones_like(Pxx)
#        else:
#            # returns mean squared coherence
#            [Cxy2, freq] = _mlab.cohere(y_in, x_in, nfft, Fs, detrend=detrend,
#                                      window=win, noverlap=noverlap, sides=sides,
#                                      scale_by_freq=False)
#        #endif
#        Cxy = _np.sqrt(_np.abs(Cxy2))

        if onesided:
            # They also return the nyquist value
#            freq = freq[:Nnyquist-1]
#            Pxx = Pxx[:Nnyquist-1]
#            Pyy = Pyy[:,:Nnyquist-1]
#            Pxy = Pxy[:,:Nnyquist-1]
            freq = freq[:Nnyquist]
            Pxx = Pxx[:Nnyquist]
            Pyy = Pyy[:,:Nnyquist]
            Pxy = Pxy[:,:Nnyquist]
        # end if
        Pyy = Pyy.T   # nfreq x nch
        Pxy = Pxy.T
        # ================================================================= #
    else:
        # Without Matlab: Welch's average periodogram method:
        if verbose:
            print('using home-brew functions for spectra/coherence calculations')
        # endif verbose

        # ============ #

        # Pre-allocate
        Pxx_seg = _np.zeros((Navr, nfft), dtype=_np.complex128)
        Pyy_seg = _np.zeros((nch, Navr, nfft), dtype=_np.complex128)
        Pxy_seg = _np.zeros((nch, Navr, nfft), dtype=_np.complex128)

        Xfft = _np.zeros((Navr, nfft), dtype=_np.complex128)
        Yfft = _np.zeros((nch, Navr, nfft), dtype=_np.complex128)

        if nTmodel:
            tx = tvec[:len(sigx)]
            # assume that one of the signals is the length of 1 window
            x_in = sigx   # reference signal is the model Doppler signal
            y_in = sigy[i0:i1,:]   # noisy long signal is the model CECE signal
        else:
            tx = tvec
            x_in = sigx[i0:i1]
            y_in = sigy[i0:i1,:]
        # end if
        
        predetrend = 0
        if predetrend:
            x_in = detrend(x_in, axis=0)
            y_in = detrend(y_in, axis=0)

        ist = _np.arange(Navr)*(nwins - noverlap)
        ist = ist.astype(int)
#        for gg in _np.arange(Navr):
        for gg in range(Navr):
            istart = ist[gg]     # Starting point of this window
            iend = istart+nwins  # End point of this window

            if nTmodel:
                xtemp = _np.copy(x_in)
            else:
                xtemp = x_in[istart:iend]
            # end if
            ytemp = y_in[istart:iend,:]

            # Windowed signal segment
            # To get the most accurate spectrum, minimally detrend
            if not predetrend:
                xtemp = win*xtemp
                ytemp = (_np.atleast_2d(win).T*_np.ones((1,nch), dtype=ytemp.dtype))*ytemp
            else:
                xtemp = win*detrend(xtemp, axis=0)
                ytemp = (_np.atleast_2d(win).T*_np.ones((1,nch), dtype=ytemp.dtype))*detrend(ytemp, axis=0)

            # The FFT output from matlab isn't normalized:
            # y_n = sum[ y_m.*exp( 2_np.pi*1i*(n/N)*m ) ]
            # The inverse is normalized::
            # y_m = (1/N)*sum[ y_n.*exp( -2_np.pi*1i*(n/N)*m ) ]
            #
            # Python normalizations are optional, pick it to match MATLAB
            Xfft[gg, :nfft] = fftmod.fft(xtemp, n=nfft, axis=0)  # defaults to last axis
            Yfft[:, gg, :nfft] = fftmod.fft(ytemp, n=nfft, axis=0).T    # nch x Navr x nfft
        #endfor loop over fft windows

        #Auto- and cross-power spectra
        Pxx_seg[:Navr, :nfft] = Xfft*_np.conj(Xfft)
        Pyy_seg[:,:Navr, :nfft] = Yfft*_np.conj(Yfft)
        Pxy_seg[:,:Navr, :nfft] = Yfft*(_np.ones((nch,1,1), dtype=Xfft.dtype)*_np.conj(Xfft))

        # Get the frequency vector
        freq = fftmod.fftfreq(nfft, 1.0/Fs)
#        freq = Fs*_np.arange(0.0, 1.0, 1.0/nfft)
#        if (nfft%2):
#            # freq = Fs*(0:1:1/(nfft+1))
#            freq = Fs*_np.arange(0.0,1.0,1.0/(nfft+1))
#        # end if nfft is odd
        if onesided:
#            freq = freq[:Nnyquist-1]  # [Hz]
#            Pxx_seg = Pxx_seg[:, :Nnyquist-1]
#            Pyy_seg = Pyy_seg[:, :, :Nnyquist-1]
#            Pxy_seg = Pxy_seg[:, :, :Nnyquist-1]
            freq = freq[:Nnyquist]  # [Hz]
            Pxx_seg = Pxx_seg[:, :Nnyquist]
            Pyy_seg = Pyy_seg[:, :, :Nnyquist]
            Pxy_seg = Pxy_seg[:, :, :Nnyquist]

            # All components but DC split their energy between positive +
            # negative frequencies: One sided spectra,
            Pxx_seg[:, 1:-1] = 2*Pxx_seg[:, 1:-1]  # [V^2/Hz],
            Pyy_seg[:, :, 1:-1] = 2*Pyy_seg[:, :, 1:-1]  # [V^2/Hz],
            Pxy_seg[:, :, 1:-1] = 2*Pxy_seg[:, :, 1:-1]  # [V^2/Hz],
            if nfft%2:  # Odd
                Pxx_seg[:, -1] = 2*Pxx_seg[:, -1]
                Pyy_seg[:, :, -1] = 2*Pyy_seg[:, :, -1]
                Pxy_seg[:, :, -1] = 2*Pxy_seg[:, :, -1]
            # endif nfft is odd
        else:
            freq = fftmod.fftshift(freq)

            Pxx_seg = fftmod.fftshift(Pxx_seg, axes=-1)
            Pyy_seg = fftmod.fftshift(Pyy_seg, axes=-1)
            Pxy_seg = fftmod.fftshift(Pxy_seg, axes=-1)
        # end if

        # Remove gain of the window function to yield the RMS Power spectrum
        # in each segment (constant peak amplitude) ... doing this after cutting the number of pionts in half if one-sided
        Pxx_seg = (1.0/(fftinfo.S1**2))*Pxx_seg  # [Vrms^2]
        Pyy_seg = (1.0/(fftinfo.S1**2))*Pyy_seg  # [Vrms^2]
        Pxy_seg = (1.0/(fftinfo.S1**2))*Pxy_seg  # [Vrms^2]

        # Compute the power spectral density from the RMS power spectrum
        # (constant noise floor)
        Pxx_seg = Pxx_seg/fftinfo.ENBW  # [V^2/Hz]
        Pyy_seg = Pyy_seg/fftinfo.ENBW  # [V^2/Hz]
        Pxy_seg = Pxy_seg/fftinfo.ENBW  # [V^2/Hz]

        # Average the different realizations: This is the output from cpsd.m
        # RMS Power spectrum
        Pxx = _np.mean(Pxx_seg, axis=0)  # [V^2/Hz]
        Pyy = _np.mean(Pyy_seg, axis=1).T  # nfft x nch
        Pxy = _np.mean(Pxy_seg, axis=1).T

        # Estimate the variance in the power spectra
#        fftinfo.varPxx = _np.var((Pxx_seg[:Navr, :Nnyquist]), axis=0)
#        fftinfo.varPyy = _np.var((Pyy_seg[:Navr, :Nnyquist]), axis=0)
#        fftinfo.varPxy = _np.var((Pxy_seg[:Navr, :Nnyquist]), axis=0)

#        # use the RMS for the standard deviation
#        fftinfo.varPxx = _np.mean(Pxx_seg**2.0, axis=0)
#        fftinfo.varPyy = _np.mean(Pyy_seg**2.0, axis=0)
#        fftinfo.varPxy = _np.mean(Pxy_seg**2.0, axis=0)

#        fftinfo.varPxy = _np.var(_np.real(Pxy_seg), axis=0) + 1j*_np.var(_np.imag(Pxy_seg), axis=0)

        # Save the cross-phase in each segmentas well
        phixy_seg = _np.angle(Pxy_seg)  # [rad], Cross-phase of each segment

        #[ phixy_seg[0:Navr,0:Nnyquist], varphi_seg[0:Navr,0:Nnyquist] ] = \
        #   varangle(Pxy_seg, fftinfo.varPxy)

        # Right way to average cross-phase:
        # mean and variance in cross-phase
        varphi_seg = _np.zeros_like(phixy_seg)
#        [phi_xy, fftinfo.varPhxy] = mean_angle(phixy_seg[0:Navr, :],
#                                               varphi_seg[0:Navr,:], dim=0)
#
#        # Now take the power  and linear spectra

        # Segmented data ... useful for making spectrograms
        fftinfo.Pxx_seg = Pxx_seg
        fftinfo.Pyy_seg = Pyy_seg
        fftinfo.Pxy_seg = Pxy_seg
        fftinfo.Xfft_seg = Xfft
        fftinfo.Yfft_seg = Yfft
        fftinfo.phixy_seg = phixy_seg
        fftinfo.varphi_seg = varphi_seg

        # ====================== #

    # endif
    # Calculate the mean-squared and complex coherence
    # take the absolute value of Cxy to get the RMS coherence
    # take the abs. value of Cxy2 and the sqrt to get the RMS coherence
    Cxy, Cxy2 = Cxy_Cxy2(Pxx, Pyy, Pxy)  # complex numbers returned

    # ========================== #
    # Uncertainty and phase part #
    # ========================== #
    # derived using error propagation for gamma^2 in
    # J.S. Bendat, Journal of Sound an Vibration 59(3), 405-421, 1978
    fftinfo.varCxy = ((1.0-Cxy*_np.conjugate(Cxy))/_np.sqrt(2*Navr))**2.0
#    fftinfo.varCxy = ((1.0-Cxy2)/_np.sqrt(2*Navr))**2.0
    fftinfo.varCxy2 = 4.0*Cxy2*fftinfo.varCxy # d/dx x^2 = 2 *x ... var:  (2*x)^2 * varx

    # Estimate the variance in the power spectra: this requires building
    # a distribution by varying the parameters used in the FFT, nwindows,
    # nfft, windowfunction, etc.  I don't do this right now
    fftinfo.varPxx = (Pxx/_np.sqrt(Navr))**2.0
    fftinfo.varPyy = (Pyy/_np.sqrt(Navr))**2.0
    fftinfo.varPxy = (Pxy/_np.sqrt(Navr))**2.0
#    fftinfo.varPxy = Pxx*Pyy*(1.0-Cxy)/Navr   # this gives nice results ... similar to above as Cxy is a measure of shared power

    # A.E. White, Phys. Plasmas, 17 056103, 2010
    # Doesn't so far give a convincing answer...
    # fftinfo.varPhxy = _np.zeros(Pxy.shape, dtype=_np.float64)
    #fftinfo.varPhxy = (_np.sqrt(1-Cxy2)/_np.sqrt(2*Navr*Cxy))**2.0
#    fftinfo.varPhxy = (_np.sqrt(1-_np.abs(Cxy*_np.conj(Cxy)))/_np.sqrt(2*Navr*_np.abs(Cxy)))**2.0
#    fftinfo.varPhxy = (_np.sqrt(1.0-Cxy2))/_np.sqrt(2*Navr*_np.sqrt(Cxy2))**2.0
    fftinfo.varPhxy = (_np.sqrt(1.0-_np.abs(Cxy2)))/_np.sqrt(2*Navr*_np.sqrt(_np.abs(Cxy2)))**2.0

    # ========================== #

    # Save the cross-phase as well
#    phi_xy = _np.angle(Pxy)
    phi_xy = _np.arctan2(Pxy.imag, Pxy.real)

    # ========================== #

    # Linear amplitude spectrum from the power spectral density
    # RMS Linear amplitude spectrum (constant amplitude values)
    fftinfo.Lxx = _np.sqrt(_np.abs(fftinfo.ENBW*Pxx))  # [V_rms]
    fftinfo.Lyy = _np.sqrt(_np.abs(fftinfo.ENBW*Pyy))  # [V_rms]
    fftinfo.Lxy = _np.sqrt(_np.abs(fftinfo.ENBW*Pxy))  # [V_rms]

    if onesided:
        # Rescale RMS values to Amplitude values (assumes a zero-mean sine-wave)
        # Just the points that split their energy into negative frequencies
        fftinfo.Lxx[1:-1] = _np.sqrt(2)*fftinfo.Lxx[1:-1]  # [V],
        fftinfo.Lyy[1:-1,:] = _np.sqrt(2)*fftinfo.Lyy[1:-1,:]  # [V],
        fftinfo.Lxy[1:-1,:] = _np.sqrt(2)*fftinfo.Lxy[1:-1,:]  # [V],
        if nfft%2:  # Odd
            fftinfo.Lxx[-1] = _np.sqrt(2)*fftinfo.Lxx[-1]
            fftinfo.Lyy[-1,:] = _np.sqrt(2)*fftinfo.Lyy[-1,:]
            fftinfo.Lxy[-1,:] = _np.sqrt(2)*fftinfo.Lxy[-1,:]
        # endif nfft/2 is odd

        # ======================================================================= #
        # Cross and auto-correlation from power spectra
        fftinfo.Rxx = Pxx.copy()
        fftinfo.Rxx[1:-1, ...] *= 0.5
        if nfft%2:
            fftinfo.Rxx[-1, ...] *= 0.5
        fftinfo.Rxx = fftmod.irfft(fftinfo.Rxx, n=nfft, axis=0)

        fftinfo.Ryy = Pyy.copy()
        fftinfo.Ryy[1:-1, ...] *= 0.5
        if nfft%2:
            fftinfo.Ryy[-1, ...] *= 0.5
        fftinfo.Ryy = fftmod.irfft(fftinfo.Ryy, n=nfft, axis=0)

        fftinfo.Rxy = Pxy.copy()
        fftinfo.Rxy[1:-1, ...] *= 0.5
        if nfft%2:
            fftinfo.Rxy[-1, ...] *= 0.5
        fftinfo.Rxy = fftmod.irfft(fftinfo.Rxy, n=nfft, axis=0)

        fftinfo.iCxy = Cxy.copy()
        fftinfo.iCxy = fftmod.irfft(fftinfo.iCxy, n=nfft, axis=0)
        # fftinfo.iCxy = fftmod.ifft(fftinfo.iCxy, n=nfft, axis=0)
        
        # ======================================================================= #
    else:
        # ======================================================================= #
        # Cross and auto-correlation from power spectra
#        fftinfo.Rxy_seg = fftmod.fftshift(_np.sqrt(nfft)*fftmod.ifft(
#                    fftmod.fftshift(Pxy_seg, axes=-1), n=nfft, axis=-1), axes=-1)

        fftinfo.Rxx = fftmod.ifft(fftmod.ifftshift(Pxx, axes=0), n=nfft, axis=0)
        fftinfo.Ryy = fftmod.ifft(fftmod.ifftshift(Pyy, axes=0), n=nfft, axis=0)
        fftinfo.Rxy = fftmod.ifft(fftmod.ifftshift(Pxy, axes=0), n=nfft, axis=0)
        fftinfo.iCxy = fftmod.ifft(fftmod.ifftshift(Cxy, axes=0), n=nfft, axis=0)
#        fftinfo.iCxy = fftmod.ifft(fftmod.ifftshift(_np.sqrt(_np.abs(Cxy2)), axes=0), n=nfft, axis=0).real

        # ======================================================================= #
    # end if
    fftinfo.Rxx *= _np.sqrt(nfft)
    fftinfo.Ryy *= _np.sqrt(nfft)
    fftinfo.Rxy *= _np.sqrt(nfft)
    fftinfo.iCxy *= _np.sqrt(nfft)

    # Calculate the normalized auto- and cross-correlations
    fftinfo.Ex = fftinfo.Rxx[0, ...].copy()    # power in the x-spectrum, int( |u(f)|^2, df)
    fftinfo.Ey = fftinfo.Ryy[0, ...].copy()    # power in the y-spectrum, int( |v(f)|^2, df)

#    fftinfo.Rxx /= fftinfo.Ex
#    fftinfo.Ryy /= fftinfo.Ey
    fftinfo.corrcoef = fftinfo.Rxy/_np.sqrt(_np.ones((nfft,1), dtype=fftinfo.Rxy.dtype)*(fftinfo.Ex*fftinfo.Ey))

    fftinfo.Rxx = fftmod.fftshift(fftinfo.Rxx, axes=0)
    fftinfo.Ryy = fftmod.fftshift(fftinfo.Ryy, axes=0)
    fftinfo.Rxy = fftmod.fftshift(fftinfo.Rxy, axes=0)
    fftinfo.iCxy = fftmod.fftshift(fftinfo.iCxy, axes=0)
    fftinfo.corrcoef = fftmod.fftshift(fftinfo.corrcoef, axes=0)
    #fftinfo.lags = (_np.asarray(range(1, nfft+1), dtype=int)-Nnyquist)/Fs
    fftinfo.lags = _np.arange(-nfft+1, nfft)/Fs
    
    # ======================================================================= #

    fftinfo.varLxx = (fftinfo.Lxx**2)*(fftinfo.varPxx/_np.abs(Pxx)**2)
    fftinfo.varLyy = (fftinfo.Lyy**2)*(fftinfo.varPyy/_np.abs(Pyy)**2)
    fftinfo.varLxy = (fftinfo.Lxy**2)*(fftinfo.varPxy/_np.abs(Pxy)**2)

    if nch == 1:
        Pyy = Pyy.flatten()
        Pxy = Pxy.flatten()
        Cxy = Cxy.flatten()
        Cxy2 = Cxy2.flatten()
        phi_xy = phi_xy.flatten()

        fftinfo.lags = fftinfo.lags.flatten()
        fftinfo.Rxx = fftinfo.Rxx.flatten()
        fftinfo.Ryy = fftinfo.Ryy.flatten()
        fftinfo.Rxy = fftinfo.Rxy.flatten()
        fftinfo.corrcoef = fftinfo.corrcoef.flatten()
        fftinfo.iCxy = fftinfo.iCxy.flatten()
        fftinfo.Lxx = fftinfo.Lxx.flatten()
        fftinfo.Lyy = fftinfo.Lyy.flatten()
        fftinfo.Lxy = fftinfo.Lxy.flatten()
        fftinfo.varLxx = fftinfo.varLxx.flatten()
        fftinfo.varLyy = fftinfo.varLyy.flatten()
        fftinfo.varLxy = fftinfo.varLxy.flatten()

        fftinfo.varCxy = fftinfo.varCxy.flatten()
        fftinfo.varCxy2 = fftinfo.varCxy2.flatten()
        fftinfo.varPxx = fftinfo.varPxx.flatten()
        fftinfo.varPyy = fftinfo.varPyy.flatten()
        fftinfo.varPxy = fftinfo.varPxy.flatten()
        fftinfo.varPhxy = fftinfo.varPhxy.flatten()
    # end if

    # Store everything
    fftinfo.nch = nch
    fftinfo.Fs = Fs
    fftinfo.Navr = Navr
    fftinfo.nwins = nwins
    fftinfo.noverlap = noverlap
    fftinfo.overlap = windowoverlap
    fftinfo.window = windowfunction
    fftinfo.minFreq = 2.0*Fs/nwins
    fftinfo.freq = freq.copy()
    fftinfo.Pxx = Pxx.copy()
    fftinfo.Pyy = Pyy.copy()
    fftinfo.Pxy = Pxy.copy()
    fftinfo.Cxy = Cxy.copy()
    fftinfo.Cxy2 = Cxy2.copy()
    fftinfo.phi_xy = phi_xy.copy()

    # ==================================================================== #

    # Plot the comparisons
    if plotit:
        if reflecting:
#            sigx = sigx[(nwins//2-1):-nwins//2]
#            sigy = sigy[(nwins//2-1):-nwins//2,:]
            sigx = sigx[(nwins-1):-nwins+1]
            sigy = sigy[(nwins-1):-nwins+1,:]
        # end if

        afont = {'fontname':'Arial','fontsize':14}

        # plot the signals
        if 'hfigSig' in kwargs:
            hfig1 = _plt.figure(kwargs['hfigSig'])
        else:
            hfig1 = _plt.figure()
        if 'axSig' in kwargs:
            _ax = kwargs['axSig']
        else:
            _ax = _plt.subplot(1,1,1)
        if _np.iscomplexobj(sigx) and _np.iscomplexobj(sigy):
            _ax.plot(tx, _np.real(sigx), 'b-')
            _ax.plot(tx, _np.imag(sigx), 'b--')
            _ax.plot(tvec, _np.real(sigy), 'r-')
            _ax.plot(tvec, _np.imag(sigy), 'r--')
        elif _np.iscomplexobj(sigx) and not _np.iscomplexobj(sigy):
            _ax.plot(tvec, sigy, 'r-')
            _ax.plot(tx, _np.real(sigx), 'b-')
            _ax.plot(tx, _np.imag(sigx), 'b--')
        elif _np.iscomplexobj(sigy) and not _np.iscomplexobj(sigx):
            _ax.plot(tx, sigx, 'b-')
            _ax.plot(tvec, _np.real(sigy), 'r-')
            _ax.plot(tvec, _np.imag(sigy), 'r--')
        else:
            _ax.plot(tx, sigx, 'b-', tvec, sigy, 'r-')
        # end if
        _ax.set_title('Input Signals', **afont)
        _ax.set_xlabel('t[s]', **afont)
        _ax.set_ylabel('sig_x,sig_y[V]', **afont)
        if tbounds is not None:
            _plt.axvline(x=tbounds[0], color='k')
            _plt.axvline(x=tbounds[1], color='k')
        # end if

        #The correlations and spectra
        if 'hfigSpec' in kwargs:
            hfig2 = _plt.figure(kwargs['hfigSpec'])
        else:
            hfig2 = _plt.figure()
        if 'axSpec' in kwargs:
            _ax1 = kwargs['axSpec'][0]
        else:
            _ax1 = _plt.subplot(2,2,1)
#        _plt.plot(1e3*fftinfo.lags, fftinfo.iCxy, 'r-')
        _ax1.plot(1e3*fftinfo.lags, fftinfo.corrcoef, 'b-')
        _plt.ylabel(r'$\rho$', **afont)
        _plt.xlabel('lags [ms]', **afont)
        _plt.title('Cross-corrrelation')

        if 'axSpec' in kwargs:
            _ax2 = kwargs['axSpec'][1]
        else:
            _ax2 = _plt.subplot(2,2,2)
#        frq = 1e-3*freq;  xlbl = 'f[KHz]'
        frq = freq;       xlbl = 'f[Hz]'
        if 0:
#            _ax2.plot(frq,_np.abs(fftinfo.Lxx), 'b-');    ylbl = r'L$_{ij}$ [I.U.]'
#            _ax2.plot(frq,_np.abs(fftinfo.Lyy), 'r-');    tlbl = 'Linear Amplitude Spectra'
#            _ax2.plot(frq,_np.abs(fftinfo.Lxy), 'k-');
            _ax2.plot(frq,_np.abs(Pxx), 'b-');    ylbl = r'P$_{ij}$ [I.U.$^2$/Hz]'
            _ax2.plot(frq,_np.abs(Pyy), 'r-');    tlbl = 'Power Spectra'
            _ax2.plot(frq,_np.abs(Pxy), 'k-');

            if onesided:
                _ax2.set_xlim(0,1.01*frq[-1])
            else:
                _ax2.set_xlim(-1.01*frq[-1],1.01*frq[-1])
            # end if
        elif onesided:
            _ax2.loglog(frq, _np.abs(Pxx), 'b-');
            _ax2.loglog(frq, _np.abs(Pyy), 'r-');    ylbl = r'P$_{ij}$ [dB/Hz]'
            _ax2.loglog(frq, _np.abs(Pxy), 'k-');    tlbl = 'Power Spectra'

            xlims = _ax2.get_xlim()
            _ax2.set_xlim(xlims[0], 1.01*frq[-1])
        else:
            _ax2.semilogy(frq, _np.abs(Pxx), 'b-');    ylbl = r'P$_{ij}$ [dB/Hz]'
            _ax2.semilogy(frq, _np.abs(Pyy), 'r-');    tlbl = 'Power Spectra'
            _ax2.semilogy(frq, _np.abs(Pxy), 'k-');

            _ax2.set_xlim(-1.01*frq[-1],1.01*frq[-1])
        # end if
        _ax2.set_title(tlbl, **afont)
        _ax2.set_ylabel(ylbl, **afont),
        _ax2.set_xlabel(xlbl, **afont)

        if 'axSpec' in kwargs:
            _ax3 = kwargs['axSpec'][2]
        else:
            _ax3 = _plt.subplot(2, 2, 3, sharex=_ax2)
#        _ax3.plot(frq, _np.sqrt(_np.abs(Cxy2)), 'k-')
#        _ax3.plot(frq, _np.abs(Cxy).real, 'k-')
#        _plt.axhline(y=1.0/_np.sqrt(Navr), color='k')
#        _ax3.set_title('Coherence', **afont)
#        _ax3.set_ylabel(r'C$_{xy}$', **afont)
#        _ax3.set_ylabel(r'$|\gamma|$', **afont)
        _ax3.plot(frq, _np.abs(Cxy2), 'k-')
        _plt.axhline(y=1.0/Navr, color='k')
        _ax3.set_title('Mean-Squared Coherence', **afont)
        _ax3.set_ylabel(r'$\gamma^2$', **afont)
        _ax3.set_xlabel(xlbl, **afont)

        if 'axSpec' in kwargs:
            _ax4 = kwargs['axSpec'][3]
        else:
            _ax4 = _plt.subplot(2, 2, 4, sharex=_ax2)
        _ax4.plot(frq, phi_xy, 'k-')
        _ax4.set_title('Cross-Phase', **afont)
        _ax4.set_ylabel(r'$\phi_{xy}$', **afont)
        _ax4.set_xlabel(xlbl, **afont)

        _plt.tight_layout()
        _plt.draw()

        fftinfo.hfig1 = hfig1
        fftinfo.hfig2 = hfig2
        fftinfo.axSig = _ax
        fftinfo.ax = [__ax for __ax in [_ax1, _ax2, _ax3, _ax4]]

        
    return freq, Pxy, Pxx, Pyy, Cxy, phi_xy, fftinfo


class fftinfosc(Struct):
    """
    A generalized spectral data holder object
    """
    def __init__(self):
        self.S1     = _np.array( [], dtype=_np.float64)
        self.S2     = _np.array( [], dtype=_np.float64)

        self.NENBW  = _np.array( [], dtype=_np.float64)
        self.ENBW   = _np.array( [], dtype=_np.float64)

        self.freq   = _np.array( [], dtype=_np.float64)
        self.Pxx    = _np.array( [], dtype = _np.complex128 )
        self.Pyy    = _np.array( [], dtype = _np.complex128 )
        self.Pxy    = _np.array( [], dtype = _np.complex128 )

        self.Cxy    = _np.array( [], dtype=_np.complex128)
        self.varcoh = _np.array( [], dtype=_np.complex128)
        self.phi_xy = _np.array( [], dtype=_np.float64)
        self.varphi = _np.array( [], dtype=_np.float64)

        self.Lxx = _np.array( [], dtype = _np.complex128 )
        self.Lyy = _np.array( [], dtype = _np.complex128 )
        self.Lxy = _np.array( [], dtype = _np.complex128 )

        self.varLxx = _np.array( [], dtype = _np.complex128 )
        self.varLyy = _np.array( [], dtype = _np.complex128 )
        self.varLxy = _np.array( [], dtype = _np.complex128 )

        #Segment data
        self.Pxx_seg  = _np.array( [], dtype = _np.complex128 )
        self.Pyy_seg  = _np.array( [], dtype = _np.complex128 )
        self.Pxy_seg  = _np.array( [], dtype = _np.complex128 )
        self.Xfft_seg = _np.array( [], dtype = _np.complex128 )
        self.Yfft_seg = _np.array( [], dtype = _np.complex128 )


# =========================================================================== #
# =========================================================================== #


def integratespectra(freq, Pxy, Pxx, Pyy, frange, varPxy=None, varPxx=None, varPyy=None):
    """
    function [Pxy_i, Pxx_i, Pyy_i, Cxy_i, ph_i, info] =
        integrate_spectrum(freq, Pxy, Pxx, Pyy, frange, varPxy, varPxx, varPyy)

     This is a simple function that integrates a power spectrum over a specified
     frequency range.  It propagates the errors from spectra variances.

     Required inputs:
       freq - [Hz], frequency vector
       Pxy - [Complex], cross-power spectrum between signals 1 and 2
       Pxx - [Real (or Complex)], auto-power of signal 1
       Pyy - [Real (or Complex)], auto-power of signal 2
       frange - [lower frequency, upper frequency] - [Hz], frequency range to
       integrate over

     Optional inputs:
       varPxy - [Complex], variance in cross-power spectrum between signals 1 and 2
       varPxx - [Real (or Complex)], variance in auto-power of signal 1
       varPyy - [Real (or Complex)], variance in auto-power of signal 2

     Outputs:
       Pxy_i - integrated cross-power
       Pxx_i - integrated auto-power of signal 1
       Pyy_i - integrated auto-power of signal 2
       Cxy_i - coherence between signal 1 and signal 2 in frequency range
               determined by frange
       ph_i - cross-phase between signal 1 and signal 2 in frequency range
               determined by frange
       info - Structure containing propagated variances

     requires:
       trapz_var  - integrates using a trapezoidal rule and propagates uncertainty
       varcoh     - calculates coherence and propagates uncertainty
       varphi     - calculates angle and propagates uncertainty
    """

    if varPyy is None:  varPyy = _np.size_like(Pyy)  # end if
    if varPxx is None:  varPxx = _np.size_like(Pxx)  # end if
    if varPxy is None:  varPxy = _np.size_like(Pxy)  # end if

    # Integrate over the frequency range specified
    # ifl = find( freq>frange(1), 1, 'first')-1
    # ifh = find( freq>frange(2), 1, 'first')-1
    # Pxy_i = trapz(freq(ifl:ifh), Pxy(ifl:ifh))
    # Pxx_i = trapz(freq(ifl:ifh), Pxx(ifl:ifh))
    # Pyy_i = trapz(freq(ifl:ifh), Pyy(ifl:ifh))

    Pxy = _ut.reshapech(Pxy)
    varPxy = _ut.reshapech(varPxy)
    Pxx = _ut.reshapech(Pxx)
    varPxx = _ut.reshapech(varPxx)
    Pyy = _ut.reshapech(Pyy)
    varPyy = _ut.reshapech(varPyy)

    inds = _np.where( (freq>=frange[0])*(freq<=frange[1]) )[0]
    [Pxy_real, varPxy_real, _, _] = \
        _ut.trapz_var(freq[inds], _np.real(Pxy[inds,:]), None, _np.real(varPxy[inds,:]), dim=0)
    [Pxy_imag, varPxy_imag, _, _] = \
        _ut.trapz_var(freq[inds], _np.imag(Pxy[inds,:]), None, _np.imag(varPxy[inds,:]), dim=0)

    Pxy_i = Pxy_real + 1j*Pxy_imag
    varPxy_i = varPxy_real + 1j*varPxy_imag

    [Pxx_i, varPxx_i, _, _] = \
        _ut.trapz_var(freq[inds], Pxx[inds,:], None, varPxx[inds,:], dim=0)
    [Pyy_i, varPyy_i, _, _] = \
        _ut.trapz_var(freq[inds], Pyy[inds,:], None, varPyy[inds,:], dim=0)

    # Calculate coherence from integrated peak
    # Cxy_i  = Pxy_i.*(Pxy_i').'./(Pxx_i.*Pyy_i); %Mean-squared Coherence between the two signals
    # Cxy_i = sqrt( abs( Cxy_i ) ); % Coherence
    meansquared = 0   #Return the coherence, not the mean-squared coherence
    [Cxy_i, varCxy_i] = varcoh(Pxy_i, varPxy_i, Pxx_i, varPxx_i, Pyy_i, varPyy_i, meansquared)

    # Calculate cross-phase from integrated peak
    # ph_i = atan( Pxy_imag./Pxy_real )    # [rad], Cross-phase
    angle_range = _np.pi
    [ph_i, varph_i] = varphi(Pxy_real, Pxy_imag, varPxy_real, varPxy_imag, angle_range)

    # Store it all for outputting
    info = Struct()
    info.frange = _np.asarray([frange[0], frange[1]])
    info.ifrange = inds
    info.Pxy_i = Pxy_i
    info.varPxy_i = varPxy_i
    info.Pxx_i = Pxx_i
    info.varPxx_i = varPxx_i
    info.Pyy_i = Pyy_i
    info.varPyy_i = varPyy_i
    info.angle_range = angle_range
    info.ph_i = ph_i
    info.varph_i = varph_i
    info.meansquared = meansquared
    info.Cxy_i = Cxy_i
    info.varCxy_i = varCxy_i

    # Cross-power weighted average frequency - (center of gravity)
    info.fweighted = _np.dot(freq[inds].reshape(len(inds),1), _np.ones((1,_np.size(Pxy,axis=1)), dtype=float))
    info.fweighted = _np.trapz( info.fweighted*_np.abs(Pxy[inds,:]))
    info.fweighted /= _np.trapz(_np.abs(Pxy[inds,:]))
    return Pxy_i, Pxx_i, Pyy_i, Cxy_i, ph_i, info


def bandpower(data, Fs, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    Fs : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = _np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * Fs
    else:
        nperseg = (2 / low) * Fs

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, Fs, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = _np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


def bandpower_multitaper(data, Fs, band, method='welch', window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    Fs : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.

      s. https://raphaelvallat.com/bandpower.html and references there in
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    from mne.time_frequency import psd_array_multitaper

    band = _np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * Fs
        else:
            nperseg = (2 / low) * Fs

        freqs, psd = welch(data, Fs, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, Fs, adaptive=True,
                                          normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = _np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


def plot_spectrum_methods(data, Fs, window_sec, band=None, dB=False):
    """Plot the periodogram, Welch's and multitaper PSD.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    Fs : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds for Welch's PSD
    dB : boolean
        If True, convert the power to dB.
    """
    from mne.time_frequency import psd_array_multitaper
    from scipy.signal import welch, periodogram
    # sns.set(style="white", font_scale=1.2) # seaborn plotting
    # Compute the PSD
    freqs, psd = periodogram(data, Fs)
    freqs_welch, psd_welch = welch(data, Fs, nperseg=window_sec*Fs)
    psd_mt, freqs_mt = psd_array_multitaper(data, Fs, adaptive=True,
                                            normalization='full', verbose=0)
    sharey = False

    # Optional: convert power to decibels (dB = 10 * log10(power))
    if dB:
        psd = 10 * _np.log10(psd)
        psd_welch = 10 * _np.log10(psd_welch)
        psd_mt = 10 * _np.log10(psd_mt)
        sharey = True

    # Start plot
    fig, (_ax1, _ax2, _ax3) = _plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=sharey)
    # Stem
    sc = 'slategrey'
    _ax1.stem(freqs, psd, linefmt=sc, basefmt=" ", markerfmt=" ")
    _ax2.stem(freqs_welch, psd_welch, linefmt=sc, basefmt=" ", markerfmt=" ")
    _ax3.stem(freqs_mt, psd_mt, linefmt=sc, basefmt=" ", markerfmt=" ")
    # Line
    lc, lw = 'k', 2
    _ax1.plot(freqs, psd, lw=lw, color=lc)
    _ax2.plot(freqs_welch, psd_welch, lw=lw, color=lc)
    _ax3.plot(freqs_mt, psd_mt, lw=lw, color=lc)
    # Labels and axes
    _ax1.set_xlabel('Frequency (Hz)')
    if not dB:
        _ax1.set_ylabel('Power spectral density (V^2/Hz)')
    else:
        _ax1.set_ylabel('Decibels (dB / Hz)')
    _ax1.set_title('Periodogram')
    _ax2.set_title('Welch')
    _ax3.set_title('Multitaper')
    if band is not None:
        _ax1.set_xlim(band)
    _ax1.set_ylim(ymin=0)
    _ax2.set_ylim(ymin=0)
    _ax3.set_ylim(ymin=0)
    # sns.despine()


def getNpeaks(Npeaks, tvec, sigx, sigy, **kwargs):
    """
    Return the specified number of peaks (ina n FFT sense) from the input signals.
    
        out = [(Linear amplitude, frequency, cross-phase), ... ]
        len(out) = Npeaks
    
    """
    kwargs.setdefault('tbounds',None)
    kwargs.setdefault('Navr', None)
    kwargs.setdefault('windowoverlap', None)
    kwargs.setdefault('windowfunction', None)
    kwargs.setdefault('useMLAB', None)
    kwargs.setdefault('plotit', None)
    kwargs.setdefault('verbose', None)
    kwargs.setdefault('detrend_style', None)
    kwargs.setdefault('onesided', True)
    fmin = kwargs.pop('fmin', None)
    fmax = kwargs.pop('fmax', None)
    minsep = kwargs.pop('minsep', 6)
    freq, Pxy, Pxx, Pyy, Cxy, phi_xy, fftinfo = fft_pwelch(tvec, sigx, sigy, **kwargs)
    Lxx = fftinfo.Lxx
    Lyy = fftinfo.Lyy
    Lxy = fftinfo.Lxy


#    fmin = 0.0 if fmin is None else fmin
#    fmax = freq[-1] if fmax is None else fmax
#    iff = _np.ones((len(freq),), dtype=bool)
#    iff[(freq<=fmin)*(freq>=fmax)] = False
#    freq = freq[iff]
#    Lyy = Lyy[iff]
#    phi_xy = phi_xy[iff]
#
#    threshold = kwargs.pop('threshold', -1)
##    mph = kwargs.pop('mph', None)
#    mph = kwargs.pop('mph', 0.5*(_np.nanmax(sigy)-_np.nanmin(sigy))/(20*Npeaks))
#    ind = _ut.detect_peaks(Lyy, mpd=int(10*minsep/(freq[10]-freq[0])), mph=mph, threshold=threshold, show=True)
#
#    out = []
#    for ii in range(Npeaks):
#        out.append([ Lyy[ind[ii]], freq[ind[ii]], phi_xy[ind[ii]] ])
#    # end for
    # build a boolean index array and replace peaks with false (+- an equivalent noise bandwidth)
    nfreq = len(freq)
    ENBW = fftinfo.ENBW # equiv. noise bandwidth
    ENBW = max((ENBW, minsep))
    iff = _np.ones((nfreq,), dtype=bool)
    irem = int(2*nfreq*ENBW/(freq[-1]-freq[0]))

    # Remove frequencies that are outside of the selected range
    fmin = 0.0 if fmin is None else fmin
    fmax = freq[-1] if fmax is None else fmax
    iff[(freq<=fmin)*(freq>=fmax)] = False
    freq = freq[iff]
    nfreq = len(freq)
    Lxx = Lxx[iff]
    Lyy = Lyy[iff]
    Lxy = Lxy[iff]
    phi_xy = phi_xy[iff]
    iff = iff[iff]

    out = []
    for ii in range(Npeaks):
        # Find the maximum peak in the cross-power spectrum
        imax = _np.argmax(Lxy)

        # Use the amplitude from the linear amplitude signal spectrum
        Ai = _np.copy(Lyy[imax])

        # freqency and phase from the big calculation
        fi = _np.copy(freq[imax])
        pi = _np.copy(phi_xy[imax])

        # Store the amplitude, frequency, and phase for output
        out.append([Ai, fi, pi])

        # Remove frequencies from the calculation that are around the current
        # peak +- an equivalent noise bandwdith
        if (imax-irem//2>=0) and (imax+irem//2<nfreq):
            iff[imax-irem//2:imax+irem//2] = False
        elif (imax+irem//2<nfreq):
            iff[:imax+irem//2] = False
        elif (imax-irem//2>=0):
            iff[-(imax+irem//2):] = False
            
        freq = freq[iff]
        nfreq = len(freq)
        Lxx = Lxx[iff]
        Lyy = Lyy[iff]
        Lxy = Lxy[iff]
        phi_xy = phi_xy[iff]
        iff = iff[iff]
    # end for
    return tuple(out)


# =========================================================================== #
# =========================================================================== #
"""
    Functions to extend the usage of the matplotlib "mlab" functions
"""


def fft_pmlab(sig1,sig2,dt,plotit=False):
    #nfft=2**_mlab.nextpow2(_np.length(sig1))
    nfft = _np.size(sig1)
    (ps1, ff) = _mlab.psd(sig1, NFFT=nfft, Fs=1./dt, detrend=_mlab.detrend_mean, \
                               sides='onesided', noverlap=0, scale_by_freq=True )
    (ps2, ff) = _mlab.psd(sig2, NFFT=nfft, Fs=1./dt, detrend=_mlab.detrend_mean, \
                               sides='onesided', noverlap=0, scale_by_freq=True )
    #(p12, ff) = mlab.csd(sig1, sig2, NFFT=sig1.len, Fs=1./dt,sides='default', scale_by_freq=False)
    (p12, ff) = _mlab.csd(sig1, sig2, NFFT=nfft, Fs=1./dt, detrend=_mlab.detrend_mean, \
                               sides='onesided', noverlap=0, scale_by_freq=True )

    if plotit:
        _plt.figure(num='Power Spectral Density')
        _plt.plot(ff*1e-9, _np.abs(ps1), 'b-')
        _plt.plot(ff*1e-9, _np.abs(ps2), 'g-')
        _plt.plot(ff*1e-9, _np.abs(p12), 'r-')
        _plt.xlabel('freq [GHz]')
        _plt.ylabel('PSD')
        _plt.show()
    #end plotit
    return ff, ps1, ps2, p12


def coh(x,y,fs,nfft=2048,fmin=0.0, fmax=500e3, detrend='mean', ov=0.67):
    """
    Calculate mean-squared coherence of data and return it below a maximum frequency
    """
    #  print('using nfft=%i\n')%(nfft)
#    Cxy, F = _mlab.cohere(x,y,NFFT=nfft,Fs=fs,detrend=detrend,pad_to=None,noverlap=int(_np.floor(nfft*ov)),window=_mlab.window_hanning)
#    ind=_np.where((_np.abs(F)<=fmax) & (_np.abs(F)>=fmin))

    window=_mlab.window_hanning
    noverlap=int(ov*nfft)
    pad_to=None
    sides='default'
    scale_by_freq=None

    Pxx, F = _mlab.psd(x, nfft, fs, detrend=detrend, window=window, noverlap=noverlap,
                 pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)
    Pyy, F = _mlab.psd(y, nfft, fs, detrend=detrend, window=window, noverlap=noverlap,
                 pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)
    Pxy, F = _mlab.csd(x, y, nfft, fs, detrend=detrend, window=window, noverlap=noverlap,
                 pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)

    Cxy2 = _np.divide(_np.absolute(Pxy)**2, Pxx*Pyy)
    Cxy2.shape = (len(F),)

    ind=_np.where((F<=fmax) & (F>=fmin))
    co=Cxy2[ind]
    fo=F[ind]
#    return co,fo
    return _np.sqrt(co),fo


def coh2(x,y,fs,nfft=4096, fmin=0, fmax=500e3, detrend='none', peak_treshold=None):
    """
    Calculate mean-squared coherence, cross-phase, and auto-power spectra
    (w.r.t x) of data and return it below a maximum frequency
    """
    fxx, f = _mlab.csd(x,x,NFFT=nfft,Fs=fs,noverlap=nfft/2,window=_mlab.window_hanning,scale_by_freq=True)
    fyy, f = _mlab.csd(y,y,NFFT=nfft,Fs=fs,noverlap=nfft/2,window=_mlab.window_hanning,scale_by_freq=True)
    fxy, f = _mlab.csd(x,y,NFFT=nfft,Fs=fs,noverlap=nfft/2,window=_mlab.window_hanning,scale_by_freq=True)

    Kxy  = _np.real(fxy)
    Qxy  = _np.imag(fxy)

    COH = _np.array([_np.abs(fxy[i]*_np.conj(fxy[i]))/(fxx[i]*fyy[i]) for i in range(len(f))])
    PHA = _np.array([_np.arctan2(Qxy[i],Kxy[i]) for i in range(len(f))])
    PSD = _np.array(_np.abs(fxx))
    ind=_np.where(_np.abs(f)<=fmax)
    co=COH[ind]
    fo=f[ind]
    do=PHA[ind]
    po=PSD[ind]
    return {'coh': co, 'f': fo, 'PS': po, 'pha':do}


def psd(x, fs, nfft=2048, fmin=None, fmax=None, detrend='none', peak_threshold=None, ov=0.67):
    """
    Calculate power spectral density of data and return spectra within frequency range
    """
    P,F=_mlab.psd(x,NFFT=nfft,Fs=fs,detrend=detrend,pad_to=None,noverlap=int(_np.floor(ov*nfft)),window=_mlab.window_hanning)
#    ind=_np.where((_np.abs(F)<=fmax) & (_np.abs(F)>=fmin))

    threshold = _np.ones(P.shape, dtype=bool)
    if fmin is not None:
        threshold = threshold & (F>=fmin)
    if fmax is not None:
        threshold = threshold & (F<=fmax)
#    threshold = (F<=fmax) & (F>=fmin)
    if peak_threshold is not None:
        threshold = threshold & (P>peak_threshold)
    ind=_np.where(threshold)
    pso=P[ind]
    fo=F[ind]
    return pso,fo


def csd(x,y,fs,nfft=2048,fmin=0,fmax=500e3, detrend='none', peak_threshold=None, ov=0.67):
    """
    Calculate cross-power spectral density of data and return spectra within frequency range

        x, y, NFFT=None, Fs=None, detrend=None, window=None,
        noverlap=None, pad_to=None, sides=None, scale_by_freq=None
    """
    P,F=_mlab.csd(x, y, NFFT=nfft, Fs=fs, detrend=detrend, pad_to=None, noverlap=int(_np.floor(ov*nfft)), window=_mlab.window_hanning)
#    ind=_np.where((_np.abs(F)<=fmax) & (_np.abs(F)>=fmin))

    threshold = _np.ones(P.shape, dtype=bool)
    if fmin is not None:
        threshold = threshold & (F>=fmin)
    if fmax is not None:
        threshold = threshold & (F<=fmax)
#    threshold = (F<=fmax) & (F>=fmin)
    if peak_threshold is not None:
        threshold = threshold & (P>peak_threshold)
    ind=_np.where(threshold)
    pso=P[ind]
    fo=F[ind]
    return pso,fo



# =========================================================================== #
# =========================================================================== #
"""
    Test functions for propagating estimated uncertainties
"""


def monticoh(Pxy, varPxy, Pxx, varPxx, Pyy, varPyy, nmonti=1000, meansquared=True):
    """
    Monti-carlo error propagation from the cross-power spectral densities to teh coherence.
    """
    nmonti = int(nmonti)
    
    sh = _np.shape(Pxy)
    Pxy = _np.atleast_2d(Pxy)
    if _np.size(Pxy, axis=0)==1:  Pxy = Pxy.T  # endif
    
    Pxx = _np.atleast_2d(Pxx)
    if _np.size(Pxx, axis=0)==1:  Pxx = Pxx.T  # endif
    
    Pyy = _np.atleast_2d(Pyy)
    if _np.size(Pyy, axis=0)==1:  Pyy = Pyy.T  # endif
    
    varPxy = _np.atleast_2d(varPxy)
    if _np.size(varPxy, axis=0)==1:  varPxy = varPxy.T  # endif
    
    varPxx = _np.atleast_2d(varPxx)
    if _np.size(varPxx, axis=0)==1:  varPxx = varPxx.T  # endif
    
    varPyy = _np.atleast_2d(varPyy)
    if _np.size(varPyy, axis=0)==1:  varPyy = varPyy.T  # endif
    
    Pxy_s, varPxy_s = Pxy.copy(), varPxy.copy()
    Pxx_s, varPxx_s = Pxx.copy(), varPxx.copy()
    Pyy_s, varPyy_s = Pyy.copy(), varPyy.copy()
    
    g2 = _np.zeros( (nmonti, _np.size(Pxy,axis=0), _np.size(Pxy, axis=1)), dtype=float)
    for ii in range(nmonti):
        Pxy = Pxy_s + _np.sqrt(varPxy_s)*_np.random.normal(0.0, 1.0, len(Pxy))
        Pxx = Pxx_s + _np.sqrt(varPxx_s)*_np.random.normal(0.0, 1.0, len(Pxx))
        Pyy = Pyy_s + _np.sqrt(varPyy_s)*_np.random.normal(0.0, 1.0, len(Pyy))
        
        g2[ii] = _np.abs( Pxy*_np.conj( Pxy ) )/( _np.abs(Pxx)*_np.abs(Pyy) )
        
    varg2 = _np.nanvar(g2, axis=0)
    g2 = _np.nanmean(g2, axis=0)
    
    if meansquared:
        return g2.reshape(sh), varg2.reshape(sh)
    else:
        return _np.sqrt(g2.reshape(sh)), _np.sqrt(varg2.reshape(sh))


def varcoh(Pxy, varPxy, Pxx, varPxx, Pyy, varPyy, meansquared=True):
    """
    function [Coh,varCoh]=varcoh(Pxy,varPxy,Pxx,varPxx,Pyy,varPyy)
    
    Calculate the coherence and propagate uncertainty from the spectra. 
    
    Only works when varPxy was formed by separating real and imaginary
    components.  As is done in fft_pwelch.m
    """
    ms = _np.imag(Pxy)
    mc = _np.real(Pxy)
    vs = _np.imag(varPxy)
    vc = _np.real(varPxy)

    Coh    = _np.array( _np.size(Pxy),dtype=_np.complex128)
    varCoh = _np.zeros_like( Coh )
    if meansquared:
        Coh = _np.abs( Pxy*_np.conj( Pxy ) )/( _np.abs(Pxx)*_np.abs(Pyy) )

        # C = ( T*T' )/(XY) = (R^2 + I^2)/(XY)
        # d()/dR = 2R/(XY)
        # d()/dI = 2I/(XY)
        # d()/dX = -(R^2+I^2)/(X^2*Y)
        # d()/dY = -(R^2+I^2)/(X*Y^2)
        # vC = C^2 * ( vR*(2R/(R^2+I^2))^2 + vI*(2I/(R^2+I^2))^2 + vX/X^2+ vY/Y2)
        varCoh = Coh**2*( vc*( 2*mc/( mc**2+ms**2) )**2 + \
                          vs*( 2*ms/( mc**2+ms**2) )**2 + \
                          varPxx*(1/Pxx)**2 + varPyy*(1/Pyy)**2 )
        
    else:  # return the complex coherence
        Coh = Pxy / _np.sqrt( _np.abs(Pxx)*_np.abs(Pyy) )
        # vardenom = ...
        # varCoh = Coh**2.0*( varPxy +
        varCoh = Coh**2*( vc*( 2*mc/( mc**2+ms**2) )**2 + \
                  vs*( 2*ms/( mc**2+ms**2) )**2 + \
                  varPxx*(1/Pxx)**2 + varPyy*(1/Pyy)**2 )
    
        # Return the coherence, not the mean-squared coherence
        varCoh = 0.25*varCoh/Coh  # (0.5*(Coh**-0.5))**2.0 * varCoh
        # Coh = _np.sqrt(Coh)


    return Coh, varCoh


def montiphi(Pxy, varPxy, nmonti=1000, angle_range=_np.pi):
    """
    Monti-carlo implementation of the cross-phase calculation from given uncertainties in spectra.
    """
    nmonti = int(nmonti)
    
    sh = _np.shape(Pxy)
    Pxy = _np.atleast_2d(Pxy)
    if _np.size(Pxy, axis=0)==1:  Pxy = Pxy.T  # endif
    
    varPxy = _np.atleast_2d(varPxy)
    if _np.size(varPxy, axis=0)==1:  varPxy = varPxy.T  # endif
    
    Pxy_s, varPxy_s = Pxy.copy(), varPxy.copy()
    
    ph = _np.zeros( (nmonti, _np.size(Pxy, axis=0), _np.size(Pxy, axis=1)), dtype=float)

    for ii in range(nmonti):
        Pxy = Pxy_s + _np.sqrt(varPxy_s)*_np.random.normal(0.0, 1.0, len(Pxy))
        
        # the angle function computes atan2( 1/n sum(sin(phi)),1/n sum(cos(phi)) )
        if angle_range>0.5*_np.pi:
            ph[ii] = _np.arctan2( _np.imag(Pxy), _np.real(Pxy) )
        else:
            ph[ii] = _np.arctan( _np.imag(Pxy) / _np.real(Pxy) )
            
        # This function  might not work becasue of wrapping issues?
        # ph[ii] = _np.unwrap(ph[ii])
        
    varph = _np.nanvar(ph, axis=0)
    ph = _np.nanmean(ph, axis=0)
    return ph.reshape(sh), varph.reshape(sh)


def varphi(Pxy_real, Pxy_imag, varPxy_real, varPxy_imag, angle_range=_np.pi):
    """
    Propagate the uncertainty in the spectra and calculate the variance in 
    the cross-phase spectra.
    """
    
    # the angle function computes atan2( 1/n sum(sin(phi)),1/n sum(cos(phi)) )
    if angle_range>0.5*_np.pi:
        ph = _np.arctan2( Pxy_imag, Pxy_real )
    else:
        ph = _np.arctan( Pxy_imag / Pxy_real )
        
    # substitute variables and propagate errors into the tangent function
    _tangent = Pxy_imag/Pxy_real   # tangent = sin / cos
    
    # vt = (tt**2)*( vsa/(msa**2) + vca/(mca**2) )  # variance in tangent
    # vt = vsa/(mca**2) + vca*msa**2/(mca**4)    # variance in tangent
    # vt = (tt**2)*( varPxy_imag/(Pxy_imag**2) + varPxy_real/(Pxy_real**2) )
    _vartang = (varPxy_imag+varPxy_real*_tangent**2)/(Pxy_real**2)
    
    # the variance of the arctangent function is related to the derivative
    #  d(arctangent)/dx = 1/(1+x^2)     using a = atan( tan(a) )
    # varph = vt/(1+tt**2)**2
    varph = _vartang/(1+_tangent**2)**2
    
    return ph, varph


# ========================================================================= #
# ========================================================================= #
"""
    Functions for taking the derivative of a signal using fft's (fft_deriv)
"""

def rescale(xx, yy, scaley=True, scalex=True):
    """ Scale the input data and save the scaling terms. """
    slope = 1.0
    offset = 0.0
    xslope = 1.0
    xoffset = 0.0

    if scaley:
        slope = _np.nanmax(yy)-_np.nanmin(yy)   # maximum 1.0
        offset = _np.nanmin(yy)                # minimum 0.0

        if slope == 0:    slope = 1.0   # end if
        yy = (yy.copy()-offset)/slope
        
    if scalex:
        xslope = _np.nanmax(xx)-_np.nanmin(xx)  # shrink so maximum is less than 1.0
        xoffset = -1e-4  # prevent 0 from being in problem

        if xslope == 0:    xslope = 1.0   # end if
        xx = (xx.copy()-xoffset)/xslope

    return xx, yy, (slope, offset, xslope, xoffset)


def unscale(xx, yy, scl, dydx=None):
    """ Unscale the data and derivative terms. """
    slope = scl[0]
    offset = scl[1]
    xslope = scl[2]
    xoffset = scl[3]
    xx = xx*xslope+xoffset
    yy = slope*yy+offset
    if dydx is not None:
        dydx = dydx*slope/xslope
        return xx, yy, dydx
    else:
        return xx, yy

    
def fft_deriv(sig, xx=None, lowpass=True, Fs_new=None, modified=True, detrend=detrend_none, window=None):
    """
    inputs:
        sig - (nx,) - signal, dependent variable
        xx  - (nx,) - grid on domain, independent variable (default: 0:nx)
        lowpass - [Hz or Bool], filter frequency pre-fft (default: nyquist freq. if True)
        modified - [Bool], See documentation below (default:True)
        detrend - [function handle], detrending function pre-LPF and pre-fft  (default: detrend_none)
        window - [function handle], window function for signal (default: None)
    outputs:
        dsdx - (nd,) - derivative of the signal
        xx   - (nd,) - independent variable of signal
                Note: nd = len(xx), downsampled signal to have twice the LPF frequency

    Documentation:
                    dfdx = ifft( wavenumber*fft(f) )

    Ringing artifacts are present in the derivative calculated by FFT's due to
    the lack of periodicity and high-frequency content in real signals.
    There are several methods to decrease ringing:
        1) use a modified wavenumber in the derivative calculation
            this decreases ringing everywhere, but ringing is still present at
            edges due to lack of periodicity
                wavenumber = 1.0j*k               if unmodified
                wavenumber = 1.0j*sin(k*dx)/dx    if modified.   s. Sunaina et al 2018 Eur. J. Phys. 39 065806
        2) use a window function
            this decreases ringing everywhere, but decreases the accuracy of
            the derivative near the edges of the domain
           (the signal is multiplied by zero at the end-points)
    """
    try:
        from .utils import downsample_efficient
    except:
        from FFT.utils import downsample_efficient

    if xx is None:
        N = len(sig)
        xx = 1.0*_np.asarray(range(0, N))

    # Low pass filter the data before calculating the FFT if requested
#    if 0:
    if lowpass:
        dxo = xx[1] - xx[0]
        if lowpass is True:
#            lowpass = 0.1*1.0/dxo
            lowpass = 0.5*1.0/dxo

#        b, a = butter_lowpass(lowpass, fnyq=0.5/dxo, order=2)
#        sig = _dsp.filtfilt(b, a, sig)
        Fs = 1.0/dxo
        if Fs_new is None:
            Fs_new = min(5.0*lowpass, Fs)

        if Fs_new<Fs:
            sig = downsample_efficient(sig, Fs=Fs, Fs_new=Fs_new, plotit=False, halforder=2, lowpass=lowpass)
            xx = xx[0] + _np.arange(0, len(xx)/Fs, 1.0/Fs_new)
            Fs = Fs_new

    # Scale the data to make it simple
    xx, sig, scl = rescale(xx, sig, scaley=True, scalex=True)
#    offset = _np.nanmean(sig, axis=0)
#    sig -= offset
    sig = detrend(sig)

    N = len(xx)
    nfft = N
    dx = xx[1] - xx[0]
    L = N*dx

    # Get the wavenumber vector
    k = fftmod.fftfreq(nfft, d=dx/L)
    k *= 2.0*_np.pi
    if modified:
        # Modified wave number
        # - Windowed with a sinc function to kill some of the ringing near the center
        # - Sunaina et al 2018 Eur. J. Phys. 39 065806
        wavenumber = 1.0j*_np.sin(k*dx)/(dx)
    else:
        # Naive fft derivative (subject to ringing)
        wavenumber = 1.0j*k
    # end if
    wavenumber /= L

    # Calculate the derivative using fft
    if window is None:
        win = _np.ones_like(sig)
    else:
        win = window(nfft)  # periodic hamming window is a good choice for the center of the domain
    # end if
    sig = win*sig # accurate at center of signal, no ringing, bad outside center

    # Calculate the derivative using fft
    ds0 = (sig[1]-sig[0])/(xx[1]-xx[0])       # beginning of boundary
    ds1 = (sig[-1]-sig[-2])/(xx[-1]-xx[-2])  # end point of boundary
    sig = _np.real(fftmod.ifft(wavenumber*fftmod.fft(sig, n=nfft), n=nfft))

    # Unnormalize the center of the window
    sig /= win  # accurate at center of signal, no ringing, bad outside center

    # discontinuity at end-points when not windowing
    sig[0] = ds0
    sig[-1] = ds1

    # Rescale back to the original data scale
#    sig += offset
    xx, _, sig = unscale(xx, sig.copy(), scl=scl, dydx=sig)

    # ======= =#

    # Low pass filter the data after calculating the FFT if requested
    if 0:
#    if lowpass:
        dx = xx[1] - xx[0]
        if lowpass is True:
            lowpass = 0.5*1.0/dx

        Fs = 1.0/dx
        if Fs_new is None:
            Fs_new = min(5.0*lowpass, Fs)

        if Fs_new<Fs:
            sig = downsample_efficient(sig, Fs=Fs, Fs_new=Fs_new, plotit=False, halforder=2, lowpass=lowpass)
            xx = xx[0] + _np.arange(0, len(xx)/Fs, 1.0/Fs_new)
            Fs = Fs_new

    return sig, xx



def test_fft_deriv(modified=True):
    """ test the derivative calculation by FFT. """
    for jj in range(1):
        if jj == 0:
            win = 'Unwindowed:'
            window = None
        elif jj == 1:
            win = 'Windowed:'
            window = _np.hamming
        for ii in range(5):
            N = int(2e3)
            L = 13.0 #interval of data
            dx = L/N
            xx = dx*_np.asarray(range(N))

            if ii == 0:
                # Test with a rectangle function
                yy = _ut.rect(2.0*xx/L-0.75)
                dy_analytic = _ut.delta(2.0*xx/L-0.75+0.5) - _ut.delta(2.0*xx/L-0.75-0.5)
                titl = '%s Box function'%(win,)
            elif ii == 1:
                # Test with a gaussian function
                yy = _np.exp(-0.5*(xx/L)*(xx/L)/(0.25*0.25))
                dy_analytic = (-1.0*(xx/L)*(1.0/L)/(0.25*0.25))*yy
                titl = '%s Gaussian function'%(win,)
            elif ii == 2:
                # Test with a line
                yy = _np.linspace(-1.2, 11.3, num=len(xx), endpoint=True)
                a = (yy[-1]-yy[0])/(xx[-1]-xx[0])
                # b = yy[0] - a*xx[0]
                dy_analytic = a*_np.ones_like(yy)
                titl = '%s Linear function'%(win,)
            elif ii == 3:
                # Test with a sine
                yy = _np.sin(xx)
                dy_analytic = _np.cos(xx)
                titl = '%s Sine function: aperiodic boundary'%(win,)
            elif ii == 4:
                # Test with a sine
                xx = 6.0*_np.pi*xx/L
                yy = _np.sin(xx)
                dy_analytic = _np.cos(xx)
                xx = xx[:-1]
                yy = yy[:-1]
                dy_analytic = dy_analytic[:-1]
                titl = '%s Sine function: periodic boundary'%(win,)
            # end if

#            # add some random noise
##            yy += 0.05*(_np.nanmax(yy)-_np.nanmin(yy))*_np.random.random(size=xx.shape)
#            yy += 0.05*yy*_np.random.random(size=xx.shape)

            dydt, xo = fft_deriv(yy, xx, modified=modified, window=window)

            _plt.figure('%s wavenumber: Test (%i,%i)'%('Modified' if modified else 'Unmodified',jj,ii+1))
            _plt.plot(xx, yy, '-', label='function')
            _plt.plot(xx, dy_analytic, '-', label='analytical der')
            _plt.plot(xo, dydt, '*', label='fft der')
            _plt.title(titl)
            _plt.legend(loc='lower left')
        # end for
    # end for

#    _plt.savefig('images/fft-der.png')
    _plt.show()


# ========================================================================= #
# ========================================================================= #


def Cxy_Cxy2(Pxx, Pyy, Pxy, ibg=None, cospectra=True): #, thresh=1.e-6):
    """ 
    Coherence and mean-squared coherence
    """
    Pxx = Pxx.copy()
    Pyy = Pyy.copy()
    Pxy = Pxy.copy()

    # Mean-squared coherence
    Pxx = _np.atleast_2d(Pxx)
    if _np.size(Pxx, axis=1) != _np.size(Pyy, axis=1):
        Pxx = Pxx.T*_np.ones( (1, _np.size(Pyy, axis=1)), dtype=Pxx.dtype)
    # end if
#    Cxy2 = Pxy*_np.conj( Pxy )/( _np.abs(Pxx)*_np.abs(Pyy) )
#    Cxy2 = _np.abs(Cxy2) # mean-squared coherence
#    Cxy = _np.sqrt(Cxy2) # RMS coherence   

    # Complex coherence
    Cxy = Pxy/_np.sqrt( _np.abs(Pxx)*_np.abs(Pyy) )
    
    # Mean-squared coherence
    Cxy2 = Cxy*_np.conjugate(Cxy)
    
    if ibg is None:
        return Cxy, Cxy2

    # Imaginary coherence   -- previously used for estimate of the correlation coefficient. 
    # iCxy = _np.imag(Cxy)/(1.0-_np.real(Cxy))   

    # Return the Maximum liklihood estimate of the coherence -- used for estimate of the correlation coefficient.
    if cospectra: 
        #
        iCxy = _np.real(Cxy)/(1.0-_np.real(Cxy))   
    else:
        # Quotient of a complex number:   multiply by complex conjugate of denominator
        #       Cxy/(1.0-Cxy) = Cxy/(1.0-Cxy)  * (1.0-Cxy*) / (1.0-Cxy*)     
        #       Cxy(1-Cxy*) / ( 1 - Cxy* - Cxy + Cxy.Cxy* )
        #       (Cxy-|Cxy|^2) / ( 1 - 2 Re{Cxy} + |Cxy|^2 )        
        iCxy = (Cxy-Cxy2)/(1.0 - 2.0*_np.real(Cxy) + Cxy2)   
        
    # Background subtracted cospectrum of the coherence
    Cprime = _np.real(Cxy-_np.mean(Cxy[:,ibg], axis=-1)) \
        /(1.0-_np.real(Cxy))
    
    # Wrong:
    # Cprime = _np.real(Cxy-_np.mean(Cxy[:,ibg], axis=-1)) \
    #    /(1.0-_np.real(Cxy-_np.mean(Cxy[:,ibg], axis=-1)))    
    return iCxy, Cprime


# ========================================================================= #
# ========================================================================= #


class fftanal(Struct):
    """
    Class based implementation of welch's averaged periodogram method.
    """
    
    afont = {'fontname':'Arial','fontsize':14}
    def __init__(self, tvec=None, sigx=None, sigy=None, **kwargs):
        """ Basic function for instantiation. """
        self.verbose = kwargs.get( 'verbose', True)
        if tvec is None or sigx is None:
            if self.verbose:
                print('Please give at least a time-vector [s]'
                      + ' and a signal vector [a.u.]')
            # end if
            return
        else:
            self.init(tvec, sigx, sigy, **kwargs)

    def init(self, tvec=None, sigx=None, sigy=None, **kwargs):
        """ Initialization function. """
        if sigy is None or sigx is sigy:
            self.nosigy = True
        else:
            self.nosigy = False

        self.tvec = tvec
        self.sigx = sigx
        self.sigy = sigy

        self.tbounds = kwargs.get( 'tbounds', [ tvec.min(), tvec.max() ] )
        self.useMLAB = kwargs.get( 'useMLAB', False )
        self.plotit  = kwargs.get( 'plotit',  False)
        self.verbose = kwargs.get( 'verbose', True)
        self.Navr    = kwargs.get( 'Navr', None)
        self.window  = kwargs.get( 'windowfunction', 'Hanning') #'SFT3F')
        if self.window is None:  self.window = 'Hanning'  # end if
        self.overlap = kwargs.get( 'windowoverlap', windows(self.window, verbose=False))
        self.tvecy   = kwargs.get( 'tvecy', None)
        self.onesided = kwargs.get('onesided', None)
        self.detrendstyle = kwargs.get('detrend', 1) # >0 mean, 0 None, <0 linear
        self.frange = kwargs.get('frange', None)
        self.axes = kwargs.get('axes', -1)

        if self.onesided is None:
            self.onesided = True
            if _np.iscomplexobj(sigx) or _np.iscomplexobj(sigy):
                self.onesided = False

        # ======== #

        # put the signals on the same time base for fourier tranfsormation
        if self.tvecy is not None:
            self.tvec, self.sigx, self.sigy = self.resample(tvec, sigx, self.tvecy, sigy)

        self.Fs = self.__Fs__(self.tvec)
        self.ibounds = self.__ibounds__(self.tvec, self.tbounds)
        self.nsig = _np.size( self.__trimsig__(tvec, self.ibounds) )

        calcNavr = False
        if self.Navr is None:
            calcNavr = True
            self.Navr = 8

        # if the window time is specified ... overwrite nwins, noverlap and Navr
        if 'minFreq' in kwargs:  # if the minimum resolvable frequency is specificed
            kwargs['tper'] = 2.0/kwargs['minFreq']
        if 'tper' in kwargs:
            self.tper = kwargs['tper']
            self.nwins = int(self.Fs*self.tper)
        else:
            calcNavr = False
            self.nwins = self.getNwins()
        self.noverlap = self.getNoverlap()

        if calcNavr:
            self.Navr = self.getNavr()

        self.win, self.winparams = self.makewindowfn(self.window, self.nwins, self.verbose)
        self.getNnyquist()
        self.getNorms()

        
    def update(self, d=None):
        if d is not None:
            if type(d) != dict:    d = d.dict_from_class()     # endif
            super(fftanal, self).__init__(d)


    def fftpwelch(self):
        """
        Call the fft_pwelch function that is defined above
        """
        self.freq, self.Pxy, self.Pxx, self.Pyy, \
        self.Cxy, self.phi_xy, self.fftinfo = \
            fft_pwelch(self.tvec, self.sigx, self.sigy, self.tbounds,
                       Navr=self.Navr, windowoverlap=self.overlap,
                       windowfunction=self.window, useMLAB=self.useMLAB,
                       plotit=self.plotit, verbose=self.verbose,
                       detrend_style=self.detrendstyle, onesided=self.onesided)
        self.update(self.fftinfo)

        
    def stft(self):
        """ Short-time fourier transform implementation. """
        if self.useMLAB:
            import scipy.signal as _dsp

            if not self.onesided or (type(self.onesided)==type('') and self.onesided.find('two')>-1):
                onesided = False
            elif self.onesided or (type(self.onesided)==type('') and self.onesided.find('one')>-1):
                onesided = True

            self.freq, self.tseg, self.Xseg = _dsp.stft(self.sigx, fs=self.Fs,
                   window=self.win, nperseg=self.nwins, noverlap=self.noverlap,
                   nfft=self.nwins, detrend=self.detrend, return_onesided=onesided,
                   boundary='zeros', padded=True, axis=self.axes)

            _, _, self.Yseg = _dsp.stft(self.sigy, fs=self.Fs,
                   window=self.win, nperseg=self.nwins, noverlap=self.noverlap,
                   nfft=self.nwins, detrend=self.detrend, return_onesided=onesided,
                   boundary='zeros', padded=True, axis=self.axes)

            self.Pstft()
            self.averagewins()
        else:
            self.pwelch()


    def pwelch(self):
        """ implement welch's averaged periodogram method from class functions. """
        self.Xstft()
        if not self.nosigy:
            self.Ystft()
        self.Pstft()
        self.averagewins()


    def crosscorr(self): #, fbnds=None):
        """
        cross correlation from the FFT
            if we calculated one-sided spectra, then we have to add back
            in the Nyquist components before inverse tranforming.
        """
        nfft = self.nwins
        for param in ['Pxx', 'Pyy', 'Pxy']:
            if hasattr(self, param):
                tmp = getattr(self, param).copy()
                if self.onesided:
                    # remaking the two-sided spectra for the auto-power
                    tmp[..., 1:-1] *= 0.5
                    if nfft%2:  # odd
                        tmp[-1] *= 0.5
                    tmp = _np.sqrt(nfft)*fftmod.irfft(tmp, n=nfft, axis=-1)
                else:
                    tmp = fftmod.ifftshift(tmp, axes=-1)
                    tmp = _np.sqrt(nfft)*fftmod.ifft(tmp, n=nfft, axis=-1)

                # cauchy energy integral == auto-correlation at zero-time lag
                if param.find('Pxx')>-1:
                    setattr(self, 'Ex', tmp[..., 0].copy())
                if param.find('Pyy')>-1:
                    setattr(self, 'Ey', tmp[..., 0].copy())

                # shift the time-series of lagged correlations to be the normal smallest to largest
                tmp = fftmod.fftshift(tmp, axes=-1)
                setattr(self, 'R'+param[1:], tmp)

        if hasattr(self, 'Rxy'):
            self.corrcoef = self.Rxy.copy()/(_np.ones((self.nch,1), dtype=self.Ey.dtype)*_np.sqrt(self.Ex*self.Ey))
        self.lags = (_np.asarray(range(1, nfft+1), dtype=int)-self.Nnyquist)/self.Fs

    
    def crosscorr_stft(self): #, fbnds=None):
        """
        cross correlation from the FFT
            if we calculated one-sided spectra, then we have to add back
            in the Nyquist components before inverse tranforming.
        """
        nfft = self.nwins

        for param in ['Pxx_seg', 'Pyy_seg', 'Pxy_seg']:
            if hasattr(self, param):
                tmp_seg = getattr(self, param).copy()

                if self.onesided:
                    # remaking the two-sided spectra for the auto-power
                    tmp_seg[...,1:-1] *= 0.5
                    if nfft%2:  # odd
                        tmp_seg[..., -1] *= 0.5
                    # end if
                    tmp_seg = _np.sqrt(nfft)*fftmod.irfft(tmp_seg, n=nfft, axis=-1)
                else:
                    tmp_seg = fftmod.ifftshift(tmp_seg, axes=-1)
                    tmp_seg = _np.sqrt(nfft)*fftmod.ifft(tmp_seg, n=nfft, axis=-1)

                # cauchy energy integral == auto-correlation at zero-time lag
                if param.find('Pxx')>-1:
                    setattr(self, 'Ex_seg', tmp_seg[..., 0].copy())
                if param.find('Pyy')>-1:
                    setattr(self, 'Ey_seg', tmp_seg[..., 0].copy())

                # shift the time-series of lagged correlations to be the normal smallest to largest
                tmp_seg = fftmod.fftshift(tmp_seg, axes=-1)
                setattr(self, 'R'+param[1:], tmp_seg)

        if hasattr(self, 'Rxy_seg'):
            self.corrcoef_seg = self.Rxy_seg.copy()/(_np.ones((self.nch,1), dtype=self.Ey_seg.dtype)*_np.sqrt(self.Ex_seg*self.Ey_seg))

        self.lags = (_np.asarray(range(1, nfft+1), dtype=int)-self.Nnyquist)/self.Fs


    def Xstft(self):
        """
        Perform the loop over averaging windows to generate the short time four. xform
            Note that the zero-frequency component is in the middle of the array (2-sided transform)
        """
        sig = self.__trimsig__(self.sigx, self.ibounds)
        tvec = self.__trimsig__(self.tvec, self.ibounds)

        self.tseg, self.freq, self.Xseg, self.Xpow = self.fft_win(sig, tvec)   # frequency [cycles/s], STFT [Navr, nfft]
        self.Xfft = _np.mean(self.Xseg, axis=0)
        return self.freq, self.Xseg

    def Ystft(self):
        """
        Perform the loop over averaging windows to generate the short time four. xform
            Note that the zero-frequency component is in the middle of the array (2-sided transform)
        """
        sig = self.__trimsig__(self.sigy, self.ibounds)
        tvec = self.__trimsig__(self.tvec, self.ibounds)
        self.tseg, self.freq, self.Yseg, self.Ypow = self.fft_win(sig, tvec)   # frequency [cycles/s], STFT [Navr, nfft]
        self.Yfft = _np.mean(self.Yseg, axis=0)
        #self.tseg = self.tbounds[0]+(self.arange(self.Navr)+0.5)*self.tper
        return self.freq, self.Yseg

    
    def Pstft(self):
        """ Calculate the spectral quantities within each segment (stft). """
        if hasattr(self,'Xseg'):
            self.Pxx_seg = self.Xseg*_np.conj(self.Xseg)

            self.Lxx_seg = _np.sqrt(_np.abs(self.ENBW*self.Pxx_seg))  # [V_rms]
            if self.onesided:
                self.Lxx_seg = _np.sqrt(2)*self.Lxx_seg # V_amp

        if hasattr(self,'Yseg'):
            self.Pyy_seg = self.Yseg*_np.conj(self.Yseg)

            self.Lyy_seg = _np.sqrt(_np.abs(self.ENBW*self.Pyy_seg))  # [V_rms]
            if self.onesided:
                self.Lyy_seg = _np.sqrt(2)*self.Lyy_seg # V_amp

        if hasattr(self, 'Xseg') and hasattr(self,'Yseg'):
            self.Pxy_seg = self.Xseg*_np.conj(self.Yseg)

            self.Lxy_seg = _np.sqrt(_np.abs(self.ENBW*self.Pxy_seg))  # [V_rms]
            if self.onesided:
                self.Lxy_seg = _np.sqrt(2)*self.Lxy_seg # V_amp

            # Save the cross-phase as well
            self.phixy_seg = _np.angle(self.Pxy_seg)  # [rad], Cross-phase of each segment

            # Mean-squared Coherence
            self.Cxy_seg, self.Cxy2_seg = Cxy_Cxy2(self.Pxx_seg, self.Pyy_seg, self.Pxy_seg)


    def averagewins(self):
        """ Average the spectral windows tgoether to create spectra and estimate uncertainties. """
        for param in ['Pxx', 'Pyy', 'Pxy']:
            if hasattr(self, param+'_seg'):
                # Use the mean of the windows
                # self.Pxx = _np.mean(self.Pxx_seg, axis=0)
                setattr(self, param, _np.mean(getattr(self, param+'_seg'), axis=0))

                # use the RMS for the standard deviation
                # self.varPxx = _np.mean(self.Pxx_seg**2.0, axis=0)
                # setattr(self, 'var'+param, _np.mean(getattr(self, param+'_seg')**2.0, axis=0) )

                # Else use the normal statistical estimate:
                # self.varPxx = (self.Pxx/_np.sqrt(self.Navr))**2.0
                setattr(self, 'var'+param, (getattr(self, param)/_np.sqrt(self.Navr))**2.0)

        if hasattr(self, 'Pxy'):
            # Cross-phase as well
            self.phi_xy = _np.angle(self.Pxy)

            # Complex coherence and Mean-squared coherence
            self.Cxy, self.Cxy2 = Cxy_Cxy2(self.Pxx, self.Pyy, self.Pxy)

            # ========================== #
            # Uncertainty and phase part

            # Estimate the variance in the power spectra: this requires building
            # a distribution by varying the parameters used in the FFT, nwindows,
            # nfft, windowfunction, etc.  I don't do this right now
            # self.varPxy = self.Pxx*self.Pyy*(1.0-self.Cxy)/self.Navr

            # A.E. White, Phys. Plasmas, 17 056103, 2010
            # Doesn't so far give a convincing answer...
            # fftinfo.varPhxy = _np.zeros(Pxy.shape, dtype=_np.float64)
            self.varPhxy = (_np.sqrt(1.0-self.Cxy2)/_np.sqrt(2.0*self.Navr*self.Cxy))**2.0

            # derived using error propagation from eq 23 for gamma^2 in
            # J.S. Bendat, Journal of Sound an Vibration 59(3), 405-421, 1978
            # fftinfo.varCxy = _np.zeros_like(Cxy)
            self.varCxy = ((1-self.Cxy2)/_np.sqrt(2*self.Navr))**2.0
            self.varCxy2 = 4.0*self.Cxy2*self.varCxy # d/dx x^2 = 2 *x ... var:  (2*x)^2 * varx


    def convert2amplitudes(self):
        """
        Calculate the linear amplitude spectra from the power spectral densities.
        
        RMS Linear amplitude spectrum (constant amplitude values).        
        """
        for param in ['Pxx', 'Pyy', 'Pxy']:
            if hasattr(self, param):
                # self.Lxx = _np.sqrt(_np.abs(self.ENBW*self.Pxx))  # [V_rms]
                tmp = _np.sqrt(_np.abs(self.ENBW*getattr(self, param)))  # [V_rms])

                if self.onesided:
                    # Rescale RMS values to Amplitude values (assumes a zero-mean sine-wave)
                    # Just the points that split their energy into negative frequencies
                    # self.Lxx[1:-1] = _np.sqrt(2)*self.Lxx[1:-1]  # [V],
                    tmp[1:-1] = _np.sqrt(2)*tmp[1:-1]   # [V],

                    if self.nfft%2:  # Odd
                        # self.Lxx[-1] = _np.sqrt(2)*self.Lxx[-1]
                        tmp[-1] = _np.sqrt(2)*tmp[-1]   # [V],  TODO:!  test!
                    # endif nfft/2 is odd
                # end if onesided
                setattr(self, 'L'+param[1:], tmp)  # [V])

                # self.varLxx = (self.Lxx**2)*(self.varPxx/_np.abs(self.Pxx)**2)
                setattr(self, 'varL'+param[1:], (tmp**2)*(getattr(self, 'var'+param)/_np.abs(getattr(self, param))**2) )


    # ====================================================================== #
    # ====================================================================== #


    def getNavr(self):
        """ Return the number of averaging windows for welch's averaged periodogram method. """
        self.Navr = fftanal._getNavr(self.nsig, self.nwins, self.noverlap)
        return self.Navr

    
    def getNwins(self):
        """ Number of points in each window / segment. """        
        self.nwins = fftanal._getNwins(self.nsig, self.Navr, self.overlap)
        return self.nwins


    def getNoverlap(self):
        """ Number of points to overlap windows. """
        self.noverlap = fftanal._getNoverlap(self.nwins, self.overlap)
        return self.noverlap

    
    def getNnyquist(self):
        """ Get the index of the nyquist frequency. """
        self.Nnyquist = self._getNnyquist(self.nwins)
        return self.Nnyquist

    
    def getNorms(self):
        """ Define normalization constants. """
        self.S1, self.S2, self.NENBW, self.ENBW = fftanal._getNorms(self.win, self.Nnyquist, self.Fs)


    # ===================================================================== #

    def integrate_spectra(self):  # TODO:  CHECK ACCURACY OF THIS!
        """
        Integrate the given spectra over the specified frequency range.
        """
        self.integrated = Struct()
        [ self.integrated.Pxy, self.integrated.Pxx, self.integrated.Pyy,
          self.integrated.Cxy, self.integrated.ph, self.integrated.info  ] = \
            integratespectra(self.freq, self.Pxy, self.Pxx, self.Pyy, self.frange,
                             self.varPxy, self.varPxx, self.varPyy)


    # ===================================================================== #


    def detrend(self, sig):
        """ Detrend function (wrapper). """
        detrender = fftanal._detrend_func(detrend_style=self.detrendstyle)
        return detrender(sig)

    
    def fft(self, sig, nfft=None, axes=None):
        """ 
        Implement the FFT.
        
        The FFT output from matlab isn't normalized:
        y_n = sum[ y_m.*exp( 2_np.pi*1i*(n/N)*m ) ]
        The inverse is normalized::
        y_m = (1/N)*sum[ y_n.*exp( -2_np.pi*1i*(n/N)*m ) ]
        """
        
        # Python normalizations are optional, pick it to match MATLAB
        if axes is None: axes = self.axes # end if
        if nfft is None: nfft = self.nfft # end if
        return fftmod.fft(sig, n=nfft, axis=axes)

    
    def ifft(self, sig, nfft=None, axes=None):
        """ 
        Implement the inverse FFT.

        The FFT output from matlab isn't normalized:
        y_n = sum[ y_m.*exp( 2_np.pi*1i*(n/N)*m ) ]
        The inverse is normalized::
        y_m = (1/N)*sum[ y_n.*exp( -2_np.pi*1i*(n/N)*m ) ]
        """
        # Python normalizations are optional, pick it to match MATLAB
        if axes is None: axes = self.axes # end if
        if nfft is None: nfft = self.nfft # end if
        return fftmod.ifft(sig, n=nfft, axis=axes)

    
    def fftshift(self, sig, axes=None):
        """ 
        Implement the FFt-shift.
        """                
        if axes is None: axes = self.axes # end if
        return fftmod.fftshift(sig, axes=axes)

    
    def ifftshift(self, sig, axes=None):
        """ 
        Implement the inverse FFt-shift.
        """        
        if axes is None: axes = self.axes # end if
        return fftmod.ifftshift(sig, axes=axes)

    
    def fft_win(self, sig, tvec=None, detrendwin=False):
        """ 
        Calculate the FFT ant auto-power spectral density of 
        the input signal (one-channel) using welch's 
        averaged periodogram method. 
        """
        x_in = sig.copy()
        if tvec is None:
            tvec = _np.linspace(0.0, 1.0, len(x_in))
        # endif
        win = self.win
        nwins = self.nwins
        Navr = self.Navr
        noverlap = self.noverlap
#        Fs = self.Fs
        Fs = self.__Fs__(tvec)
        Nnyquist = self.Nnyquist
        nfft  = nwins

        # Define normalization constants for the window
        S1 = self.S1
        S2 = self.S2
        ENBW = self.ENBW        # Equivalent noise bandwidth
        
        # detrend the whole background, like most programs do it
        if not detrendwin:
            x_in = self.detrend(x_in)

        # ===== #

        ist = _np.arange(Navr)*(nwins-noverlap)
        Xfft = _np.zeros((Navr, nfft), dtype=_np.complex128)
        tt = _np.zeros( (Navr,), dtype=float)
        pseg = _np.zeros( (Navr,), dtype=float)
        for gg in _np.arange(Navr):
            istart = ist[gg]  #Starting point of this window
            iend   = istart+nwins                 #End point of this window

            if gg == 0:
                self.tper = tvec[iend]-tvec[istart]
            # endif
            tt[gg] = _np.mean(tvec[istart:iend])
            xtemp = x_in[istart:iend]

            # Windowed signal segment:
            # - To get the most accurate spectrum, background subtract
            # xtemp = win*_dsp.detrend(xtemp, type='constant')
            if detrendwin:
                # this is only good when the background is not evolving!
                xtemp = self.detrend(xtemp)
            # end if
            xtemp = win*xtemp
            pseg[gg] = _np.trapz(xtemp*_np.conj(xtemp), x=tvec[istart:iend]).real
            Xfft[gg,:] = self.fft(xtemp, nfft)

        freq = fftmod.fftfreq(nfft, 1.0/Fs)
        if self.onesided:
            freq = freq[:Nnyquist]  # [Hz]
            Xfft = Xfft[:,:Nnyquist]

            # Real signals equally split their energy between positive and negative frequencies
            Xfft[:, 1:-1] = _np.sqrt(2)*Xfft[:, 1:-1]
            if nfft%2:  # odd
                Xfft[:,-1] = _np.sqrt(2)*Xfft[:,-1]
        else:
            freq = self.fftshift(freq, axes=0)
            Xfft = self.fftshift(Xfft, axes=-1)

        # Remove gain of the window function to yield the RMS Power spectrum
        # in each segment (constant peak amplitude)
        Xfft /= S1   # Vrms
        pseg /= S2

        # Compute the spectral density from the RMS spectrum
        # (constant noise floor)
        Xfft /= _np.sqrt(ENBW)  # [V/Hz^0.5]
        return tt, freq, Xfft, pseg

    # ===================================================================== #
    # ===================================================================== #


    def plotall(self):
        """ Make a summary plot. """        
        # The input signals versus time
        self.fig = _plt.figure(figsize=(15,15))

        self.ax1 = _plt.subplot(2,3,1)    # time-series
        self.ax2 = _plt.subplot(2,3,2)    # correlation coeff
        self.ax3 = _plt.subplot(2,3,3)    # power spectra
        self.ax4 = _plt.subplot(2,3,4, sharex=self.ax2)    # spectrogram
        self.ax5 = _plt.subplot(2,3,5, sharex=self.ax3) # phase coherence
        self.ax6 = _plt.subplot(2,3,6, sharex=self.ax3) # cross-phase


        _ax1 = self.plottime(_ax=self.ax1)
        _ax2 = self.plotCorr(_ax=self.ax2)
        _ax3 = self.plotPxy(_ax=self.ax3)
        _ax4 = self.plotspec(param='Pxy', logscale=True, _ax=self.ax4)
#        _ax4, _ax2 = self.plotCorrelations(_ax=[self.ax4, self.ax2])
        _ax5 = self.plotCxy(_ax=self.ax5)
        _ax6 = self.plotphxy(_ax=self.ax6)

        _plt.tight_layout()
        _plt.draw()
        return _ax1, _ax2, _ax3, _ax4, _ax5, _ax6


    def plotspec(self, param='Pxy', logscale=False, _ax=None, vbnds=None, cmap=None):  # spectrogram
        """ Plot the spectra calculated using welch's averaged periodogram method. """
        # Minimum resolvable frequency, Maximum resolvable frequency (or bounded if the freq vector has been trimmed)
        fbounds = [1e-3*max((2.0*self.Fs/self.nwins, self.freq.min())), 1e-3*min((self.Fs/2.0, self.freq.max()))]

        _ax = fftanal._plotspec(self.tseg, self.freq, getattr(self, param+'_seg').copy(),
                                logscale=logscale, _ax=_ax, vbnds=vbnds, cmap=cmap,
                                titl=param, ylbl='freq [KHz]', xlbl='time [s]',
                                tbounds=self.tbounds, fbounds=fbounds)  # spectrogram
        return _ax


    def plotCorrelations(self, axs=None):
        """ Plot the correlation coefficients and auto-/cross-correlation functions."""
        plotCorr = fftanal._plotCorr
        if axs is None:
            _plt.figure()
            _ax1 = _plt.subplot(4,1,1)
            _ax2 = _plt.subplot(4,1,2, sharex=_ax1, sharey=_ax1)
            _ax3 = _plt.subplot(4,1,3, sharex=_ax1, sharey=_ax1)
            _ax4 = _plt.subplot(4,1,4, sharex=_ax1)

            axs = [_ax1, _ax2, _ax3, _ax4]
        # end if
        axs = _np.atleast_1d(axs)

        if len(axs) == 1:
            _ax = plotCorr(self.lags, self.corrcoef, _ax=axs[0], scl=1e6, afont=self.afont, titl=None, ylbl=r'$\rho_{xy}$', fmt='k-')
            return _ax
        elif len(axs) == 2:
            _ax1 = plotCorr(self.lags, self.Rxx, _ax=axs[0], scl=1e6, afont=self.afont, titl='Correlations', xlbl=None, ylbl=r'$R_{xx}$', fmt='b-')
            plotCorr(self.lags, self.Ryy, _ax=axs[0], scl=1e6, afont=self.afont, titl=None, xlbl=None, ylbl=r'$R_{yy}$', fmt='r-')
            plotCorr(self.lags, self.Rxy, _ax=axs[0], scl=1e6, afont=self.afont, titl=None, xlbl=None, ylbl=r'$R_{xy}$', fmt='k-')
            _ax2 = plotCorr(self.lags, self.corrcoef, _ax=axs[1], scl=1e6, afont=self.afont, titl='Cross-Correlation', ylbl=r'$\rho_{xy}$', fmt='k-')
            return _ax1, _ax2
        elif len(axs) == 3:
            _ax1 = plotCorr(self.lags, self.Rxx, _ax=axs[0], scl=1e6, afont=self.afont, titl='Auto-Correlation', xlbl=None, ylbl=r'$R_{xx}$', fmt='b-')
            _ax2 = plotCorr(self.lags, self.Ryy, _ax=axs[1], scl=1e6, afont=self.afont, titl='Auto-Correlation', xlbl=None, ylbl=r'$R_{yy}$', fmt='r-')
            _ax3 = plotCorr(self.lags, self.Rxy, _ax=axs[2], scl=1e6, afont=self.afont, titl='Cross-Correlation', xlbl=None, ylbl=r'$R_{xy}$', fmt='k-')
            return _ax1, _ax2, _ax3
        else:
            _ax1 = plotCorr(self.lags, self.Rxx, _ax=axs[0], scl=1e6, afont=self.afont, titl='Cross-Correlation', xlbl='', ylbl=r'$R_{xx}$', fmt='b-')
            _ax2 = plotCorr(self.lags, self.Ryy, _ax=axs[1], scl=1e6, afont=self.afont, titl=None, xlbl='', ylbl=r'$R_{yy}$', fmt='r-')
            _ax3 = plotCorr(self.lags, self.Rxy, _ax=axs[2], scl=1e6, afont=self.afont, titl=None, xlbl='', ylbl=r'$R_{xy}$', fmt='k-')
            _ax4 = plotCorr(self.lags, self.corrcoef, _ax=axs[3], scl=1e6, afont=self.afont, titl=None, ylbl=r'$\rho_{xy}$', fmt='k-')
            return _ax1, _ax2, _ax3, _ax4


    # ===================================================================== #


    def plottime(self, _ax=None):
        """ Plot the signals input. """
        _ax = fftanal._plotSignal([self.tvec, self.tvec], [self.sigx, self.sigy],
                   _ax=_ax, scl=1.0, afont=self.afont, titl='Input Signals',
                   ylbl='Sigx, Sigy', fmt='k-', tbounds=self.tbounds)
        return _ax

    
    def plotCorr(self, _ax=None):
        """ Plot the correlation coefficients. """
        _ax = fftanal._plotCorr(self.lags, self.corrcoef, _ax=_ax, scl=1e6, afont=self.afont, titl=None, ylbl=r'$\rho_{xy}$', fmt='k-')
        return _ax

    
    def plotPxy(self, _ax=None):
        """ Plot the power spectra. """
        _ax = fftanal._plotlogAmp(self.freq, self.Pxx, self.Pyy, self.Pxy, afont=self.afont, _ax=_ax, scl=1e-3)
        return _ax

    
    def plotCxy(self, _ax=None):
        """ Plot the mean-squared coherence from the FFT. """        
        _ax = fftanal._plotMeanSquaredCoherence(self.freq, self.Cxy2, afont=self.afont, _ax=_ax, scl=1e-3, Navr=self.Navr)
        return _ax

    
    def plotphxy(self, _ax=None):
        """ Plot the phase spectra calculated from the FFT. """
        _ax = fftanal._plotPhase(self.freq, self.phi_xy, afont=self.afont, _ax=_ax, scl=1e-3)
        return _ax

    # ===================================================================== #
    # ===================================================================== #


    def __calcAmp__(self, tvec, sigx, sigy, tbounds, nn=8, ol=0.5, ww='hanning'):
        """ Calculate the amplitude spectra and plot it. """
        # The amplitude is most accurately calculated by using several windows
        self.frqA, self.Axy, self.Axx, self.Ayy, self.aCxy, _, _ = \
            fft_pwelch(tvec, sigx, sigy, tbounds, Navr=nn, windowoverlap=ol,
                       windowfunction=ww, useMLAB=self.useMLAB, plotit=0,
                       verbose=self.verbose, detrend_style=self.detrendstyle,
                       onesided=self.onesided)
        self.__plotAmp__()


    def __calcPh1__(self, tvec, sigx, sigy, tbounds, nn=1, ol=0.0, ww='box'):
        """ Calculate the phase spectra and plot it. """
        # The amplitude is most accurately calculated by using several windows
        self.frqP, _, _, _, _, self.ph, _ = \
            fft_pwelch(tvec, sigx, sigy, tbounds, Navr=nn, windowoverlap=ol,
                       windowfunction=ww, useMLAB=self.useMLAB, plotit=0,
                       verbose=self.verbose, detrend_style=self.detrendstyle,
                       onesided=self.onesided)
        self.__plotPh1__()

    
    def __plotAmp__(self, _ax=None):
        """ Plot the linear amplitude spectra. """        
        fftanal._plotlogAmp(self.frqA, self.Axx, self.Ayy, self.Axy, afont=self.afont, _ax=_ax, scl=1e-3)

        
    def __plotPh1__(self, _ax=None):
        """ Plot the phase spectra. """
        fftanal._plotPhase(self.frqP, self.ph, afont=self.afont, _ax=_ax, scl=1e-3)

        
    # ===================================================================== #

    
    def __preallocateFFT__(self):
        """ Required spectral quantities. """
        #Inputs
        self.tvec = _np.array([], dtype=_np.float64)

        #Outputs
        self.freq = _np.array([], dtype=_np.float64)
        self.Pxy = _np.array([], dtype = _np.complex128)
        self.Pxx = _np.array([], dtype = _np.complex128)
        self.Pyy = _np.array([], dtype = _np.complex128)
        self.varPxy = _np.array([], dtype = _np.complex128)
        self.varPxx = _np.array([], dtype = _np.complex128)
        self.varPyy = _np.array([], dtype = _np.complex128)

        self.Coh = _np.array([], dtype=_np.float64)
        self.varCoh = _np.array([], dtype=_np.float64)

        self.phi = _np.array([], dtype=_np.float64)
        self.varphi = _np.array([], dtype=_np.float64)
    #end __preallocateFFT__

    # ===================================================================== #
    # ===================================================================== #

    
    @staticmethod
    def resample(tvx, sigx, tvy, sigy):
        """ Resample input vectors so that they are sampled at the same data-rate. """
        try:
            from .utils import upsample
        except:
            from FFT.utils import upsample

        Fsx = fftanal.__Fs__(tvx)
        Fsy = fftanal.__Fs__(tvy)
        if len(sigx) > len(sigy):
            sigy = upsample(sigy, Fsy, Fsx)
            tvec = tvx
        elif len(sigy) > len(sigx):
            sigx = upsample(sigx, Fsx, Fsy)
            tvec = tvy
        return tvec, sigx, sigy

    
    @staticmethod
    def __Fs__(tvec):
        """ Calculate the sampling rate from the time-vector.  """
        return (len(tvec)-1)/(tvec[-1]-tvec[0])

    
    @staticmethod
    def __ibounds__(tvec, tbounds):
        """ Return the index bounds given time-bounds. """
        # ib1 = int(_np.floor((tbounds[0]-tvec[0])*fftanal.__Fs__(tvec)))
        # ib2 = int(_np.floor(1 + (tbounds[1]-tvec[0])*fftanal.__Fs__(tvec)))
        ib1 = _np.argmin(_np.abs(tbounds[0]-tvec[0]))
        ib2 = _np.argmin(_np.abs(tbounds[1]-tvec[0]))  
        return [ib1, ib2]

    
    @staticmethod
    def __trimsig__(sigt, ibounds):
        """ Trim the input signal to given index bounds. """
        return sigt[ibounds[0]:ibounds[1]]

    # ===================================================================== #
    # ===================================================================== #
    """
    Must know two of these inputs to determine third
     k = # windows, M = Length of data to be segmented, L = length of segments,
          K = (M-NOVERLAP)/(L-NOVERLAP)
               Navr  = (nsig-noverlap)/(nwins-noverlap)
               nwins = (nsig - noverlap)/Navr + noverlap
               noverlap = (nsig - nwins*Navr)/(1-Navr)
       noverlap = windowoverlap*nwins
               nwins = nsig/(Navr-Navr*windowoverlap + windowoverlap)

    """

    
    @staticmethod
    def makewindowfn(windowfunction, nwins, verbose=True):
        """ Define windowing function for apodization. """
        win, winparams = windows(windowfunction, nwins=nwins, verbose=verbose, msgout=True)
        return win, winparams
    
    @staticmethod
    def _getNwins(nsig, Navr, windowoverlap):
        """ Return the segment length (integer number of points) from the signal length, number of averaging windodws, and fractional window overlap. 
        
        Navr = (nsig - noverlap) // (nwins - noverlap) + 1   
             = (nsig - windowoverlap*nwins) // (nwins - noverlap) + 1           
        
            (Navr - 1)*(nwins - noverlap) = (nsig - windowoverlap*nwins) 
            (Navr - 1)*nwins - (Navr - 1)*noverlap = nsig - windowoverlap*nwins
            (Navr - 1 + windowoverlap)*nwins - (Navr - 1)*windowoverlap*nwins = nsig             
            (Navr - 1 + windowoverlap - (Navr - 1)*windowoverlap)*nwins = nsig             
        
            
        nwins = nsig / {Navr - 1 + windowoverlap - (Navr - 1)*windowoverlap}
              = nsig / {Navr - 1 + windowoverlap - (Navr*windowoverlap - 1*windowoverlap)}
              = nsig / {Navr - 1 + windowoverlap + (-Navr*windowoverlap + windowoverlap)}              
              = nsig / (Navr - Navr*windowoverlap + 2*windowoverlap - 1))              
              
        """
        nwins = int(_np.floor(nsig*1.0/(Navr-Navr*windowoverlap + 2*windowoverlap - 1)))
        if nwins>=nsig:
            nwins = nsig
        return nwins

    
    @staticmethod
    def _getNoverlap(nwins, windowoverlap):
        """ Convert percentage window overlap and length of the window to points of window overlap. """
        return int( _np.ceil( windowoverlap*nwins ) )

    
    @staticmethod
    def _getNavr(nsig, nwins, noverlap):
        """ Return the required number of averaging windows given the signal length, length of the window function, and points of overlap. 
        
        Navr = (nsig - noverlap) // (nwins - noverlap) + 1   
        """        
        if nwins>= nsig:
            return int(1)
        else:
            return int(1 + (nsig-noverlap)//(nwins-noverlap))


    @staticmethod
    def __getNwins(nsig, Navr, noverlap):
        """ 
        Return the required window length given the signal length, number of averaging windows, and points of overlap.
        
        
        Navr = (nsig - noverlap) // (nwins - noverlap) + 1        
            Navr - 1 = (nsig - noverlap) // (nwins - noverlap)
            (Navr - 1)*(nwins - noverlap) = (nsig - noverlap) 
            (Navr - 1)*nwins - (Navr - 1)*noverlap = (nsig - noverlap)             
            (Navr - 1)*nwins = (nsig - noverlap) + (Navr - 1)*noverlap
        
        nwins = (nsig - noverlap)/(Navr - 1) + noverlap
                             
        """
        nwins = int((nsig-noverlap)/(Navr-1)+noverlap)
        if nwins>= nsig:
            return nsig
        else:
            return nwins

    
    @staticmethod
    def __getNoverlap(nsig, nwins, Navr):
        """ 
        Return the number of points of overlap given the signal length, window length, and number of averaging windows. 
        
        Navr = (nsig - noverlap) // (nwins - noverlap) + 1        
            Navr - 1 = (nsig - noverlap) // (nwins - noverlap)
            (Navr - 1)*(nwins - noverlap) = (nsig - noverlap) 
            (Navr - 1)*nwins - (Navr - 1)*noverlap = (nsig - noverlap)             
            (Navr - 1)*nwins - (Navr - 2)*noverlap = nsig             
             - (Navr - 2)*noverlap = nsig - (Navr - 1)*nwins
             (2 - Navr)*noverlap = nsig + (1 - Navr)*nwins             

        noverlap = ( nsig + (1 - Navr)*nwins )/(2 - Navr)
                 = ( (Navr - 1)*nwins - nsig )/(Navr - 2)        
        """
        if nwins>= nsig:
            return 0
        else:
            return int( ( (Navr - 1)*nwins - nsig )/(Navr - 2) )

    
    @staticmethod
    def _getMINoverlap(nsig, nwins, Navr):
        """ Calculate the minimum allowable overlap that satisfies the constant overlap add constraint. """        
        noverlap = 1
        while fftanal._checkCOLA(nsig, nwins, noverlap) == False and noverlap<1e4:
            noverlap += 1
        return noverlap

    
    @staticmethod
    def _getMAXoverlap(nsig, nwins, Navr):
        """ Calculate the maximum allowable overlap that satisfies the constant overlap add constraint. """
        noverlap = _np.copy(nwins)-1
        while fftanal._checkCOLA(nsig, nwins, noverlap) == False and noverlap>0:
            noverlap -= 1
        return noverlap

    
    @staticmethod
    def _checkCOLA(nsig, nwins, noverlap):
        """ Check the constant overlap add condition for the given parameters.
        
        If satisfied, then the input signal can be reconstructed from the nonmodified spectra. 
        """
        return (nsig - nwins) % (nwins - noverlap) == 0

    
    @staticmethod
    def _getNnyquist(nfft):
        """ Return index of the Nyquist component. Note--> this function is probably wrong. """
        Nnyquist = nfft//2       # Even
        if nfft % 2:        # Odd
            # Nnyquist = nfft//2 +1
            Nnyquist = (nfft+1)//2
        return Nnyquist

    
    @staticmethod
    def _getS1(win):
        """ sum of the componenets of the window function. """
        return _np.sum(win)

    @staticmethod
    def _getS2(win):
        """ sum of the squared componenets of the window function. """        
        return _np.sum(win**2.0)

    
    @staticmethod
    def _getNENBW(Nnyquist, S1, S2):
        """ Normalized equivalent noise bandwidth. """        
        return Nnyquist*1.0*S2/(S1**2) # Normalized equivalent noise bandwidth

    
    @staticmethod
    def _getENBW(Fs, S1, S2):
        """ Effective noise bandwidth. """
        return Fs*S2/(S1**2)  # Effective noise bandwidth

    
    @staticmethod
    def _getNorms(win, Nnyquist, Fs):
        """ Return the normalization constants for spectra. """
        S1 = fftanal._getS1(win)
        S2 = fftanal._getS2(win)

        # Normalized equivalent noise bandwidth
        NENBW = fftanal._getNENBW(Nnyquist, S1, S2)
        ENBW = fftanal._getENBW(Fs, S1, S2) # Effective noise bandwidth
        return S1, S2, NENBW, ENBW

    
    # ===================================================================== #

    
    @staticmethod
    def intspectra(freq, sigft, ifreq=None, ispan=None, ENBW=None):
        """
        This function integrates the spectra over a specified range.
        """

        if ifreq is None:
            ifreq = _np.argmax(_np.abs(sigft), axis=0)

            if ENBW is not None:
                ispan = 2*_np.where(freq>=ENBW)[0][0]
            elif ispan is None:
                ispan = 6
            # end if
            ilow = ifreq-ispan//2
            ihigh = ifreq+ispan//2
        elif ispan is None:
            ilow = 0
            ihigh = len(sigft)
        # end
        Isig = _np.trapz(sigft[ilow:ihigh], freq[ilow:ihigh], axis=0)
        Ivar = _np.zeros_like(Isig)   # TODO!: decide if you want to use trapz_var or not
        return Isig, Ivar

    
    @staticmethod
    def _detrend_func(detrend_style=None):
        """ Return detrending function. """
        if detrend_style is None:  detrend_style = 0  # end if

        if detrend_style > 0:
            detrend = detrend_mean    # ------- Mean detrending ========= #
        elif detrend_style < 0:
            detrend = detrend_linear  # ------ Linear detrending ======== #
        else:  # detrend_style == 0:
            detrend = detrend_none    # -------- No detrending ========== #
        # end if
        return detrend

    
    # ===================================================================== #

    
    @staticmethod
    def _fft_win(sig, **kwargs):
        """ implementation of spectrogram method. """
        x_in = sig.copy()
        tvec = kwargs.get('tvec', None)
        detrendwin = kwargs.get('detrendwin', False)
        onesided = kwargs.get('onesided', False)
        win = kwargs['win']
        nwins = kwargs['nwins']
        Navr = kwargs['Navr']
        noverlap = kwargs['noverlap']
        Nnyquist = kwargs['Nnyquist']
        detrender = fftanal._detrend_func(detrend_style=kwargs['detrend_style'])

        # Define normalization constants for the window
        S1 = kwargs['S1']
        S2 = kwargs['S2']
        ENBW = kwargs['ENBW']  # Equivalent noise bandwidth

        if tvec is None:
            tvec = _np.linspace(0.0, 1.0, len(x_in))
        # endif
        Fs = kwargs.get('Fs', fftanal.__Fs__(tvec))
        nfft  = nwins

        nch = 1
        if len(x_in.shape)>1:
            _, nch = x_in.shape
        # end if

        # ===== #
        # detrend the whole background, like most programs do it
        if not detrendwin:
            x_in = detrender(x_in)
        # end if
        # ===== #

        ist = _np.arange(Navr)*(nwins-noverlap)
        Xfft = _np.zeros((nch, Navr, nfft), dtype=_np.complex128)
        tt = _np.zeros( (Navr,), dtype=float)
        pseg = _np.zeros( (nch, Navr,), dtype=float)
        for gg in _np.arange(Navr):
            istart = ist[gg]  #Starting point of this window
            iend   = istart+nwins                 #End point of this window

            tt[gg] = _np.mean(tvec[istart:iend])
            xtemp = x_in[istart:iend, ...]

            # Windowed signal segment:
            # - To get the most accurate spectrum, background subtract
            # xtemp = win*_dsp.detrend(xtemp, type='constant')
            if detrendwin:
                # this is only good when the background is not evolving!
                xtemp = detrender(xtemp, axes=0)
            # end if
            xtemp = (_np.atleast_2d(win).T*_np.ones((1,nch), dtype=xtemp.dtype))*xtemp

            pseg[..., gg] = _np.trapz(xtemp**2.0, x=tvec[istart:iend], axes=0)
            Xfft[..., gg,:nfft] = fftmod.fft(xtemp, n=nfft, axes=0).T  # nch, Navr, nfft
        #endfor loop over fft windows

        freq = fftmod.fftfreq(nfft, 1.0/Fs)
        if onesided:
            freq = freq[:Nnyquist]  # [Hz]
            Xfft = Xfft[...,:Nnyquist]

            # Real signals equally split their energy between positive and negative frequencies
            Xfft[..., 1:-1] = _np.sqrt(2)*Xfft[..., 1:-1]
            if nfft%2:  # odd
                Xfft[...,-1] = _np.sqrt(2)*Xfft[...,-1]
            # endif
        else:
            freq = fftmod.fftshift(freq, axes=0)
            Xfft = fftmod.fftshift(Xfft, axes=-1)
        # end if

        # in the case of one-channel input, don't over expand stuff
        pseg = pseg.squeeze()
        Xfft = Xfft.squeeze()

        # Remove gain of the window function to yield the RMS Power spectrum
        # in each segment (constant peak amplitude)
        Xfft /= S1   # Vrms
        pseg /= S2

        # Compute the spectral density from the RMS spectrum
        # (constant noise floor)
        Xfft /= _np.sqrt(ENBW)  # [V/Hz^0.5]
        return tt, freq, Xfft, pseg

    
    @staticmethod
    def _plotspec(tseg, freq, Pxy_seg, logscale=False, _ax=None, vbnds=None, cmap=None, tbounds=None,
                 titl=r'P$_{xy}', ylbl='freq [KHz]', xlbl='time [s]', fbounds=None):  # spectrogram
        """ Plot the spectrogram. """        

        spec = _np.abs(Pxy_seg).astype(float)
        if _ax is None:
            _plt.figure()
            _ax = _plt.gca()
        if vbnds is None:            vbnds = [spec.min(), spec.max()]        # endif
        if cmap is None:             cmap = 'RdBu'                           # endif
        if tbounds is None:          tbounds = [tseg.min(), tseg.max()]        # end if
        if fbounds is None:          fbounds = [freq.min(), freq.max()]        # end if

        _ax.set_title(titl)
        _ax.set_ylabel(ylbl)
        _ax.set_xlabel(xlbl)

        if logscale:
            spec = 10.0*_np.log10(spec)
            _ax.set_yscale('symlog', linthreshy=0.01)
        # endif

        # bin starts are plotted, not bin centers
        tbin = tseg-0.5*(tseg[2]-tseg[1])
        fbin = 1e-3*(freq-0.5*(freq[2]-freq[1]))
        _plt.pcolor(tbin, fbin, spec.T, cmap=cmap, vmin=vbnds[0], vmax=vbnds[1])
#        _plt.pcolormesh(tseg, 1e-3*freq, spec.T, cmap=cmap, vmin=vbnds[0], vmax=vbnds[1])
        _plt.xlim(tuple(tbounds))
        _plt.ylim(tuple(fbounds))
        _plt.colorbar()
        _plt.draw()
        return _ax

    
    @staticmethod
    def _plotSignal(tvec, sig, _ax=None, scl=1.0, afont=None, titl='Input Signal', ylbl='Signal', fmt='k-', tbounds=None):
        """ Plot the input signals. """        
        _plot_quantity = fftanal._plot_quantity
        if scl == 1e6:        xlbl = 't [us]'
        elif scl == 1e3:      xlbl = 't [ms]'
        else:                 xlbl = 't [s]'
        # end if
        if len(sig)==2:
            tx = tvec[0]
            ty = tvec[1]
            sigx = sig[0]
            sigy = sig[1]
            if _np.iscomplexobj(sigx) and _np.iscomplexobj(sigy):
                _ax = _plot_quantity(tx, sigx.real, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=xlbl, fmt='b-')
                _ax = _plot_quantity(tx, sigx.imag, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=xlbl, fmt='b--')
                _ax = _plot_quantity(ty, sigy.real, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=xlbl, fmt='r-')
                _ax = _plot_quantity(ty, sigy.imag, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=xlbl, fmt='r--')
            elif _np.iscomplexobj(sigx) and not _np.iscomplexobj(sigy):
                _ax = _plot_quantity(ty, sigy, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=xlbl, fmt='r-')
                _ax = _plot_quantity(tx, sigx.real, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=xlbl, fmt='b-')
                _ax = _plot_quantity(tx, sigx.imag, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=xlbl, fmt='b--')
            elif _np.iscomplexobj(sigy) and not _np.iscomplexobj(sigx):
                _ax = _plot_quantity(tx, sigx, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=xlbl, fmt='b-')
                _ax = _plot_quantity(ty, sigy.real, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=xlbl, fmt='r-')
                _ax = _plot_quantity(ty, sigy.imag, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=xlbl, fmt='r--')
            else:
                _ax = _plot_quantity(tx, sigx, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=xlbl, fmt='b'+fmt[1])
                _ax = _plot_quantity(ty, sigy, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=xlbl, fmt='r'+fmt[1])
            # end if
        else:
            if _np.iscomplexobj(sigx):
                _ax = _plot_quantity(tvec, sig.real, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=xlbl, fmt=fmt[0]+'-')
                _ax = _plot_quantity(tvec, sig.imag, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=xlbl, fmt=fmt[0]+'--')
            else:
                _ax = _plot_quantity(tvec, sig, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=xlbl, fmt=fmt)
            # end if
        # end if

        if tbounds is not None:
            _ax.axvline(x=tbounds[0], color=fmt[0])
            _ax.axvline(x=tbounds[1], color=fmt[0])
        # end if
        return _ax

    
    @staticmethod
    def _plotCorr(lags, corrcoef, _ax=None, scl=1e6, afont=None, titl='Cross-Correlation', ylbl=r'$\rho_{xy}$', fmt='k-'):
        """ Plot the correlation coefficient. """        
        _plot_quantity = fftanal._plot_quantity
        if scl == 1e6:        xlbl = 'lags [us]'
        elif scl == 1e3:      xlbl = 'lags [ms]'
        else:                 xlbl = 'lags [s]'
        # end if
        _ax = _plot_quantity(lags, corrcoef, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=xlbl, fmt=fmt)
        return _ax

    
    @staticmethod
    def _plotCoherence(freq, Cxy, _ax=None, scl=1e-3, afont=None, titl='Complex Coherence', ylbl=r'|$\gamma_{xy}$|', Navr=None):
        """ Plot the root-mean-squared coherence versus frequency. 
        
        Calculated from the complex coherence.
        """        
        _plot_quantity = fftanal._plot_quantity
        _ax = _plot_quantity(freq, _np.abs(Cxy), _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=None, fmt='k-')
        if Navr is not None:
            _ax.axhline(y=1.0/_np.sqrt(Navr), color='k')
        return _ax

    
    @staticmethod
    def _plotRMSCoherence(freq, Cxy2, _ax=None, scl=1e-3, afont=None, titl='RMS Coherence', ylbl=r'$\gamma_{xy}$', Navr=None):
        """ Plot the root-mean-squared coherence versus frequency. """        
        _plot_quantity = fftanal._plot_quantity
        _ax = _plot_quantity(freq, _np.sqrt(_np.abs(Cxy2)), _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=None, fmt='k-')
        if Navr is not None:
            _ax.axhline(y=1.0/_np.sqrt(Navr), color='k')
        return _ax

    
    @staticmethod
    def _plotMeanSquaredCoherence(freq, Cxy2, _ax=None, scl=1e-3, afont=None, titl='Mean Squared-Coherence', ylbl=r'$\gamma_{xy}^2$', Navr=None):
        """ Plot the mean-squared coherence versus frequency. """
        _plot_quantity = fftanal._plot_quantity
        _ax = _plot_quantity(freq, _np.abs(Cxy2), _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=None, fmt='k-')
        if Navr is not None:
            _ax.axhline(y=1.0/Navr, color='k')
        return _ax

    
    @staticmethod
    def _plotsemilogAmp(freq, Axx, Ayy, Axy, _ax=None, scl=1e-3, afont=None, titl='Power Spectra', ylbl=r'P$_{ij}$ [dB/Hz]'):
        """ Plot the logarithmic ampmlitude spectra using semilogx. """
        _plot_quantity = fftanal._plot_quantity
        if _ax is None:
            _plt.figure()
            _ax = _plt.subplot(1,1,1)
        _ax = _plot_quantity(freq, 10*_np.log10(_np.abs(Axx)), _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=None, fmt='b-', plothandle=_ax.semilogx)
        _ax = _plot_quantity(freq, 10*_np.log10(_np.abs(Ayy)), _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=None, fmt='r-', plothandle=_ax.semilogx)
        _ax = _plot_quantity(freq, 10*_np.log10(_np.abs(Axy)), _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=None, fmt='k-', plothandle=_ax.semilogx)
        return _ax

    
    @staticmethod
    def _plotlogAmp(freq, Axx, Ayy, Axy, _ax=None, scl=1e-3, afont=None, titl='Power Spectra', ylbl=r'P$_{ij}$ [dB/Hz]'):
        """ Plot the logarithmic amplitude spectra on a linear scale. """
        _plot_quantity = fftanal._plot_quantity
        _ax = _plot_quantity(freq, 10*_np.log10(_np.abs(Axx)), _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=None, fmt='b-')
        _ax = _plot_quantity(freq, 10*_np.log10(_np.abs(Ayy)), _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=None, fmt='r-')
        _ax = _plot_quantity(freq, 10*_np.log10(_np.abs(Axy)), _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=None, fmt='k-')
        return _ax

    
    @staticmethod
    def _plotAmp(freq, Axx, Ayy, Axy, _ax=None, scl=1e-3, afont=None, titl='Power Spectra', ylbl=r'P$_{ij}$ [I.U./Hz]'):
        """ Plot the amplitude spectra. """
        _plot_quantity = fftanal._plot_quantity
        _ax = _plot_quantity(freq, _np.abs(Axx), _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=None, fmt='b-')
        _ax = _plot_quantity(freq, _np.abs(Ayy), _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=None, fmt='r-')
        _ax = _plot_quantity(freq, _np.abs(Axy), _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=None, fmt='k-')
        return _ax

    
    @staticmethod
    def _plotPhase(freq, phase, _ax=None, scl=1e-3, afont=None, titl='Cross-Phase', ylbl=r'$\phi_{xy}$', fmt='k-'):
        """ Plot the phase angle. """
        return fftanal._plot_quantity(freq, phase, _ax=_ax, scl=scl, afont=afont, titl=titl, ylbl=ylbl, xlbl=None, fmt=fmt)

    
    @staticmethod
    def _plot_quantity(freq, quant, _ax=None, scl=1e-3, afont=None, titl='', ylbl='', xlbl=None, fmt='k-', plothandle=None):
        """ Plot a generic quantity with given labels """
        if _ax is None:
            _plt.figure()
            _ax = _plt.subplot(1,1,1)
        if afont is None:              afont = {'fontname':'Arial','fontsize':14}
        if plothandle is None:         plothandle = _ax.plot
        if xlbl is None:
            if scl == 1e-6:            xlbl = 'f [MHz]'
            elif scl == 1e-3:          xlbl = 'f [KHz]'
            else:                      xlbl = 'f [Hz]'

        plothandle(scl*freq, quant,fmt)
        if ylbl is not None:    _ax.set_ylabel(ylbl, **afont)     # end if
        if len(xlbl)>0:         _ax.set_xlabel(xlbl, **afont)     # end if
        if titl is not None:    _ax.set_title(titl, **afont)      # end if
        xlims = _ax.get_xlim()
        if xlims[0] == 0:
            _ax.set_xlim(0,1.01*scl*freq[-1])
        else:
            _ax.set_xlim(-1.01*scl*freq[-1], 1.01*scl*freq[-1])

        _plt.draw()
    #    _plt.show()
        return _ax

    # ===================================================================== #
    # ===================================================================== #


    @staticmethod
    def __testFFTanal__(useMLAB=True, plotext=True):
        """ Compare the mlab versus pure python implementation of welch's averaged periodogram method """
        ft1 = test_fftanal(useMLAB, plotit=not plotext, nargout=1)
        ft2 = test_fftanal(not useMLAB, plotit=not plotext, nargout=1,
                           tstsigs=(ft1.tvec, ft1.sigx, ft1.sigy))

        # ------
        
        # Plot the comparisons
        if plotext:
            afont = {'fontname':'Arial','fontsize':14}
            #The input signals versus time
            _plt.figure()
            _ax1 = _plt.subplot(2,2,1)
            _ax1.plot(ft1.tvec, ft1.sigx,'b-',ft1.tvec, ft1.sigy,'r-')
            # _ax1.plot(ft1.tvec, ft1.sigx,'m--',ft1.tvec, ft1.sigy,'m--')
            _ax1.set_title('Input Signals',**afont)
            _ax1.set_xlabel('t[s]',**afont)
            _ax1.set_ylabel('sig_x,sig_y[V]',**afont)
            _plt.axvline(x=ft1.tbounds[0],color='k')
            _plt.axvline(x=ft1.tbounds[1],color='k')

            _ax2 = _plt.subplot(2,2,2)
            _ax2.plot(1e-3*ft1.freq, _np.abs(ft1.Pxy), 'k-')
            _ax2.plot(1e-3*ft2.freq, _np.abs(ft2.Pxy), 'm--')
            _ax2.set_title('Power Spectra Comparison', **afont)
            _ax2.set_ylabel(r'P$_{ij}$ [dB/Hz]', **afont),
            _ax2.set_xlabel('f[kHz]', **afont)
            if ft1.onesided:
                _ax2.set_xlim(0,1.01e-3*ft1.freq[-1])
            else:
                _ax2.set_xlim(-1.01e-3*ft1.freq[-1],1.01e-3*ft1.freq[-1])
            # end if

            _ax3 = _plt.subplot(2, 2, 3, sharex=_ax2)
            _ax3.plot(1e-3*ft1.freq, _np.sqrt(_np.abs(ft1.Cxy2)), 'k-')
            _ax3.plot(1e-3*ft2.freq, _np.sqrt(_np.abs(ft2.Cxy2)), 'm-')
#            _plt.axhline(y=1.0/_np.sqrt(ft1.Navr), color='k')
            _ax3.plot(1e-3*ft1.freq, ft1.Cxy, 'k--')
            _ax3.plot(1e-3*ft2.freq, ft2.Cxy, 'm--')
            _plt.axhline(y=1.0/_np.sqrt(ft1.Navr), color='k')
            _ax3.set_title('Coherence', **afont)
            _ax3.set_ylabel(r'C$_{xy}$', **afont)
            _ax3.set_xlabel('f[kHz]', **afont)
            if ft2.onesided:
                _ax3.set_xlim(0,1.01e-3*ft2.freq[-1])
            else:
                _ax3.set_xlim(-1.01e-3*ft2.freq[-1],1.01e-3*ft2.freq[-1])


            _ax4 = _plt.subplot(2, 2, 4, sharex=_ax2)
            _ax4.plot(1e-3*ft1.freq, ft1.phi_xy, 'k-')

            # this is mlab version dependent:
            if 0:
                _ax4.plot(1e-3*ft2.freq, -1.0*ft2.phi_xy, 'm--')
            else:
                _ax4.plot(1e-3*ft2.freq, ft2.phi_xy, 'm--')
            # end if
            _ax4.set_title('Phase', **afont)
            _ax4.set_ylabel(r'$\phi_{xy}$',**afont)
            _ax4.set_xlabel('f[kHz]',**afont)
            if ft1.onesided:
                _ax4.set_xlim(0,1.01e-3*ft1.freq[-1])
            else:
                _ax4.set_xlim(-1.01e-3*ft1.freq[-1],1.01e-3*ft1.freq[-1])
            # end if

            _plt.tight_layout()
            _plt.draw()

        return ft1, ft2

#end class fftanal


def test_data():
    """ Generate test-data """
    import scipy.signal as _dsp

    #Minimize the spectral leakage:
    df = 5.0   #Hz
    N  = 2**14 #Numper of points in time-series
#    N  = 2**20 #Numper of points in time-series
    tvec = (1.0/df)*_np.arange(0.0,1.0,1.0/(N))
    Fs = 1.0/(df*N)

    #Sine-wave
    _np.random.seed()
#     nx = int(N / 100)
#     sigx = _np.sin(2.0*_np.pi*(df*2000.0)*tvec[:nx])     #Shifted time-series
#     sigx = _np.sin(2.0*_np.pi*(df*30.0)*tvec)     #Shifted time-series

    #Square-wave
    sigx = _dsp.square(2.0*_np.pi*(df*30.0)*tvec)    #Shifted square wave

    sigx *= 0.1
#    sigx += 0.01*_np.random.standard_normal( (sigx.shape[0],) )
    sigx += 7.0

    #Noisy phase-shifted sine-wave
    _np.random.seed()
#    sigy = _np.sin(2.0*_np.pi*(df*2000.0)*tvec-_np.pi/4.0)
    nch = 1
    sigy = _np.zeros((len(tvec), nch), dtype=_np.float64)
    for ii in range(nch):
        sigy[:,ii] = _np.sin(2.0*_np.pi*((ii+1)*df*30.0)*tvec-_np.pi/4.0-ii*_np.pi/16)/(ii+1)
        sigy[:,ii] += ii
    sigy *= 0.007
#    sigy += 0.07*_np.random.standard_normal( (tvec.shape[0],nch) )
    sigy += 2.5

    return Fs, tvec, sigx, sigy

# ========================================================================== #
# ==========================================================================


def test_fftpwelch(useMLAB=True, plotit=True, nargout=0, tstsigs = None, verbose=True):
    """ Test welch's averaged periodogram method implemented in the function. """

    ##Generate test data for the no input case:
    # import scipy.signal as _dsp

    if tstsigs is None:
        Fs, tvec, sigx, sigy = test_data()
        df = (len(tvec)+1)/(len(tvec)*(tvec[-1]-tvec[0]))
    else:
        tvec = tstsigs[0].copy()
        sigx = tstsigs[1].copy()
        sigy = tstsigs[2].copy()

#    detrend_style = 0 # None     # matches mlab and my version together
    detrend_style = 1 # Mean     # Results in coherence > 1 in 1st non-zero freq bin for mlab, but my version works
#    detrend_style = -1 # Linear   # Definitely doesn't work well for the test signals: breaks both

    fft_pwelch(tvec,sigx,sigy, [tvec[0],tvec[-1]], Navr = 8, windowfunction = 'hamming', detrend_style=detrend_style, useMLAB=False, plotit=plotit, verbose=verbose)

    fft_pwelch(tvec,sigx,sigy, [tvec[0],tvec[-1]], minFreq=15*df, detrend_style=detrend_style, useMLAB=False, plotit=plotit, verbose=verbose)


def test_fftanal(useMLAB=False, plotit=True, nargout=0, tstsigs = None):
    """ Test welch's averaged periodogram method implemented in the class object. """
    ##Generate test data for the no input case:

    if tstsigs is None:
        #Minimize the spectral leakage:
        df = 5.0   #Hz
#        N  = 2**12 #Numper of points in time-series
        N  = 2**19
        tvec = (1.0/df)*_np.arange(0.0,1.0,1.0/(N))

        #Sine-wave
        _np.random.seed()
        sigx = _np.sin(2.0*_np.pi*(df*30.0)*tvec)     #Shifted time-series
        sigx *= 0.005
        sigx += 7.0
        sigx += 0.02*_np.random.standard_normal( (tvec.shape[0],) )
#        sigx += _np.random.uniform( low=-0.01, high=0.01, size=(tvec.shape[0],) )

        _np.random.seed()
        #Noisy phase-shifted sine-wave
        sigy = _np.sin(2.0*_np.pi*(df*30.0)*tvec-_np.pi/4.0)
        sigy *= 0.005
        sigy += 0.02*_np.random.standard_normal( (tvec.shape[0],) )
#        sigy += _np.random.uniform(low=-.01, high=0.01, size=(tvec.shape[0],) )
        sigy += 2.5

        #Square-wave
        #sigx = 10.0 + _dsp.square(2.0*_np.pi*(df*100.0)*tvec)    #Shifted square wave
        #sigy = sigx
    else:
        tvec = tstsigs[0].copy()
        sigx = tstsigs[1].copy()
        sigy = tstsigs[2].copy()
    # endif
    ft = None

    # Test using the fftpwelch function
    ft = fftanal(tvec,sigx,sigy,tbounds = [tvec[0],tvec[-1]],
            Navr = 8, windowfunction = 'hamming',
            useMLAB=useMLAB, plotit=plotit, verbose=True,
            detrend_style=1, onesided=True)
    ft.fftpwelch()

    if nargout>0:
        return ft


def create_turb_spectra(addwhitenoise=False):
    """ function for making a gaussian noise distribution """
    val = 0.005
    sigma = 1.0/500e3  #
    mu = 0.0           # centered spectra
    kfact = 5.0/3.0
    Fs = 1e6
    nfft = 2**14
    lags = (_np.asarray(range(nfft))-nfft//2).astype(float)
    lags /= Fs

    Rxy = _np.exp(-kfact*(lags-mu)**2.0/(2*sigma*sigma))
#    Rxy /= sigma*_np.sqrt(2*_np.pi)
    Rxy /= _np.nanmax(Rxy)
    Rxy *= val

    fft_pwelch(lags, Rxy, Rxy, plotit=True)
#    if addwhitenoise:
#        Rxy = fftmod.ifftshift(Rxy)
#        Rxy[0] += 2.0*min((_np.max(Rxy), val))
#        Rxy = fftmod.fftshift(Rxy)
#    # end if

    freq = fftmod.fftfreq(nfft, d=1.0/Fs)
    Pxy =fftmod.fft(Rxy, n=nfft)

    freq = fftmod.fftshift(freq)
    Pxy = fftmod.fftshift(Pxy)

    _plt.figure()
    _ax1 = _plt.subplot(2,1,1)
    _ax2 = _plt.subplot(2,1,2)
    if addwhitenoise:
        Pxy += 0.25*_np.nanmax(Pxy)*_np.random.uniform(low=-1.0, high=1.0, size=Pxy.shape)
        Rxy2 = fftmod.ifft(fftmod.ifftshift(Pxy), n=nfft).real

        _ax1 .plot(1e6*lags, Rxy, 'b-', 1e6*lags, Rxy2, 'r-')
    else:
        _ax1 .plot(1e6*lags, Rxy, '-')
    # end if

    _ax1 .set_xlabel('lags [us]')
    _ax1 .set_ylabel('Rxy')
    _ax1 .set_title('input correlations')
    _ax2 .plot(1e-3*freq, _np.abs(Pxy), '-')
    _ax2 .set_ylabel('Pxy')
#    _ax2 .plot(1e-3*freq, 10.0*_np.log10(_np.abs(Pxy)), '-')
#    _ax2 .set_ylabel('Pxy [dB]')
    _ax2 .set_xlabel('freq [KHz]')
    _ax2 .set_title('Power spectra')

    
def test():
    """ test the fft-analysis class """
    tst = fftanal(verbose=True)
    ft1, ft2 = tst.__testFFTanal__()
    return ft1, ft2


if __name__ == "__main__":
    # fts = test()
    # test_fftpwelch()

#    fts = test_fftanal(nargout=1)

    # test_fft_deriv(modified=False)
    # test_fft_deriv(modified=True)
    # test_fft_deriv(xx=2*_np.pi*_np.linspace(-1.5, 3.3, num=650, endpoint=False))

    Fs, tvec, sigx, sigy = test_data()
    df = (len(tvec)+1)/(len(tvec)*(tvec[-1]-tvec[0]))
    window_sec = 2.0/(15*df)  # minimum window length to see frequencies at 15 frequency bins
    plot_spectrum_methods(data=sigx, Fs=Fs, window_sec=window_sec, band=None, dB=False)
    plot_spectrum_methods(data=sigy, Fs=Fs, window_sec=window_sec, band=None, dB=False)

# ========================================================================== #
# ========================================================================== #

