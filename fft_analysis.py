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
import scipy.signal as _dsp
from scipy import linalg as _la
import matplotlib.mlab as _mlab
import matplotlib.pyplot as _plt

from pybaseutils.Struct import Struct
from pybaseutils.utils import detrend_mean, detrend_none, detrend_linear
from pybaseutils import utils as _ut


# ========================================================================== #
# ========================================================================== #

def fft_pwelch(tvec, sigx, sigy, tbounds=None, Navr=None, windowoverlap=None,
               windowfunction=None, useMLAB=None, plotit=None, verbose=None,
               detrend_style=None, onesided=True, **kwargs):
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
    """

    if Navr is None:
        Navr = 8
    # endif
    if windowoverlap is None:
        windowoverlap=0.5
    # endif
    if windowfunction is None:
        windowfunction = 'hamming'
    # endif
    if useMLAB is None:
        useMLAB=False
    # endif
    if plotit is None:
        plotit=True
    # endif
    if verbose is None:
        verbose=False
    # endif
    if detrend_style is None:
        detrend_style=1
    # endif
    if tbounds is None:
        tbounds = [tvec[0], tvec[-1]]
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
    # Used previously:
    # nwins    = floor( (nsig-Navr+1)/(1+(1-windowoverlap)*(Navr-1)) - 1)
    # nfft     = nwins
    # nfft     = 2.0*nfft
    # noverlap = floor( windowoverlap*nwins ) #Number of points to overlap

#    sigx = _np.atleast_2d(sigx) # multi-channel input only supported for sigy
    sigy = _np.atleast_2d(sigy)
    if _np.shape(sigy)[1] == len(tvec):
        sigy = sigy.T
    # end if
    nch = _np.size(sigy, axis=1)
    nTmodel = False
#    if len(sigx) != len(sigy):
    if _np.size(sigx, axis=0) != _np.size(sigy, axis=0):
        nTmodel = True
        nwins = _np.size(sigx, axis=0)
    else:
        # Heliotron-J
        nwins = int(_np.floor(nsig*1.0/(Navr-Navr*windowoverlap + windowoverlap)))
    # end if

    if nwins>=nsig:
        Navr = 1
        nwins = nsig
    # endif
    # nfft     = max(2^12,2^nextpow2(nwins))
    nfft = nwins

    # Number of points to overlap
    noverlap = int( _np.ceil( windowoverlap*nwins ) )
#    noverlap = int( windowoverlap*nwins )
    if nTmodel:
        Navr  = int( (nsig-noverlap)/(nwins-noverlap) )
    # end if
    Nnyquist = nfft//2 + 1
    if (nfft%2):  # odd
       Nnyquist = (nfft+1)//2
    # end if the remainder of nfft / 2 is odd

    # Remember that since we are not dealing with infinite series, the lowest
    # frequency we actually resolve is determined by the period of the window
    # fhpf = 1.0/(nwins*dt)  # everything below this should be set to zero (when background subtraction is applied)

    # ==================================================================== #

    class fftinfosc(Struct):
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
    # end class

    # =================================================================== #

    # Define windowing function for apodization
    if windowfunction.lower() == 'hamming':
        if verbose:
            print('Using a Hamming window function')
        # endif verbose
#        win = _np.hamming(nwins)  # periodic hamming window?
        win = _np.hamming(nwins+1)  # periodic hamming window
    elif windowfunction.lower() == 'hanning':
        if verbose:
            print('Using a Hanning window function')
        # endif verbose
#        win = _np.hanning(nwins)  # periodic hann window?
        win = _np.hanning(nwins+1)  # periodic hann window
    elif windowfunction.lower() == 'blackman':
        if verbose:
            print('Using a Blackman type window function')
        # endif verbose
#        win = _np.blackman(nwins)  # periodic blackman window?
        win = _np.blackman(nwins+1)  # periodic blackman window?
    elif windowfunction.lower() == 'bartlett':
        if verbose:
            print('Using a Bartlett type window function')
        # endif verbose
#        win = _np.bartlett(nwins)  # periodic Bartlett window?
        win = _np.bartlett(nwins+1)  # periodic Bartlett window?
    else:
        if verbose:
            print('Defaulting to a box window function')
        # endif verbose
        # No window function (actually a box-window)
#        win = _np.ones( (nwins,), dtype=_np.float64)
        win = _np.ones( (nwins+1,), dtype=_np.float64)
    # endif windowfunction.lower()
    win = win[:-1]  # truncate last point to make it periodic

    # Instantiate the information class that will be output
    fftinfo = fftinfosc()
    fftinfo.ibnds = [i0, i1]    # time-segment

    # Define normalization constants
    fftinfo.S1 = _np.sum( win )
    fftinfo.S2 = _np.sum(win**2)

    # Normalized equivalent noise bandwidth
    fftinfo.NENBW = Nnyquist*1.0*fftinfo.S2/(fftinfo.S1**2)
    fftinfo.ENBW = Fs*fftinfo.S2/(fftinfo.S1**2)  # Effective noise bandwidth


    # ================================================================ #
    if detrend_style is None: detrend_style = 0 # endif

    if detrend_style == 0:
        detrend = detrend_none    # -------- No detrending ========== #
    elif detrend_style > 0:
        detrend = detrend_mean    # ------- Mean detrending ========= #
    elif detrend_style < 0:
        detrend = detrend_linear  # ------ Linear detrending ======== #
    # endif

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
#            # Does nnot work very well.  amplitude is all wrong, and coherence is very low
#            sigx = _np.hstack((sigx, _np.zeros((nsig-len(sigx)+1,), dtype=sigx.dtype)))
            sigx = _np.tile(sigx, _np.size(sigy, axis=0)//len(sigx)+1)
            sigx = sigx[:len(tvec)]
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
            freq = freq[:Nnyquist-1]
            Pxx = Pxx[:Nnyquist-1]
            Pyy = Pyy[:,:Nnyquist-1]
            Pxy = Pxy[:,:Nnyquist-1]
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
            xtemp = win*detrend(xtemp, axis=0)
            ytemp = (_np.atleast_2d(win).T*_np.ones((1,nch), dtype=ytemp.dtype))*detrend(ytemp, axis=0)
            # xtemp = win*_dsp.detrend(x_in[istart:iend], type='constant')
            # ytemp = win*_dsp.detrend( y_in[istart:iend], type='constant' )

            # The FFT output from matlab isn't normalized:
            # y_n = sum[ y_m.*exp( 2_np.pi*1i*(n/N)*m ) ]
            # The inverse is normalized::
            # y_m = (1/N)*sum[ y_n.*exp( -2_np.pi*1i*(n/N)*m ) ]
            #
            # Python normalizations are optional, pick it to match MATLAB
            Xfft[gg, :nfft] = _np.fft.fft(xtemp, n=nfft, axis=0)  # defaults to last axis
            Yfft[:, gg, :nfft] = _np.fft.fft(ytemp, n=nfft, axis=0).T    # nch x Navr x nfft
        #endfor loop over fft windows

        #Auto- and cross-power spectra
        Pxx_seg[:Navr, :nfft] = Xfft*_np.conj(Xfft)
        Pyy_seg[:,:Navr, :nfft] = Yfft*_np.conj(Yfft)
        Pxy_seg[:,:Navr, :nfft] = Yfft*(_np.ones((nch,1,1), dtype=Xfft.dtype)*_np.conj(Xfft))

        # Get the frequency vector
        freq = _np.fft.fftfreq(nfft, 1.0/Fs)
#        freq = Fs*_np.arange(0.0, 1.0, 1.0/nfft)
#        if (nfft%2):
#            # freq = Fs*(0:1:1/(nfft+1))
#            freq = Fs*_np.arange(0.0,1.0,1.0/(nfft+1))
#        # end if nfft is odd
        if onesided:
            freq = freq[:Nnyquist-1]  # [Hz]
            Pxx_seg = Pxx_seg[:, :Nnyquist-1]
            Pyy_seg = Pyy_seg[:, :, :Nnyquist-1]
            Pxy_seg = Pxy_seg[:, :, :Nnyquist-1]

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
            freq = _np.fft.fftshift(freq)

            Pxx_seg = _np.fft.fftshift(Pxx_seg, axes=-1)
            Pyy_seg = _np.fft.fftshift(Pyy_seg, axes=-1)
            Pxy_seg = _np.fft.fftshift(Pxy_seg, axes=-1)
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
#        fftinfo.varPxx = _np.var((Pxx_seg[:Navr, :Nnyquist-1]), axis=0)
#        fftinfo.varPyy = _np.var((Pyy_seg[:Navr, :Nnyquist-1]), axis=0)
#        fftinfo.varPxy = _np.var((Pxy_seg[:Navr, :Nnyquist-1]), axis=0)

#        # use the RMS for the standard deviation
#        fftinfo.varPxx = _np.mean(Pxx_seg**2.0, axis=0)
#        fftinfo.varPyy = _np.mean(Pyy_seg**2.0, axis=0)
#        fftinfo.varPxy = _np.mean(Pxy_seg**2.0, axis=0)

#        fftinfo.varPxy = _np.var(_np.real(Pxy_seg), axis=0) + 1j*_np.var(_np.imag(Pxy_seg), axis=0)

        # Save the cross-phase in each segmentas well
        phixy_seg = _np.angle(Pxy_seg)  # [rad], Cross-phase of each segment

        #[ phixy_seg[0:Navr,0:Nnyquist-1], varphi_seg[0:Navr,0:Nnyquist-1] ] = \
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
    #       Cxy2  = Pxy.*(Pxy').'./(Pxx.*Pyy)  # Mean-squared Coherence between the two signals
#    Cxy2 = Pxy*_np.conj( Pxy )/( _np.abs(Pxx)*_np.abs(Pyy) ) # mean-squared coherence
##        Cxy2 = _np.abs( Cxy2 )
#    Cxy = Pxy/_np.sqrt( _np.abs(Pxx)*_np.abs(Pyy) )  # complex coherence
#
    Cxy, Cxy2 = Cxy_Cxy2(Pxx, Pyy, Pxy)

    # ========================== #
    # Uncertainty and phase part #
    # ========================== #
    # derived using error propagation from eq 23 for gamma^2 in
    # J.S. Bendat, Journal of Sound an Vibration 59(3), 405-421, 1978
    # fftinfo.varCxy2 = _np.zeros_like(Cxy2)
    fftinfo.varCxy = ((1-Cxy2)/_np.sqrt(2*Navr))**2.0
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
    fftinfo.varPhxy = (_np.sqrt(1.0-Cxy2))/_np.sqrt(2*Navr*_np.sqrt(Cxy2))**2.0

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
    # end if
    fftinfo.varLxx = (fftinfo.Lxx**2)*(fftinfo.varPxx/_np.abs(Pxx)**2)
    fftinfo.varLyy = (fftinfo.Lyy**2)*(fftinfo.varPyy/_np.abs(Pyy)**2)
    fftinfo.varLxy = (fftinfo.Lxy**2)*(fftinfo.varPxy/_np.abs(Pxy)**2)

    if nch == 1:
        Pyy = Pyy.flatten()
        Pxy = Pxy.flatten()
        Cxy = Cxy.flatten()
        Cxy2 = Cxy2.flatten()
        phi_xy = phi_xy.flatten()

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
    fftinfo.overlap = windowoverlap
    fftinfo.window = windowfunction
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
        afont = {'fontname':'Arial','fontsize':14}

        #The input signals versus time
        _plt.figure()
        _ax1 = _plt.subplot(2,2,1)
        if _np.iscomplexobj(sigx) and _np.iscomplexobj(sigy):
            _ax1.plot(tx, _np.real(sigx), 'b-')
            _ax1.plot(tx, _np.imag(sigx), 'b--')
            _ax1.plot(tvec, _np.real(sigy), 'r-')
            _ax1.plot(tvec, _np.imag(sigy), 'r--')
        elif _np.iscomplexobj(sigx) and not _np.iscomplexobj(sigy):
            _ax1.plot(tvec, sigy, 'r-')
            _ax1.plot(tx, _np.real(sigx), 'b-')
            _ax1.plot(tx, _np.imag(sigx), 'b--')
        elif _np.iscomplexobj(sigy) and not _np.iscomplexobj(sigx):
            _ax1.plot(tx, sigx, 'b-')
            _ax1.plot(tvec, _np.real(sigy), 'r-')
            _ax1.plot(tvec, _np.imag(sigy), 'r--')
        else:
            _ax1.plot(tx, sigx, 'b-', tvec, sigy, 'r-')
        # end if
        _ax1.set_title('Input Signals', **afont)
        _ax1.set_xlabel('t[s]', **afont)
        _ax1.set_ylabel('sig_x,sig_y[V]', **afont)
        if tbounds is not None:
            _plt.axvline(x=tbounds[0], color='k')
            _plt.axvline(x=tbounds[1], color='k')

        _ax2 = _plt.subplot(2,2,2)
        _ax2.plot(1e-3*freq,_np.abs(fftinfo.Lxx), 'b-')
        _ax2.plot(1e-3*freq,_np.abs(fftinfo.Lyy), 'r-')
        _ax2.plot(1e-3*freq,_np.abs(fftinfo.Lxy), 'k-')
        _ax2.set_title('Linear Amplitude Spectra', **afont)
        _ax2.set_ylabel(r'L$_{ij}$ [I.U.]', **afont),
#        _ax2.plot(1e-3*freq,_np.abs(Pxx), 'b-')
#        _ax2.plot(1e-3*freq,_np.abs(Pyy), 'r-')
#        _ax2.plot(1e-3*freq,_np.abs(Pxy), 'k-')
##        _ax2.semilogy(1e-3*freq.flatten(), _np.abs(Pxx.flatten()), 'b-')
##        _ax2.semilogy(1e-3*freq.flatten(), _np.abs(Pyy.flatten()), 'r-')
##        _ax2.semilogy(1e-3*freq.flatten(), _np.abs(Pxy.flatten()), 'k-')
#        _ax2.plot(1e-3*freq, 10*_np.log10(_np.abs(Pxx)), 'b-')
#        _ax2.plot(1e-3*freq, 10*_np.log10(_np.abs(Pyy)), 'r-')
#        _ax2.plot(1e-3*freq, 10*_np.log10(_np.abs(Pxy)), 'k-')
#        _ax2.set_title('Power Spectra', **afont)
#        _ax2.set_ylabel(r'P$_{ij}$ [dB/Hz]', **afont),
        _ax2.set_xlabel('f[kHz]', **afont)
        if onesided:
            _ax2.set_xlim(0,1.01e-3*freq[-1])
        else:
            _ax2.set_xlim(-1.01e-3*freq[-1],1.01e-3*freq[-1])
        # end if

        # _plt.setp(ax1.get_xticklabels(), fontsize=6)

        # _ax4.text(0.20, 0.15, 'P_{xx}', 'Color', 'b', 'units', 'normalized',
        #           'FontSize', 14, 'FontName', 'Arial')
        # _ax4.text(0.20, 0.35, 'P_{yy}', 'Color', 'r', 'units', 'normalized',
        #           'FontSize', 14, 'FontName', 'Arial')
        # _ax4.text(0.20, 0.55, 'P_{xy}', 'Color', 'k', 'units', 'normalized',
        #           'FontSize', 14, 'FontName', 'Arial')

        _ax3 = _plt.subplot(2, 2, 3, sharex=_ax2)
        _ax3.plot(1e-3*freq, _np.sqrt(_np.abs(Cxy2)), 'k-')
        _plt.axhline(y=1.0/_np.sqrt(Navr), color='k')
#        _ax3.plot(1e-3*freq, Cxy, 'k-')
#        _plt.axhline(y=1.0/Navr, color='k')
        _ax3.set_title('Cross-Coherence', **afont)
        _ax3.set_ylabel(r'C$_{xy}$', **afont)
        _ax3.set_xlabel('f[kHz]', **afont)
        if onesided:
            _ax3.set_xlim(0,1.01e-3*freq[-1])
        else:
            _ax3.set_xlim(-1.01e-3*freq[-1],1.01e-3*freq[-1])
        # end if

        # _ax4.text(0.80, 0.90, 'C_{xy}', 'Color', 'k', 'units', 'normalized',
        #          'FontSize', 14, 'FontName', 'Arial')

        _ax4 = _plt.subplot(2, 2, 4, sharex=_ax2)
        _ax4.plot(1e-3*freq, phi_xy, 'k-')
        _ax4.set_title('Cross-Phase', **afont)
        _ax4.set_ylabel(r'$\phi_{xy}$', **afont)
        _ax4.set_xlabel('f[kHz]', **afont)
        if onesided:
            _ax4.set_xlim(0,1.01e-3*freq[-1])
        else:
            _ax4.set_xlim(-1.01e-3*freq[-1],1.01e-3*freq[-1])
        # end if
        # _ax4.text(0.80, 0.90, '\phi_{xy}', 'Color', 'k', 'units',
        #           'normalized', 'FontSize', 14, 'FontName', 'Arial')

        _plt.tight_layout()
        _plt.draw()
        # _plt.show()
    # endif plotit

    return freq, Pxy, Pxx, Pyy, Cxy, phi_xy, fftinfo
# end fft_pwelch


def stft(tt, y_in, tper=1e-3, returnclass=True, **kwargs):

    # Instantiate the fft analysis wrapper class (but don't run it)
    Ystft = fftanal()

    # Initialize the class with settings and variables
    Ystft.init(tt, y_in, tper=1e-3, **kwargs)

    # Perform the loop over averaging windows to generate the short time four. xform
    #   Note that the zero-frequency component is in the middle of the array (2-sided transform)
    freq, Yft = Ystft.stft(Yfft=False)   # frequency [cycles/s], STFT [Navr, nfft]

    if returnclass:
        return Ystft
    else:
        return freq, Yft
    # end if
# end if

#    if detrend is None: detrend = detrend_mean   # endif
#    if win is None:  win = _np.hamming(nwins)   # endif
##    if
#    if nfft is None: nfft=nwins  # endif
#
#    Yfft = _np.zeros((Navr, nfft), dtype=_np.complex128)
#
#    ist = _np.arange(Navr)*(nwins - noverlap)
#    ist = ist.astype(int)
#    for gg in _np.arange(Navr):
#        istart = ist[gg]     # Starting point of this window
#        iend = istart+nwins  # End point of this window
#
#        ytemp = y_in[istart:iend]
#
#        # Windowed signal segment
#        # To get the most accurate spectrum, minimally detrend
#        ytemp = win*detrend(ytemp)
#        # xtemp = win*_dsp.detrend(x_in[istart:iend], type='constant')
#        # ytemp = win*_dsp.detrend( y_in[istart:iend], type='constant' )
#
#        # The FFT output from matlab isn't normalized:
#        # y_n = sum[ y_m.*exp( 2_np.pi*1i*(n/N)*m ) ]
#        # The inverse is normalized::
#        # y_m = (1/N)*sum[ y_n.*exp( -2_np.pi*1i*(n/N)*m ) ]
#        #
#        # Python normalizations are optional, pick it to match MATLAB
#        Yfft[gg, 0:nfft] = _np.fft.fft(ytemp, n=nfft)
#    #endfor loop over fft windows
#
#    return freq, YFFT
# end def

# =========================================================================== #
# =========================================================================== #

#def getwindowfunction(windowfunction='None', nwins=None, periodic=False, verbose=False):
#    # Define windowing function for apodization
#    if windowfunction.lower() == 'hamming':
#        if verbose:
#            print('Using a Hamming window function')
#        # endif verbose
#        win = _np.hamming(nwins)  # periodic hamming window?
#        name = 'hamming'
#    elif windowfunction.lower() == 'hanning':
#        if verbose:
#            print('Using a Hann window function')
#        # endif verbose
#        win = _np.hanning(nwins)  # periodic hann window?
#        name = 'hanning'
#    elif windowfunction.lower() == 'blackman':
#        if verbose:
#            print('Using a Blackman type window function')
#        # endif verbose
#        win = _np.blackman(nwins)  # periodic blackman window?
#        COLA=[2.0/3.0, 3.0/4.0, 4.0/5.0, 5.0/6.0, 6.0/7.0, 8.0/9.0, 9.0/10.0] # etc
#    elif windowfunction.lower().find('bart')>-1 and windowfunction.lower().find('han')>-1:
#        if verbose:
#            print('Using a Bartlett-Hann type window function')
#        # endif verbose
#        COLA = [1.0/2.0, 3.0/4.0, 5.0/6.0, 7.0/8.0, 9.0/10.0, 11.0/12.0, 13.0/14.0] # etc.
#        win = _np.barthann( (nwins,), dtype=_np.float64)
#    elif windowfunction.lower().find('bart')>-1:   #  == 'bartlett':
#        if verbose:
#            print('Using a Bartlett type window function')
#        # endif verbose
#        name = 'bartlett'
#        COLA = [1.0/2.0, 3.0/4.0, 5.0/6.0, 7.0/8.0, 9.0/10.0, 11.0/12.0, 13.0/14.0] # etc.
#        win = _np.bartlett( (nwins,), dtype=_np.float64)
#    elif windowfunction.lower().find('tukey')>-1:
#        if verbose:
#            print('Using a Tukey type window function')
#        # endif verbose
#        name = 'tukey'
#        COLA = [3.0/4.0, 5.0/6.0, 7.0/8.0, 9.0/10.0, 11.0/12.0, 13.0/14.0] # etc.
##        win = _np.tukey( alpha=0.5, (nwins,), dtype=_np.float64)
#    else:
#        if verbose:
#            print('Using a rectangular window function')
#        # endif verbose
#        # No window function (actually a box-window)
#        name = 'rect'
#        COLA = [0.0, 1.0/2.0, 2.0/3.0, 3.0/4.0, 4.0/5.0, 5.0/6.0] # etc.
#        win = _np.ones( (nwins,), dtype=_np.float64)
#    # endif windowfunction.lower()
#    if periodic:
#         win = win[0:-1]  # truncate last point to make it periodic
#    # end if
#    rov = {'rect':0.0, 'welch':29.3, 'bartlett':50.0, 'hanning':50.0,
#           'hamming':50.0, 'nutall3':64.7, 'nuttall4':70.5, 'nuttall3a':61.2,
#           'kaiser3':61.9, 'nuttall3b':59.8, 'nuttall4a':68.0, 'bh92':66.1,
#           'nuttall4b':66.3, 'kaiser4':67.0, 'nuttall4c':65.6, 'kaiser5':70.5,
#           'sft3f':66.7, 'sft3m':65.5, 'ftni':65.6, 'sft4f':75.0, 'sft5f':78.5,
#           'sft4m':72.1, 'fthp':72.3, 'hft70':72.2, 'ftsrs':75.4, 'sft5m':76.0,
#           'hft90d':76.0,'hft95':75.6, 'hf5116d':78.2, 'hft144d':79.9, 'hft169d':81.2,
#           'hft196d':82.3, 'hft223d':83.3, 'hf5248d':84.1}
#    try:
#        rov = rov[name]
#    except:
#        rov = COLA[0]
#    return win, rov

def fft_pmlab(sig1,sig2,dt,plotit=False):
    #nfft=2**_mlab.nextpow2(np.length(sig1))
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

# =========================================================================== #

def _preconvolve_fft(a, b):
    a = _np.asarray(a)
    b = _np.asarray(b)
    if _np.prod(a.ndim) > 1 or _np.prod(b.ndim) > 1:
        raise ValueError('Can only vectorize vectors')
    if len(b) > len(a):
        a, b = b, a
    n = len(a)
    # Pad vector
    c = _np.hstack((_np.zeros(n/2), b, _np.zeros(n/2 + len(a) - len(b) + 1)))
    return c

def convolve_fft(a, b, mode='valid'):
    """
    Convolution between two 1D signals.

    Parameters
    ----------
    a : np.array, shape(n)
    b : np.array, shape(m)
        If len(b) > len(a), a, b = b, a

    Output
    ------
    r : np.array
    """
    c = _preconvolve_fft(a, b)
    # Convolution of signal:
    return _dsp.fftconvolve(c, a, mode=mode)

def cross_correlation_fft(a, b, mode='valid'):
    """
    Cross correlation between two 1D signals. Similar to np.correlate, but
    faster.

    Parameters
    ----------
    a : np.array, shape(n)
    b : np.array, shape(m)
        If len(b) > len(a), a, b = b, a

    Output
    ------
    r : np.array
        Correlation coefficients. Shape depends on mode.
    """
    c = _preconvolve_fft(a, b)
    # Convolution of reverse signal:
    return _dsp.fftconvolve(c, a[::-1], mode=mode)


def align_signals(a, b):
    """Finds optimal delay to align two 1D signals
    maximizes hstack((zeros(shift), b)) = a

    Parameters
    ----------
    a : np.array, shape(n)
    b : np.array, shape(m)

    Output
    ------
    shift : int
        Integer that maximizes hstack((zeros(shift), b)) - a = 0
    """
    # check inputs
    a = _np.asarray(a)
    b = _np.asarray(b)
    if _np.prod(a.ndim) > 1 or _np.prod(b.ndim) > 1:
        raise ValueError('Can only vectorize vectors')
    # longest first
    sign = 1
    if len(b) > len(a):
        sign = -1
        a, b = b, a
    r = cross_correlation_fft(a, b)
    shift = _np.argmax(r) - len(a) + len(a) / 2
    # deal with odd / even lengths (b doubles in size by cross_correlation_fft)
    if len(a) % 2 and len(b) % 2:
        shift += 1
    if len(a) > len(b) and len(a) % 2 and not(len(b) % 2):
        shift += 1
    return sign * shift

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

# =========================================================================== #
# =========================================================================== #

def ccf(x1,x2,fs):
    """
    Return the cross-correlation function and lags betwen two signals (x1, x2)
    - a little slower than cross_correlation_fft, but also returns time-lags
    """
    npts=len(x1)
    lags=_np.arange(-npts+1,npts)
    tau=-lags/float(fs)         # time-lags in input scales
    ccov = _np.correlate(x1-x1.mean(), x2-x2.mean(), mode='full') # cross-covariance
    co = ccov / (npts * x1.std() * x2.std())  # normalized cross-covariance
    return tau, co


def ccf_sh(x1, x2, fs, nav):
    """
    Returns the average cross-correlation within a sliding window
    inputs:
        x1 - data-series 1
        x2 - data-series 2
        fs - sampling frequency for scaling index-lags to time-lags
        nav - window length for each time-window (in samples)
    outputs:
        tau - time-lags
        csh - average cross-correlatin between x1 and x2
    """
    _, xs1, _ =_ut.sliding_window_1d(x1,x1,nav,ss=None)
    _, xs2, _ =_ut.sliding_window_1d(x1,x2,nav,ss=None)

    # Calculate the cross-correlation and time-lag between the time-series
    # within each window along the time-series data
    co=_np.zeros((2*len(xs1)-1,nav))
    for ii in range(0, nav):
        tau, co[:,ii]=ccf(xs1[:,ii],xs2[:,ii],fs)
    # end for

    # The average cross-correlation within each sliding window
    csh=_np.mean(co,1)
    return tau, csh


def ccf_test():
    fs=1e5
    N=2048
    f=1e3
    phi=50*_np.pi/180        #phase lag phi>0 : x2 lags behind, phi<0 : x2 is ahead
    t=_np.arange(0,N)*1./fs
    x1=_np.sin(2*_np.pi*f*t)+_np.random.normal(0,1,N)
    x2=_np.sin(2*_np.pi*f*t+phi)+_np.random.normal(0,1,N)
    tau,co=ccf(x1,x2,fs)
    print('expect max at t=%2.3f us' % (-phi/(2*_np.pi*f)*1e6))
    _plt.figure(1)
    _plt.clf()
    _plt.subplot(2,1,1)
    _plt.plot(t,x1,t,x2)
    _plt.legend(['x1','x2'])
    _plt.subplot(2,1,2)
    _plt.plot(tau*1e6,co)
    _plt.show()

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
# end def integratespectra


def getNpeaks(Npeaks, tvec, sigx, sigy, **kwargs):
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
        # end if
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

def cog(x,fs, fmin=None, fmax=None):
    """
    Center of gravity of data from PSD of input data
    - power spectral density weighted average of frequency
    """
    if fmax is None: fmax = fs  # end if
    n=len(x)
    freq=_np.fft.fftshift(_np.fft.fftfreq(n,1/fs))
    spec=_np.fft.fftshift(_np.fft.fft(x))/_np.sqrt(n/2)
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
#        freq=_np.fft.fftshift(_np.fft.fftfreq(n,1./self.fs))
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

def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = _np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = _la.eigh(R)
    # sort eigenvalue in decreasing order
    idx = _np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return _np.dot(evecs.T, data.T).T, evals, evecs

def test_PCA(data, dims_rescaled_data=2):
    '''
    test by attempting to recover original data array from
    the eigenvectors of its covariance matrix & comparing that
    'recovered' array with the original data
    '''
    m, n = data.shape
    _ , _ , eigenvectors = PCA(data, dim_rescaled_data=2)
    data_recovered = _np.dot(eigenvectors, m).T
    data_recovered += data_recovered.mean(axis=0)
    assert _np.allclose(data, data_recovered)


def plot_pca(data):
    clr1 =  '#2026B2'
    fig = _plt.figure()
    ax1 = fig.add_subplot(111)
    data_resc, data_orig = PCA(data)
    ax1.plot(data_resc[:, 0], data_resc[:, 1], '.', mfc=clr1, mec=clr1)
    _plt.show()

# =========================================================================== #

def monticoh(Pxy, varPxy, Pxx, varPxx, Pyy, varPyy, nmonti=1000, meansquared=True):

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

    Pxy_s = Pxy.copy()
    varPxy_s = varPxy.copy()

    Pxx_s = Pxx.copy()
    varPxx_s = varPxx.copy()

    Pyy_s = Pyy.copy()
    varPyy_s = varPyy.copy()

    g2 = _np.zeros( (nmonti, _np.size(Pxy,axis=0), _np.size(Pxy, axis=1)), dtype=float)
    for ii in range(nmonti):

       Pxy = Pxy_s + _np.sqrt(varPxy_s)*_np.random.normal(0.0, 1.0, len(Pxy))
       Pxx = Pxx_s + _np.sqrt(varPxx_s)*_np.random.normal(0.0, 1.0, len(Pxx))
       Pyy = Pyy_s + _np.sqrt(varPyy_s)*_np.random.normal(0.0, 1.0, len(Pyy))

       g2[ii] = _np.abs( Pxy*_np.conj( Pxy ) )/( _np.abs(Pxx)*_np.abs(Pyy) )
    # end for
    varg2 = _np.nanvar(g2, axis=0)
    g2 = _np.nanmean(g2, axis=0)

    if meansquared:
        return g2.reshape(sh), varg2.reshape(sh)
    else:
        return _np.sqrt(g2.reshape(sh)), _np.sqrt(varg2.reshape(sh))
    # end if
# end def


def varcoh(Pxy, varPxy, Pxx, varPxx, Pyy, varPyy, meansquared=True):
    # function [Coh,varCoh]=varcoh(Pxy,varPxy,Pxx,varPxx,Pyy,varPyy)
    # Only works when varPxy was formed by separating real and imaginary
    # components.  As is done in fft_pwelch.m

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

#        if meansquared is False:
#            # Return the coherence, not the mean-squared coherence
#            varCoh = 0.25*varCoh/Coh  # (0.5*(Coh**-0.5))**2.0 * varCoh
#            Coh = _np.sqrt(Coh)
#        # endif
    else:  # return the complex coherence
        Coh = Pxy / _np.sqrt( _np.abs(Pxx)*_np.abs(Pyy) )
#        vardenom = ...
#        varCoh = Coh**2.0*( varPxy +

        varCoh = Coh**2*( vc*( 2*mc/( mc**2+ms**2) )**2 + \
                  vs*( 2*ms/( mc**2+ms**2) )**2 + \
                  varPxx*(1/Pxx)**2 + varPyy*(1/Pyy)**2 )
        # Return the coherence, not the mean-squared coherence
        varCoh = 0.25*varCoh/Coh  # (0.5*(Coh**-0.5))**2.0 * varCoh
        Coh = _np.sqrt(Coh)
    # end if

    return Coh, varCoh
# end function varcoh

# ================= #

def montiphi(Pxy, varPxy, nmonti=1000, angle_range=_np.pi):

    nmonti = int(nmonti)

    sh = _np.shape(Pxy)
    Pxy = _np.atleast_2d(Pxy)
    if _np.size(Pxy, axis=0)==1:  Pxy = Pxy.T  # endif

    varPxy = _np.atleast_2d(varPxy)
    if _np.size(varPxy, axis=0)==1:  varPxy = varPxy.T  # endif

    Pxy_s = Pxy.copy()
    varPxy_s = varPxy.copy()
    ph = _np.zeros( (nmonti, _np.size(Pxy,axis=0), _np.size(Pxy, axis=1)), dtype=float)

    for ii in range(nmonti):

       Pxy = Pxy_s + _np.sqrt(varPxy_s)*_np.random.normal(0.0, 1.0, len(Pxy))

       # the angle function computes atan2( 1/n sum(sin(phi)),1/n sum(cos(phi)) )

       if angle_range>0.5*_np.pi:
           ph[ii] = _np.arctan2( _np.imag(Pxy), _np.real(Pxy) )
       else:
           ph[ii] = _np.arctan( _np.imag(Pxy) / _np.real(Pxy) )
       # endif
#       # This function  might not work becasue of wrapping issues?
#       ph[ii] = _np.unwrap(ph[ii])
    # end for
    varph = _np.nanvar(ph, axis=0)
    ph = _np.nanmean(ph, axis=0)

    return ph.reshape(sh), varph.reshape(sh)

def varphi(Pxy_real, Pxy_imag, varPxy_real, varPxy_imag, angle_range=_np.pi):

#def varphi(Pxy, varPxy, angle_range=_np.pi):
#   Pxy_real = _np.real(Pxy)
#   Pxy_imag = _np.imag(Pxy)
#
#   varPxy_real = _np.real(varPxy)
#   varPxy_imag = _np.imag(varPxy)

   # the angle function computes atan2( 1/n sum(sin(phi)),1/n sum(cos(phi)) )
   if angle_range>0.5*_np.pi:
       ph = _np.arctan2( Pxy_imag, Pxy_real )
   else:
       ph = _np.arctan( Pxy_imag / Pxy_real )
   # endif

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

# ========================================================================== #


def mean_angle(phi, vphi=None, dim=0, angle_range=0.5*_np.pi, vsyst=None):
    # Proper way to average a phase angle is to convert from polar (imaginary)
    # coordinates to a cartesian representation and average the components.

   if vphi is None:
       vphi = _np.zeros_like(phi)
   # endif
   if vsyst is None:
       vsyst = _np.zeros_like(phi)
   # endif

   nphi = _np.size(phi, dim)
   complex_phase = _np.exp(1.0j*phi)
   complex_var = vphi*(_np.abs(complex_phase))**2
   complex_vsy = vsyst*(_np.abs(complex_phase))**2

   # Get the real and imaginary parts of the complex phase
   ca = _np.real(complex_phase)
   sa = _np.imag(complex_phase)

   # Take the mean and variance of these components
   # mca = _np.mean(ca, dim)
   # msa = _np.mean(sa, dim)
   #
   # vca = _np.var(ca, dim) + _np.sum(complex_var, dim)/(nphi**2)
   # vsa = _np.var(sa, dim) + _np.sum(complex_var, dim)/(nphi**2)

   mca = _np.nanmean(ca, axis=dim)
   msa = _np.nanmean(sa, axis=dim)

   # Stat error
   vca = _np.nanvar(ca, axis=dim) + _np.nansum(complex_var, axis=dim)/(nphi**2)
   vsa = _np.nanvar(sa, axis=dim) + _np.nansum(complex_var, axis=dim)/(nphi**2)

   # Add in systematic error
   vca += (_np.nansum( _np.sqrt(complex_vsy), axis=dim )/nphi)**2.0
   vsa += (_np.nansum( _np.sqrt(complex_vsy), axis=dim )/nphi)**2.0

   mean_phi, var_phi = varphi(Pxy_real=mca, Pxy_imag=msa,
                              varPxy_real=vca, varPxy_imag=vsa, angle_range=angle_range)
   return mean_phi, var_phi
# end mean_angle

#   # the angle function computes atan2( 1/n sum(sin(phi)),1/n sum(cos(phi)) )
#   if angle_range > 0.5*_np.pi:
#       mean_phi = _np.arctan2(msa, mca)
#   else:
#       mean_phi = _np.arctan(msa/mca)
#   # endif
#
#   # substitute variables and propagate errors into the tangent function
#   tt = msa/mca   # tangent = sin / cos
#   # vt = (tt**2)*( vsa/(msa**2) + vca/(mca**2) )  # variance in tangent
#   vt = vsa/(mca**2) + vca*msa**2/(mca**4)    # variance in tangent
#
#   # the variance of the arctangent function is related to the derivative
#   #  d(arctangent)/dx = 1/(1+x^2)     using a = atan( tan(a) )
#   var_phi = vt/(1+tt**2)**2
#   return mean_phi, var_phi
## end mean_angle


def unwrap_tol(data, scal=_np.pi, atol=None, rtol=None, itol=None):
    if atol is None and rtol is None:       atol = 0.2    # endif
    if atol is None and rtol is not None:   atol = rtol*scal   # endif
    if itol is None: itol = 1 # endif
    tt = _np.asarray(range(len(data)))
    ti = tt[::itol]
    diffdata = _np.diff(data[::itol])/scal
    diffdata = _np.sign(diffdata)*_np.floor(_np.abs(diffdata) + atol)
    data[1:] = data[1:]-_np.interp(tt[1:], ti[1:], scal*_np.cumsum(diffdata))
    return data
#end unwrap_tol


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

def downsample_efficient(u_t, Fs, Fs_new, plotit=False):
    """
     The proper way to downsample a signal.
       First low-pass filter the signal
       Interpolate / Decimate the signal down to the new sampling frequency
    """

    tau = 2/Fs_new
    nt  = len(u_t)
#    tt  = _np.arange(0, nt, 1)/Fs
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
#    ti = _np.arange(0, nt/Fs, 1/Fs_new)

    u_n = _ut.interp(xi=_np.arange(0, nt, 1)/Fs,
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
        _ax1.plot(_np.arange(0, nt/Fs, 1/Fs_new), u_n, 'b-')

        _ax3 = _plt.subplot(3, 1, 3, sharex=_ax1)
        _ax3.plot(_np.arange(0, nt, 1)/Fs, u_t, 'k')
        _ax3.set_ylabel('Filt. Signal', color='k')
        _ax3.set_xlabel('t [s]')
#        _plt.show(hfig, block=False)
        _plt.draw()
#        _plt.show()
    # endif plotit

    return u_n
# end def downsample_efficient

# ========================================================================= #
# ========================================================================= #


def rescale(xx, yy, scaley=True, scalex=True):
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
    # end if
    return xx, yy, (slope, offset, xslope, xoffset)

def unscale(xx, yy, scl, dydx=None):
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

def fft_deriv(sig, xx=None, nfft=None, lowpass=True, modified=True):
    if xx is None:
        N = 101
        xx = 1.0*_np.asarray(range(0, N))
    # end if

    # Scale the data to make it simple to calculate
    xx, sig, scl = rescale(xx, sig, scaley=True, scalex=True)

    N = len(xx)
    dx = xx[1] - xx[0]
    L = N*dx
    if nfft is None: nfft = N # end if

    # Get the wavenumber vector
    k = _np.fft.fftfreq(nfft, d=dx/L)
    k *= 2.0*_np.pi
    if modified:
        # Modified wave number
        # - Windowed witha  sinc function to kill the ringing
        # - Sunaina et al 2018 Eur. J. Phys. 39 065806
        wavenumber = 1.0j*_np.sin(k*dx)/(dx)
    else:
        # Naive fft derivative (subject to ringing)
        wavenumber = 1.0j*k
    # end if
    wavenumber /= L

    # Low pass filter the data before calculating the FFT if requested
    if lowpass:
        if lowpass is True:     lowpass = 0.1*1.0/dx    # end if

        b, a = butter_lowpass(lowpass, fnyq=0.5/dx, order=2)

#        sig = butter_lowpass_filter(sig, cutoff=lowpass, fs=1.0/dx, order=5)
        sig = _dsp.filtfilt(b, a, sig)
    # end if

    # Calculate the derivative using fft
    dsdx = _np.real(_np.fft.ifft(wavenumber*_np.fft.fft(sig, n=nfft), n=nfft))

    # Rescale back to the original data scale
    xx, yy, dsdx = unscale(xx, sig, scl=scl, dydx=dsdx)
    return dsdx
# end def


#def test_fft_deriv(xx=None, nfft=256):
def test_fft_deriv(nfft=512):

    N = 101 #number of points
    L = 2 * _np.pi #interval of data
#    L = 5.3 #interval of data
    dx = L/N
    xx = dx*_np.asarray(range(N))
#    xx = _np.arange(0.0, L, L/float(N)) #this does not include the endpoint
    # end if

#    # Test with a rectangle function
#    yy = _ut.rect(xx/L)
#    dy_analytical = _ut.delta(xx/L+0.5) - _ut.delta(xx/L-0.5)

#    # Test with a gaussian function
#    yy = _np.exp(-0.5*(xx/L)*(xx/L)/(0.25*0.25))
#    dy_analytical = (-1.0*(xx/L)*(1.0/L)/(0.25*0.25))*yy

#    # Test with a line
#    yy = _np.linspace(-1.2, 11.3, num=len(xx), endpoint=True)
#    a = (yy[-1]-yy[0])/(xx[-1]-xx[0])
##    b = yy[0] - a*xx[0]
#    dy_analytical = a*_np.ones_like(yy)

    # Test with a sine
    yy = _np.sin(xx)
    dy_analytical = _np.cos(xx)

    #add some random noise
    yy += 0.10*(_np.nanmax(yy)-_np.nanmin(yy))*_np.random.random(size=xx.shape)

    dydt = fft_deriv(yy, xx, nfft=len(xx))

    _plt.figure()
    _plt.plot(xx, yy, '-', label='function')
    _plt.plot(xx, dy_analytical, '-', label='analytical der')
    _plt.plot(xx, dydt, '-', label='fft der')
    _plt.legend(loc='lower left')

#    _plt.savefig('images/fft-der.png')
    _plt.show()
# end def test_fft_deriv()

# ========================================================================= #
# ========================================================================= #

def butter_bandpass(x,fs=4e6,lf=1000,hf=500e3,order=3,disp=0):
    nyq=0.5*fs
    low=lf/nyq
    high=hf/nyq
    b,a = _dsp.butter(order,[low, high], btype='band')
    y = _dsp.lfilter(b,a,x)
    w,h=_dsp.freqz(b,a,worN=2000)
    #_plt.plot(fs*0.5/_np.pi*w,_np.abs(h))
    #_plt.show()
    return y

#This is a lowpass filter design subfunction
def butter_lowpass(cutoff, fnyq, order=5):
    normal_cutoff = cutoff / fnyq
    # b, a = _dsp.butter(order, normal_cutoff, btype='low', analog=False)
    b, a = _dsp.butter(order, normal_cutoff, btype='low')
    return b, a
#end butter_lowpass

#This is a subfunction for filtering the data
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = _dsp.lfilter(b, a, data)
    return y
#end butter_lowpass_filter

def complex_filtfilt(filt_n,filt_d,data):
    # dRR = np.mean(data.real)+_dsp.filtfilt(filt_n, filt_d, data.real-np.mean(data.real) ) #LPF injected signal
    # dII = np.mean(data.imag)+_dsp.filtfilt(filt_n, filt_d, data.imag-np.mean(data.imag) ) #LPF injected signal
    dRR = _dsp.filtfilt(filt_n, filt_d, data.real ) #LPF injected signal
    dII = _dsp.filtfilt(filt_n, filt_d, data.imag ) #LPF injected signal
    data = dRR+1j*dII
    return data
#end complex_filtfilt

def Cxy_Cxy2(Pxx, Pyy, Pxy, ibg=None): #, thresh=1.e-6):
    Pxx = Pxx.copy()
    Pyy = Pyy.copy()
    Pxy = Pxy.copy()

##    Pxx[Pxx<thresh] = _np.nan
##    Pyy[Pyy<thresh] = _np.nan
#    Pxy[Pxy<thresh] = 0.0
#    Pxy[_np.abs(Pxy*_np.conj(Pxy))<thresh*_np.abs(Pxx)*_np.abs(Pyy)] = 0.0

#    rotmat = 0
#    sh = Pxx.shape
#    Pxx = _np.atleast_2d(Pxx.copy())
#    Pyy = _np.atleast_2d(Pyy.copy())
#    Pxy = _np.atleast_2d(Pxy.copy())
#    if _np.size(Pxx, axis=0) == 1:
#        rotmat = 1
#        Pxx = Pxx.T
#        Pyy = Pyy.T
#        Pxy = Pxy.T
#    Cxy2 = _np.zeros_like(Pxy)
#    Cxy = _np.zeros_like(Pxy)
#
#    ithresh = _np.where(_np.abs(Pxx*Pyy)>thresh*thresh)[0]
#    Pxx = Pxx[ithresh]
#    Pyy = Pyy[ithresh]
#    Pxy = Pxy[ithresh]

    # Mean-squared coherence
    Pxx = _np.atleast_2d(Pxx)
    if _np.size(Pxx, axis=1) != _np.size(Pyy, axis=1):
        Pxx = Pxx.T*_np.ones( (1, _np.size(Pyy, axis=1)), dtype=Pxx.dtype)
    # end if
    Cxy2 = _np.abs( Pxy*_np.conj( Pxy ) )/( _np.abs(Pxx)*_np.abs(Pyy) )
#    Cxy2 = _np.abs( Pxy*_np.conj( Pxy ) )/( _np.abs(Pxx*Pyy) )
#    Cxy = _np.sqrt(Cxy2) # RMS coherence

    # Complex coherence
    Cxy = Pxy/_np.sqrt( _np.abs(Pxx)*_np.abs(Pyy) )
#    Cxy = Pxy/_np.sqrt( _np.abs(Pxx*Pyy) )

#    if rotmat:
#        Cxy = Cxy.T
#        Cxy2 = Cxy2.T
#    # end if
#    Cxy = Cxy.reshape(sh)
#    Cxy2 = Cxy2.reshape(sh)

    if ibg is None:
        return Cxy, Cxy2

    # Imaginary coherence
    iCxy = _np.imag(Cxy)/(1.0-_np.real(Cxy))

    # Background subtracted coherence
    Cprime = _np.real(Cxy-_np.mean(Cxy[:,ibg], axis=-1)) \
        /(1.0-_np.real(Cxy-_np.mean(Cxy[:,ibg], axis=-1)))
    return iCxy, Cprime

# ========================================================================= #
# ========================================================================= #

#class fftmch(fftanal):

class fftanal(Struct):

    afont = {'fontname':'Arial','fontsize':14}
    def __init__(self, tvec=None, sigx=None, sigy=None, **kwargs):

        self.verbose = kwargs.get( 'verbose', True)
        if tvec is None or sigx is None:
            if self.verbose:
                print('Please give at least a time-vector [s]'
                      + ' and a signal vector [a.u.]')
            # end if
            return
        else:
            self.init(tvec, sigx, sigy, **kwargs)
            self.fftpwelch()
        # endif
    # end __init__

    def init(self, tvec=None, sigx=None, sigy=None, **kwargs):
        if sigy is None or sigx is sigy:
            self.nosigy = True
        else:
            self.nosigy = False
        #endif

        self.tvec = tvec
        self.sigx = sigx
        self.sigy = sigy

        # == #

        self.tbounds = kwargs.get( 'tbounds', [ tvec.min(), tvec.max() ] )
        self.useMLAB = kwargs.get( 'useMLAB', False )
        self.plotit  = kwargs.get( 'plotit',  False)
        self.verbose = kwargs.get( 'verbose', True)
        self.Navr    = kwargs.get( 'Navr', 8)
        self.overlap = kwargs.get( 'windowoverlap', 0.5)
        self.window  = kwargs.get( 'windowfunction', 'hamming')
        self.tvecy   = kwargs.get( 'tvecy', None)
        self.onesided = kwargs.get('onesided', True)
        self.detrendstyle = kwargs.get('detrend', 1) # >0 mean, 0 None, <0 linear
        self.frange = kwargs.get('frange', None)
        self.axes = kwargs.get('axes', -1)

        if self.tvecy is not None:
            self.tvec, self.sigx, self.sigy = self.resample(tvec, sigx, self.tvecy, sigy)
        # end if

        self.Fs = self.__Fs__(self.tvec)
        self.ibounds = self.__ibounds__(self.tvec, self.tbounds)
        self.nsig = _np.size( self.__trimsig__(tvec, self.ibounds) )

        # if the window time is specified ... overwrite nwins, noverlap and Navr
        if 'tper' in kwargs:
            self.tper = kwargs['tper']
            self.nwins = int(self.Fs*self.tper)
            self.noverlap = self.getNoverlap()
            self.Navr = self.getNavr()
        else:
            self.nwins = self.getNwins()
            self.noverlap = self.getNoverlap()
        # end if
        self.win = self.makewindowfn(self.window, self.nwins, self.verbose)
        self.getNnyquist()
        self.getNorms()
    # end def init

    def update(self, d=None):
        if d is not None:
            if type(d) != dict:    d = d.dict_from_class()     # endif
            super(fftanal, self).__init__(d)
        # endif
    # end def update

    # =========================================================== #

    def fftpwelch(self):
        # Call the fft_pwelch function that is defined above
        self.freq, self.Pxy, self.Pxx, self.Pyy, \
        self.Cxy, self.phi_xy, self.fftinfo = \
            fft_pwelch(self.tvec, self.sigx, self.sigy, self.tbounds,
                       Navr=self.Navr, windowoverlap=self.overlap,
                       windowfunction=self.window, useMLAB=self.useMLAB,
                       plotit=self.plotit, verbose=self.verbose,
                       detrend_style=self.detrendstyle, onesided=self.onesided)
        self.update(self.fftinfo)

    def stft(self):
        if self.useMLAB:
            if not self.onesided or (type(self.onesided)==type('') and self.onesided.find('two')>-1):
                onesided = False
            elif self.onesided or (type(self.onesided)==type('') and self.onesided.find('one')>-1):
                onesided = True
            # end if
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
        # end if

    def pwelch(self):
        self.Xstft()
        if not self.nosigy:
            self.Ystft()
        self.Pstft()
        self.averagewins()

    # =============== #

    def crosscorr(self, fbnds=None):
        # cross correlation from the FFT
        #     if we calculated one-sided spectra, then we have to add back
        #     in the Nyquist components before inverse tranforming.

        nfft = self.nwins
        inds = _np.ones( (nfft,), dtype=bool)

        if hasattr(self,'Pxx'):
            Pxx_seg = self.Pxx_seg.copy()
            Pxx = self.Pxx.copy()

            if self.onesided:
                # remaking the two-sided spectra for the auto-power
                i0 = int(len(Pxx)+1)
                inds[i0] = False
                Pxx[1:-1] *= 0.5
                Pxx_seg[:,1:-1] *= 0.5
                Pxx = _ut.cylsym_even(Pxx)
                Pxx_seg = _ut.cylsym_even(Pxx_seg.T).T
            # endif
            Pxx = Pxx[inds]
            Pxx_seg = Pxx_seg[:, inds]

            Pxx = _np.fft.ifftshift(Pxx)
            Pxx_seg = _np.fft.ifftshift(Pxx_seg, axes=-1)

        if hasattr(self,'Pyy'):
            Pyy_seg = self.Pyy_seg.copy()
            Pyy = self.Pyy.copy()

            if self.onesided:
                # remaking the two-sided spectra for the auto-power
                i0 = int(len(Pyy)+1)
                inds[i0] = False
                Pyy[1:-1] *= 0.5
                Pyy_seg[:,1:-1] *= 0.5
                Pyy = _ut.cylsym_even(Pyy)
                Pyy_seg = _ut.cylsym_even(Pyy_seg.T).T
            # endif
            Pyy = Pyy[inds]
            Pyy_seg = Pyy_seg[:, inds]

            Pyy = _np.fft.ifftshift(Pyy)
            Pyy_seg = _np.fft.ifftshift(Pyy_seg, axes=-1)

        if hasattr(self,'Pxy'):
            Pxy_seg = self.Pxy_seg.copy()
            Pxy = self.Pxy.copy()

            if self.onesided:
                # remaking the two-sided spectra for the cross-power ...
                #  this doesn't work if the input signals were complex!
                i0 = int(len(Pxy)+1)
                inds[i0] = False
                Pxy[1:-1] *= 0.5   # is this right?
                Pxy_seg[:,1:-1] *= 0.5
                Pxy = _ut.cylsym_even(Pxy)
                Pxy_seg = _ut.cylsym_even(Pxy_seg.T).T
            # endif
            Pxy = Pxy[inds]
            Pxy_seg = Pxy_seg[:, inds]

            Pxy = _np.fft.ifftshift(Pxy)
            Pxy_seg = _np.fft.ifftshift(Pxy_seg, axes=-1)

            # ==== #

            Cxy = self.Cxy.copy()
            Cxy_seg = self.Cxy_seg.copy()
            if self.onesided:
                i0 = int(len(Cxy)+1)
                inds[i0] = False
                Cxy = _ut.cylsym_even(Cxy)
                Cxy_seg = _ut.cylsym_even(Cxy_seg.T).T
            # endif
            Cxy = Cxy[inds]
            Cxy_seg = Cxy_seg[:, inds]

            Cxy = _np.fft.ifftshift(Cxy)
            Cxy_seg = _np.fft.ifftshift(Cxy_seg, axes=-1)
        # end if

        # ==================== #
        freq = _np.fft.fftfreq(self.nwins, 1.0/self.Fs)
        freq = _np.fft.fftshift(freq, axes=-1)

        if fbnds is None:
            ibnds = [0,-1]
            nfft = self.nwins
        else:
            ibnds = self.__ibounds__(self.freq, fbnds)
            nfft = len(self.freq[ibnds])
        # end if

#        nfft -= sum(~inds)
        mult = 1.0     # TODO: get these normalizations right, or figure out what is wrong
        mult *= 0.5
        mult *= 0.5
        mult *= nfft
#        mult *= 2.0
#        mult *= self.Fs

#        mult *= self.ENBW
#        mult *= self.ENBW
#        mult *= self.S1**2.0
        mult *= self.S2
        print(self.S1, self.S2, self.S1**2.0/self.S2, self.ENBW, nfft, self.Fs)

        if hasattr(self,'Pxx'):
            self.Rxx_seg = _np.fft.ifft(Pxx_seg[:,ibnds], n=nfft, axis=-1)
            self.Rxx = _np.fft.ifft(Pxx[ibnds], n=nfft)

            self.Rxx_seg *= mult
            self.Rxx *= mult

            self.Rxx_seg = _np.fft.fftshift(self.Rxx_seg, axes=-1)
            self.Rxx = _np.fft.fftshift(self.Rxx)

        if hasattr(self,'Pyy'):
            self.Ryy_seg = _np.fft.ifft(Pyy_seg[:,ibnds], n=nfft, axis=-1)
            self.Ryy = _np.fft.ifft(Pyy[ibnds], n=nfft)

            self.Ryy_seg *= mult
            self.Ryy *= mult

            self.Ryy_seg = _np.fft.fftshift(self.Ryy_seg, axes=-1)
            self.Ryy = _np.fft.fftshift(self.Ryy)

        if hasattr(self,'Pxy'):
            self.Rxy_seg = _np.fft.ifft(Pxy_seg[:,ibnds], n=nfft, axis=-1)
            self.Rxy = _np.fft.ifft(Pxy[ibnds], n=nfft)

            self.Rxy_seg *= mult
            self.Rxy *= mult

            self.Rxy_seg = _np.fft.fftshift(self.Rxy_seg, axes=-1)
            self.Rxy = _np.fft.fftshift(self.Rxy)

            # ======= #
            Pnorm = _np.dot(_np.sqrt(self.Xpow*self.Ypow).reshape((self.Navr,1)), _np.ones((1,nfft), dtype=float))
            Pnorm *= nfft
            Pnorm *= 0.5
            Pnorm *= self.S2
            self.corrcoef_seg = self.Rxy_seg.copy()/Pnorm
            Pnorm = _np.sqrt(_np.mean(self.Xpow*self.Ypow, axis=0))
            Pnorm *= nfft
            Pnorm *= 0.5
            Pnorm *= self.S2
            self.corrcoef = self.Rxy.copy()/Pnorm

#            self.corrcoef_seg = _np.fft.ifft(Cxy_seg, n=nfft, axis=1)
#            self.corrcoef = _np.fft.ifft(Cxy, n=nfft)
#
#            self.corrcoef_seg = _np.fft.fftshift(self.corrcoef_seg, axes=1)
#            self.corrcoef = _np.fft.fftshift(self.corrcoef)
        # end if

        self.lags = _np.asarray(_np.arange(-nfft//2, nfft//2), dtype=_np.float64)
        self.lags /= -1*float(self.Fs)

#        self.lags = _np.linspace(-nfft/self.Fs, nfft/self.Fs, nfft)
    # end def

    # =============== #

    def Xstft(self):
        # Perform the loop over averaging windows to generate the short time four. xform
        #   Note that the zero-frequency component is in the middle of the array (2-sided transform)
        sig = self.__trimsig__(self.sigx, self.ibounds)
        tvec = self.__trimsig__(self.tvec, self.ibounds)

        self.tseg, self.freq, self.Xseg, self.Xpow = self.fft_win(sig, tvec)   # frequency [cycles/s], STFT [Navr, nfft]
        return self.freq, self.Xseg

    def Ystft(self):
        # Perform the loop over averaging windows to generate the short time four. xform
        #   Note that the zero-frequency component is in the middle of the array (2-sided transform)
        sig = self.__trimsig__(self.sigy, self.ibounds)
        tvec = self.__trimsig__(self.tvec, self.ibounds)
        self.tseg, self.freq, self.Yseg, self.Ypow = self.fft_win(sig, tvec)   # frequency [cycles/s], STFT [Navr, nfft]

        #self.tseg = self.tbounds[0]+(self.arange(self.Navr)+0.5)*self.tper
        return self.freq, self.Yseg

    def Pstft(self):
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
#            self.Cxy2_seg = _np.abs( self.Pxy_seg*_np.conj( self.Pxy_seg ) )/( _np.abs(self.Pxx_seg)*_np.abs(self.Pyy_seg) )
##            self.Cxy_seg = _np.sqrt(self.Cxy2_seg)   # RMS coherence
#
#            # Complex coherence
#            self.Cxy_seg = self.Pxy_seg/_np.sqrt( _np.abs(self.Pxx_seg)*_np.abs(self.Pyy_seg) )
#
            self.Cxy_seg, self.Cxy2_seg = Cxy_Cxy2(self.Pxx_seg, self.Pyy_seg, self.Pxy_seg)
    # end def

    # =============== #

    def averagewins(self):
        if hasattr(self, 'Pxx_seg'):
            self.Pxx = _np.mean(self.Pxx_seg, axis=0)

            # use the RMS for the standard deviation
#            self.varPxx = _np.mean(self.Pxx_seg**2.0, axis=0)

            # Else use the normal statistical estimate:
            self.varPxx = (self.Pxx/_np.sqrt(self.Navr))**2.0

        if hasattr(self, 'Pyy_seg'):
            self.Pyy = _np.mean(self.Pyy_seg, axis=0)

            # use the RMS for the standard deviation
#            self.varPyy = _np.mean(self.Pyy_seg**2.0, axis=0)

            # Else use the normal statistical estimate:
            self.varPyy = (self.Pyy/_np.sqrt(self.Navr))**2.0

        if hasattr(self, 'Pxy_seg'):
            self.Pxy = _np.mean(self.Pxy_seg, axis=0)

#            # Mean-squared coherence
#            self.Cxy2 = _np.abs( self.Pxy*_np.conj( self.Pxy ) )/( _np.abs(self.Pxx)*_np.abs(self.Pyy) )
##            self.Cxy = _np.sqrt(self.Cxy2) # RMS coherence
#
#            # Complex coherence
#            self.Cxy = self.Pxy/_np.sqrt( _np.abs(self.Pxx)*_np.abs(self.Pyy) )
#
            self.Cxy, self.Cxy2 = Cxy_Cxy2(self.Pxx, self.Pyy, self.Pxy)
            # ========================== #
            # Uncertainty and phase part

            # derived using error propagation from eq 23 for gamma^2 in
            # J.S. Bendat, Journal of Sound an Vibration 59(3), 405-421, 1978
            # fftinfo.varCxy = _np.zeros_like(Cxy)
            self.varCxy = ((1-self.Cxy2)/_np.sqrt(2*self.Navr))**2.0
            self.varCxy2 = 4.0*self.Cxy2*self.varCxy # d/dx x^2 = 2 *x ... var:  (2*x)^2 * varx

            # Estimate the variance in the power spectra: this requires building
            # a distribution by varying the parameters used in the FFT, nwindows,
            # nfft, windowfunction, etc.  I don't do this right now
#            self.varPxy = self.Pxx*self.Pyy*(1.0-self.Cxy)/self.Navr

            # use the RMS for the standard deviation
#            self.varPxy = _np.mean(self.Pxy_seg**2.0, axis=0)

            # Else use the normal statistical estimate:
            self.varPxy = (self.Pxy/_np.sqrt(self.Navr))**2.0


            # A.E. White, Phys. Plasmas, 17 056103, 2010
            # Doesn't so far give a convincing answer...
            # fftinfo.varPhxy = _np.zeros(Pxy.shape, dtype=_np.float64)
            self.varPhxy = (_np.sqrt(1.0-self.Cxy2)/_np.sqrt(2.0*self.Navr*self.Cxy))**2.0

            self.phi_xy = _np.angle(self.Pxy)        # Save the cross-phase as well
        # end if

    # =============== #

    def convert2amplitudes(self):
        # Linear amplitude spectrum from the power spectral density
        # RMS Linear amplitude spectrum (constant amplitude values)
        #   ONLY WORKS FOR ONE-SIDED SPECTRUM
        if hasattr(self,'Pxx'):
            self.Lxx = _np.sqrt(_np.abs(self.ENBW*self.Pxx))  # [V_rms]
            if self.onesided:
                # Rescale RMS values to Amplitude values (assumes a zero-mean sine-wave)
                # Just the points that split their energy into negative frequencies
                self.Lxx[1:-1] = _np.sqrt(2)*self.Lxx[1:-1]  # [V],

                if self.nfft%2:  # Odd
                    self.Lxx[-1] = _np.sqrt(2)*self.Lxx[-1]
                # endif nfft/2 is odd
            self.varLxx = (self.Lxx**2)*(self.varPxx/_np.abs(self.Pxx)**2)

        if hasattr(self,'Pyy'):
            self.Lyy = _np.sqrt(_np.abs(self.ENBW*self.Pyy))  # [V_rms]
            if self.onesided:
                self.Lyy[1:-1] = _np.sqrt(2)*self.Lyy[1:-1]  # [V],

                if self.nfft%2:  # Odd
                    self.Lyy[-1] = _np.sqrt(2)*self.Lyy[-1]
                # endif nfft/2 is odd
            # end if
            self.varLyy = (self.Lyy**2)*(self.varPyy/_np.abs(self.Pyy)**2)

        if hasattr(self, 'Pxy'):
            self.Lxy = _np.sqrt(_np.abs(self.ENBW*self.Pxy))  # [V_rms]
            if self.onesided:
                self.Lxy[1:-1] = _np.sqrt(2)*self.Lxy[1:-1]  # [V],
                if self.nfft%2:  # Odd
                    self.Lxy[-1] = _np.sqrt(2)*self.Lxy[-1]
                # endif nfft/2 is odd
            # end if
            self.varLxy = (self.Lxy**2)*(self.varPxy/_np.abs(self.Pxy)**2)
    # end def

    # =========================================================== #

    @staticmethod
    def resample(tvx, sigx, tvy, sigy):
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
        return (len(tvec)-1)/(tvec[-1]-tvec[0])

    @staticmethod
    def __ibounds__(tvec, tbounds):
        ib1 = int(_np.floor((tbounds[0]-tvec[0])*fftanal.__Fs__(tvec)))
        ib2 = int(_np.floor(1 + (tbounds[1]-tvec[0])*fftanal.__Fs__(tvec)))
        return [ib1, ib2]

    @staticmethod
    def __trimsig__(sigt, ibounds):
        return sigt[ibounds[0]:ibounds[1]]

    # =========================================================== #

    @staticmethod
    def makewindowfn(windowfunction, nwins, verbose=True):
        #Define windowing function for apodization
        if windowfunction.lower() == 'hamming':
            if verbose:
                print('Using a Hamming window function')
            #endif verbose
            win = _np.hamming(nwins)  # periodic hamming window?
            # win = _np.hamming(nwins+1)  # periodic hamming window
            # win = win[0:-1]  # truncate last point to make it periodic
        elif windowfunction.lower() == 'hanning':
            if verbose:
                print('Using a Hanning window function')
            # endif verbose
            win = _np.hanning(nwins) #periodic hann window?
            # win = _np.hanning(nwins+1)  # periodic hann window
            # win = win[0:-1]  # truncate last point to make it periodic
        elif windowfunction.lower() == 'blackman':
            if verbose:
                print('Using a Blackman type window function')
            # endif verbose
            win = _np.blackman(nwins)  # periodic blackman window?
            # win = win[0:-1]  # truncate last point to make it periodic
        elif windowfunction.lower() == 'bartlett':
            if verbose:
                print('Using a Bartlett type window function')
            # endif verbose
            win = _np.bartlett(nwins)  # periodic Bartlett window?
            # win = win[0:-1]  # truncate last point to make it periodic
        else:
            if verbose:
                print('Defaulting to a box window function')
            # endif verbose
            win = _np.ones( (nwins,), dtype=_np.float64)   # Box-window
            # win = win[0:-1]  # truncate last point to make it periodic
        # endif windowfunction.lower()

        return win

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
    def _getNavr(nsig, nwins, noverlap):
        if nwins>= nsig:
            return int(1)
        else:
            return (nsig-noverlap)//(nwins-noverlap)
    # end def getNavr

    @staticmethod
    def _getNwins(nsig, Navr, noverlap):
        nwins = (nsig-noverlap)//Navr+noverlap
        if nwins>= nsig:
            return nsig
        else:
            return nwins
    # end def getNwins

    @staticmethod
    def _getNoverlap(nsig, nwins, Navr):
        if nwins>= nsig:
            return 0
        else:
            return (nsig-nwins*Navr)//(1-Navr)
        # end if
    # end def getNoverlap

    @staticmethod
    def _getMINoverlap(nsig, nwins, Navr):
        noverlap = 1
        while fftanal._checkCOLA(nsig, nwins, noverlap) == False and noverlap<1e4:
            noverlap += 1
        # end while
        return noverlap

    @staticmethod
    def _getMAXoverlap(nsig, nwins, Navr):
        noverlap = _np.copy(nwins)-1
        while fftanal._checkCOLA(nsig, nwins, noverlap) == False and noverlap>0:
            noverlap -= 1
        # end while
        return noverlap

    @staticmethod
    def _checkCOLA(nsig, nwins, noverlap):
        return (nsig - nwins) % (nwins-noverlap) == 0

    @staticmethod
    def _getNnyquist(nfft):
        Nnyquist = nfft//2 + 1
        if (nfft%2):  # odd
           Nnyquist = (nfft+1)//2
        # end if the remainder of nfft / 2 is odd
        return Nnyquist

    @staticmethod
    def _getS1(win):
        return _np.sum(win)

    @staticmethod
    def _getS2(win):
        return _np.sum(win**2.0)

    @staticmethod
    def _getNENBW(Nnyquist, S1, S2):
        return Nnyquist*1.0*S2/(S1**2)

    @staticmethod
    def _getENBW(Fs, S1, S2):
        return Fs*S2/(S1**2)  # Effective noise bandwidth

    # ========== #

    def getNavr(self):
        self.Navr = self._getNavr(self.nsig, self.nwins, self.noverlap)
        return self.Navr
    # end def getNavr

    def getNwins(self):
        self.nwins = int(_np.floor(self.nsig*1.0/(self.Navr-self.Navr*self.overlap + self.overlap)))
        if self.nwins>=self.nsig:
#            self.nwins = self.nsig.copy()
            self.nwins = self.nsig
        # end if
        return self.nwins
    # end def getNwins

    def getNoverlap(self):
        # Number of points to overlap
        self.noverlap = int( _np.ceil( self.overlap*self.nwins ) )
        return self.noverlap

    def getNnyquist(self):
        self.Nnyquist = self._getNnyquist(self.nwins)
        return self.Nnyquist

    def getNorms(self):
        # Define normalization constants
        self.S1 = self._getS1(self.win)
        self.S2 = self._getS2(self.win)

        # Normalized equivalent noise bandwidth
        self.NENBW = self._getNENBW(self.Nnyquist, self.S1, self.S2)
        self.ENBW = self._getENBW(self.Fs, self.S1, self.S2)
    # end def

    # ===================================================================== #

    def integrate_spectra(self):  # TODO:  CHECK ACCURACY OF THIS!
        self.integrated = Struct()
        [ self.integrated.Pxy_i, self.integrated.Pxx_i, self.integrated.Pyy_i,
          self.integrated.Cxy_i, self.integrated.ph_i, self.integrated.info  ] = \
            integratespectra(self.freq, self.Pxy, self.Pxx, self.Pyy, self.frange,
                             self.varPxy, self.varPxx, self.varPyy)
    # end def

    @staticmethod
    def intspectra(freq, sigft, ifreq=None, ispan=None):
        """
        This function integrates the spectra over a specified range.
        """
        if ifreq is None:
#            xmax  = _np.amax(_np.abs(sigspec), axis=0)
            ifreq = _np.argmax(_np.abs(sigft), axis=0)

            ispan = 6
            ilow = ifreq-ispan//2
            ihigh = ifreq+ispan//2
        elif ispan is None:
            ilow = 0
            ihigh = len(sigft)
        # end
        Isig = _np.trapz(sigft[ilow:ihigh], freq[ilow:ihigh], axis=0)
        Ivar = _np.zeros_like(Isig)
        return Isig, Ivar

    # ===================================================================== #

    def detrend(self, sig):
        if self.detrendstyle>0:
            detrender = detrend_mean
        elif self.detrendstyle<0:
            detrender = detrend_linear
        else:
            detrender = detrend_none
        # endif
        return detrender(sig)
    # end def

    def fft(self, sig, nfft=None, axes=None):
        #The FFT output from matlab isn't normalized:
        # y_n = sum[ y_m.*exp( 2_np.pi*1i*(n/N)*m ) ]
        # The inverse is normalized::
        # y_m = (1/N)*sum[ y_n.*exp( -2_np.pi*1i*(n/N)*m ) ]
        #
        # Python normalizations are optional, pick it to match MATLAB
        if axes is None: axes = self.axes # end if
        if nfft is None: nfft = self.nfft # end if
        return _np.fft.fft(sig, n=nfft, axis=axes)

    def ifft(self, sig, nfft=None, axes=None):
        #The FFT output from matlab isn't normalized:
        # y_n = sum[ y_m.*exp( 2_np.pi*1i*(n/N)*m ) ]
        # The inverse is normalized::
        # y_m = (1/N)*sum[ y_n.*exp( -2_np.pi*1i*(n/N)*m ) ]
        #
        # Python normalizations are optional, pick it to match MATLAB
        if axes is None: axes = self.axes # end if
        if nfft is None: nfft = self.nfft # end if
        return _np.fft.ifft(sig, n=nfft, axis=axes)

    def fftshift(self, sig, axes=None):
       if axes is None: axes = self.axes # end if
       return _np.fft.fftshift(sig, axes=axes)

    def ifftshift(self, sig, axes=None):
       if axes is None: axes = self.axes # end if
       return _np.fft.ifftshift(sig, axes=axes)

    def fft_win(self, sig, tvec=None):
        x_in = sig.copy()
        if tvec is None:
            tvec = _np.linspace(0.0, 1.0, len(x_in))
        # endif
        win = self.win
        nwins = self.nwins
        Navr = self.Navr
        noverlap = self.noverlap
        Fs = self.Fs
        Nnyquist = self.Nnyquist
        nfft  = nwins

        # Define normalization constants
        S1 = self.S1
        S2 = self.S2

        # Equivalent noise bandwidth
        ENBW = self.ENBW

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
#            print(tvec[iend]-tvec[istart])
            tt[gg] = _np.mean(tvec[istart:iend])
#            print(tt[gg])
            xtemp = x_in[istart:iend]

            # Windowed signal segment:
            # - To get the most accurate spectrum, background subtract
            # xtemp = win*_dsp.detrend(xtemp, type='constant')
            xtemp = win*self.detrend(xtemp)
            pseg[gg] = _np.trapz(xtemp**2.0, x=tvec[istart:iend])
            Xfft[gg,:] = self.fft(xtemp, nfft)
        #endfor loop over fft windows

        freq = _np.fft.fftfreq(nfft, 1.0/Fs)
        if self.onesided:
#            freq = freq[:Nnyquist]  # [Hz]
#            Xfft = Xfft[:,:Nnyquist]
            freq = freq[:Nnyquist-1]  # [Hz]
            Xfft = Xfft[:,:Nnyquist-1]

            # Real signals equally split their energy between positive and negative frequencies
            Xfft[:, 1:-1] = _np.sqrt(2)*Xfft[:, 1:-1]
            if nfft%2:  # odd
                Xfft[:,-1] = _np.sqrt(2)*Xfft[:,-1]
            # endif
        else:
            freq = self.fftshift(freq, axes=0)
            Xfft = self.fftshift(Xfft, axes=-1)
        # end if

        # Remove gain of the window function to yield the RMS Power spectrum
        # in each segment (constant peak amplitude)
        Xfft /= S1   # Vrms
        pseg /= S2

        # Compute the spectral density from the RMS spectrum
        # (constant noise floor)
        Xfft /= _np.sqrt(ENBW)  # [V/Hz^0.5]

        return tt, freq, Xfft, pseg

    # ===================================================================== #

    def plotall(self):

        # The input signals versus time
        self.fig = _plt.figure()
        self.ax1 = _plt.subplot(2,2,1)
        _ax1 = self.ax1
        _ax1.plot(  self.tvec,  self.sigx,'b-', self.tvec, self.sigy,'r-')
        _ax1.set_title('Input Signals',**self.afont)
        _ax1.set_xlabel('t[s]',**self.afont)
        _ax1.set_ylabel('sig_x,sig_y[V]',**self.afont)
        _plt.axvline(x=self.tbounds[0],color='k')
        _plt.axvline(x=self.tbounds[1],color='k')

        self.ax2 = _plt.subplot(2,2,2)
        _ax2 = self.ax2
#        _ax2.plot(1e-3*self.freq, _np.abs( self.Pxx ), 'b-')
#        _ax2.plot(1e-3*self.freq, _np.abs( self.Pyy ), 'r-')
#        _ax2.plot(1e-3*self.freq, _np.abs( self.Pxy ), 'k-')
#        _ax2.set_ylabel(r'P_{ij} [V$**2$/Hz]',**self.afont),
        _ax2.plot(1e-3*self.freq, 10*_np.log10( _np.abs( self.Pxx ) ), 'b-')
        _ax2.plot(1e-3*self.freq, 10*_np.log10( _np.abs( self.Pyy ) ), 'r-')
        _ax2.plot(1e-3*self.freq, 10*_np.log10( _np.abs( self.Pxy ) ), 'k-')
        _ax2.set_ylabel('P_{ij} [dB/Hz]',**self.afont),
        _ax2.set_title('Power Spectra',**self.afont)
        _ax2.set_xlabel('f[kHz]',**self.afont)
        if self.onesided:
            _ax2.set_xlim(0,1.01e-3*self.freq[-1])
        else:
            _ax2.set_xlim(-1.01e-3*self.freq[-1],1.01e-3*self.freq[-1])
        # end if

        #_ax2.text(0.20, 0.15, 'P_{xx}', 'Color', 'b', 'units', 'normalized',
        #          'FontSize', 14, 'FontName', 'Arial')
        # _ax2.text(0.20, 0.35, 'P_{yy}', 'Color', 'r', 'units', 'normalized',
        #           'FontSize', 14, 'FontName', 'Arial')
        # _ax2.text(0.20, 0.55, 'P_{xy}', 'Color', 'k', 'units', 'normalized',
        #           'FontSize', 14, 'FontName', 'Arial')

        self.ax3 = _plt.subplot(2,2,3, sharex=self.ax2)
        _ax3 = self.ax3
        _ax3.plot(1e-3*self.freq, self.Cxy2,'k-')
        _ax3.axhline(y=1./self.Navr,color='k')
#        _ax3.plot(1e-3*self.freq, self.Cxy,'k-')
#        _ax3.axhline(y=1./_np.sqrt(self.Navr),color='k')
        _ax3.set_title('Cross-Coherence',**self.afont)
        _ax3.set_ylabel('C_{xy}',**self.afont)
        _ax3.set_xlabel('f[kHz]',**self.afont)
        if self.onesided:
            _ax3.set_xlim(0,1.01e-3*self.freq[-1])
        else:
            _ax3.set_xlim(-1.01e-3*self.freq[-1],1.01e-3*self.freq[-1])
        # end if

        # _ax3.text(0.80, 0.90, 'C_{xy}', 'Color', 'k', 'units', 'normalized',
        #           'FontSize', 14, 'FontName', 'Arial')

        self.ax4 = _plt.subplot(2,2,4, sharex=self.ax2)
        _ax4 = self.ax4
        _ax4.plot(1e-3*self.freq,self.phi_xy,'k-')
        _ax4.set_title('Cross-Phase',**self.afont)
        _ax4.set_ylabel('\phi_{xy}',**self.afont)
        _ax4.set_xlabel('f[kHz]',**self.afont)
        if self.onesided:
            _ax4.set_xlim(0,1.01e-3*self.freq[-1])
        else:
            _ax4.set_xlim(-1.01e-3*self.freq[-1],1.01e-3*self.freq[-1])
        # end if
        # _ax4.text(0.80, 0.90, '\phi_{xy}', 'Color', 'k', 'units', 'normalized',
        #           'FontSize', 14, 'FontName', 'Arial')

        _plt.tight_layout()
        _plt.draw()
        # _plt.show()

    # ===================================================================== #

    def plotspec(self, param='Cxy', logscale=False, vbnds=None, cmap=None):  # spectrogram

        self.fig = _plt.figure()
        _ax = _plt.gca()
#        _ax1 = _plt.subplot(2,1,1)
#        _ax1.set_ylabel('Signal')
#        if param.lower().find('x')>-1:
#            _plt.plot( self.tvec, self.sigx, 'b-')
#        if param.lower().find('y')>-1:
#            _plt.plot( self.tvec, self.sigy, 'r-')
#
#        _plt.axvline(x=self.tbounds[0], color='k')
#        _plt.axvline(x=self.tbounds[1], color='k')
        # _plt.xlim((self.tvec.min(), self.tvec.max()))

#        _ax2 = _plt.subplot(2,1,2, sharex=_ax1)
        _ax.set_title(param)
        _ax.set_ylabel('freq [KHz]')
        _ax.set_xlabel('time [s]')

        spec = getattr(self, param+'_seg').copy()
        spec = _np.abs(spec).astype(float)
        if logscale:
            spec = 20.0*_np.log10(spec)
            _ax.set_yscale('symlog', linthreshy=0.01)
        # endif
        if vbnds is None:
            vbnds = [spec.min(), spec.max()]
        # endif
        if cmap is None:
#            cmap = _plt.cm.gist_heat
             cmap = 'RdBu'
        # endif

        # bin starts are plotted, not bin centers
        tbin = self.tseg-0.5*(self.tseg[2]-self.tseg[1])
        fbin = 1e-3*(self.freq-0.5*(self.freq[2]-self.freq[1]))
        _plt.pcolor(tbin, fbin, spec.T, cmap=cmap, vmin=vbnds[0], vmax=vbnds[1])
#        _plt.pcolormesh(self.tseg, 1e-3*self.freq, spec.T, cmap=cmap, vmin=vbnds[0], vmax=vbnds[1])
        _plt.xlim(tuple(self.tbounds))
        _plt.ylim((1e-3*min(0, 1.1*self.freq.min()), 1e-3*self.freq.max()))
        _plt.colorbar()
        _plt.draw()
    # end def

    # ===================================================================== #

    def plotcorr(self):

#        self.Rxy_np = _np.correlate(self.__trimsig__(self.sigx, self.ibounds), self.__trimsig__(self.sigy, self.ibounds), mode='full')
#        _plt.plot(self.lags, self.Rxy_np, 'm--')

        _plt.figure()
        _ax1 = _plt.subplot(4,1,1)
        _plt.ylabel(r'R$_{x,x}$')
        _plt.title(r'Cross-Correlation')
        Rxx = self.Rxx.copy()
        Rxx = _np.real(Rxx)
#        Rxx = _np.abs(Rxx)
        _plt.plot(self.lags, Rxx, 'b-')

        _plt.subplot(4,1,2, sharex=_ax1, sharey=_ax1)
        _plt.ylabel(r'R$_{y,y}$')
        Ryy = self.Ryy.copy()
        Ryy = _np.real(Ryy)
#        Ryy = _np.abs(Ryy)
        _plt.plot(self.lags, Ryy, 'r-')

        _plt.subplot(4,1,3, sharex=_ax1, sharey=_ax1)
        _plt.ylabel(r'R$_{x,y}$')
        Rxy = self.Rxy.copy()
        Rxy = _np.real(Rxy)   # phase delay works with two-sided spectra!
#        Rxy = _np.imag(Rxy)  # imag part zero wtih two-sided spectra
#        Rxy = _np.abs(Rxy)
        _plt.plot(self.lags, Rxy, 'k-')
#        _plt.xlabel('lags [ms]')

        _plt.subplot(4,1,4, sharex=_ax1)
#        _plt.figure()
        _plt.xlabel('lags [ms]')
        _plt.ylabel(r'$\rho_{x,y}$')
        corrcoef = self.corrcoef.copy()
        corrcoef = _np.real(corrcoef)
#        corrcoef = _np.abs(corrcoef)
        _plt.plot(self.lags, corrcoef, 'k-')



    # ===================================================================== #

    def plottime(self):
        # The input signals versus time
        self.fig = _plt.figure()
        _plt.plot(  self.tvec,  self.sigx,'b-', self.tvec, self.sigy, 'r-')
        _plt.title('Input Signals', **self.afont)
        _plt.xlabel('t[s]', **self.afont)
        _plt.ylabel('sig_x,sig_y[V]', **self.afont)
        _plt.axvline(x=self.tbounds[0], color='k')
        _plt.axvline(x=self.tbounds[1], color='k')
        _plt.draw()
#        _plt.show()

    # ===================================================================== #

    def plotPxy(self):
        self.fig = _plt.figure()
        _plt.plot(1e-3*self.freq, 10*_np.log10(_np.abs(self.Pxx)), 'b-')
        _plt.plot(1e-3*self.freq, 10*_np.log10(_np.abs(self.Pyy)), 'r-')
        _plt.plot(1e-3*self.freq, 10*_np.log10(_np.abs(self.Pxy)), 'k-')
        _plt.title('Power Spectra', **self.afont)
        _plt.ylabel('P_{ij} [dB/Hz]', **self.afont)
        _plt.xlabel('f[kHz]', **self.afont)
        if self.onesided:
            _plt.xlim(0,1.01e-3*self.freq[-1])
        else:
            _plt.xlim(-1.01e-3*self.freq[-1],1.01e-3*self.freq[-1])
        # end if
        _plt.draw()
#        _plt.show()

    # ===================================================================== #

    def plotCxy(self):
        self.fig = _plt.figure()
        _plt.plot(1e-3*self.freq, self.Cxy, 'k-')
        _plt.axhline(y=1./self.Navr, color='k')
        _plt.title('Cross-Coherence', **self.afont)
        _plt.ylabel('C_{xy}', **self.afont)
        _plt.xlabel('f[kHz]', **self.afont)
        if self.onesided:
            _plt.xlim(0,1.01e-3*self.freq[-1])
        else:
            _plt.xlim(-1.01e-3*self.freq[-1],1.01e-3*self.freq[-1])
        # end if
        _plt.draw()
#        _plt.show()

    # ===================================================================== #

    def plotphxy(self):
        self.fig = _plt.figure()
        _plt.plot(1e-3*self.freq, self.phi_xy, 'k-')
        _plt.title('Cross-Phase', **self.afont)
        _plt.ylabel('\phi_{xy}', **self.afont)
        _plt.xlabel('f[kHz]', **self.afont)
        if self.onesided:
            _plt.xlim(0,1.01e-3*self.freq[-1])
        else:
            _plt.xlim(-1.01e-3*self.freq[-1],1.01e-3*self.freq[-1])
        # end if
        _plt.draw()
#        _plt.show()

    # ===================================================================== #

    def __calcAmp__(self, tvec, sigx, sigy, tbounds, nn=8, ol=0.5,
                    ww='hanning'):

        # The amplitude is most accurately calculated by using several windows
        self.frqA, self.Axy, self.Axx, self.Ayy, self.aCxy, _, _ = \
            fft_pwelch(tvec, sigx, sigy, tbounds, Navr=nn, windowoverlap=ol,
                       windowfunction=ww, useMLAB=self.useMLAB, plotit=0,
                       verbose=self.verbose, detrend_style=self.detrendstyle,
                       onesided=self.onesided)
        self.__plotAmp__()

    def __plotAmp__(self):
        self.fig = _plt.figure()
        _plt.plot(1e-3*self.frqA, 10*_np.log10(_np.abs(self.Axx)), 'b-')
        _plt.plot(1e-3*self.frqA, 10*_np.log10(_np.abs(self.Ayy)), 'r-')
        _plt.plot(1e-3*self.frqA, 10*_np.log10(_np.abs(self.Axy)), 'k-')
        _plt.title('Power Spectra', **self.afont)
        _plt.ylabel('P_{ij} [dB/Hz]', **self.afont),
        _plt.xlabel('f[kHz]', **self.afont)
        if self.onesided:
            _plt.xlim(0,1.01e-3*self.frqA[-1])
        else:
            _plt.xlim(-1.01e-3*self.frqA[-1],1.01e-3*self.frqA[-1])
        # end if
        _plt.draw()
#        _plt.show()

    # ===================================================================== #

    def __calcPh1__(self, tvec, sigx, sigy, tbounds, nn=1, ol=0.0, ww='box'):

        # The amplitude is most accurately calculated by using several windows
        self.frqP, _, _, _, _, self.ph, _ = \
            fft_pwelch(tvec, sigx, sigy, tbounds, Navr=nn, windowoverlap=ol,
                       windowfunction=ww, useMLAB=self.useMLAB, plotit=0,
                       verbose=self.verbose, detrend_style=self.detrendstyle,
                       onesided=self.onesided)
        self.__plotPh1__()

    def __plotPh1__(self):
        self.fig = _plt.figure()
        _plt.plot(1e-3*self.frqP, self.ph,'k-')
        _plt.title('Cross-Phase', **self.afont)
        _plt.ylabel('\phi_{xy}', **self.afont)
        _plt.xlabel('f[kHz]', **self.afont)
        if self.onesided:
            _plt.xlim(0,1.01e-3*self.frqP[-1])
        else:
            _plt.xlim(-1.01e-3*self.frqP[-1],1.01e-3*self.frqP[-1])
        # end if
        _plt.draw()
#        _plt.show()

    # ===================================================================== #

    def __preallocateFFT__(self):
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

    @staticmethod
    def __testFFTanal__(useMLAB=True, plotext=True):
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
            _ax1.plot(ft1.tvec, ft1.sigx,'m--',ft1.tvec, ft1.sigy,'m--')
            _ax1.set_title('Input Signals',**afont)
            _ax1.set_xlabel('t[s]',**afont)
            _ax1.set_ylabel('sig_x,sig_y[V]',**afont)
            _plt.axvline(x=ft1.tbounds[0],color='k')
            _plt.axvline(x=ft1.tbounds[1],color='k')

            _ax2 = _plt.subplot(2,2,2)
#            _ax2.plot(1e-3*ft1.freq, 10*_np.log10(_np.abs(ft1.Pxy)), 'k-')
#            _ax2.plot(1e-3*ft2.freq, 10*_np.log10(_np.abs(ft2.Pxy)), 'm--')
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
            # end if


            # _ax4.text(0.80, 0.90, 'C_{xy}', 'Color', 'k', 'units', 'normalized',
            #          'FontSize', 14, 'FontName', 'Arial')

            _ax4 = _plt.subplot(2, 2, 4, sharex=_ax2)
            _ax4.plot(1e-3*ft1.freq, ft1.phi_xy, 'k-')
            _ax4.plot(1e-3*ft2.freq, -1.0*ft2.phi_xy, 'm--')
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
            # _plt.show()
        # endif plotit
        return ft1, ft2
    # end def testFFtanal

    # ===================================================================== #

#end class fftanal

# ==========================================================================

def test_fftpwelch(useMLAB=True, plotit=True, nargout=0, tstsigs = None):
    ##Generate test data for the no input case:

    if tstsigs is None:
        #Minimize the spectral leakage:
        df = 5.0   #Hz
        N  = 2**14 #Numper of points in time-series
#        N  = 2**20 #Numper of points in time-series
        tvec = (1.0/df)*_np.arange(0.0,1.0,1.0/(N))

        #Sine-wave
        _np.random.seed()
#        nx = int(N / 100)
#        sigx = _np.sin(2.0*_np.pi*(df*2000.0)*tvec[:nx])     #Shifted time-series
#        sigx = _np.sin(2.0*_np.pi*(df*30.0)*tvec)     #Shifted time-series
        #Square-wave
        sigx = _dsp.square(2.0*_np.pi*(df*30.0)*tvec)    #Shifted square wave

        sigx *= 0.1
        sigx += 0.01*_np.random.standard_normal( (sigx.shape[0],) )
        sigx += 7.0

        #Noisy phase-shifted sine-wave
        _np.random.seed()
#        sigy = _np.sin(2.0*_np.pi*(df*2000.0)*tvec-_np.pi/4.0)
        nch = 3
        sigy = _np.zeros((len(tvec), nch), dtype=_np.float64)
        for ii in range(nch):
            sigy[:,ii] = _np.sin(2.0*_np.pi*((ii+1)*df*30.0)*tvec-_np.pi/2.0-_np.pi/4.0-ii*_np.pi/16)/(ii+1)
        sigy *= 0.007
        sigy += 0.07*_np.random.standard_normal( (tvec.shape[0],nch) )
        sigy += 2.5
    else:
        tvec = tstsigs[0].copy()
        sigx = tstsigs[1].copy()
        sigy = tstsigs[2].copy()
    # endif

#    detrend_style = 0 # None     # matches mlab and my version together
    detrend_style = 1 # Mean     # Results in coherence > 1 in 1st non-zero freq bin for mlab, but my version works
#    detrend_style = -1 # Linear   # Definitely doesn't work well for the test signals: breaks both
    fft_pwelch(tvec,sigx,sigy, [tvec[0],tvec[-1]], Navr = 8, windowoverlap = 2.0/3.0, windowfunction = 'hamming', detrend_style=detrend_style, useMLAB=True, plotit=True, verbose=True)
    fft_pwelch(tvec,sigx,sigy, [tvec[0],tvec[-1]], Navr = 8, windowoverlap = 2.0/3.0, windowfunction = 'hamming', detrend_style=detrend_style, useMLAB=False, plotit=True, verbose=True)

    fft_pwelch(tvec,sigx,sigy, [tvec[0],tvec[-1]], Navr = 16, windowoverlap = 2.0/3.0, windowfunction = 'hamming', detrend_style=detrend_style, useMLAB=True, plotit=True, verbose=True)
    fft_pwelch(tvec,sigx,sigy, [tvec[0],tvec[-1]], Navr = 16, windowoverlap = 2.0/3.0, windowfunction = 'hamming', detrend_style=detrend_style, useMLAB=False, plotit=True, verbose=True)

#    [freq,Pxy] = fft_pwelch(tvec,Zece[:,1],Zece[:,2],[0.1,0.3],useMLAB=True,plotit=True)
#end testFFTanal

def test_fftanal(useMLAB=True, plotit=True, nargout=0, tstsigs = None):
    ##Generate test data for the no input case:

    if tstsigs is None:
        #Minimize the spectral leakage:
        df = 5.0   #Hz
        N  = 2**12 #Numper of points in time-series
        tvec = (1.0/df)*_np.arange(0.0,1.0,1.0/(N))

        #Sine-wave
        sigx = _np.sin(2.0*_np.pi*(df*30.0)*tvec)     #Shifted time-series
        sigx *= 0.004
        sigx += 7.0

        #Noisy phase-shifted sine-wave
        sigy = _np.sin(2.0*_np.pi*(df*30.0)*tvec-_np.pi/4.0)
        sigy *= 0.007
        sigy += 0.05*_np.random.standard_normal( (tvec.shape[0],) )
        sigy += 2.5

        #Square-wave
        #sigx = 10.0 + _dsp.square(2.0*_np.pi*(df*100.0)*tvec)    #Shifted square wave

        #sigy = sigx

        fs = df*N
#        N = 1e5
        amp = 2.0 * _np.sqrt(2)
        noise_power = 0.01 * fs / 2
        time = _np.arange(N) / float(fs)
        mod = 500*_np.cos(2*_np.pi*0.25*time)
        carrier = amp * _np.sin(2*_np.pi*3e3*time + mod)
        noise = _np.random.normal(scale=_np.sqrt(noise_power), size=time.shape)
        noise *= _np.exp(-time/5)
        sigz = carrier + noise
    else:
        tvec = tstsigs[0].copy()
        sigx = tstsigs[1].copy()
        sigy = tstsigs[2].copy()
        sigz = sigx.copy() * sigy.copy()
    # endif
    # ------
    #
    #    _plt.Figure, set(gca,'FontSize',14,'FontName','Arial'), hold all
    #    xlabel('t[s]'), ylabel('Y_{test} [a.u.]')
    #    plot(tvec,[sigx;sigy],'-'),
    #    axis([-0.01*tvec[0],1.01*tvec[-1],0,15])
    #
    # -----

    # Test using the fftpwelch function
    ft = fftanal(tvec,sigx,sigy,tbounds = [tvec[0],tvec[-1]],
            Navr = 8, windowoverlap = 0.5, windowfunction = 'hamming',
            useMLAB=useMLAB, plotit=plotit, verbose=True,
            detrend_style=0, onesided=False)
#            detrend_style=0, onesided=True)

    ft.plotall()
#    if not useMLAB:
#         # test using the pwelch class methods
#        ft2 = fftanal()
#        ft2.init(tvec, sigx, sigy, tbounds=[tvec[1], tvec[-1]],
#                 Navr=8, windowoverlap=0.5, windowfunction='hamming',
#                 useMLAB=False, plotit=plotit, detrend=1,
#                 onesided=False)
##                 onesided=True)
#        ft2.pwelch()
#        ft2.plotall()
##        ft2.plotspec('Pxy')
##        ft2.plotspec('Cxy', vbnds=[0, 1.0])
##        ft2.plotspec('Lxy')
##        ft2.plotspec('Lyy')
#
#        ft2.crosscorr()
#        ft2.plotcorr()

    if nargout>0:
        return ft

#    fft_pwelch(tvec,sigx,sigy, [tvec[1],tvec[-1]], Navr = 8, windowoverlap = 0.5, windowfunction = 'hamming', useMLAB=False, plotit=True, verbose=True)


#    [freq,Pxy] = fft_pwelch(tvec,Zece[:,1],Zece[:,2],[0.1,0.3],useMLAB=True,plotit=True)
#end testFFTanal

def test():
    tst = fftanal(verbose=True)
    ft1, ft2 = tst.__testFFTanal__()
    return ft1, ft2

if __name__ == "__main__":
#    fts = test()
#    test_fftpwelch()

    test_fft_deriv()
#    test_fft_deriv(xx=2*_np.pi*_np.linspace(-1.5, 3.3, num=650, endpoint=False))
# ========================================================================== #
# ========================================================================== #

