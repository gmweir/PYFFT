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

import scipy.signal as _dsp
import numpy as _np
import matplotlib.mlab as _mlab
import matplotlib.pyplot as _plt

#import pybaseutils as _pyut

# Local module testing
#from pybaseutils.Struct import Struct
#from pybaseutils import utils as _ut

# Normal use
from .Struct import Struct
from . import utils as _ut

# ========================================================================== #
# ========================================================================== #

def fft_pwelch(tvec, sigx, sigy, tbounds, Navr=None, windowoverlap=None,
               windowfunction=None, useMLAB=None, plotit=None, verbose=None,
               detrend_style=None):
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
      sigx    - [V], signal X,
      sigy    - [V], signal Y,
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
        detrend_style=None
    # endif

    # Matlab returns the power spectral denstiy in [V^2/Hz], and doesn't
    # normalize it's FFT by the number of samples or the power in the windowing
    # function.  These can be handled by controlling the inputs and normalizing
    # the output:
    dt   = 0.5*(tvec[2]-tvec[0])       #[s],  time-step in time-series
    Fs   = 1.0/dt                #[Hz], sampling frequency

    # ==================================================================== #
    # ==================================================================== #

    # Detrend the two signals to get FFT's
    i0 = int( _np.floor( Fs*(tbounds[0]-tvec[0] ) ) )
    i1 = int( _np.floor( Fs*(tbounds[1]-tvec[0] ) ) )
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

    # Heliotron-J
    nwins = int(_np.floor(nsig*1.0/(Navr-Navr*windowoverlap + windowoverlap)))
    if nwins>=nsig:
        Navr = 1
        nwins = nsig
    # endif
    # nfft     = max(2^12,2^nextpow2(nwins))
    nfft = nwins

    # Number of points to overlap
    noverlap = int( _np.ceil( windowoverlap*nwins ) )
    Nnyquist = nfft//2 + 1
    if (nfft%2):  # odd
       Nnyquist = (nfft+1)//2
    # end if the remainder of nfft / 2 is odd

    # Remember that since we are not dealing with infinite series, the lowest
    # frequency we actually resolve is determined by the period of the window
    # fhpf = 1.0/(nwins*dt)  # everything below this should be set to zero (when background subtraction is applied)

    # ==================================================================== #

    class fftinfosc():
        def __init__(self):
            self.S1     = _np.array( [], dtype=_np.float64)
            self.S2     = _np.array( [], dtype=_np.float64)

            self.NENBW  = _np.array( [], dtype=_np.float64)
            self.ENBW   = _np.array( [], dtype=_np.float64)

            self.freq   = _np.array( [], dtype=_np.float64)
            self.Pxx    = _np.array( [], dtype = _np.complex128 )
            self.Pyy    = _np.array( [], dtype = _np.complex128 )
            self.Pxy    = _np.array( [], dtype = _np.complex128 )

            self.Cxy    = _np.array( [], dtype=_np.float64)
            self.varcoh = _np.array( [], dtype=_np.float64)
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
        win = _np.hamming(nwins)  # periodic hamming window?
        # win = _np.hamming(nwins+1)  # periodic hamming window
        # win = win[0:-1]  # truncate last point to make it periodic
    elif windowfunction.lower() == 'hanning':
        if verbose:
            print('Using a Hanning window function')
        # endif verbose
        win = _np.hanning(nwins)  # periodic hann window?
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
        # No window function (actually a box-window)
        win = _np.ones( (nwins,), dtype=_np.float64)
        # win = win[0:-1]  # truncate last point to make it periodic
    # endif windowfunction.lower()

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
        # Use MLAB for the power spectral density calculations
        if verbose:
            print('using matlab built-ins for spectra/coherence calculations')
        # endif verbose

        x_in = sigx[i0:i1]
        y_in = sigy[i0:i1]

        # Power spectral density (auto-power spectral density), and
        # cross-power spectral density of signal 1 and signal 2,
        # Pyy = Yfft.*conj(Yfft), and the linear amplitude spectrum, Lxx:

        # [V^2/Hz], RMS power spectral density calculation
        Pxx, freq = _mlab.csd(x_in, x_in, nfft, Fs, detrend=detrend,
                              window=win, noverlap=noverlap,
                              scale_by_freq=True)
        Pyy, freq = _mlab.csd(y_in, y_in, nfft, Fs, detrend=detrend,
                              window=win, noverlap=noverlap,
                              scale_by_freq=True)
        Pxy, freq = _mlab.csd(x_in, y_in, nfft, Fs, detrend=detrend,
                              window=win, noverlap=noverlap,
                              scale_by_freq=True)

        # Get the coherence
        if (Navr==1):
            Cxy2 = _np.ones_like(Pxx)
        else:
            # returns mean squared coherence
            [Cxy2, freq] = _mlab.cohere(x_in, y_in, nfft, Fs, detrend=detrend,
                                      window=win, noverlap=noverlap,
                                      scale_by_freq=True)

#            # Force Mean detrending
#            [Cxy2, freq] = _mlab.cohere(x_in, y_in, nfft, Fs,
#                                       detrend=detrend_mean, window=win,
#                                       noverlap=noverlap, scale_by_freq=True)
        #endif

        # Linear amplitude spectrum from the power spectral density
        # RMS linear amplitude spectrum
        fftinfo.Lxx = _np.sqrt(_np.abs(fftinfo.ENBW*Pxx))  # [V_rms],
        fftinfo.Lyy = _np.sqrt(_np.abs(fftinfo.ENBW*Pyy))  # [V_rms],
        fftinfo.Lxy = _np.sqrt(_np.abs(fftinfo.ENBW*Pxy))  # [V_rms],

        # Convert from RMS to Amplitude
        # Just the points that split their energy into negative frequencies
        fftinfo.Lxx[1:-2] = _np.sqrt(2)*fftinfo.Lxx[1:-2]  # [V],
        fftinfo.Lyy[1:-2] = _np.sqrt(2)*fftinfo.Lyy[1:-2]  # [V],
        fftinfo.Lxy[1:-2] = _np.sqrt(2)*fftinfo.Lxy[1:-2]  # [V],
        if nfft%2:  # Odd
            fftinfo.Lxx[-1] = _np.sqrt(2)*fftinfo.Lxx[-1]
            fftinfo.Lyy[-1] = _np.sqrt(2)*fftinfo.Lyy[-1]
            fftinfo.Lxy[-1] = _np.sqrt(2)*fftinfo.Lxy[-1]
        # endif nfft/2 is odd

        fftinfo.varPxx = _np.zeros(_np.shape(Pxx), dtype=_np.complex128) #c = chi2conf(CL,k) # confidence interval from confidence level and number of obs.
        fftinfo.varPyy = _np.zeros(_np.shape(Pyy), dtype=_np.complex128)
        fftinfo.varPxy = _np.zeros(_np.shape(Pxy), dtype=_np.complex128)

        # ================================================================= #
    else:
        # Without Matlab: Welch's average periodogram method:
        if verbose:
            print('using home-brew functions for spectra/coherence calculations')
        # endif verbose

        # ============ #

        # Pre-allocate
        Pxy_seg = _np.zeros((Navr, nfft), dtype=_np.complex128)
        Pxx_seg = _np.zeros((Navr, nfft), dtype=_np.complex128)
        Pyy_seg = _np.zeros((Navr, nfft), dtype=_np.complex128)

        x_in = sigx[i0:i1]
        y_in = sigy[i0:i1]

        Xfft = _np.zeros((Navr, nfft), dtype=_np.complex128)
        Yfft = _np.zeros((Navr, nfft), dtype=_np.complex128)

        ist = _np.arange(Navr)*(nwins - noverlap)
        ist = ist.astype(int)
        for gg in _np.arange(Navr):
            istart = ist[gg]     # Starting point of this window
            iend = istart+nwins  # End point of this window

            xtemp = x_in[istart:iend]
            ytemp = y_in[istart:iend]

            # Windowed signal segment
            # To get the most accurate spectrum, minimally detrend
            xtemp = win*detrend(xtemp)
            ytemp = win*detrend(ytemp)
            # xtemp = win*_dsp.detrend(x_in[istart:iend], type='constant')
            # ytemp = win*_dsp.detrend( y_in[istart:iend], type='constant' )

            # The FFT output from matlab isn't normalized:
            # y_n = sum[ y_m.*exp( 2_np.pi*1i*(n/N)*m ) ]
            # The inverse is normalized::
            # y_m = (1/N)*sum[ y_n.*exp( -2_np.pi*1i*(n/N)*m ) ]
            #
            # Python normalizations are optional, pick it to match MATLAB
            Xfft[gg, 0:nfft] = _np.fft.fft(xtemp, n=nfft)
            Yfft[gg, 0:nfft] = _np.fft.fft(ytemp, n=nfft)
        #endfor loop over fft windows

        #Auto- and cross-power spectra
        Pxx_seg[0:Navr, 0:nfft] = Xfft*_np.conj(Xfft)
        Pyy_seg[0:Navr, 0:nfft] = Yfft*_np.conj(Yfft)
        Pxy_seg[0:Navr, 0:nfft] = Xfft*_np.conj(Yfft)

        #Normalize, Average and Remove contributions above the Nyquist frequency
        #freq = Fs*(0:1/(nfft):1)
        freq = Fs*_np.arange(0.0, 1.0, 1.0/nfft)
        if (nfft%2):
            # freq = Fs*(0:1:1/(nfft+1))
            freq = Fs*_np.arange(0.0,1.0,1.0/(nfft+1))
        # end if nfft is odd
        freq = freq[0:Nnyquist]  # [Hz]
#        freq = freq[0:Nnyquist-1]  # [Hz]
#        freq = freq.reshape(1,Nnyquist-1)

        # Remove gain of the window function to yield the RMS Power spectrum
        # in each segment (constant peak amplitude)
        Pxx_seg = Pxx_seg[0:Navr, 0:Nnyquist] #-1]
        Pyy_seg = Pyy_seg[0:Navr, 0:Nnyquist] #-1]
        Pxy_seg = Pxy_seg[0:Navr, 0:Nnyquist] #-1]
        Pxx_seg = (1.0/(fftinfo.S1**2))*Pxx_seg  # [Vrms^2]
        Pyy_seg = (1.0/(fftinfo.S1**2))*Pyy_seg  # [Vrms^2]
        Pxy_seg = (1.0/(fftinfo.S1**2))*Pxy_seg  # [Vrms^2]

        # All components but DC split their energy between positive +
        # negative frequencies: One sided spectra,
        Pxx_seg[0:Navr, 1:-2] = 2*Pxx_seg[0:Navr, 1:-2]  # [V^2/Hz],
        Pyy_seg[0:Navr, 1:-2] = 2*Pyy_seg[0:Navr, 1:-2]  # [V^2/Hz],
        Pxy_seg[0:Navr, 1:-2] = 2*Pxy_seg[0:Navr, 1:-2]  # [V^2/Hz],
        if nfft%2:  # Odd
            Pxx_seg[0:Navr, -1] = 2*Pxx_seg[0:Navr, -1]
            Pyy_seg[0:Navr, -1] = 2*Pyy_seg[0:Navr, -1]
            Pxy_seg[0:Navr, -1] = 2*Pxy_seg[0:Navr, -1]
        # endif nfft is odd

        # Compute the power spectral density from the RMS power spectrum
        # (constant noise floor)
        Pxx_seg = Pxx_seg/fftinfo.ENBW  # [V^2/Hz]
        Pyy_seg = Pyy_seg/fftinfo.ENBW  # [V^2/Hz]
        Pxy_seg = Pxy_seg/fftinfo.ENBW  # [V^2/Hz]

        # Average the different realizations: This is the output from cpsd.m
        # RMS Power spectrum
        Pxx = _np.mean((Pxx_seg[0:Navr,0:Nnyquist]), axis=0)  # [V^2/Hz]
        Pyy = _np.mean((Pyy_seg[0:Navr,0:Nnyquist]), axis=0)
        Pxy = _np.mean((Pxy_seg[0:Navr,0:Nnyquist]), axis=0)

        # Estimate the variance in the power spectra
#        fftinfo.varPxx = _np.var((Pxx_seg[0:Navr, 0:Nnyquist]), axis=0)
#        fftinfo.varPyy = _np.var((Pyy_seg[0:Navr, 0:Nnyquist]), axis=0)
#        fftinfo.varPxy = _np.var((Pxy_seg[0:Navr, 0:Nnyquist]), axis=0)

        # use the RMS for the standard deviation
        fftinfo.varPxx = _np.mean(Pxx_seg[0:Navr, 0:Nnyquist]**2.0, axis=0)
        fftinfo.varPyy = _np.mean(Pyy_seg[0:Navr, 0:Nnyquist]**2.0, axis=0)
        fftinfo.varPxy = _np.mean(Pxy_seg[0:Navr, 0:Nnyquist]**2.0, axis=0)

#        fftinfo.varPxy = \
#            (_np.var(_np.real(Pxy_seg[0:Navr, 0:Nnyquist]), axis=0)
#            + 1j*_np.var(_np.imag(Pxy_seg[0:Navr, 0:Nnyquist]), axis=0))

        # Save the cross-phase as well
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
#        # Cxy2  = Pxy.*(Pxy').'./(Pxx.*Pyy)  # Coherence between the two signals
#        [Cxy2, fftinfo.varCxy2] = varcoh(Pxy, fftinfo.varPxy,
#                                       Pxx, fftinfo.varPxx,
#                                       Pyy, fftinfo.varPyy)
#        fftinfo.varPxy = Pxx*Pyy*(1.0-Cxy2)/Navr
#
        Cxy2 = _np.abs( Pxy*_np.conj( Pxy ) )/( _np.abs(Pxx)*_np.abs(Pyy) )
#        [Cxy2, fftinfo.varCxy2] = monticoh(Pxy, fftinfo.varPxy, Pxx, fftinfo.varPxx,
#                                         Pyy, fftinfo.varPyy, meansquared=True)
#
        # Segment data
        fftinfo.Pxx_seg = Pxx_seg
        fftinfo.Pyy_seg = Pyy_seg
        fftinfo.Pxy_seg = Pxy_seg
        fftinfo.Xfft_seg = Xfft
        fftinfo.Yfft_seg = Yfft
        fftinfo.phixy_seg = phixy_seg
        fftinfo.varphi_seg = varphi_seg
    # endif

    # ========================== #
    # Uncertainty and phase part

    # derived using error propagation from eq 23 for gamma^2 in
    # J.S. Bendat, Journal of Sound an Vibration 59(3), 405-421, 1978
    # fftinfo.varCxy2 = _np.zeros_like(Cxy2)
    Cxy = _np.sqrt(Cxy2)
    fftinfo.varCxy = ((1-Cxy2)/_np.sqrt(2*Navr))**2.0

    # Estimate the variance in the power spectra: this requires building
    # a distribution by varying the parameters used in the FFT, nwindows,
    # nfft, windowfunction, etc.  I don't do this right now
    fftinfo.varPxy = Pxx*Pyy*(1.0-Cxy)/Navr

    # A.E. White, Phys. Plasmas, 17 056103, 2010
    # Doesn't so far give a convincing answer...
    # fftinfo.varPhxy = _np.zeros(Pxy.shape, dtype=_np.float64)
    #fftinfo.varPhxy = (_np.sqrt(1-Cxy2)/_np.sqrt(2*Navr*Cxy))**2.0
    fftinfo.varPhxy = (_np.sqrt(1-Cxy2)/_np.sqrt(2*Navr*Cxy))**2.0

    phi_xy = _np.angle(Pxy)        # Save the cross-phase as well
    # phi_xy = _np.arctan(_np.imag(Pxy)/_np.real(Pxy))

    # phi_xy, fftinfo.varPhxy = montiphi(Pxy, varPxy)

#        phi_xy, fftinfo.varPhxy = varphi(_np.real(Pxy), _np.imag(Pxy),
#                 (_np.real(Pxy)/_np.abs(Pxy))**2.0*fftinfo.varPxy,
#                 (_np.imag(Pxy)/_np.abs(Pxy))**2.0*fftinfo.varPxy, angle_range=_np.pi)


    # ========================== #


#    if onesided:
#        pass
#    else:
#        pass
#    # end if

    # Linear amplitude spectrum from the power spectral density
    # RMS Linear amplitude spectrum (constant amplitude values)
    fftinfo.Lxx = _np.sqrt(_np.abs(fftinfo.ENBW*Pxx))  # [V_rms]
    fftinfo.Lyy = _np.sqrt(_np.abs(fftinfo.ENBW*Pyy))  # [V_rms]
    fftinfo.Lxy = _np.sqrt(_np.abs(fftinfo.ENBW*Pxy))  # [V_rms]

    # Rescale RMS values to Amplitude values (assumes a zero-mean sine-wave)
    # Just the points that split their energy into negative frequencies
    fftinfo.Lxx[1:-2] = _np.sqrt(2)*fftinfo.Lxx[1:-2]  # [V],
    fftinfo.Lyy[1:-2] = _np.sqrt(2)*fftinfo.Lyy[1:-2]  # [V],
    fftinfo.Lxy[1:-2] = _np.sqrt(2)*fftinfo.Lxy[1:-2]  # [V],
    if nfft%2:  # Odd
        fftinfo.Lxx[-1] = _np.sqrt(2)*fftinfo.Lxx[-1]
        fftinfo.Lyy[-1] = _np.sqrt(2)*fftinfo.Lyy[-1]
        fftinfo.Lxy[-1] = _np.sqrt(2)*fftinfo.Lxy[-1]
    # endif nfft/2 is odd

    fftinfo.varLxx = (fftinfo.Lxx**2)*(fftinfo.varPxx/_np.abs(Pxx)**2)
    fftinfo.varLyy = (fftinfo.Lyy**2)*(fftinfo.varPyy/_np.abs(Pyy)**2)
    fftinfo.varLxy = (fftinfo.Lxy**2)*(fftinfo.varPxy/_np.abs(Pxy)**2)

    # Store everything
    fftinfo.Navr = Navr
    fftinfo.overlap = windowoverlap
    fftinfo.window = windowfunction
    fftinfo.freq = freq
    fftinfo.Pxx = Pxx
    fftinfo.Pyy = Pyy
    fftinfo.Pxy = Pxy
    fftinfo.Cxy = Cxy
    fftinfo.phi_xy = phi_xy

    # ==================================================================== #

    # Plot the comparisons
    if plotit:
        afont = {'fontname':'Arial','fontsize':14}

        #The input signals versus time
        _plt.figure()
        _ax1 = _plt.subplot(2,2,1)
        _ax1.plot(tvec, sigx, 'b-', tvec, sigy, 'r-')
        _ax1.set_title('Input Signals', **afont)
        _ax1.set_xlabel('t[s]', **afont)
        _ax1.set_ylabel('sig_x,sig_y[V]', **afont)
        _plt.axvline(x=tbounds[0], color='k')
        _plt.axvline(x=tbounds[1], color='k')

        _ax2 = _plt.subplot(2,2,2)
#        _ax2.plot(1e-3*freq,_np.abs(Pxx), 'b-')
#        _ax2.plot(1e-3*freq,_np.abs(Pyy), 'r-')
#        _ax2.plot(1e-3*freq,_np.abs(Pxy), 'k-')

#        _ax2.semilogy(1e-3*freq.flatten(), _np.abs(Pxx.flatten()), 'b-')
#        _ax2.semilogy(1e-3*freq.flatten(), _np.abs(Pyy.flatten()), 'r-')
#        _ax2.semilogy(1e-3*freq.flatten(), _np.abs(Pxy.flatten()), 'k-')
        _ax2.plot(1e-3*freq, 10*_np.log10(_np.abs(Pxx)), 'b-')
        _ax2.plot(1e-3*freq, 10*_np.log10(_np.abs(Pyy)), 'r-')
        _ax2.plot(1e-3*freq, 10*_np.log10(_np.abs(Pxy)), 'k-')
        _ax2.set_title('Power Spectra', **afont)
        _ax2.set_ylabel(r'P$_{ij}$ [dB/Hz]', **afont),
        _ax2.set_xlabel('f[kHz]', **afont)
        _ax2.set_xlim(0, 1.01e-3*freq[-1])

        # _plt.setp(ax1.get_xticklabels(), fontsize=6)

        # _ax4.text(0.20, 0.15, 'P_{xx}', 'Color', 'b', 'units', 'normalized',
        #           'FontSize', 14, 'FontName', 'Arial')
        # _ax4.text(0.20, 0.35, 'P_{yy}', 'Color', 'r', 'units', 'normalized',
        #           'FontSize', 14, 'FontName', 'Arial')
        # _ax4.text(0.20, 0.55, 'P_{xy}', 'Color', 'k', 'units', 'normalized',
        #           'FontSize', 14, 'FontName', 'Arial')

        _ax3 = _plt.subplot(2, 2, 3, sharex=_ax2)
        _ax3.plot(1e-3*freq, Cxy, 'k-')
        _plt.axhline(y=1.0/Navr, color='k')
        _ax3.set_title('Cross-Coherence', **afont)
        _ax3.set_ylabel(r'C$_{xy}$', **afont)
        _ax3.set_xlabel('f[kHz]', **afont)
        _ax3.set_xlim(0, 1.01e-3*freq[-1])

        # _ax4.text(0.80, 0.90, 'C_{xy}', 'Color', 'k', 'units', 'normalized',
        #          'FontSize', 14, 'FontName', 'Arial')

        _ax4 = _plt.subplot(2, 2, 4, sharex=_ax2)
        _ax4.plot(1e-3*freq, phi_xy, 'k-')
        _ax4.set_title('Cross-Phase', **afont)
        _ax4.set_ylabel(r'$\phi_{xy}$', **afont)
        _ax4.set_xlabel('f[kHz]', **afont)
        _ax4.set_xlim(0, 1.01e-3*freq[-1])
        # _ax4.text(0.80, 0.90, '\phi_{xy}', 'Color', 'k', 'units',
        #           'normalized', 'FontSize', 14, 'FontName', 'Arial')

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

def detrend_mean(x):
    return x - _np.mean(x)

def detrend_none(x):
    return x

def detrend_linear(x, nargout=1):
    """Remove the best fit line from x"""
    # I'm going to regress x on xx=range(len(x)) and return
    # x - (b*xx+a)
#    xx = _np.arange(len(x), typecode=x.typecode())
    xx = _np.arange(len(x)) #, dtype=type(x))
    X = _np.transpose(_np.array([xx]+[x]))
    C = _np.cov(X)
    b = C[0,1]/C[0,0]
    a = _np.mean(x) - b*_np.mean(xx)
    if nargout == 1:
        return x-(b*xx+a)
    elif nargout == 2:
        return b, a
    # endif

# =========================================================================== #


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
    from scipy import signal
    c = _preconvolve_fft(a, b)
    # Convolution of signal:
    return signal.fftconvolve(c, a, mode=mode)

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
    from scipy import signal
    c = _preconvolve_fft(a, b)
    # Convolution of reverse signal:
    return signal.fftconvolve(c, a[::-1], mode=mode)

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

    # Cross-power weighted average frequency
    info.fweighted = _np.dot(freq[inds].reshape(len(inds),1), _np.ones((1,_np.size(Pxy,axis=1)), dtype=float))
    info.fweighted = _np.trapz( info.fweighted*_np.abs(Pxy[inds,:]))
    info.fweighted /= _np.trapz(_np.abs(Pxy[inds,:]))
    return Pxy_i, Pxx_i, Pyy_i, Cxy_i, ph_i, info
# end def integratespectra


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

    Coh    = _np.array( _np.size(Pxy),dtype=_np.float64)
    varCoh = _np.zeros_like( Coh )
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

    if meansquared is False:
        # Return the coherence, not the mean-squared coherence
        varCoh = 0.25*varCoh/Coh  # (0.5*(Coh**-0.5))**2.0 * varCoh
        Coh = _np.sqrt(Coh)
    # endif

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
    u_n = _ut.interp( tt, u_t, ei=None, xo=ti)
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


# ========================================================================= #
# ========================================================================= #


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
        if sigy is None:
            sigy = sigx
        #endif

        self.tvec = tvec
        self.sigx = sigx
        self.sigy = sigy

        # == #

        self.tbounds = kwargs.get( 'tbounds', [ tvec[0], tvec[-1] ] )
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


    # =========================================================== #

    def fftpwelch(self):
        # Call the fft_pwelch function that is defined above
        self.freq, self.Pxy, self.Pxx, self.Pyy, \
        self.Cxy, self.phi_xy, self.fftinfo = \
            fft_pwelch(self.tvec, self.sigx, self.sigy, self.tbounds,
                       Navr=self.Navr, windowoverlap=self.overlap,
                       windowfunction=self.window, useMLAB=self.useMLAB,
                       plotit=self.plotit, verbose=self.verbose)

    def pwelch(self):
        self.Xstft()
        self.Ystft()
        self.Pstft()
        self.averagewins()

    # =============== #

    def crosscorr(self):

        # cross correlation from the FFT
        #     if we calculated one-sided spectra, then we have to add back
        #     in the Nyquist components before inverse tranforming.

#        freq = self.freq.copy()
#        if self.onesided:
#            freq = _ut.cylsym_odd(freq)
##            freq = _np.concatenate((freq, freq+freq[-1]))
#            freq, inds = _np.unique(freq, return_index=True)
#            freq = _np.fft.ifftshift(freq)
#        # endif
#        i0 = _np.where(freq>0)[0][0]-1
#        print(freq[i0])
        nfft = self.nwins
#        nfft = len(freq)
        inds = _np.ones( (nfft,), dtype=bool)

        if hasattr(self,'Pxx'):
            Pxx_seg = self.Pxx_seg.copy()
            Pxx = self.Pxx.copy()

            if self.onesided:
                i0 = int(len(Pxx)+1)
                inds[i0] = False
                Pxx[1:-2] *= 0.5
                Pxx_seg[:,1:-2] *= 0.5
                Pxx = _ut.cylsym_even(Pxx)
                Pxx_seg = _ut.cylsym_even(Pxx_seg.T).T
            # endif
            Pxx = Pxx[inds]
            Pxx_seg = Pxx_seg[:, inds]

            Pxx = _np.fft.ifftshift(Pxx)
            Pxx_seg = _np.fft.ifftshift(Pxx_seg, axes=1)

        if hasattr(self,'Pyy'):
            Pyy_seg = self.Pyy_seg.copy()
            Pyy = self.Pyy.copy()

            if self.onesided:
                i0 = int(len(Pyy)+1)
                inds[i0] = False
                Pyy[1:-2] *= 0.5
                Pyy_seg[:,1:-2] *= 0.5
                Pyy = _ut.cylsym_even(Pyy)
                Pyy_seg = _ut.cylsym_even(Pyy_seg.T).T
            # endif
            Pyy = Pyy[inds]
            Pyy_seg = Pyy_seg[:, inds]

            Pyy = _np.fft.ifftshift(Pyy)
            Pyy_seg = _np.fft.ifftshift(Pyy_seg, axes=1)

        if hasattr(self,'Pxy'):
            Pxy_seg = self.Pxy_seg.copy()
            Pxy = self.Pxy.copy()

            if self.onesided:
                i0 = int(len(Pxy)+1)
                inds[i0] = False
#                Pxy[1:-2] *= 0.5
#                Pxy_seg[:,1:-2] *= 0.5
                Pxy = _ut.cylsym_even(Pxy)
                Pxy_seg = _ut.cylsym_even(Pxy_seg.T).T
            # endif
            Pxy = Pxy[inds]
            Pxy_seg = Pxy_seg[:, inds]

            Pxy = _np.fft.ifftshift(Pxy)
            Pxy_seg = _np.fft.ifftshift(Pxy_seg, axes=1)

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
            Cxy_seg = _np.fft.ifftshift(Cxy_seg, axes=1)
        # end if

#        nfft -= sum(~inds)
        nfft = len(self.freq)
        mult = 1.0
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
            self.Rxx_seg = _np.fft.ifft(Pxx_seg, n=nfft, axis=1)
            self.Rxx = _np.fft.ifft(Pxx, n=nfft)

            self.Rxx_seg *= mult
            self.Rxx *= mult

            self.Rxx_seg = _np.fft.fftshift(self.Rxx_seg, axes=1)
            self.Rxx = _np.fft.fftshift(self.Rxx)

        if hasattr(self,'Pyy'):
            self.Ryy_seg = _np.fft.ifft(Pyy_seg, n=nfft, axis=1)
            self.Ryy = _np.fft.ifft(Pyy, n=nfft)

            self.Ryy_seg *= mult
            self.Ryy *= mult

            self.Ryy_seg = _np.fft.fftshift(self.Ryy_seg, axes=1)
            self.Ryy = _np.fft.fftshift(self.Ryy)

        if hasattr(self,'Pxy'):
            self.Rxy_seg = _np.fft.ifft(Pxy_seg, n=nfft, axis=1)
            self.Rxy = _np.fft.ifft(Pxy, n=nfft)

            self.Rxy_seg *= mult
            self.Rxy *= mult

            self.Rxy_seg = _np.fft.fftshift(self.Rxy_seg, axes=1)
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

        self.lags = _np.linspace(-nfft/self.Fs, nfft/self.Fs, nfft)
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
            self.Lxx_seg = _np.sqrt(2)*self.Lxx_seg # V_amp

        if hasattr(self,'Yseg'):
            self.Pyy_seg = self.Yseg*_np.conj(self.Yseg)

            self.Lyy_seg = _np.sqrt(_np.abs(self.ENBW*self.Pyy_seg))  # [V_rms]
            self.Lyy_seg = _np.sqrt(2)*self.Lyy_seg # V_amp

        if hasattr(self, 'Xseg') and hasattr(self,'Yseg'):
            self.Pxy_seg = self.Xseg*_np.conj(self.Yseg)

            self.Lxy_seg = _np.sqrt(_np.abs(self.ENBW*self.Pxy_seg))  # [V_rms]
            self.Lxy_seg = _np.sqrt(2)*self.Lxy_seg # V_amp

            # Save the cross-phase as well
            self.phixy_seg = _np.angle(self.Pxy_seg)  # [rad], Cross-phase of each segment

            # Coherence
            self.Cxy2_seg = _np.abs( self.Pxy_seg*_np.conj( self.Pxy_seg ) )/( _np.abs(self.Pxx_seg)*_np.abs(self.Pyy_seg) )
            self.Cxy_seg = _np.sqrt(self.Cxy2_seg)
    # end def

    # =============== #

    def averagewins(self):
        if hasattr(self, 'Pxx_seg'):
            self.Pxx = _np.mean(self.Pxx_seg, axis=0)

            # use the RMS for the standard deviation
            self.varPxx = _np.mean(self.Pxx_seg**2.0, axis=0)

        if hasattr(self, 'Pyy_seg'):
            self.Pyy = _np.mean(self.Pyy_seg, axis=0)

            # use the RMS for the standard deviation
            self.varPyy = _np.mean(self.Pyy_seg**2.0, axis=0)

        if hasattr(self, 'Pxy_seg'):
            self.Pxy = _np.mean(self.Pxy_seg, axis=0)

            # use the RMS for the standard deviation
            self.varPxy = _np.mean(self.Pxy_seg**2.0, axis=0)

            # Mean-squared coherence
            self.Cxy2 = _np.abs( self.Pxy*_np.conj( self.Pxy ) )/( _np.abs(self.Pxx)*_np.abs(self.Pyy) )
            self.Cxy = _np.sqrt(self.Cxy2)

            # ========================== #
            # Uncertainty and phase part

            # derived using error propagation from eq 23 for gamma^2 in
            # J.S. Bendat, Journal of Sound an Vibration 59(3), 405-421, 1978
            # fftinfo.varCxy = _np.zeros_like(Cxy)
            self.varCxy = ((1-self.Cxy2)/_np.sqrt(2*self.Navr))**2.0

            # Estimate the variance in the power spectra: this requires building
            # a distribution by varying the parameters used in the FFT, nwindows,
            # nfft, windowfunction, etc.  I don't do this right now
            self.varPxy = self.Pxx*self.Pyy*(1.0-self.Cxy)/self.Navr

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

            # Rescale RMS values to Amplitude values (assumes a zero-mean sine-wave)
            # Just the points that split their energy into negative frequencies
            self.Lxx[1:-2] = _np.sqrt(2)*self.Lxx[1:-2]  # [V],

            if self.nfft%2:  # Odd
                self.Lxx[-1] = _np.sqrt(2)*self.Lxx[-1]
            # endif nfft/2 is odd
            self.varLxx = (self.Lxx**2)*(self.varPxx/_np.abs(self.Pxx)**2)

        if hasattr(self,'Pyy'):
            self.Lyy = _np.sqrt(_np.abs(self.ENBW*self.Pyy))  # [V_rms]
            self.Lyy[1:-2] = _np.sqrt(2)*self.Lyy[1:-2]  # [V],

            if self.nfft%2:  # Odd
                self.Lyy[-1] = _np.sqrt(2)*self.Lyy[-1]
            # endif nfft/2 is odd
            self.varLyy = (self.Lyy**2)*(self.varPyy/_np.abs(self.Pyy)**2)

        if hasattr(self, 'Pxy'):
            self.Lxy = _np.sqrt(_np.abs(self.ENBW*self.Pxy))  # [V_rms]
            self.Lxy[1:-2] = _np.sqrt(2)*self.Lxy[1:-2]  # [V],
            if self.nfft%2:  # Odd
                self.Lxy[-1] = _np.sqrt(2)*self.Lxy[-1]
            # endif nfft/2 is odd
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
        return len(tvec)/(tvec[-1]-tvec[0])

    @staticmethod
    def __ibounds__(tvec, tbounds):
        ib1 = _np.floor(1 + (tbounds[0]-tvec[0])*fftanal.__Fs__(tvec))
        ib2 = _np.floor(1 + (tbounds[1]-tvec[0])*fftanal.__Fs__(tvec))
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
    # end def getNoverlap

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
            self.nwins = self.nsig.copy()
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

    def integrate_spectra(self):
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

    def fft(self, sig, nfft):
        #The FFT output from matlab isn't normalized:
        # y_n = sum[ y_m.*exp( 2_np.pi*1i*(n/N)*m ) ]
        # The inverse is normalized::
        # y_m = (1/N)*sum[ y_n.*exp( -2_np.pi*1i*(n/N)*m ) ]
        #
        # Python normalizations are optional, pick it to match MATLAB
        return _np.fft.fft(sig, n=nfft)

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

            # Windowed signal segment:
            # - To get the most accurate spectrum, background subtract
            # xtemp = win*_dsp.detrend(xtemp, type='constant')
            xtemp = win*self.detrend(xtemp)
            pseg[gg] = _np.trapz(xtemp**2.0, x=tvec[istart:iend])
            xtemp = self.fft(xtemp, nfft)
            Xfft[gg, :] = _np.fft.fftshift(xtemp)
#            Xfft[gg, 0:nfft] = _np.fft.fftshift(xtemp)
        #endfor loop over fft windows
        freq = _np.fft.fftfreq(nfft, 1.0/Fs)
        freq = _np.fft.fftshift(freq)

        # Remove gain of the window function to yield the RMS Power spectrum
        # in each segment (constant peak amplitude)
        Xfft /= S1   # Vrms
        pseg /= S2

        # Compute the spectral density from the RMS spectrum
        # (constant noise floor)
        Xfft /= _np.sqrt(ENBW)  # [V/Hz^0.5]

        if self.onesided:
            freq = _np.fft.ifftshift(freq)
            Xfft = _np.fft.ifftshift(Xfft, axes=1)

            freq = freq[:Nnyquist]  # [Hz]
            Xfft = Xfft[:Navr, :Nnyquist]  # [Hz]
#            freq = freq[:Nnyquist-1]  # [Hz]
#            Xfft = Xfft[:Navr, :Nnyquist-1]  # [Hz]

            # Real signals equally split their energy between positive and negative frequencies
            Xfft[:, 1:-2] = _np.sqrt(2)*Xfft[:, 1:-2]
            if nfft%2:  # odd
                Xfft[:,-1] = _np.sqrt(2)*Xfft[:,-1]
            # endif
        # end if

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
        _ax2.plot(1e-3*self.freq, 10*_np.log10( _np.abs( self.Pxx ) ),'b-')
        _ax2.plot(1e-3*self.freq, 10*_np.log10( _np.abs( self.Pyy ) ),'r-')
        _ax2.plot(1e-3*self.freq, 10*_np.log10( _np.abs( self.Pxy ) ),'k-')
        _ax2.set_title('Power Spectra',**self.afont)
        _ax2.set_ylabel('P_{ij} [dB/Hz]',**self.afont),
        _ax2.set_xlabel('f[kHz]',**self.afont)
        _ax2.set_xlim(0,1.01e-3*self.freq[-1])

        #_ax2.text(0.20, 0.15, 'P_{xx}', 'Color', 'b', 'units', 'normalized',
        #          'FontSize', 14, 'FontName', 'Arial')
        # _ax2.text(0.20, 0.35, 'P_{yy}', 'Color', 'r', 'units', 'normalized',
        #           'FontSize', 14, 'FontName', 'Arial')
        # _ax2.text(0.20, 0.55, 'P_{xy}', 'Color', 'k', 'units', 'normalized',
        #           'FontSize', 14, 'FontName', 'Arial')

        self.ax3 = _plt.subplot(2,2,3, sharex=self.ax2)
        _ax3 = self.ax3
        _ax3.plot(1e-3*self.freq, self.Cxy,'k-')
        _ax3.axhline(y=1./self.Navr,color='k')
        _ax3.set_title('Cross-Coherence',**self.afont)
        _ax3.set_ylabel('C_{xy}',**self.afont)
        _ax3.set_xlabel('f[kHz]',**self.afont)
        _ax3.set_xlim(0,1.01e-3*self.freq[-1])

        # _ax3.text(0.80, 0.90, 'C_{xy}', 'Color', 'k', 'units', 'normalized',
        #           'FontSize', 14, 'FontName', 'Arial')

        self.ax4 = _plt.subplot(2,2,4, sharex=self.ax2)
        _ax4 = self.ax4
        _ax4.plot(1e-3*self.freq,self.phi_xy,'k-')
        _ax4.set_title('Cross-Phase',**self.afont)
        _ax4.set_ylabel('\phi_{xy}',**self.afont)
        _ax4.set_xlabel('f[kHz]',**self.afont)
        _ax4.set_xlim(0,1.01e-3*self.freq[-1])
        # _ax4.text(0.80, 0.90, '\phi_{xy}', 'Color', 'k', 'units', 'normalized',
        #           'FontSize', 14, 'FontName', 'Arial')

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
        _plt.xlim(0, 1.01e-3*self.freq[-1])
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
        _plt.xlim(0, 1.01e-3*self.freq[-1])
        _plt.draw()
#        _plt.show()

    # ===================================================================== #

    def plotphxy(self):
        self.fig = _plt.figure()
        _plt.plot(1e-3*self.freq, self.phi_xy, 'k-')
        _plt.title('Cross-Phase', **self.afont)
        _plt.ylabel('\phi_{xy}', **self.afont)
        _plt.xlabel('f[kHz]', **self.afont)
        _plt.xlim(0, 1.01e-3*self.freq[-1])
        _plt.draw()
#        _plt.show()

    # ===================================================================== #

    def __calcAmp__(self, tvec, sigx, sigy, tbounds, nn=8, ol=0.5,
                    ww='hanning'):

        # The amplitude is most accurately calculated by using several windows
        self.frqA, self.Axy, self.Axx, self.Ayy, self.aCxy, _, _ = \
            fft_pwelch(tvec, sigx, sigy, tbounds, Navr=nn, windowoverlap=ol,
                       windowfunction=ww, useMLAB=self.useMLAB, plotit=0,
                       verbose=self.verbose)
        self.__plotAmp__()

    def __plotAmp__(self):
        self.fig = _plt.figure()
        _plt.plot(1e-3*self.frqA, 10*_np.log10(_np.abs(self.Axx)), 'b-')
        _plt.plot(1e-3*self.frqA, 10*_np.log10(_np.abs(self.Ayy)), 'r-')
        _plt.plot(1e-3*self.frqA, 10*_np.log10(_np.abs(self.Axy)), 'k-')
        _plt.title('Power Spectra', **self.afont)
        _plt.ylabel('P_{ij} [dB/Hz]', **self.afont),
        _plt.xlabel('f[kHz]', **self.afont)
        _plt.xlim(0, 1.01e-3*self.frqA[-1])
        _plt.draw()
#        _plt.show()

    # ===================================================================== #

    def __calcPh1__(self, tvec, sigx, sigy, tbounds, nn=1, ol=0.0, ww='box'):

        # The amplitude is most accurately calculated by using several windows
        self.frqP, _, _, _, _, self.ph, _ = \
            fft_pwelch(tvec, sigx, sigy, tbounds, Navr=nn, windowoverlap=ol,
                       windowfunction=ww, useMLAB=self.useMLAB, plotit=0,
                       verbose=self.verbose)
        self.__plotPh1__()

    def __plotPh1__(self):
        self.fig = _plt.figure()
        _plt.plot(1e-3*self.frqP, self.ph,'k-')
        _plt.title('Cross-Phase', **self.afont)
        _plt.ylabel('\phi_{xy}', **self.afont)
        _plt.xlabel('f[kHz]', **self.afont)
        _plt.xlim(0, 1.01e-3*self.frqP[-1])
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
            _ax1.set_title('Input Signals',**afont)
            _ax1.set_xlabel('t[s]',**afont)
            _ax1.set_ylabel('sig_x,sig_y[V]',**afont)
            _plt.axvline(x=ft1.tbounds[0],color='k')
            _plt.axvline(x=ft1.tbounds[1],color='k')

            _ax2 = _plt.subplot(2,2,2)
            _ax2.plot(1e-3*ft1.freq, 10*_np.log10(_np.abs(ft1.Pxy)), 'k-')
            _ax2.plot(1e-3*ft2.freq, 10*_np.log10(_np.abs(ft2.Pxy)), 'm--')
            _ax2.set_title('Power Spectra Comparison', **afont)
            _ax2.set_ylabel(r'P$_{ij}$ [dB/Hz]', **afont),
            _ax2.set_xlabel('f[kHz]', **afont)
            _ax2.set_xlim(0, 1.01e-3*ft1.freq[-1])

            _ax3 = _plt.subplot(2, 2, 3, sharex=_ax2)
            _ax3.plot(1e-3*ft1.freq, ft1.Cxy, 'k-')
            _ax3.plot(1e-3*ft2.freq, ft2.Cxy, 'm--')
            _plt.axhline(y=1.0/ft1.Navr, color='k')
            _ax3.set_title('Coherence', **afont)
            _ax3.set_ylabel(r'C$_{xy}$', **afont)
            _ax3.set_xlabel('f[kHz]', **afont)
            _ax3.set_xlim(0, 1.01e-3*ft2.freq[-1])

            # _ax4.text(0.80, 0.90, 'C_{xy}', 'Color', 'k', 'units', 'normalized',
            #          'FontSize', 14, 'FontName', 'Arial')

            _ax4 = _plt.subplot(2, 2, 4, sharex=_ax2)
            _ax4.plot(1e-3*ft1.freq, ft1.phi_xy, 'k-')
            _ax4.plot(1e-3*ft2.freq, -1.0*ft2.phi_xy, 'm--')
            _ax4.set_title('Phase', **afont)
            _ax4.set_ylabel(r'$\phi_{xy}$',**afont)
            _ax4.set_xlabel('f[kHz]',**afont)
            _ax4.set_xlim(0,1.01e-3*ft1.freq[-1])

            _plt.draw()
            # _plt.show()
        # endif plotit
    # end def testFFtanal

    # ===================================================================== #

#end class fftanal

# ========================================================================== #


def test_fftanal(useMLAB=True, plotit=True, nargout=0, tstsigs = None):
    ##Generate test data for the no input case:

    if tstsigs is None:
        #Minimize the spectral leakage:
        df = 5.0   #Hz
        N  = 2**12 #Numper of points in time-series
        tvec = (1.0/df)*_np.arange(0.0,1.0,1.0/(N))

        #Sine-wave
        sigx = _np.sin(2.0*_np.pi*(df*30.0)*tvec)     #Shifted time-series
        sigx *= 0.005
        sigx += 7.0

        #Noisy phase-shifted sine-wave
        sigy = _np.sin(2.0*_np.pi*(df*30.0)*tvec-_np.pi/4.0)
        sigy *= 0.005
        sigy += 0.05*_np.random.standard_normal( (tvec.shape[0],) )
        sigy += 2.5

        #Square-wave
        #sigx = 10.0 + _dsp.square(2.0*_np.pi*(df*100.0)*tvec)    #Shifted square wave

        #sigy = sigx
    else:
        tvec = tstsigs[0]
        sigx = tstsigs[1]
        sigy = tstsigs[2]
    # endif
    # ------
    #
    #    _plt.Figure, set(gca,'FontSize',14,'FontName','Arial'), hold all
    #    xlabel('t[s]'), ylabel('Y_{test} [a.u.]')
    #    plot(tvec,[sigx;sigy],'-'),
    #    axis([-0.01*tvec[0],1.01*tvec[-1],0,15])
    #
    # -----

    ft = fftanal(tvec,sigx,sigy,tbounds = [tvec[1],tvec[-1]],
            Navr = 8, windowoverlap = 0.5, windowfunction = 'hamming',
            useMLAB=useMLAB, plotit=plotit, verbose=True)

    if not useMLAB:
        ft2 = fftanal()
        ft2.init(tvec, sigx, sigy, tbounds=[tvec[1], tvec[-1]],
                 Navr=8, windowoverlap=0.5, windowfunction='hamming')
        ft2.pwelch()
        ft2.plotall()
        ft2.plotspec('Pxy')
        ft2.plotspec('Cxy', vbnds=[0, 1.0])
        ft2.plotspec('Lxy')
        ft2.plotspec('Lyy')

        ft2.crosscorr()
        ft2.plotcorr()

    if nargout>0:
        return ft

#    fft_pwelch(tvec,sigx,sigy, [tvec[1],tvec[-1]], Navr = 8, windowoverlap = 0.5, windowfunction = 'hamming', useMLAB=False, plotit=True, verbose=True)


#    [freq,Pxy] = fft_pwelch(tvec,Zece[:,1],Zece[:,2],[0.1,0.3],useMLAB=True,plotit=True)
#end testFFTanal

def test():
    tst = fftanal(verbose=True)
    tst.__testFFTanal__()


if __name__ == "__main__":
    test()
    test_fftanal()

# ========================================================================== #
# ========================================================================== #

