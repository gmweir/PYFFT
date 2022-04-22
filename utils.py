# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:36:37 2022

@author: gawe
"""
# ========================================================================= #
# ========================================================================= #

import numpy as _np
import matplotlib.pyplot as _plt
import scipy.signal as _dsp
import pybaseutils.utils as _ut
from pybaseutils.utils import detrend_mean, detrend_none, detrend_linear


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


# ========================================================================== #


def mean_angle(phi, vphi=None, dim=0, angle_range=0.5*_np.pi, vsyst=None):
    """
      Proper way to average a phase angle is to convert from polar (imaginary)
      coordinates to a cartesian representation and average the components.
    """

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














# ========================================================================= #
# ========================================================================= #



