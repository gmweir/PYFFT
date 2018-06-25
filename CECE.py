# ========================================================================== #
#
#
# ========================================================================== #
# ========================================================================== #
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:13:05 2016

@author: gawe
"""
# ========================================================================== #
# ========================================================================== #

from __future__ import absolute_import, with_statement, absolute_import, \
                       division, print_function, unicode_literals

import numpy as _np
import matplotlib.pyplot as _plt
import scipy.signal as _dsp
import scipy.stats  as stat

#from pybaseutils.data import data as DataStruct
import IO as _io
#from .HeatPulse_Funcs import __HeatPulse_FFTbase__ #, loadFFTdata
from .fft_analysis import butter_lowpass

from W7X.mdsfetch import QMF as _qmfp_base_
from W7X.fetch import QMF as _qmfr_base_

try:
    __import__('scipy.signal.iirnotch')
    iirnotch = _dsp.iirnotch   # scipy 0.19.1
except:
    from .notch_filter import iirnotch
    pass
# end try

__metaclass__ = type

# ========================================================================== #

def loadHDF5data(filname, flds):
    """
    flds = ['COIL', 'QME', 'QMJ', 'QTB', 'XPLOG', 'QMEZ', 'ECRH']
    """
    all_data = _io.loadHDF5(filname, False, True)
    return tuple([all_data[flds[ii]] for ii in range(len(flds)) if flds[ii] in all_data])

# ===========================0 #

_plt.rc('font',size = 16)
_figsize = _plt.rcParams["figure.figsize"]
_figsize = (1.2*_figsize[0],1.2*_figsize[1])

class Data(DataStruct):
#class Data(object):
    _fig_size = _figsize
    clrs = 'bgrcmyk'
    def __init__(self, d=None, verbose=False):
        d = _np.copy(d)

        if 'Zrad' in d: d = self.loadQMEZ(d, verbose)    # endif
        if 'ece_roa' in d:
            d['xx'] = d.pop('ece_roa')
        elif 'ece_freq' in d:
            d['xx'] = d.pop('ece_freq')
        else:
            d['xx'] = _np.asarray(range(d['nch']), dtype=_np.float64)
        # endif
        if 'igch' in d and len(d['igch'])>0:
            imsk = _np.ones( (d['nch'],), dtype=bool)
            imsk[d['igch']-1] = False
            d['xx'] = d['xx'][imsk]
#            d['yy'] = d['yy'][:,imsk] # already ignored
#        # endif

        super(Data, self).__init__(self, tt=d, yy=[], xx=None, systvar=None, statvar=None, verbose=None)
        self.update(d)
        self.verbose = verbose
        if self.verbose: print("Initialized a base data class")  # end if
    # end def __init__

    def loadQMEZ(self, d=None, verbose=False):
        # QMEZ
        if 'Zrad' in d: d['yy'] = d.pop('Zrad')     # endif
        if 'varZ' in d: d['systvar'] = d.pop('varZ')     # endif
        return d

    # end def __init__
#        self.nt = 0
#        self.Fs = 0
#        self.nch = 0
#
#        self.tt = None
#        self.ff
#        self.xx
#
#        self.sig = None
#        self.systvar
#        self.statvar

#        active
#
#        LO_2x1
#        IF
#        RF_2x8
#        BW_2x8
#        ifbandwidth
#        LPF_None
#        HPF_None
#
#        antenna
#        wgfilter_2x2
#
#        coupling
#        Fs_0.002
#        nt
#        nch_32
#
#        filter_4x16
#        _t - tt
#        _f - ff
#        _s - sig
#
#        att
#        gains_2x16
#        ADC
#        sens
#        varsens
#
#        comment_''
#        shot_shotidentifier
#        host_mds-data-2
#        treename
#        tree_mdsplus
#        RP_class
#
#        verbose

    def update(self, d=None):
        if d is not None:
            if type(d) != dict:    d = d.dict_from_class()     # endif
            super(Data, self).__init__(d)
        # endif
    # end def update

    def loadHDF5fld(self, sfile, fld):
        d = _io.loadHDF5(sfile, fld)
        self.update(d)
# end class Data


#class CECE(__HeatPulse_FFTbase__):
#    def __init__(self, QMFdat={}):
#        if type(QMFdat) != dict:    QMFdat = QMFdat.dict_from_class()     # endif
#
#        # Populate the class with input data from the dictionary
#        super(CECE, self).__init__(HPdata=QMFdat)
#
#        self.freq = None   # self.frequencyAxis
##        self.
#
#    # end def


class CECE(Data):
    # now we define the spectral functions. We make use of the inbuilt scipy.signal.csd for estimating spectral densities.
    # this helps avoid the convention unpleasantness which we can find using ffts.

    def estimate_Sxy(self,x,y,fs,wl = 1024):
        return _dsp.csd(y,x,fs,nperseg = wl,return_onesided=False)

    def estimate_Gxy(self,x,y,fs,wl = 4096):
        return _dsp.csd(y,x,fs,nperseg = wl,return_onesided=True)

    def estimate_gamma(self,x,y,fs,wl = 4096):
        f,Gxy = self.estimate_Gxy(x,y,fs,wl)
        f,Gxx = self.estimate_Gxy(x,x,fs,wl)
        f,Gyy = self.estimate_Gxy(y,y,fs,wl)

        gamma = _np.abs(Gxy)/_np.sqrt(Gxx*Gyy)
        return f,gamma

    def estimate_Gxynorm(self,x,y,fs,wl = 4096):
        f,Gxy = self.estimate_Gxy(x,y,fs,wl)
        f,Gxx = self.estimate_Gxy(x,x,fs,wl)
        f,Gyy = self.estimate_Gxy(y,y,fs,wl)

        Gxynorm = Gxy/_np.sqrt(Gxx*Gyy)
        return f,Gxynorm

    def integrateSx(self, freq, Sx, f0 = None,f1 = None):

        _Sx = Sx.squeeze()

        df = _np.abs(freq[1]-freq[0])

        if (f0 != None) and (f1 != None):
            idx = _np.argwhere((_np.abs(freq) > f0) & (_np.abs(freq) < f1))
            return _np.sum(_Sx[idx])*df

        else:
            return _np.sum(_Sx)*df

    def getCorrelations(self, nTmode = 'amplitude', reflCh = [7,8], reflConst = [1,1]):
        """

        d.getCorrelations(self, backgroundData = None, nTmode = 'amplitude', reflCh = [7,8], refConst = [1,1])

        Here we calculate the cross spectra using time averaging to boost the statistics
        The data is windowed by a Hanning window, unless hanning = False, then a square
        window is used.

        If the data is not divisable by windowLength, then the remainder data will
        not contribute to the spectra.

        windowLength is a property of the data since it is shared between auto and cross
        spectra. If you want it to be different from 4096, set it before coputing the spectra
        """

        if self.verbose:
            print('making cross-correlation spectrum and coherence')

        wl = self.windowLength
        fs = self.Fs
        nw = self.Navr = len(self.sig[0,:])//wl

        if self.diagnostic == 'nTphase':

            if (reflCh[0] in self.channels) and (reflCh[1] in self.channels):

                idx1 = _np.argwhere(_np.array(self.channels) == reflCh[0])
                idx2 = _np.argwhere(_np.array(self.channels) == reflCh[1])

                idx1 = idx1.squeeze()
                idx2 = idx2.squeeze()

                c = (reflConst[0]*self.sig[idx1,:] + 1j*reflConst[1]*self.sig[idx2,:])

                if nTmode == 'amplitude':
                    if self.verbose:
                        print('nTphase selected in amplitude mode')

                    if self.verbose:
                        print('taking amplitude of reflectometer channels')

                    self.sig[idx1] = _np.abs(c)
                    self.nch = self.nch-1
                    self.nb = (self.nch*(self.nch-1))//2

                    del self.channels[idx2]

                elif nTmode == 'IQ':

                    if self.verbose:
                        print('nTphase selected in IQ mode')

                    if self.verbose:
                        print('making reflectometer channels 1 complex trace')


                    self.sig = self.sig + 1j*_np.zeros(self.sig.shape)
                    self.sig[idx1] = c
                    self.nch = self.nch-1
                    self.nb = (self.nch*(self.nch-1))//2
                    del self.channels[idx2]


                elif nTmode == 'phase':

                    if self.verbose:
                        print('nTphase selected in phase mode')

                    if (reflCh[0] in self.channels) and (reflCh[1] in self.channels):
                        if self.verbose:
                            print('taking unwrapped reflectometer phase')

                    self.sig[idx1] = _np.angle(c).squeeze()

                    self.nch = self.nch-1
                    self.nb = (self.nch*(self.nch-1))//2
                    del self.channels[idx2]

                elif nTmode == 'unwrappedPhase':
                    if self.verbose:
                        print('nTphase selected in unwrapped phase mode')

                    if (reflCh[0] in self.channels) and (reflCh[1] in self.channels):
                        if self.verbose:
                            print('taking unwrapped reflectometer phase')

                    self.sig[_np.min([idx1,idx2])] = _np.unwrap(_np.angle(c).squeeze())

                    self.nch = self.nch-1
                    self.nb = (self.nch*(self.nch-1))//2
                    del self.channels[_np.max([idx1,idx2])]
                else:
                    print('correlation mode not recognied. No action taken.')

            else:
                if self.verbose:
                    print('reflectometer channels not found.')
        else:
            pass


        self.Cxy      = _np.zeros((self.nch,self.nch,wl))
        self.rho      = _np.zeros((self.nch,self.nch,wl))
        self.Sxy      = _np.zeros((self.nch,self.nch,wl)) + 1j *_np.zeros((self.nch,self.nch,wl))

        self.Gxy      = _np.zeros((self.nch,self.nch,wl/2+1)) + 1j *_np.zeros((self.nch,self.nch,wl/2+1))
        self.Gxy_norm = _np.zeros((self.nch,self.nch,wl/2+1)) + 1j *_np.zeros((self.nch,self.nch,wl/2+1))
        self.gamma    = _np.zeros((self.nch,self.nch,wl/2+1))
        self.theta    = _np.zeros((self.nch,self.nch,wl/2+1))

        self.stDev = _np.zeros(self.nch)
        self.mean  = _np.zeros(self.nch)

        # first do the diagonals
        for i in range(self.nch):

            if self.verbose:    
                print('calculating auto spectral densities: ch ' + str(self.channels[i]))

            freq_S,dum = self.estimate_Sxy(self.sig[i,:],self.sig[i,:],fs,wl)
            self.Sxy[i,i,:] = dum

            freq_G,dum = self.estimate_Gxy(self.sig[i,:],self.sig[i,:],fs,wl)
            self.Gxy[i,i,:] = dum
            self.theta[i,i,:] = _np.angle(self.Gxy[i,i,:],deg=True)

            freq_G,dum = self.estimate_Gxynorm(self.sig[i,:],self.sig[i,:],fs,wl)
            self.Gxy_norm[i,i,:] = dum
            self.gamma[i,i,:] = _np.abs(self.Gxy_norm[i,i,:])

            self.Cxy[i,i,:] = _np.real(_np.fft.ifft(self.Sxy[i,i,:])*fs)

            self.rho[i,i,:] = (_np.real(_np.fft.ifft(self.Sxy[i,i,:]))*fs /
                               _np.mean(self.sig[i,:]**2))

            self.stDev[i] = _np.sqrt(_np.mean(self.sig[i,:]**2))
            self.mean[i]  = _np.mean(self.sig[i,:])

        # then half the cross terms.
        # Use hermitian symmetry to fill in the other half of matrix
        for i in range(self.nch):
            for j in range(self.nch-i-1):

                idx1 = i
                idx2 = j+i+1

                if self.verbose:
                    print('correlating channel',self.channels[idx1],self.channels[idx2])


                freq_S,dum = self.estimate_Sxy(self.sig[idx1,:],self.sig[idx2,:],fs,wl)
                self.Sxy[idx1,idx2,:] = dum
                self.Sxy[idx2,idx1,:] = _np.conj(dum)

                freq_G,dum = self.estimate_Gxy(self.sig[idx1,:],self.sig[idx2,:],fs,wl)
                self.Gxy[idx1,idx2,:] = dum
                self.Gxy[idx2,idx1,:] = _np.conj(dum)
                self.theta[idx1,idx2,:] = _np.angle(self.Gxy[idx1,idx2,:],deg=True)
                self.theta[idx2,idx1,:] = _np.angle(self.Gxy[idx2,idx1,:],deg=True)

                freq_G,dum = self.estimate_Gxynorm(self.sig[idx1,:],self.sig[idx2,:],fs,wl)
                self.Gxy_norm[idx1,idx2,:] = dum
                self.Gxy_norm[idx2,idx1,:] = _np.conj(dum)
                self.gamma[idx1,idx2,:] = _np.abs(self.Gxy_norm[idx1,idx2,:])
                self.gamma[idx2,idx1,:] = _np.abs(self.Gxy_norm[idx2,idx1,:])



                self.Cxy[idx1,idx2,:] = _np.real(_np.fft.ifft(self.Sxy[idx1,idx2,:]))*fs
                self.Cxy[idx2,idx1,:] = _np.real(_np.fft.ifft(self.Sxy[idx2,idx1,:]))*fs

                self.rho[idx1,idx2,:] = (_np.real(_np.fft.ifft(self.Sxy[idx1,idx2,:]))*fs /
                                         _np.sqrt(_np.mean(self.sig[idx1,:]**2) *
                                                 _np.mean(self.sig[idx2,:]**2)))

                self.rho[idx2,idx1,:] = (_np.real(_np.fft.ifft(self.Sxy[idx2,idx1,:]))*fs /
                                         _np.sqrt(_np.mean(self.sig[idx1,:]**2) *
                                                 _np.mean(self.sig[idx2,:]**2)))

        # store the frequency axes
        self.freq_S = freq_S
        self.freq_G = freq_G

        self.CxyEr = 0.0
        self.rhoEr = 0.0
        self.SxyEr = 0.0
        self.GxyEr      = self.Gxy/_np.sqrt(nw)

        self.Gxy_normEr = (1-self.gamma**2)/_np.sqrt(2*nw)  # derived using error propagation from eq 23 for gamma^2 in
                                                           # J.S. Bendat, Journal of Sound an Vibration 59(3), 405-421, 1978
                                                           # standard deviation!

        self.gammaEr    = (1-self.gamma**2)/_np.sqrt(2*nw)  # derived using error propagation from eq 23 for gamma^2 in
                                                           # J.S. Bendat, Journal of Sound an Vibration 59(3), 405-421, 1978
                                                           # standard deviation!

        # standard deviation in degrees of phase angle from
        # A.E. White, Phys. Plasmas, 17 056103, 2010
        # Doesn't so far give a convincing answer...
        self.thetaEr = _np.sqrt(1-self.gamma**2)/_np.sqrt(2*nw*self.gamma)*180/_np.pi

        # make a timebase for the delay correlation
        dt = 1/fs
        df = _np.abs(freq_S[1]-freq_S[0])

        self.tau = freq_S*dt/df

        # sort the time lag correlations acording to time
        tauIdx   = _np.argsort(self.tau)
        self.tau = self.tau[tauIdx]
        self.Cxy = self.Cxy[:,:,tauIdx]
        self.rho = self.rho[:,:,tauIdx]

    def smooth(self,t,a, n=1000) :
        """
        axis, data = smooth(self,t,a, n=1000)
        t - signal axis (e.g. time)
        a - signal
        n - # of points in smooth
        """
        Navr = len(a)//n
        data = _np.zeros(Navr)
        axis = _np.zeros(Navr)
        for i in range(Navr):
            data[i]  = _np.mean(a[i*n:(i+1)*n])
            axis[i] = t[i*n+n//2]
        return axis,data

    def fluctuationAmplitudeEstimate(self,fLims = [5e3,100e3], BIF = None, P0 = None, calibrationOverlap = False):
        """
        Uses 3 methods to estimate the fluctuation amplitude present in each correlation.
        Correlations must already have been calculated.
        """

        # If this value is not specified, then we calculate it via a fit
        if BIF == None:
            print('no BIF constant supplied, calculating from power fit')
            if calibrationOverlap:
                self.getDecimatedPower(chList = self.channels,saveMean = True,
                                       filterFreq = [fLims[0],fLims[1]],
                                       nInt = 10000, nPower = 1000)
            else:
                self.getDecimatedPower(chList = self.channels,saveMean = True,
                                       filterFreq = [fLims[1],fLims[1] + (fLims[1]-fLims[0])],
                                       nInt = 10000, nPower = 1000)

            self.BIF = _np.zeros(self.nch)
            self.P0 = _np.zeros(self.nch)
            for i in range(self.nch):
                m,c,r,p,e = stat.linregress(self.decimatedMean[i,:].squeeze(),
                                            self.decimatedPower[i,:].squeeze())
                self.BIF[i] = (fLims[1]-fLims[0])/m**2
                self.P0[i]  = -c/m

        else:
            self.BIF = BIF
            self.P0  = P0

        # first calculate all the spectral integrals within the supplied limits

        self.fIntLims   = fLims
        self.SxyInt      = _np.zeros((self.nch,self.nch)) + 1j*_np.zeros((self.nch,self.nch))
        self.GxyInt      = _np.zeros((self.nch,self.nch)) + 1j*_np.zeros((self.nch,self.nch))
        self.Gxy_normInt = _np.zeros((self.nch,self.nch)) + 1j*_np.zeros((self.nch,self.nch))
        self.gammaInt    = _np.zeros((self.nch,self.nch))
        self.Gxy_normPrimeInt = _np.zeros((self.nch,self.nch)) + 1j*_np.zeros((self.nch,self.nch))
        self.gammaPrimeInt   = _np.zeros((self.nch,self.nch)) + 1j*_np.zeros((self.nch,self.nch))

        self.dT2_T2 = _np.zeros(self.Gxy_norm.shape) + 1j*_np.zeros(self.Gxy_norm.shape)
        self.dT2_T2_int = _np.zeros((self.nch,self.nch)) + 1j*_np.zeros((self.nch,self.nch))

        for i in range(self.nch):
            for j in range(self.nch):

                self.SxyInt[i,j] = self.integrateSx(self.freq_S, self.Sxy[i,j,:],
                                                    fLims[0],fLims[1])

                self.GxyInt[i,j] = self.integrateSx(self.freq_G, self.Gxy[i,j,:],
                                                    fLims[0],fLims[1])

                self.Gxy_normInt[i,j] = self.integrateSx(self.freq_G, self.Gxy_norm[i,j,:],
                                                         fLims[0],fLims[1])

                self.gammaInt[i,j] = self.integrateSx(self.freq_G, self.gamma[i,j,:],
                                                      fLims[0],fLims[1])

                if i != j:
                    self.Gxy_normPrimeInt[i,j] = self.integrateSx(self.freq_G, self.Gxy_norm[i,j,:]/(1-self.Gxy_norm[i,j,:]),
                                                                  fLims[0],fLims[1])

                    self.gammaPrimeInt[i,j] = self.integrateSx(self.freq_G, self.gamma[i,j,:]/(1-self.gamma[i,j,:]),
                                                               fLims[0],fLims[1])

                # divide by the IF bandwidth in kHz, special comaprison to previous papers/GENE output
                self.dT2_T2[i,j,:] = self.Gxy_norm[i,j,:]/(1-self.Gxy_norm[i,j,:])/_np.sqrt(self.BIF[i]*1e-3*self.BIF[j]*1e-3)

                self.dT2_T2_int[i,j] = self.integrateSx(self.freq_G*1e-3, self.dT2_T2[i,j,:], fLims[0]*1e-3,fLims[1]*1e-3)

        print('estimating fluctuation levels via several methods:\n'
              'a: integral of cross-spectral power divided by signal mean\n'
              'b: integral of gamma\n'
              'c: integral of Gxy_norm\n'
              'd: integral of gamma/(1-gamma)\n'
              'e: integral of Gxy_norm/(1-Gxy_norm)\n'
              'f: integral of dT2/T2')

        self.dT_Ta   = _np.zeros((self.nch,self.nch))
        self.dT_Tb   = _np.zeros((self.nch,self.nch))
        self.dT_Tc   = _np.zeros((self.nch,self.nch))
        self.dT_Td   = _np.zeros((self.nch,self.nch))
        self.dT_Te   = _np.zeros((self.nch,self.nch))
        self.dT_Tf   = _np.zeros((self.nch,self.nch))
        self.dT_T_er = _np.zeros((self.nch,self.nch))

        for i in range(self.nch):
            for j in range(self.nch):

                # cross spectral power divided by the signal mean
                self.dT_Ta[i,j] = (_np.sqrt(_np.abs(self.GxyInt[i,j])) /
                                   _np.sqrt((self.mean[i]-self.P0[i])*(self.mean[j]-self.P0[j])))

                # scaled integral of gamma
                self.dT_Tb[i,j] = _np.sqrt(_np.abs(self.gammaInt[i,j])/_np.sqrt(self.BIF[i]*self.BIF[j]))

                # scaled integral of Gxy_norm
                self.dT_Tc[i,j] = _np.sqrt(_np.abs(self.Gxy_normInt[i,j])/_np.sqrt(self.BIF[i]*self.BIF[j]))

                # scaled integral of gamma/(1-gamma)
                self.dT_Td[i,j] = _np.sqrt(_np.abs(self.gammaPrimeInt[i,j])/_np.sqrt(self.BIF[i]*self.BIF[j]))

                # scaled integral of Gxy_norm/(1-Gxy_norm)
                self.dT_Te[i,j] = _np.sqrt(_np.abs(self.Gxy_normPrimeInt[i,j])/_np.sqrt(self.BIF[i]*self.BIF[j]))

                self.dT_Tf[i,j] = _np.sqrt(_np.abs(self.dT2_T2_int[i,j]))

                self.dT_T_er[i,j] = _np.sqrt(1/_np.sqrt(self.BIF[i]*self.BIF[j]))*((fLims[1]-fLims[0])/self.Navr)**0.25





    def design_notch(self, f0, Q=30.0, plotit=True):
        """
        Make a band rejection / Notch filter for MHD or electronic noise

        f0 = 80.0428e3      # [Hz], frequency to reject
        Q = 30.0            # Quality factor of digital filter
        """
        w0 = f0/(0.5*self.Fs)   # Normalized frequency

        # Design the notch
        notchb, notcha = iirnotch(w0, Q)

        if plotit:
            # Frequency response
            w, h = _dsp.freqz(notchb,notcha)
            freq = w*self.Fs/(2.0*_np.pi)    # Frequency axis

            # Plot the response of the filter
            fig, ax = _plt.subplots(1,1, figsize=(8,6))
            ax[0].plot(1e-3*freq, 20*_np.log10(_np.abs(h)), color='blue')
            ax[0].set_title('Frequency Response of Notch filter (%4.1f KHz, Q=%i)'%(f0*1e-3, Q))
            ax[0].set_ylabel('Amplitude [dB]')
            xlims = [0, int(1e-3*self.Fs/2)]
            ax[0].set_xlim(xlims)
            ax[0].set_ylim([-25, 10])
            ax[0].grid()
            ax[1].plot(1e-3*freq, _np.unwrap(_np.angle(h))*180.0/_np.pi, color='green')
            ax[1].set_ylabel('Angle [deg]', color='green')
            ax[1].set_xlabel('Frequency [KHz]')
            ax[1].set_xlim(xlims)
            ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
            ax[1].set_ylim([-90, 90])
            ax[1].grid()
            _plt.show()
        # end if
        return notchb, notcha
    # end def design_notch

    def design_lpf(self, fLPF, lpf_order = 1.0, plotit=True):
        """
        Make a low pass filter for the case of small bandwidth fluctuations

        fLPF = 2*intb[1]      # [Hz], frequency to reject
        lpf_order = 1       # Order of low pass filter
        """
        # Design the LPF
        lpfb, lpfa = butter_lowpass(fLPF, 0.5*self.Fs, order=lpf_order)

        if plotit:
            # Frequency response
            w, h = _dsp.freqz(lpfb,lpfa)
            freq = w*self.Fs/(2.0*_np.pi)    # Frequency axis

            # Plot the response of the filter
            fig, ax = _plt.subplots(1,1, figsize=(8,6))
            ax[0].plot(1e-3*freq, 20*_np.log10(_np.abs(h)), color='blue')
            ax[0].set_title('Frequency Response of Low pass filter (%4.1f KHz order=%i)'%(fLPF*1e-3, lpf_order))
            ax[0].set_ylabel('Amplitude [dB]')
            xlims = [0, int(1e-3*self.Fs/2)]
            ax[0].set_xlim(xlims)
            ax[0].set_ylim([-25, 10])
            ax[0].grid()
            ax[1].plot(1e-3*freq, _np.unwrap(_np.angle(h))*180.0/_np.pi, color='green')
            ax[1].set_ylabel('Angle [deg]', color='green')
            ax[1].set_xlabel('Frequency [KHz]')
            ax[1].set_xlim(xlims)
            ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
            ax[1].set_ylim([-90, 90])
            ax[1].grid()
            _plt.show()
        # end if
        return lpfb, lpfa
    # end def design_lpf

    def lowPassFilter(self,cutoff, order = 2):

        if self.verbose:
            print('low pass filtering the data with order',order,' butterworth filter at:',cutoff*1e-6,'MHz')

        nyq = 0.5 * self.Fs
        normal_cutoff = cutoff / nyq

        # record filter parameters
        self.filterOrder = order
        self.fFilter     = cutoff

        b, a = _dsp.butter(order, normal_cutoff, btype='low', analog=False)
        self.sig = _dsp.lfilter(b, a, self.sig, axis = 1)

        self.BVid = cutoff

        if self.verbose:
            print('Done!')

    def lpFilter(self,data,cutoff,order = 2, axis = 0):
        """copy of lowPassFilter, but works with arbitrary data passed as an argument."""

        nyq = 0.5 * self.Fs
        normal_cutoff = cutoff / nyq

        b, a = _dsp.butter(order, normal_cutoff, btype='low', analog=False)
        data = _dsp.lfilter(b, a, data, axis = axis)
        return data

    def bpFilter(self,data,f0,f1,order = 2, axis = 0):

        nyq = 0.5 * self.Fs
        normal_f0 = f0 / nyq
        normal_f1 = f1 / nyq

        b, a = _dsp.butter(order, [normal_f0,normal_f1], btype='bandpass', analog=False)
        data = _dsp.lfilter(b, a, data, axis = axis)
        return data

    def chebyNotchFilter(self,centre, bw, gPass = 0.4, gStop = 20, maxRipple = 0.1):

        nyq = 0.5 * self.Fs
        passBand = [(centre-0.5*bw)/ nyq , (centre+0.5*bw)/nyq]
        stopBand = [(centre-0.1*bw)/ nyq , (centre+0.1*bw)/nyq]
        order, Wn = _dsp.cheb1ord(passBand, stopBand, gPass, gStop, analog=False)

        if self.verbose:
            print('notch filtering the data with centre',centre*1e-3,' and bandwidth of ',bw*1e-3,'kHz')

        b, a = _dsp.cheby1(order, maxRipple, Wn, btype='bandstop', analog=False)
        self.sig = _dsp.lfilter(b, a, self.sig, axis = 1)

    def squareNotchFilter(self,centre,bw, binary = True):

        if binary:
            if self.verbose:
                print('recasting array so that number of timepoints is a power of 2')
            nt = 2**int(_np.log2(self.nt))
            idx = _np.arange(nt,self.nt)

            if self.verbose:
                print('original  :', self.nt)
            self.nt = nt

            if self.verbose:
                print('new       :', self.nt)
            self.sig = _np.delete(self.sig,idx,axis = 1)
            self.time = _np.delete(self.time,idx)

            if self.verbose:
                print('data shape: ', self.sig.shape)

        freq = _np.fft.fftfreq(self.sig.shape[1],1.0/self.Fs)
        filterFunction = _np.zeros(self.sig.shape)+1

        if self.verbose:
            print('forming filter function')
        for i in range(len(freq)):
            for j in range(len(centre)):
                if ((_np.abs(freq[i]) < (centre[j]+ bw[j]/2)) and
                    (_np.abs(freq[i]) > (centre[j]-bw[j]/2)) ):
                    filterFunction[:,i] = 0

        if self.verbose:
            print('notch filtering the data ')
        self.sig = _np.real(_np.fft.ifft(_np.fft.fft(self.sig,axis = 1)*filterFunction,axis = 1))

# end class

class QMFR(CECE):
    debug = True

                            
    def get_data_quickanal(self, tstart, tend, tstep=None, overlap=0.5):
        from W7X.fetch import getZOOMsig        
        if tstep is None: tstep = 4096/self.Fs # end if

        nsig = int((tend-tstart)*self.Fs)
        nwins = int(tstep*self.Fs)
        noverlap = int(overlap*nwins)
        Navr = (nsig-noverlap)//(nwins-noverlap)
        istart = 0
        
        Sxy = 0.0
        
        Sxy_avg = 0.0
        Sxy_var = 0.0
        Gxy_avg = 0.0
        Gxy_var = 0.0
        Corr_avg = 0.0
        Corr_var = 0.0
        mean = 0.0
        mean_var = 0.0
        msq = 0.0
        self.nch = 16
        
        for ii in range(Navr):
            tst = self._utcstart + 1e9*istart/self.Fs
            tnd = tst+1e9*nwins/self.Fs
            tt, self.sig = getZOOMsig(tstart=tst, tend=tnd, nch=self.nch, bg_subtr=True, 
                                 expprog=0, getparlog=False, verbose=self.debug)         # check BGSUBTR!   
            istart += self.nwins

            # ================================= #

            self.diagnostic='CECE'
#            IRfft = _fft.fftanal(tt, tmpRF-tmpRF.mean(), tmpIF-tmpRF.mean(), tbounds=tb, Navr=Nw, windowoverlap=overlap)     
            self.getCorrelations()
            Sxy = self.Sxy.copy()
            Gxy = self.Gxy.copy()
            
            if ii == 0:  
                self.freq_S = self.freq_S.copy()
                self.freq_G = self.freq_G.copy()
            # end if
            if 1:
                # Running mean and variance (vs frequency)
                #    n = n + 1
                #    m_prev = m
                #    m = m + (x_i - m) / n
                #    S = S + (x_i - m) * (x_i - m_prev)
        #        m_prev = Nxy_avg.copy()    
        #        Nxy_avg += (NRfft.Pxy - Nxy_avg) / (ii+1)
        #        Nxy_var += (NRfft.Pxy - Nxy_avg) * (NRfft.Pxy - m_prev)        

                sigs.append(self.mean) 
                
                m_prev = _np.copy(Sxy_avg)
                Sxy_avg += (Sxy - Sxy_avg) / (ii+1)
                Sxy_var += (Sxy - Sxy_avg) * (self.Sxy - m_prev)        
                
                m_prev = _np.copy(Gxy_avg)
                Gxy_avg += (Gxy - Gxy_avg) / (ii+1)
                Gxy_var += (Gxy - Gxy_avg) * (Gxy - m_prev)        
                
                m_prev = _np.copy(mean)    
                mean += (self.mean - mean) / (ii+1)
                mean_var += (self.mean**2.0 - mean) * (self.mean**2.0 - m_prev)     # variance in the mean
#                msq += (self.stDev**2.0 - mean) * (self.stDev**2.0 - m_prev)     # actually a variance in the mean

                m_prev = _np.copy(msq)    
                msq += (self.stDev**2.0 - msq) / (ii+1) # square to remove "R" in RMS, continue the mean calculation
                
                m_prev = _np.copy(Corr_avg)            
                Corr_avg += (self.Cxy - Corr_avg) / (ii+1)
                Corr_var += (self.Cxy - Corr_avg) * (self.Cxy - m_prev)                            


            # endif            
        # end for
        self.Cxy = Corr_avg.copy()
        self.Sxy = Sxy_avg.copy()
        self.Gxy = Gxy_avg.copy()
        self.mean = mean.copy()
        self.stDev = _np.sqrt(msq)   # take square root again to put the "R" back in RMS
        
        # =============== Now mimic the data analysis of CECE with the ensemble averages ======== #
        Gxy_norm = _np.zeros_like(self.Gxy)
        gamma = _np.zeros_like(self.Gxy)
        rho = _np.zeros_like(self.Cxy)
        for ii in range(self.nch):
            for jj in range(self.nch):                
                Gxy_norm[ii,jj] = self.Gxy[ii,jj].copy()/_np.sqrt(self.Gxy[ii,ii].copy()*self.Gxy[jj,jj].copy())
                gamma[ii,jj] = _np.abs(self.Gxy[ii,jj].copy())/_np.sqrt(self.Gxy[ii,ii].copy()*self.Gxy[jj,jj].copy())
                rho[ii,jj] = self.Cxy[ii,jj].copy() / _np.sqrt(self.stDev[ii,ii]**2.0 * self.stDev[jj,jj]**2.0 )
            # end for
        # end for
        self.Gxy_norm = Gxy_norm
        self.gamma = gamma
        self.rho = rho
        self.theta = _np.angle(self.Gxy, deg=True)

        self.GxyEr = self.Gxy/_np.sqrt(self.Navr)
        
        self.Gxy_normEr = (1.0-self.gamma**2)/_np.sqrt(2.0*self.Navr)  # derived using error propagation from eq 23 for gamma^2 in
                                                           # J.S. Bendat, Journal of Sound an Vibration 59(3), 405-421, 1978
                                                           # standard deviation!

        self.gammaEr    = (1.0-self.gamma**2)/_np.sqrt(2.0*self.Navr)  # derived using error propagation from eq 23 for gamma^2 in
                                                           # J.S. Bendat, Journal of Sound an Vibration 59(3), 405-421, 1978
                                                           # standard deviation!

        # standard deviation in degrees of phase angle from
        # A.E. White, Phys. Plasmas, 17 056103, 2010
        # Doesn't so far give a convincing answer...
        self.thetaEr = _np.sqrt(1.0-self.gamma**2.0)/_np.sqrt(2.0*self.Navr*self.gamma)*180.0/_np.pi

    def getTnorm(self,fLims = [5e3,100e3]):
        # =============== Now calculate fluctuation level with the ensemble averages ======== #
        if not hasattr(self,'ece_bw'):  self.ece_bw = 150e6*_np.ones((self.nch,), dtype=_np.float64) # end if
        self.fluctuationAmplitudeEstimate(self,fLims=[5e3,100e3], BIF=self.ece_bw.copy(), P0=0.0)
    # end def

        
    def convert_TimeRelT12utc(self, tt, expprog=None):
        from W7X.fetch import zero_time
        
        if not hasattr(self,'T1'):
            self.T1 = zero_time(self.utcstart, expprog) # 
        # end if
        return 1e9*tt + self.T1        

#
#    def getTimeIntervalfromID(self):
#        return _jsnut.getTIfromID(self.XPPROGID)
#        
#    def getIDfromTimeInterval(self, tstart, tend):
#        return _jsnut.getIDfromTimeInterval(tstart, tend)
#            
#    def get_settings(self, tstart, tend=None):
#        from W7X.fetch import getZOOMsig    
#        if tend is None:  tend = tstart+100e-3 # end if
#        parlog = getZOOMsig(tstart, tend, nch=16, bg_subtr=True, 
#                            expprog=0, getparlog=True, verbose=self.debug)
#        self.update(parlog)
#
#
#    def get_data(self, tstart, tend=None):
#        from W7X.fetch import getZOOMsig        
#        if tend is None:  tnd = tstart+100e-3; else: tnd = tend    # end if
#        
#        tt, sig = getZOOMsig(tstart=tstart, tend=tnd, nch=16, bg_subtr=True, 
#                             expprog=0, getparlog=False, verbose=self.debug)
#
#        if tend is None:   self.Fs = (len(tt) - 1) / (tt[-1] - tt[0])   # end if
#        return tt, sig                            


        
    def plot_sigs(self):
        hfig0, ax0 = _plt.subplots(self.nch,1, figsize=(8,6), sharex=True)
        hfig1, ax1 = _plt.subplots(self.nch,1, figsize=(8,6), sharex=True)
        ax0[0].set_title('RF')
        ax1[0].set_title('IF')

        for ii in range(self.nch):
            # plot the signals
            ax0[ii].plot(self.tt, tmpRF, '-')
            ax0[ii].set_ylabel('%5s'%(fils[ii][5:]))
            ax1[ii].plot(self.tt, tmpIF, '-')
            ax1[ii].set_ylabel('%5s'%(fils[ii][5:]))
            if ii == self.nch-1:
                ax0[ii].set_xlabel('t [ms]')
                ax1[ii].set_xlabel('t [ms]')
    # end def plot_sigs

    def plot_spectra(self):
        freq = self.freq
        i0 = self.ibounds[0]
        i1 = self.ibounds[1]

        hfig, ax = _plt.subplots(5,1, figsize=(8,6))
        #ax[0].set_xlabel('freq [KHz]')
        ax[0].set_ylabel('Cross Power')
        ax[1].set_ylabel('Phase')
        ax[2].set_ylabel('Coh')
        ax[2].set_xlabel('freq [KHz]')
        ax[0].get_shared_x_axes().join(ax[0], ax[1], ax[2])

        ax[3].set_ylabel('Coh')
        ax[4].set_ylabel('Phase Diff')
        ax[4].set_xlabel('freq [GHz]')
        ax[3].get_shared_x_axes().join(ax[3], ax[4])

        ax[0].axvline(x=1e-3*freq[i0], linewidth=2, color='k')
        ax[0].axvline(x=1e-3*freq[i1], linewidth=2, color='k')
        ax[1].axvline(x=1e-3*freq[i0], linewidth=2, color='k')
        ax[1].axvline(x=1e-3*freq[i1], linewidth=2, color='k')
        ax[3].plot(self.ece_freq, self.Cxy, 'o')
        ax[3].axhline(y=self.CohLim, linewidth=2, color='k')
        ax[3].text(_np.average(self.ece_freq), 0.05,
                  '%i to %i GHz'%(int(1e-3*freq[i0]),int(1e-3*freq[i1])),
                  fontsize=12)
        ax[4].plot(freq, self.phxy, 'o')
    # end def plot_spectra
# end class RADCECE
#
#
#class POLCECE(RADCECE):
#    nch = 8
#    nrad = 2
#
#    def
#    def w7xpcecemask(self):
#        imask1 = _np.ones(range(self.nch), dtype=bool)
#        imask2 = _np.ones(range(self.nch), dtype=bool)
#
#        imask1[8:] = False
#        imask2[:8] = False
#        imask2[16:] = False
#        return imask1, imask2
#
##    def plot_sigs(self, imask1=None, imask2=None):
##        if imask1 or imask2 is None:
##            imask1, imask2 = self.w7xpcecemask()
##        # end if
#
#    def plot_sigs(self):
#        hfig0, ax0 = _plt.subplots(self.nch,self.nrad, figsize=(8,6), sharex=True)
#        for jj in range(self.nrad):
#            ax0[jj].set_title('RAD%0i'%(jj,))
#            for ii in range(self.nch):
#                sp = self.nrad*ii
#                ch = jj*self.nch + ii
#                # plot the signals
#                ax0[sp].plot(self.tt, self.Trad[:,ch], '-')
#                ax0[sp].set_ylabel('CH%0i'%(ch,))
#            # end for
#            ax0[sp].set_xlabel('t [ms]')
#    # end def plot_sigs
#
#    def plot_spectra(self):
#        freq = self.freq
#
#        hfig, ax = _plt.subplots(5,1, figsize=(8,6))
#        #ax[0].set_xlabel('freq [KHz]')
#        ax[0].set_ylabel('Cross Power')
#        ax[1].set_ylabel('Phase')
#        ax[2].set_ylabel('Coh')
#        ax[2].set_xlabel('freq [KHz]')
#        ax[0].get_shared_x_axes().join(ax[0], ax[1], ax[2])
#
#        ax[3].set_ylabel('Coh')
#        ax[4].set_ylabel('Phase Diff')
#        ax[4].set_xlabel('freq [GHz]')
#        ax[3].get_shared_x_axes().join(ax[3], ax[4])
#
#        ax[0].axvline(x=1e-3*freq[i0], linewidth=2, color='k')
#        ax[0].axvline(x=1e-3*freq[i1], linewidth=2, color='k')
#        ax[1].axvline(x=1e-3*freq[i0], linewidth=2, color='k')
#        ax[1].axvline(x=1e-3*freq[i1], linewidth=2, color='k')
#        ax[3].plot(self.ece_freq, Cxy, 'o')
#        ax[3].axhline(y=CohLim, linewidth=2, color='k')
#        ax[3].text(_np.average(freqs), 0.05,
#                  '%i to %i GHz'%(int(1e-3*freq[i0]),int(1e-3*freq[i1])),
#                  fontsize=12)
#        ax[4].plot(freqs, phxy, 'o')
#    # end def plot_spectra
## end class POLCECE