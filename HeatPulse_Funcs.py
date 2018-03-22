# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:03:30 2016

@author: gawe

This is the driver function for heat pulse propagation analysis.

The hp_fft code calculates the amplitude and phase of a perturbation that is
"""
# ========================================================================== #

from __future__ import absolute_import, with_statement, absolute_import, division, print_function, unicode_literals
__metaclass__ = type

# ========================================================================== #
# ========================================================================== #

import numpy as _np
import os as _os
import h5py as _h5
import matplotlib.pyplot as _plt

import IO as _io
try:
    from ..Struct import Struct
    from .. import plt_utils as _pltut
    from ..FIT import fitting_dev as _fd
    from . import fft_analysis as _fft
except:
    from pybaseutils import Struct
    from pybaseutils import plt_utils as _pltut
    from pybaseutils.FIT import fitting_dev as _fd
    from pybaseutils import fft_analysis as _fft
# end try
# ========================================================================== #
# ========================================================================== #

def load_data(filname):
    all_data = _io.loadHDF5(filname, False, True)

    flds = ['COIL', 'QME', 'QMJ', 'QTB', 'XPLOG', 'QMEZ', 'ECRH']
    return tuple([all_data[flds[ii]] for ii in range(len(flds)) if flds[ii] in all_data])

def loadFFTdata(filname):
    tmp = _io.loadHDF5(filname, 'FFTResults')

    FFTdata = {}
    FFTdata["nch"] = tmp["nch"]
    FFTdata["usech"] = tmp["usech"]
    FFTdata["chnum"] = tmp["chnum"]
    FFTdata["ece_roa"] = tmp["ece_roa"]
    FFTdata["ece_freq"] = tmp["ece_freq"]
    FFTdata["ece_bw"] = tmp["ece_bw"]
    FFTdata["tau"] = tmp["tau"]
    FFTdata["uth"] = tmp["uth"]
    FFTdata["Amp"] = tmp["Amp"]
    FFTdata["varA"] = tmp["varA"]
    FFTdata["Phase"] = tmp["Phase"]
    FFTdata["varP"] = tmp["varP"]
    FFTdata["Coh"] = tmp["Coh"]
    FFTdata["varC"] = tmp["varC"]
    FFTdata["Txy"] = tmp["Txy"]
    FFTdata["Vxy"] = tmp["Vxy"]
    FFTdata["Tnn"] = tmp["Tnn"]
    FFTdata["fmods"] = _np.atleast_1d(tmp["fmods"])
    del tmp
    return FFTdata
    # return tuple([HPdata[flds[ii]] for ii in range(len(flds)) if flds[ii] in HPdata])

def loadHPdata(filname):
    HPdata = _io.loadHDF5(filname, False, True)

    flds = ['Results', 'Inputs', 'DataIn']
    return tuple([HPdata[flds[ii]] for ii in range(len(flds)) if flds[ii] in HPdata])
    # return loadHDF5(filname, True, True)

def fit_neprofile(QTBdat, rvec, loggradient=True, plotit=False):
    logne, varlogne, dlnnedrho, vardlnnedrho = \
        _fd.fit_TSneprofile(QTBdat, rvec, loggradient, plotit)
    return logne, varlogne, dlnnedrho, vardlnnedrho

# ========================================================================== #

_plt.rc('font',size = 16)
_figsize = _plt.rcParams["figure.figsize"]
_figsize = (1.2*_figsize[0],1.2*_figsize[1])

class __HeatPulse_base__(Struct):
    _fig_size = _figsize
    clrs = 'bgrcmyk'
    def __init__(self, d=None):

        if d is not None:
            super(__HeatPulse_base__, self).__init__(d)
        # endif
        if self.verbose: print("Initialized a heat pulse base class")  # end if
    # end def __init__


    #  ====================================================================  #

    def _sortECEdat_(self, sortby='RF'):
        """

        Default to sorting by ece resonance frequency
        """
        if sortby.lower() == 'rf':
            _isort = _np.argsort(self.ece_freq.reshape((self.nch,), order='C'))
        elif sortby.lower().find('abs')>-1:
            _isort = _np.argsort(_np.abs(self.ece_roa).reshape((self.nch,), order='C'))
        elif sortby.lower() == 'roa' or sortby.lower()=='r/a' or sortby.lower()=='reff':
            _isort = _np.argsort(self.ece_roa.reshape((self.nch,), order='C'))
        # endif
        return _isort

    def __sortECEdat__(self, _isort):
        self.ece_freq = self.ece_freq[_isort]
        self.ece_bw = self.ece_bw[_isort]
        # self.Trad = self.Trad[ :,_isort]
        if self.isroa:
            self.ece_roa = self.ece_roa[_isort]
        # endif
        if hasattr(self, 'tau'):
            self.tau = self.tau[_isort]
        if hasattr(self, 'uth'):
            self.uth = self.uth[_isort]

        # ECRH signal only has 1 channel: No sorting necessary
        # self.Txx = self.Txx[_isort,:]  # 1 ch x nharm (x Navr)
        # self.Vxx = self.Vxx[_isort,:]

        self.Tnn = self.Tnn[_isort,:]
        self.Txy = self.Txy[_isort,:]
        self.Vxy = self.Vxy[_isort,:]
        self.Amp = self.Amp[_isort,:]
        self.varA = self.varA[_isort,:]
        self.Coh = self.Coh[_isort,:]
        self.varC = self.varC[_isort,:]
        self.Phase = self.Phase[_isort,:]
        self.varP = self.varP[_isort,:]
        self.usech = self.usech[_isort]
        self.chnum = self.chnum[_isort]
    # end def

    def __chfilter__(self):
        info = Struct()
        info.ece_freq = self.ece_freq[self.usech]
        info.ece_bw = self.ece_bw[self.usech]
        if self.isroa:
            info.ece_roa = self.ece_roa[self.usech]
        # endif
        if hasattr(self, 'tau'):
            info.tau = self.tau[self.usech]
        if hasattr(self, 'uth'):
            info.uth = self.uth[self.usech]

#        # ECRH signal only has 1 channel: No sorting necessary
#        info.Txx = self.Txx[self.usech,:] # 1 ch x nharm (x Navr)
#        info.Vxx = self.Vxx[self.usech,:]

        info.Tnn = self.Tnn[self.usech,:]
        info.Txy = self.Txy[self.usech,:]
        info.Vxy = self.Vxy[self.usech,:]
        info.Amp = self.Amp[self.usech,:]
        info.varA = self.varA[self.usech,:]
        info.Coh = self.Coh[self.usech,:]
        info.varC = self.varC[self.usech,:]
        info.Phase = self.Phase[self.usech,:]
        info.varP = self.varP[self.usech,:]
        info.usech = self.usech[self.usech]
        info.chnum = self.chnum[self.usech]
        info.nch = len(info.ece_freq)
        return info
    # end def

    # ==== #

#    def __sortHPdat__(self, _isort):
#        self.__sortAmpdat__(_isort)
#        self.__sortPhadat__(_isort)
#        self.usech = self.usech[_isort]
#        self.chnum = self.chnum[_isort]
#        self.ece_freq = self.ece_freq[_isort]
#    # end def
#
#    def __sortAmpdat__(self, _isort):
#        self.Amp = self.Amp[_isort]
#        self.varA = self.varA[_isort]
#        self.roaA = self.roaA[_isort]
#    # end def
#
#    def __sortPhadat__(self, _isort):
#        self.Phase = self.Phase[_isort]
#        self.varP = self.varP[_isort]
#        self.roaP = self.roaP[_isort]
#    # end def

    #  ====================================================================  #

    def load_data(self, filn):
        Results, Inputs, DataIn = loadHPdata(filn)

        self.__dict__.update(Inputs)
        self.__dict__.update(DataIn)
        self.__dict__.update(Results)
    # end def load_data

    #  ====================================================================  #
    # Basic plotting section

    def plottime(self):
        # The input signals versus time
        # hTEMP = (self.sfilename).replace('HPRESULTS','HP_Trad')
        hfig = _plt.figure(figsize=self._fig_size)
        _plt.plot(self.tt,  self.Trad, '-')
        _plt.title('ECE Signal', **self.afont)
        _plt.xlabel('t[s]', **self.afont)
        _plt.ylabel('T_{rad} [KeV]', **self.afont)
        _plt.axvline(x=self.tbounds[0], color='k')
        _plt.axvline(x=self.tbounds[1], color='k')
        _plt.draw()
#        _plt.show()
        return hfig

    def plotPxyf(self, ch):
        hfig = _plt.figure(figsize=self._fig_size)
        _plt.plot(1e-3*self.freq, 10*_np.log10(_np.abs(self.Pxx[:,ch])), 'b-')
        _plt.plot(1e-3*self.freq, 10*_np.log10(_np.abs(self.Pyy[:,ch])), 'r-')
        _plt.plot(1e-3*self.freq, 10*_np.log10(_np.abs(self.Pxy[:,ch])), 'k-')
        _plt.title('Power Spectra', **self.afont)
        _plt.ylabel('P_{ij} [dB/Hz]', **self.afont),
        _plt.xlabel('f[kHz]', **self.afont)
        _plt.xlim(0, 1.01e-3*self.freq[-1])
        _plt.draw()
#        _plt.show()
        return hfig

    def plotphxyf(self, ch=0):
        hfig = _plt.figure(figsize=self._fig_size)
        _plt.plot(1e-3*self.freq, self.phi_xy[:,ch], 'k-')
        _plt.title('Cross-Phase', **self.afont)
        _plt.ylabel('\phi_{xy}', **self.afont)
        _plt.xlabel('f[kHz]', **self.afont)
        _plt.xlim(0, 1.01e-3*self.freq[-1])
        _plt.draw()
#        _plt.show()
        return hfig
    #end def

    def plotCxyf(self, ch=0):
        hfig = _plt.figure(figsize=self._fig_size)
        _plt.plot(1e-3*self.freq, self.Cxyf[:, ch], 'k-')
        _plt.axhline(y=1./self.Navr, color='k')
        _plt.title('Cross-Coherence', **self.afont)
        _plt.ylabel('C_{xy}', **self.afont)
        _plt.xlabel('f[kHz]', **self.afont)
        _plt.xlim(0, 1.01e-3*self.freq[-1])
        _plt.draw()
#        _plt.show()
        return hfig
    #end def



    #  ====================================================================  #
    # Data saving section


    def save_dat(self, Inputs=None, DataIn=None, FFTResults=None, HPResults=None):

        h5dict = {}
        if Inputs is not None:
            if type(Inputs) != dict:  Inputs = Inputs.dict_from_class()  # end if
            h5dict['Inputs'] = Inputs
        if DataIn is not None:
            if type(DataIn) != dict:  DataIn = DataIn.dict_from_class()  # end if
            h5dict['DataIn'] = DataIn
        if FFTResults is not None:
            if type(FFTResults) != dict:  FFTResults = FFTResults.dict_from_class()  # end if
            h5dict['FFTResults'] = FFTResults
        if HPResults is not None:
            if type(HPResults) != dict:  HPResults = HPResults.dict_from_class()  # end if
            h5dict['HPResults'] = HPResults
        # end if

        try:
            if not _os.path.exists(self.savedir):
                _os.mkdir(self.savedir)
            #endif

            print('#-- Attempting to save processed data to HDF5 file --#')
            sfilename = _os.path.join(self.savedir, self.sfilename+'.hdf5' )
            with _h5.File( sfilename, 'a') as sfil:
                _io.saveHDF5.__recursively_save_dict_contents_to_group__(sfil, h5dict)
            #close HDF5 file
            print('Successfully saved dictionary to HDF5 file')
            print('Under name '+self.sfilename)
        except:
            print('failed to save the data ...')
        # end try
        return sfilename

    def save_fig(self, hfig=None, hNAME=None):
        if hfig is None: hfig = _plt.gcf()  # end if
        if hNAME is None: hNAME=hfig.number  # end if
        sfilename = _os.path.join(self.savedir, hNAME)
        try:
            print('#-- Attempting to save figure %s --#'%(hfig,))
            try:
                _plt.figure(hfig.number)
            except:
                _plt.figure(hNAME)
            # end try
            # _pltut.savefig(sfilename, ext='eps', close=False, dotsperinch=100, transparency=True)
            _pltut.savefig(sfilename, ext='png', close=self.closefig, dotsperinch=100, transparency=True)
            print('#-- Successfully saved figure %s --#'%(hfig,))
        except:
            print('failed to save the figure ... ')
        # end try
        return sfilename
    # end def save_fig
# end class __HeatPulse_base__

# ========================================================================== #


class __HeatPulse_FFTbase__(__HeatPulse_base__):

    def __init__(self, runinfo={}, HPdata={}):
        if type(runinfo) != dict:   runinfo = runinfo.dict_from_class()   # endif
        if type(HPdata) != dict:    HPdata = HPdata.dict_from_class()     # endif

        # Call the initialization function of the super class to set up everything
        super(__HeatPulse_base__, self).__init__(runinfo)
        super(__HeatPulse_base__, self).__init__(HPdata)

        self.afont = self.afont.dict_from_class()
    # end def __init__

    # ==================================== #

    def _mkfftdict_(self, RemExtraKeys=[]):
        FFTresults = self.dict_from_class()

        # Remove keys that have overlap with runinfo and QMEinfo to keep the HDF5 file sane
        keys2remove = []

        # Keys that overlap with runinfo
        keys2remove.extend(['saveit','intno2per','harms','afont','overlap','sfilename',
                            'savedir','winfun','usesegs','vmcfil','fmod','xpname',
                            'plotit','igch','verbose','DutyCycle','tbounds','fwid'])

        # Remove extra keys specific to this instantiation
        keys2remove.extend(RemExtraKeys)

        # Remove some basic ones that are automatically generated internally
        keys2remove.extend(['noverlap','nsig','isroa','nf','_refsig','closefig','nharms','nwins'])
        keys2keep = list(set(FFTresults.keys()) - set(keys2remove))

        return {k: FFTresults[k] for k in keys2keep}

    # ====================================================================== #
    # ====================================================================== #

    def PreCheck(self):
        self.chnum = _np.int64(_np.linspace(1, self.nch, self.nch))
        self.usech = _np.ones((self.nch,), dtype=bool)
        if self.igch is not None:
            self.usech[self.igch-1] = False
        # end if igch
    # end def PreCheck

    # ====================================================================== #

    def _PWELCH_settings_(self):
        self.ibounds = _np.floor(1+self.Fs*(self.tbounds-self.tt[0]))
        self.ibounds = self.ibounds.astype(int)
        self.nsig = len(self.tt[self.ibounds[0]:self.ibounds[1]])
        self.nwins = _np.floor(self.intno2per*(2/self.fmod)*self.Fs)  #Length of the FFT window (integer number of periods)
        self.nwins = self.nwins.astype(int)
        self.noverlap = _np.ceil(self.overlap*self.nwins) #Number of points to overlap
        self.noverlap = self.noverlap.astype(int)
        self.Navr  = _np.floor((self.nsig-self.noverlap)/(self.nwins-self.noverlap)) #Number of time-windows

        self.Navr = _np.asarray(self.Navr, dtype=_np.int64)
        if not hasattr(self, 'useMLAB'):
            self.useMLAB = False  # use version written by GMW
        # end if

    # end def _FFTsettings_

    def _PWELCH_preallocate(self):
        self.nharms = len(self.harms)
        if self.usesegs:
            self.Txx = _np.zeros((self.nharms, self.Navr), dtype=_np.float64)
            self.Amp = _np.zeros((self.nch, self.nharms, self.Navr), dtype=_np.float64)
            self.Txy = _np.zeros((self.nch, self.nharms, self.Navr), dtype=_np.complex128)
        else:
            self.Txx = _np.zeros((self.nharms,), dtype=_np.float64)
            self.Amp = _np.zeros((self.nch, self.nharms), dtype=_np.float64)
            self.Txy = _np.zeros((self.nch, self.nharms), dtype=_np.complex128)
        # endif
        self.Vxx = _np.zeros_like(self.Txx)

        self.Vxy = _np.zeros_like(self.Txy)
        self.Tnn = _np.zeros_like(self.Txy)

        self.varA = _np.zeros_like(self.Amp)
        self.Coh = _np.zeros_like(self.Amp)
        self.varC = _np.zeros_like(self.Amp)
        self.Phase = _np.zeros_like(self.Amp)
        self.varP = _np.zeros_like(self.Amp)

        self._ifk = _np.zeros(_np.shape(self.harms), dtype=_np.int64)
    # end def _PWELCH_preallocate

    def _getharmindex_(self, fftinfo=None):

        self.nf = len(self.freq)
        _dT = self.nf/(self.freq[-1]-self.freq[0])

        # self._ifw = 1+_np.round( _dT*( 0.5*self.fwid ) )
        self._ifw = 1+_np.floor( _dT*( 0.5*self.fwid ) )
        self._ifw = self._ifw.astype(int)

        Pxx = None
        if fftinfo is not None:
            Pxx = _np.abs((fftinfo.Pxx.reshape((self.nf,), order='C')).copy())
        # end if

        for jj, kk in enumerate(self.harms):

            itemp = _np.where(self.freq>kk*self.fmod)[0][0]
            if Pxx is None:
                # self._ifk[jj] = 1+_np.round(_dT*(kk*self.fmod-self.freq[0]))
                # self._ifk[jj] = _np.floor(_dT*(kk*self.fmod-self.freq[0]))
                self._ifk[jj] = itemp.copy()
            else:
                _isl = _np.arange(itemp-2*self._ifw, itemp+2*self._ifw, dtype=int)
                self._ifk[jj] = _np.argmax(Pxx[_isl])
                self._ifk[jj] += _isl[0]
            # endif

            print('Using frequency %1.3f for harmonic %i: '%(self.freq[self._ifk[jj]], kk))
        # endfor
        self.fmods = self.freq[self._ifk]
    # end def _getharmindex _

    def _HP_preallocate_(self, fftinfo):
        if self.usesegs:
            self.Pxx = _np.real((fftinfo.Pxx_seg.reshape((self.nf, self.Navr), order='C')).copy())
            self.vPxx = _np.real((fftinfo.varPxx.reshape((self.nf, self.Navr), order='C')).copy())
            self.Pxy = _np.zeros((self.nf, self.nch, self.Navr), dtype=_np.complex128)
        else:
            self.Pxx = _np.real((fftinfo.Pxx.reshape((self.nf,), order='C')).copy())
            self.vPxx = _np.real((fftinfo.varPxx.reshape((self.nf,), order='C')).copy())
            self.Pxy = _np.zeros((self.nf, self.nch), dtype=_np.complex128)
        # endif
        self.vPxy = _np.zeros_like(self.Pxy)
        self.Pnn = _np.zeros_like(self.Pxy)
        self.Pyy = _np.zeros_like(self.Pxy)
        self.vPyy = _np.zeros_like(self.Pxy)
    # end def _HP_preallocate

    # Integrate spectra
    def integrate_spectra(self, fftinfo, frange):
        if self.usesegs:
            Pxy_i = _np.zeros((self.Navr,1), dtype=_np.complex128)
            varPxy_i = _np.zeros_like(Pxy_i)
            Pxx_i = _np.zeros((self.Navr,1), dtype=_np.float64)
            varPxx_i = _np.zeros_like(Pxx_i)
            Pyy_i = _np.zeros_like(Pxx_i)
            varPyy_i = _np.zeros_like(Pxx_i)
            Cxy_i = _np.zeros_like(Pxx_i)
            varCxy_i = _np.zeros_like(Pxx_i)
            ph_i = _np.zeros_like(Pxx_i)
            varph_i = _np.zeros_like(Pxx_i)
            for kk in range(self.Navr):
                Pxy_i[kk], Pxx_i[kk], Pyy_i[kk], Cxy_i[kk], ph_i[kk], info = _fft.integratespectra(
                    fftinfo.freq, fftinfo.Pxy_seg[kk,:], fftinfo.Pxx_seg[kk,:], fftinfo.Pyy_seg[kk,:], frange)

                varPxy_i[kk] = info.varPxy_i
                varPxx_i[kk] = info.varPxx_i
                varPyy_i[kk] = info.varPyy_i
                varCxy_i[kk] = info.varCxy_i
                varph_i[kk] = info.varph_i
            # end for
        else:
            Pxy_i, Pxx_i, Pyy_i, Cxy_i, ph_i, info = _fft.integratespectra(
                fftinfo.freq, fftinfo.Pxy, fftinfo.Pxx, fftinfo.Pyy, frange,
                varPxy=fftinfo.varPxy, varPxx=fftinfo.varPxx, varPyy=fftinfo.varPyy)

            varPxy_i = info.varPxy_i
            varPxx_i = info.varPxx_i
            varPyy_i = info.varPyy_i
            varCxy_i = info.varCxy_i
            varph_i = info.varph_i
        # end if

        # isl = info.ifrange
        return Pxy_i, Pxx_i, Pyy_i, Cxy_i, ph_i, varPxy_i, varPxx_i, varPyy_i, varCxy_i, varph_i

    def _integrate_spectra(self, ii, fftinfo):
        for jj in range(self.nharms):
            frange = _np.asarray([self.freq[self._ifk[jj]-self._ifw],
                                  self.freq[self._ifk[jj]+self._ifw]])
            _isl = _np.arange(self._ifk[jj]-self._ifw, self._ifk[jj]+self._ifw, 1, dtype=int)
            [Txy, Txx, Amp, Coh, Phase, Vxy, Vxx, varA, varC, varP ] = \
                self.integrate_spectra(fftinfo, frange)

            if self.usesegs:
                Tnn = 0.5*fftinfo.ENBW* \
                    (fftinfo.Pyy_seg[:self.Navr, _isl[0]-1]+fftinfo.Pyy_seg[:self.Navr, _isl[-1]+1]).T
            else:
                Tnn = 0.5*fftinfo.ENBW*(fftinfo.Pyy[_isl[0]-1]+fftinfo.Pyy[_isl[-1]+1]).T
            # end if
            self.Tnn[ii,jj] =  _np.asscalar(Tnn.copy())
            self.Txy[ii,jj] =  _np.asscalar(Txy.copy())
            self.Vxy[ii,jj] =  _np.asscalar(Vxy.copy())

            # Auto power spectra are real by definition ... discard the
            # floating point complex component if it is still around
            self.Amp[ii,jj] = _np.real(Amp.copy())
            self.varA[ii,jj] = _np.real(varA.copy())
            self.Coh[ii,jj] = _np.real(Coh.copy())
            self.varC[ii,jj] = _np.real(varC.copy())
            self.Phase[ii,jj] = _np.real(Phase.copy())
            self.varP[ii,jj] = _np.real(varP.copy())

            if ii == 0:
                self.Vxx[jj] = _np.real(Vxx.copy())
                self.Txx[jj] = _np.real(Txx.copy())
            # endif
        #end loop over harmonics
    # end def _integrate_spectra

    def _PWELCH_ch(self, ii, iref=0, plotAlias=None):
        if self.verbose:
            print('Working on channel %s'%(str(ii+1).zfill(2),))
        # end if

        [self.freq, _, _, _, _, _, fftinfo] = \
            _fft.fft_pwelch(self.tt, self._refsig, self._sig[:,ii], self.tbounds,
                            Navr=self.Navr, windowoverlap=self.overlap,
                            windowfunction=self.winfun, useMLAB=self.useMLAB, plotit=False, verbose=False)

        if ii == 0:
            # self._getharmindex_()
            self._getharmindex_(fftinfo=fftinfo)
            self._HP_preallocate_(fftinfo)
        # endif

        if self.usesegs:
            self.Pxy[:, ii, :] = (fftinfo.Pxy_seg.reshape((self.nf, self.Navr), order='C')).copy()
            self.Pyy[:, ii, :] = (fftinfo.Pyy_seg.reshape((self.nf, self.Navr), order='C')).copy()
            self.vPxy[:, ii, :] = (fftinfo.varPxy_seg.reshape((self.nf, self.Navr), order='C')).copy()
            self.vPyy[:, ii, :] = (fftinfo.varPyy_seg.reshape((self.nf, self.Navr), order='C')).copy()
        else:
            self.Pxy[:,ii] = (fftinfo.Pxy.reshape((self.nf,), order='C')).copy()
            self.Pyy[:,ii] = (fftinfo.Pyy.reshape((self.nf,), order='C')).copy()
            self.vPxy[:,ii] = (fftinfo.varPxy.reshape((self.nf,), order='C')).copy()
            self.vPyy[:,ii] = (fftinfo.varPyy.reshape((self.nf,), order='C')).copy()
        # endif

        # Integrate the spectra over the modulation frequency (and harmonics if you wish)
        self._integrate_spectra(ii, fftinfo)

        if self.plotit and not self.usesegs:
            if plotAlias is not None:
                # hfig, hSPEC = self.plot_HPinfo(ii, fftinfo)
                hfig, hSPEC = plotAlias(ii, fftinfo)
                if self.saveit:
                    self.closefig = True
                    if (ii == iref):  self.closefig = False  # endif
                    self.save_fig(hfig, hSPEC)
                # end if saveit
            # end if plotAlias
        # end if plotit
    # end def _PWELCH_ch

    def _PWELCH_chloop(self, iref=0, plotAlias=None):
        self._PWELCH_settings_()
        self._PWELCH_preallocate()

        # Loop over channels
        for ii in range(self.nch):
            self._PWELCH_ch(ii, iref, plotAlias)
        # end for loop over channels

        # ============== Mean squared coherence =============== #
        # Up to now we are using mean-squared coherence
        self.Coh = _np.sqrt(self.Coh)   # Linear coherence
        self.varC = ((1.0-self.Coh**2.0)/_np.sqrt(2*self.Navr))**2.0

        # =============== Phase angle ==================== #
        # A.E. White, Phys. Plasmas, 17 056103, 2010
        # Doesn't so far give a convincing answer...
        # fftinfo.varPhxy = _np.zeros(Pxy.shape, dtype=_np.float64)
        self.varP = (_np.sqrt(1.0-self.Coh**2)/_np.sqrt(2.0*self.Navr*self.Coh))**2.0

        self.Phase = _np.angle(self.Txy)        # Save the cross-phase as well
        if not self.useMLAB:
            self.Phase *= -1   # The ECE is lagging the ECRH ... there is a discrepency between my version and mlab's
        # end if

#      This part is now included in the spectra integration part
#        #Coh = _np.abs( Txy*_np.conj(Txy) )/( _np.abs(Txx)*_np.abs(Amp) )
#        #varC = Vxy*(2*_np.abs(Txy)/(_np.abs(Txx)*_np.abs(Amp)))**2 + \
#        #       Vxx*(  _np.abs( Txy*_np.conj(Txy) )/( _np.abs(Txx)**2*_np.abs(Amp) ) )**2 + \
#        #       varA*(  _np.abs( Amp*_np.conj(Amp) )/( _np.abs(Amp)**2*_np.abs(Txx) ) )**2
#        [self.Coh, self.varC] = \
#            _fft.varcoh(self.Txy, self.Vxy, self.Txx, self.Vxx, self.Amp, self.varA)
#
#        self.Coh, self.varCoh = _fft.monticoh(self.Txy, self.Vxy, self.Txx, self.Vxx,
#                                              self.Amp, self.varA, nmonti=10e3, meansquared=True)
#
#
#        self.Phase, self.varP = _fft.montiphi(self.Txy, self.Vxy, nmonti=10e3)
#
##        self.Phase, self.varP = _fft.varphi(_np.real(self.Txy), _np.imag(self.Txy),
##                                            _np.real(self.Vxy), _np.imag(self.Vxy))
#
##        self.Phase = _np.angle(self.Txy)
#        self.Phase = _np.arctan2(_np.imag(self.Txy), _np.real(self.Txy))
#
#        _tangent = _np.imag(self.Txy)/_np.real(self.Txy)
#        _vartang = (_np.imag(self.Vxy)+_np.real(self.Vxy)*_tangent**2)/(_np.real(self.Txy)**2)
#        self.varP = _vartang/(1+_tangent**2)**2

        for ii in range(self.nharms):
            ph = self.Phase[:,ii].copy()
            while (ph>0.3).any() or (ph<-2*_np.pi-0.3).any():
               # ph[ph>0.3] = ph[ph>0.3] - 2.0*_np.pi
               if (ph>0.3).any():
                   ph[ph>0.3] = ph[ph>0.3] - 2*_np.pi
               if (ph<-2*_np.pi-0.3).any():
                   ph[ph<-2*_np.pi-0.3] = ph[ph<-2*_np.pi-0.3] + 2*_np.pi

#               mph = _np.nanmean(ph)
#               if (ph>mph+2*_np.pi).any():
#                   ph[ph>mph+2*_np.pi] = ph[ph>mph+2*_np.pi] - 2*_np.pi
#               if (ph<mph-2*_np.pi).any():
#                   ph[ph<mph-2*_np.pi] = ph[ph<mph-2*_np.pi] + 2*_np.pi
            # end while
            self.Phase[:, ii] = ph.copy()
        # end for
#        self.Phase = _np.unwrap(self.Phase, axis=0)

        # ========== Variance in Amplitude perturbation ====== #

        # Estimate the variance in the power spectra: this requires building
        # a distribution by varying the parameters used in the FFT, nwindows,
        # nfft, windowfunction, etc.  I don't do this right now.
        # This part is from Anne White's thesis
        self.varA = self.Txx*self.Amp*(1.0-self.Coh)/self.Navr

        # ========== Convert to Logarithmic perturbation ====== #
        #Convert to an RMS perturbation
        self.varA = _np.abs(0.25*self.varA/self.Amp)
        self.Amp = _np.sqrt(_np.abs(self.Amp))
        self.Txx = _np.sqrt(_np.abs(self.Txx))
        self.Tnn = _np.float64(_np.sqrt(_np.abs(self.Tnn)))
        self.RMSECHpower = self.Txx.copy()

        # ======= Convert from RMS to Amplitude perturbation ========= #
        self.varA = self.varA/(self.DutyCycle)
        self.Amp = self.Amp/_np.sqrt(self.DutyCycle)
        self.Txx = self.Txx/_np.sqrt(self.DutyCycle)
        self.Tnn = self.Tnn/_np.sqrt(self.DutyCycle)
        self.ModECHpower = self.Txx.copy()

        # ======= Convert to logarithmic amplitude perturbation ======= #
        self.varA = self.varA/self.Amp**2
        self.Amp = _np.log(self.Amp)
        self.Tnn = _np.log(self.Tnn)
    # end def _PWELCH_chloop
    #  ====================================================================  #

# end class __HeatPulse_FFTbase__

# ========================================================================== #
# ========================================================================== #
