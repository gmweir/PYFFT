"""
In this module:
    spectrogram()
    stft()
    test_stft()

    STFT class (under development)
"""
# ========================================================================== #
# ========================================================================== #


# This section is to improve python compataibilty between py27 and py3
from __future__ import absolute_import, with_statement, absolute_import, division, print_function, unicode_literals
__metaclass__ = type

# ========================================================================== #

import numpy as _np
import matplotlib.pyplot as _plt

try:
    from .fft_analysis import fftanal
except:
    from FFT.fft_analysis import fftanal
# end try


# import scipy.signal as _dsp
# import sys
# #import scipy.io.wavfile as wav
# import matplotlib

# #matplotlib.use('Qt4Agg')
# #matplotlib.rcParams['backend.qt4']='PySide'

# #from PySide.QtGui import QApplication, QMainWindow
# #from ParseMathExp import eval_expr
# #from BuildQtGui import make_gui

# ========================================================================== #
# ========================================================================== #
"""
    Spectrograms based on the short-time fourier transform

"""


def specgram(t, s, wl = 512, hanning = True, overlap = True, windowAverage = None):
    """
    Annoyed with the lack of flexibility of the matplotlib specgram, I wrote this program to make my
    own spectrogram. It simply retruns the spectrogram array with time and frequency vectors

    INPUTS:
        t  - the time vector of the signal (same length as s)
        s  - the 1D time signal for which to make a spectrogram
        wl - the window length

    RETURNS:
        time        - time axis of the spectrogram
        frequency   - frequency axis of the spectrogram
        spectrogram - the spectrogram array

    """

    if windowAverage != None:
        overlap = False
    # end if

    s = s.flatten()
    n  = len(s)
    dt = _np.abs(t[1]-t[0])

    if overlap:
        nWindows = 2*(n - (n % wl))//wl - 1
    else:
        nWindows = (n - (n % wl))//wl - 1
    # end if
    spectrogram = _np.zeros((wl,nWindows))

    print('nWindows = ', nWindows)
    try:
        nProg = 100
        a = _np.floor_divide(nWindows/nProg)
        print('[',' '*(nWindows/a) , ']')
        progressBar = True
    except:
        progressBar = False
    # end try

    for i in range(nWindows):

            if progressBar:
                if ((i % a) == 0):
                    print('.',end='')
                # end if
            # end if

            if overlap:
                idx1 = i*wl//2
                idx2 = idx1+wl
            else:
                idx1 = i*wl
                idx2 = idx1+wl
            # end if

            # first Fourier transform each block within the data. Either with a Hanning window or otherwise
            if hanning:
                spectrogram[:,i] = _np.sqrt(8.0/3.0)*_np.abs(_np.fft.fft(_np.hanning(wl)*(s[idx1:idx2])))**2/wl
            else:
                spectrogram[:,i] = _np.abs(_np.fft.fft(s[idx1:idx2]))**2/wl
            # end if

    if windowAverage != None:

        spectrogramAverage = _np.zeros((wl,nWindows/windowAverage))
        for i in range(nWindows/windowAverage):
            spectrogramAverage[:,i] = _np.mean(spectrogram[:,i*windowAverage:(i+1)*windowAverage],axis=1)
        # end for

        fAxis = _np.fft.fftfreq(wl,dt)
        time  = _np.linspace(t[0]+wl*dt/2, t[0]+wl*dt*((nWindows-1) + 1/2), num = nWindows/windowAverage)
        return time, fAxis, spectrogramAverage

    else:
        fAxis = _np.fft.fftfreq(wl,dt)
        if overlap == False:
            time  = _np.linspace(t[0]+wl*dt/2, t[0]+wl*dt*((nWindows-1) + 1/2), num = nWindows)
        else:
            time  = _np.linspace(t[0]+wl*dt/2, t[0]+wl*dt*((nWindows/2-1) + 1/2), num = nWindows)
        # end if
        return time, fAxis, spectrogram
    # end if
# end spectrogram function

# ========================================================================== #
# ========================================================================== #


def stft(tt, y_in, tper=None, returnclass=True, **kwargs):

    if tper is None:
        # tper = 1e-3
        tper = (tt[-1]-tt[0])/20
        if tper< tt[2]-tt[1]:
            # tper = (tt[2]-tt[1])*len(tt)
            print('check your stft window size')
        # end if
    # end if

    # Instantiate the fft analysis wrapper class (but don't run it)
    Ystft = fftanal()

    # Initialize the class with settings and variables
    Ystft.init(tt, y_in, tper=tper, **kwargs)

    # Perform the loop over averaging windows to generate the short time four. xform
    #   Note that the zero-frequency component is in the middle of the array
    #   if the input signal is complex (2-sided transform)
    Ystft.stft()   # frequency [cycles/s], STFT [Navr, nfft]

    if returnclass:
        return Ystft
    else:
        twin = _np.linspace(tt[0], tt[-1], num=Ystft.Navr, endpoint=True)
        return twin, Ystft.freq, Ystft.Xseg
    # end if
# end def

# ========================================================================== #

def test_case(case=3, npts=2e3):
    if case == 1:
        tt = _np.linspace(0, 1.0, num=npts, endpoint=True)
        dt = tt[2]-tt[1]
        f0 = 0.1/dt
        print(f0)
        y_in = _np.sin(2*_np.pi*f0*tt-0.0)
        tper = 200*dt
    elif case == 2:
        _t0 = _np.linspace(0, 1.0, num=npts, endpoint=True)
        _t1 = _np.linspace(_t0[-1]+_t0[1]-_t0[0], 2.0*_t0[-1], num=npts, endpoint=True)
        tt = _np.asarray(_t0.tolist() + _t1.tolist())
        dt = tt[2]-tt[1]
        f0 = 0.05/dt
        f1 = 0.08/dt
        print((f0, f1))
        y_in = _np.asarray(
            (_np.sin(2*_np.pi*f0*_t0-0.0)).tolist()
          + (_np.sin(2*_np.pi*f1*_t1-0.0)).tolist()
                          )
    elif case == 3:
        tt = _np.linspace(0, 1.0, num=npts, endpoint=True)
        Fs = 1.0/(tt[1]-tt[0])
        fstart = 100
        fend = 200
        f_in = _np.linspace(fstart, fend, num=len(tt), endpoint=True)

        phase_in = _np.cumsum(f_in/Fs)
        y_in = _np.sin(2.0*_np.pi*phase_in)
    # end if
    return tt, y_in


def test_specgram(windowfunction='hanning', npts=2e3, Nper = 21):
    """
    Error in spectrogram output unfortunately
    """

    tt, y_in = test_case(case=3, npts=npts)

    overlap = True
    if windowfunction is None:
        hanning = False
        # overlap = False
    else:
        hanning = True
        # overlap = True
    # end if

    # end if
    tper = (tt[-1]-tt[0])/(Nper-1)

    # specgram(t, s, wl = 512, hanning = True, overlap = True, windowAverage = None)
    twin, freq, spec = specgram(tt, y_in, wl = int(tper/(tt[1]-tt[0])), hanning = hanning, overlap = overlap, windowAverage = None)

    norm = _np.abs(spec*_np.conj(spec))
    Stot = _np.mean(spec, axis = 1).squeeze()

    _plt.figure()
    _ax1 = _plt.subplot(3,1,1)
    _ax1.plot(tt, y_in, '-')
    _ax1.set_xlabel('t [s]')
    _ax1.set_ylabel('input')

    _ax2 = _plt.subplot(3,1,2)
    # _ax2.imshow(10*_np.log10(norm), aspect='auto', interpolation='none', cmap=_plt.cm.hot,
                # extent = [tt[0], tt[-1], stft_y.freq.min(), stft_y.freq.max()])
    _ax2.pcolormesh(twin, freq, 10*_np.log10(norm), shading='nearest', cmap=_plt.cm.rainbow) # , cmap=_plt.cm.hot)
    _ax2.set_xlabel('t [s]')
    _ax2.set_ylabel('f [Hz]')

    _ax3 = _plt.subplot(3,1,3)
    _ax3.plot(freq, 10*_np.log10(_np.abs(Stot*_np.conj(Stot))), '-')
    _ax3.set_xlabel('f [s]')
    _ax3.set_ylabel('|FFT[y]|')
# end test_stft


def test_stft(windowfunction=None, npts=2e3, Nper = 21):
    tt, y_in = test_case(case=3, npts=npts)
    if windowfunction is None:
        if 1:
            # windowoverlap = 0.00  # optimized value
            windowfunction = 'None' # gives no window
        else:
            # windowoverlap = None # optimized value
            windowfunction = None  # Defaults to hanning
        # endif
    # end if

    # end if
    tper = (tt[-1]-tt[0])/(Nper-1)

    # stft_y = stft(tt, y_in, tper = tper, returnclass=True, windowoverlap=windowoverlap, windowfunction=windowfunction)
    stft_y = stft(tt, y_in, tper = tper, returnclass=True, windowfunction=windowfunction)
    sss = stft_y.Xseg
    # norm = _np.absolute(sss).transpose()
    norm = _np.abs(sss*_np.conj(sss)).transpose()
    twin = _np.linspace(tt[0], tt[-1], num=stft_y.Navr, endpoint=True)

    _plt.figure()
    _ax1 = _plt.subplot(3,1,1)
    _ax1.plot(tt, y_in, '-')
    _ax1.set_xlabel('t [s]')
    _ax1.set_ylabel('input')

    _ax2 = _plt.subplot(3,1,2)
    # _ax2.imshow(10*_np.log10(norm), aspect='auto', interpolation='none', cmap=_plt.cm.hot,
                # extent = [tt[0], tt[-1], stft_y.freq.min(), stft_y.freq.max()])
    _ax2.pcolormesh(twin, stft_y.freq, 10*_np.log10(norm), shading='nearest', cmap=_plt.cm.rainbow) # , cmap=_plt.cm.hot)
    _ax2.set_xlabel('t [s]')
    _ax2.set_ylabel('f [Hz]')

    _ax3 = _plt.subplot(3,1,3)
    _ax3.plot(stft_y.freq, 10*_np.log10(_np.abs(stft_y.Xfft*_np.conj(stft_y.Xfft))), '-')
    _ax3.set_xlabel('f [s]')
    _ax3.set_ylabel('|FFT[y]|')
# end test_stft

# ========================================================================= #
# ========================================================================= #


class STFT(object):
    """Computes the short time fourier transform of a signal
    How to use:
    1.) Pass the signal into the class,
    2.) Call stft() to get the transformed data
    3.) Call freq_axis() and time_axis() to get the freq and time values for each index in the array
    """
    def __init__(self, data, fs, win_size, fft_size, overlap_fac=0.5):
        """Computes a bunch of information that will be used in all of the STFT functions"""
        self.data = _np.array(data, dtype=_np.float32)
        self.fs = _np.int32(fs)
        self.win_size = _np.int32(win_size)
        self.fft_size = _np.int32(fft_size)
        self.overlap_fac = _np.float32(1 - overlap_fac)

        self.hop_size = _np.int32(_np.floor(self.win_size * self.overlap_fac))
        self.pad_end_size = self.fft_size
        self.total_segments = _np.int32(_np.ceil(len(self.data) / _np.float32(self.hop_size)))
        self.t_max = len(self.data) / _np.float32(self.fs)

    def stft(self, scale='log', ref=1.0, clip=None):
        """Perform the STFT and return the result"""

        # Todo: changing the overlap factor doens't seem to preserve energy, need to fix this
        window = _np.hanning(self.win_size) * self.overlap_fac * 2
        inner_pad = _np.zeros((self.fft_size * 2) - self.win_size)

        proc = _np.concatenate((self.data, _np.zeros(self.pad_end_size)))
        result = _np.empty((self.total_segments, self.fft_size), dtype=_np.float32)

        for ii in range(self.total_segments):
            current_hop = self.hop_size * ii
            segment = proc[current_hop:current_hop+self.win_size]
            windowed = segment * window
            padded = _np.append(windowed, inner_pad)
            spectrum = _np.fft.fft(padded) / self.fft_size
            autopower = _np.abs(spectrum * _np.conj(spectrum))
            result[ii, :] = autopower[:self.fft_size]

        if scale == 'log':
            result = self.dB(result, ref)

        if clip is not None:
            _np.clip(result, clip[0], clip[1], out=result)

        return result

    def dB(self, data, ref=1.0):
        """Return the dB equivelant of the input data"""
        return 20*_np.log10(data / ref)

    def freq_axis(self):
        """Returns a list of frequencies which correspond to the bins in the returned data from stft()"""
        return _np.arange(self.fft_size) / _np.float32(self.fft_size * 2) * self.fs

    def time_axis(self):
        """Returns a list of times which correspond to the bins in the returned data from stft()"""
        return _np.arange(self.total_segments) / _np.float32(self.total_segments) * self.t_max


#def create_ticks_optimum(axis, num_ticks, resolution, return_errors=False):
    #""" Try to divide <num_ticks> ticks evenly across the axis, keeping ticks to the nearest <resolution>"""
    #max_val = axis[-1]
    #hop_size = max_val / _np.float32(num_ticks)

    #indicies = []
    #ideal_vals = []
    #errors = []

    #for i in range(num_ticks):
        #current_hop = resolution * round(float(i*hop_size)/resolution)
        #index = _np.abs(axis-current_hop).argmin()

        #indicies.append(index)
        #ideal_vals.append(current_hop)
        #errors.append(_np.abs(current_hop - axis[index]))

    #if return_errors:
        #return indicies, ideal_vals, errors
    #else:
        #return indicies, ideal_vals


#class StftGui(QMainWindow):
    #"""The gui for the STFT application"""
    #def __init__(self, filename, parent=None):
        #super(StftGui, self).__init__(parent)
        #self.ui = Ui_MainWindow()
        #self.ui.setupUi(self)

        #self.ui.render.clicked.connect(self.on_render)
        #self.ui.mpl.onResize.connect(self.redraw)
        #self.init(filename)

    #def init(self, filename):
        #self.fs, self.data = wav.read(filename)
        #if len(self.data.shape) > 1:
            ## if there are multiple channels, pick the first one.
            #self.data = self.data[...,0]

        #self.on_render()

    #def redraw(self, *args, **kwargs):
        #fig = self.ui.mpl.fig
        #fig.tight_layout()
        #self.ui.mpl.draw()

    ##def on_render(self):
        ### get data from GUI
        ##downsample_fac = int(eval_expr(self.ui.downsample_fac.text()))
        ##win_size = int(eval_expr(self.ui.win_size.text()))
        ##fft_size = int(eval_expr(self.ui.fft_size.text()))
        ##overlap_fac = float(eval_expr(self.ui.overlap_fac.text()))
        ##clip_min, clip_max = float(eval_expr(self.ui.clip_min.text())), float(eval_expr(self.ui.clip_max.text()))
        ##x_tick_num, x_res = int(eval_expr(self.ui.x_num_ticks.text())), float(eval_expr(self.ui.x_resolution.text()))
        ##x_tick_rotation = int(eval_expr(self.ui.x_tick_rotation.text()))
        ##y_ticks_num, y_res = int(eval_expr(self.ui.y_num_ticks.text())), float(eval_expr(self.ui.y_resolution.text()))


        #if downsample_fac > 1:
            #downsampled_data = _dsp.decimate(self.data, downsample_fac, ftype='fir')
            #downsampled_fs = self.fs / downsample_fac
        #else:
            #downsampled_data = self.data
            #downsampled_fs = self.fs

        #ft = Stft(downsampled_data, downsampled_fs, win_size=win_size, fft_size=fft_size, overlap_fac=overlap_fac)
        #result = ft.stft(clip=(clip_min, clip_max))

        #x_ticks, x_tick_labels = create_ticks_optimum(ft.freq_axis(), num_ticks=x_tick_num, resolution=x_res)
        #y_ticks, y_tick_labels = create_ticks_optimum(ft.time_axis(), num_ticks=y_ticks_num, resolution=y_res)

        #fig = self.ui.mpl.fig
        #fig.clear()
        #ax = fig.add_subplot(111)

        #img = ax.imshow(result, origin='lower', cmap='jet', interpolation='none', aspect='auto')

        #ax.set_xticks(x_ticks)
        #ax.set_xticklabels(x_tick_labels, rotation=x_tick_rotation)
        #ax.set_yticks(y_ticks)
        #ax.set_yticklabels(y_tick_labels)

        #if self.ui.x_grid.isChecked():
            #ax.xaxis.grid(True, linestyle='-', linewidth=1)

        #if self.ui.y_grid.isChecked():
            #ax.yaxis.grid(True, linestyle='-', linewidth=1)

        #ax.set_xlabel('Frequency [Hz]')
        #ax.set_ylabel('Time [s]')

        #fig.colorbar(img)
        #fig.tight_layout()

        #self.ui.mpl.draw()

        #self.ui.sampling_freq.setText('%d' % downsampled_fs)
        #self.ui.data_length.setText('%.2f' % ft.t_max)
        #self.ui.freq_res.setText('%s' % (downsampled_fs * 0.5 / _np.float32(ft.fft_size)))

# ========================================================================== #
# ========================================================================== #

class __spectrogram(object):
    """
    The spectrogram function from bout utils with adaptations:

    Creates spectrograms using the Gabor transform to maintain time and
    frequency resolution

    written by: Jarrod Leddy
    updated:    23/06/2016

    adapted and updated by GMW/2021
    """
    pass
# end class


def spectrogram(data, dx, sigma, clip=1.0, optimise_clipping=True, nskip=1.0):
    """Creates spectrograms using the Gabor transform to maintain time
    and frequency resolution

    .. note:: Very early and very late times will have some issues due
          to the method - truncate them after taking the spectrogram
          if they are below your required standards

    .. note:: If you are seeing issues at the top or bottom of the
          frequency range, you need a longer time series


    Parameters
    ----------
    data : array_like
        The time series you want spectrogrammed
    dt : float
        Time resolution
    sigma : float
        Used in the Gabor transform, will balance time and frequency
        resolution suggested value is 1.0, but may need to be adjusted
        manually until result is as desired:

            - If bands are too tall raise sigma
            - If bands are too wide, lower sigma
    clip : float, optional
        Makes the spectrogram run faster, but decreases frequency
        resolution. clip is by what factor the time spectrum should be
        clipped by --> N_new = N / clip
    optimise_clip : bool
        If true (default) will change the data length to be 2^N
        (rounded down from your inputed clip value) to make FFT's fast
    nskip : float
        Scales final time axis, skipping points over which to centre
        the gaussian window for the FFTs

    Returns
    -------
    tuple : (array_like, array_like, array_like)
        A tuple containing the spectrogram, frequency and time

    """
    # from builtins import range
    # from scipy import fftpack

    n = data.size
    nnew = int(n / nskip)
    xx = _np.arange(n) * dx
    xxnew = _np.arange(nnew) * dx * nskip
    sigma = sigma * dx

    n_clipped = int(n / clip)

    # check to see if n_clipped is near a 2^n factor for speed
    if optimise_clipping:
        nn = n_clipped
        two_count = 1
        while 1:
            nn = nn / 2.0
            if nn <= 2.0:
                n_clipped = 2 ** two_count
                print("clipping window length from ", n, " to ", n_clipped, " points")
                break
            else:
                two_count += 1
    else:
        print("using full window length: ", n_clipped, " points")

    halfclip = int(n_clipped / 2)
    spectra = _np.zeros((nnew, halfclip))

    # omega = fftpack.fftfreq(n_clipped, dx)
    omega = _np.fft.fftfreq(n_clipped, dx)
    omega = omega[0:halfclip]

    for i in range(nnew):
        beg = i * nskip - halfclip
        end = i * nskip + halfclip - 1

        if beg < 0:
            end = end - beg
            beg = 0
        elif end >= n:
            end = n - 1
            beg = end - n_clipped + 1

        gaussian = (
            1.0
            / (sigma * 2.0 * _np.pi)
            * _np.exp(-0.5 * _np.power((xx[beg:end] - xx[i * nskip]), 2.0) / (2.0 * sigma))
        )
        # fftt = abs(fftpack.fft(data[beg:end] * gaussian))
        fftt = abs(_np.fft.fft(data[beg:end] * gaussian))
        fftt = fftt[:halfclip]
        spectra[i, :] = fftt

    return (_np.transpose(spectra), omega, xxnew)


def test_spectrogram(n, d, s):
    """Function used to test the performance of spectrogram with various
    values of sigma

    Parameters
    ----------
    n : int
        Number of points
    d : float
        Grid spacing
    s : float
        Initial sigma

    """

    import matplotlib.pyplot as plt

    nskip = 10
    xx = _np.arange(n) / d
    test_data = _np.sin(2.0 * _np.pi * 512.0 * xx * (1.0 + 0.005 * _np.cos(xx * 50.0))) + 0.5 * _np.exp(
        xx
    ) * _np.cos(2.0 * _np.pi * 100.0 * _np.power(xx, 2))
    test_sigma = s
    dx = 1.0 / d

    s1 = test_sigma * 0.1
    s2 = test_sigma
    s3 = test_sigma * 10.0

    (spec2, omega2, xx) = spectrogram(test_data, dx, s2, clip=5.0, nskip=nskip)
    (spec3, omega3, xx) = spectrogram(test_data, dx, s3, clip=5.0, nskip=nskip)
    (spec1, omega1, xx) = spectrogram(test_data, dx, s1, clip=5.0, nskip=nskip)

    levels = _np.linspace(_np.min(spec1), _np.max(spec1), 100)
    plt.subplot(311)
    plt.contourf(xx, omega1, spec1, levels=levels)
    plt.ylabel("frequency")
    plt.xlabel(r"$t$")
    plt.title(r"Spectrogram of $sin(t + cos(t) )$ with $\sigma=$%3.1f" % s1)

    levels = _np.linspace(_np.min(spec2), _np.max(spec2), 100)
    plt.subplot(312)
    plt.contourf(xx, omega2, spec2, levels=levels)
    plt.ylabel("frequency")
    plt.xlabel(r"$t$")
    plt.title(r"Spectrogram of $sin(t + cos(t) )$ with $\sigma=$%3.1f" % s2)

    levels = _np.linspace(_np.min(spec3), _np.max(spec3), 100)
    plt.subplot(313)
    plt.contourf(xx, omega3, spec3, levels=levels)
    plt.ylabel("frequency")
    plt.xlabel(r"$t$")
    plt.title(r"Spectrogram of $sin(t + cos(t) )$ with $\sigma=$%3.1f" % s3)
    plt.tight_layout()
    plt.show()



# ========================================================================== #
# ========================================================================== #

# choi-williams

# ========================================================================== #
# ========================================================================== #


if __name__ == '__main__':
    #import os.path as path
    #make_gui('sfft_gui')
    #from sfft_gui import Ui_MainWindow


    #filename = path.join('media','mike_chirp.wav')
    ##filename = 'mike_annoying.wav'
    ##filename = 'New Seal and New Spring_conv.wav'

    #app = QApplication(sys.argv)
    #win = StftGui(filename)
    #win.show()
    #app.exec_()

    # test_stft()
    # test_specgram()

    test_spectrogram(2048, 2048.0, 0.01 )  # array size, divisions per unit, sigma of gaussian
# end if

# ========================================================================= #
# ========================================================================= #

