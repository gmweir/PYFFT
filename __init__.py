# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:06:36 2018

@author: gawe
"""
# ========================================================================== #
# ========================================================================== #

import sys

__version__ = "2021.04.22.17"
__all__ = ['fft_analysis', 'spectrogram', 'filters', 'hilbert', 'laplace', 'fft', 'utils', 'windows', 'multitaper'] # analysis:ignore

# from . import fft_analysis as fft
from . import fft
from . import utils
from . import spectrogram
from . import filters
from . import hilbert
from . import laplace
from . import fft_analysis
from . import windows
from . import multitaper
# from .multitaper import multitaper

# sys.modules['multitaper.mtcross'] = python.multitaper.mtcross
# sys.modules['multitaper.mtspec'] = python.multitaper.mtspec
# sys.modules['multitaper.utils'] = python.multitaper.utils

# ========================================================================== #

#try:
#    import FFT.fft_analysis as _fft
#    from FFT.fft_analysis import fft_pwelch, detrend_none, detrend_mean, detrend_linear  # analysis:ignore
#    from FFT.fft_analysis import fftanal, unwrap_tol  # analysis:ignore
#except:
#    import fft_analysis as _fft
#    from .fft_analysis import fft_pwelch, detrend_none, detrend_mean, detrend_linear  # analysis:ignore
#    from .fft_analysis import fftanal, unwrap_tol  # analysis:ignore
## end try

#import fft_analysis as _fft
from .fft_analysis import fftanal, fft_pwelch # analysis:ignore

from .utils import detrend_none, detrend_mean, detrend_linear, unwrap_tol  # analysis:ignore
from .utils import upsample, downsample, downsample_efficient # analysis:ignore

# ========================================================================== #

from .filters import butter_lowpass_filter, butter_bandpass  # analysis:ignore

# from .hilbert import hilbert, hilbert_1d  # analysis:ignore
# from .laplace import laplace, laplace_1d  # analysis:ignore

from .spectrogram import stft, specgram


#from . import kPyWavelet as kpywt
#from . import CECE # analysis:ignore

try:
    from . import pycwt
except:
    pass
# end try

# ========================================================================== #

# import numpy as _np


# ========================================================================== #
# ========================================================================== #