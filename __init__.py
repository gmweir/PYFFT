# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:06:36 2018

@author: gawe
"""
# ========================================================================== #
# ========================================================================== #

__version__ = "2018.02.15.17"
__all__ = ['fft_analysis', 'spectrogram', 'filters', 'hilbert', 'laplace'] # analysis:ignore

from . import fft_analysis as fft
from . import spectrogram
from . import filters
from . import hilbert
from . import laplace

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
from .fft_analysis import fft_pwelch, detrend_none, detrend_mean, detrend_linear  # analysis:ignore
from .fft_analysis import fftanal, unwrap_tol  # analysis:ignore

# ========================================================================== #

from .filters import butter_lowpass_filter, butter_bandpass, upsample, downsample, downsample_efficient  # analysis:ignore

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
# ========================================================================== #