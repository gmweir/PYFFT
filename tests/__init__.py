"""
These tests were initially shamelessly stolen from the python bindings for FFTW
--> https://github.com/pyFFTW

The idea is to add tests for my own simplified wrapper that picks the FFT interface based on what is installed
    FFTW --> pyfftw
    numpy --> numpy.fft
    scipy --> scipy.fftpack

"""