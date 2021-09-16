# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:15:38 2019

@author: gawe


  These functions are just for understanding how these algorithms work.  They
  are not optimized at all. You should use the built-in functions from
  np scipy or FFT

"""
# =========================================================================== #
# =========================================================================== #


from __future__ import absolute_import, with_statement, absolute_import, \
                       division, print_function, unicode_literals
import numpy as _np
import matplotlib.pyplot as _plt
import pybaseutils.utils as _ut
__metaclass__ = type

import cmath   # compute_dft_complex
import math    # compute_dft_real_pair, complex_dft

import random  # DFT plus above
pi2 = cmath.pi * 2.0  # DFT

# =========================================================================== #
# =========================================================================== #


def compute_dft_complex(sig):
    """
    Discrete Fourier transform (Python)
    by Project Nayuki, 2017. Public domain.
    https://www.nayuki.io/page/how-to-implement-the-discrete-fourier-transform




    Computes the discrete Fourier transform (DFT) of the given complex vector.
    'sig' is a sequence of numbers (integer, float, or complex).
    Returns a list of complex numbers as output, having the same length.
    """
    n = len(sig)
    output = []
    for k in range(n):  # For each output element
        s = complex(0)
        for t in range(n):  # For each input element
            angle = 2j * cmath.pi * t * k / n
            s += sig[t] * cmath.exp(-angle)
        output.append(s)
    return output


def compute_dft_real_pair(inreal, inimag):
    """
    (Alternate implementation using only real numbers.)
    Computes the discrete Fourier transform (DFT) of the given complex vector.
    'inreal' and 'inimag' are each a sequence of n floating-point numbers.
    Returns a tuple of two lists of floats - outreal and outimag, each of length n.
    """
    assert len(inreal) == len(inimag)
    n = len(inreal)
    outreal = []
    outimag = []
    for k in range(n):  # For each output element
        sumreal = 0.0
        sumimag = 0.0
        for t in range(n):  # For each input element
            angle = 2 * math.pi * t * k / n
            sumreal +=  inreal[t] * math.cos(angle) + inimag[t] * math.sin(angle)
            sumimag += -inreal[t] * math.sin(angle) + inimag[t] * math.cos(angle)
        outreal.append(sumreal)
        outimag.append(sumimag)
    return (outreal, outimag)


def ForwardDFT(fnList):
    """
    Alternative implementation
    Discrete Fourier Transform (DFT)
    FB - 20141227
    """
    N = len(fnList)
    FmList = []
    for m in range(N):
        Fm = 0.0
        for n in range(N):
            Fm += fnList[n] * cmath.exp(- 1j * pi2 * m * n / N)
        FmList.append(Fm / N)
    return FmList

def InverseDFT(FmList):
    N = len(FmList)
    fnList = []
    for n in range(N):
        fn = 0.0
        for m in range(N):
            fn += FmList[m] * cmath.exp(1j * pi2 * m * n / N)
        fnList.append(fn)
    return fnList

# ====== #

def dft(x, sign=-1):
    """
    DFT using discrete summation
       X(n) = \sum_k W^{nk} x(k),  W = e^{-j2\pi/N}
    where N need not be power of 2.  The choice of e^{-j2\pi/N} or
    e^{j2\pi/N} is made by "sign=-1" or "sign=1" respectively.
    """
    N = len(x)
    W = [_np.exp(sign * 2j * _np.pi * i / N)
          for i in range(N)]          # exp(-j...) is default
    X = [sum(W[n * k % N] * x[k] for k in range(N))
          for n in range(N)]
    return X
# end def dft


def idft(X):
    """
    Inverse DFT with normalization by N, so that x == idft(dft(x)) within
    round-off errors.
    """
    N, x = len(X), dft(X, sign=1)       # e^{j2\pi/N}
    for i in range(N):
        x[i] /= float(N)
    return x
# end def idft


def test():
    # TEST
    print("Input Sine Wave Signal:")
    N = 360 # degrees (Number of samples)
    a = float(random.randint(1, 100))
    f = float(random.randint(1, 100))
    p = float(random.randint(0, 360))
    print("frequency = " + str(f))
    print("amplitude = " + str(a))
    print("phase ang = " + str(p))
    print('\n')
    fnList = []
    for n in range(N):
        t = float(n) / N * pi2
        fn = a * math.sin(f * t + p / 360 * pi2)
        fnList.append(fn)

    print("ForwardDFT Calculation Results:")
    FmList = ForwardDFT(fnList)
    threshold = 0.001
    for (i, Fm) in enumerate(FmList):
        if abs(Fm) > threshold:
            print("frequency = " + str(i))
            print("amplitude = " + str(abs(Fm) * 2.0))
            p = int(((cmath.phase(Fm) + pi2 + pi2 / 4.0) % pi2) / pi2 * 360 + 0.5)
            print("phase ang = " + str(p))
            print('\n')

    ### Recreate input signal from DFT results and compare to input signal
    ##fnList2 = InverseDFT(FmList)
    ##for n in range(N):
    ##    print((fnList[n], fnList2[n].real))
# end def test


# ============================================ #
# An implementation of the FFT


def complex_dft(xr, xi, n):
    pi = 3.141592653589793
    rex = [0] * n
    imx = [0] * n
    for k in range(0, n):  # exclude n
        rex[k] = 0
        imx[k] = 0
    for k in range(0, n):  # for each value in freq domain
        for i in range(0, n):  # correlate with the complex sinusoid
            sr =  math.cos(2 * pi * k * i / n)
            si = -math.sin(2 * pi * k * i / n)
            rex[k] += xr[i] * sr - xi[i] * si
            imx[k] += xr[i] * si + xi[i] * sr
    return rex, imx

# FFT version based on the original BASIC program
def fft_basic(rex, imx, n):
    pi = 3.141592653589793
    m = int(math.log(n, 2))  # float to int
    j = n / 2

    # bit reversal sorting
    for i in range(1, n - 1):  # [1,n-2]
        if i >= j:
            # swap i with j
            print "swap %d with %d"%(i, j)
            rex[i], rex[j] = rex[j], rex[i]
            imx[i], imx[j] = imx[j], imx[i]
        k = n / 2
        while (1):
            if k > j:
                break
            j -= k
            k /= 2
        j += k

    for l in range(1, m + 1):  # each stage
        le = int(math.pow(2, l))  # 2^l
        le2 = le / 2
        ur = 1
        ui = 0
        sr =  math.cos(pi / le2)
        si = -math.sin(pi / le2)
        for j in range(1, le2 + 1):  # [1, le2] sub DFT
            for i in xrange(j - 1, n - 1, le):  #  for butterfly
                ip = i + le2
                tr = rex[ip] * ur - imx[ip] * ui
                ti = rex[ip] * ui + imx[ip] * ur
                rex[ip] = rex[i] - tr
                imx[ip] = imx[i] - ti
                rex[i] += tr
                imx[i] += ti
            tr = ur
            ur = tr * sr - ui * si
            ui = tr * si + ui * sr

def print_list(l):
    n = len(l)
    print("[%d]: {"%(n,))
    for i in xrange(0, n):
        print(l[i])
    print("}")

# ======================================================================== #
# ======================================================================== #


def fft(x, sign=-1):
    """
    FFT using Cooley-Tukey algorithm where N = 2^n.  The choice of
    e^{-j2\pi/N} or e^{j2\pi/N} is made by 'sign=-1' or 'sign=1'
    respectively.  Since I prefer Engineering convention, I chose
    'sign=-1' as the default.

    FFT is performed as follows:
    1. bit-reverse the array.
    2. partition the data into group of m = 2, 4, 8, ..., N data points.
    3. for each group with m data points,
        1. divide into upper half (section A) and lower half (section B),
            each containing m/2 data points.
        2. divide unit circle by m.
        3. apply "butterfly" operation
                |a| = |1  w||a|     or      a, b = a+w*b, a-w*b
                |b|   |1 -w||b|
            where a and b are data points of section A and B starting from
            the top of each section, and w is data points along the unit
            circle starting from z = 1+0j.
    FFT ends after applying "butterfly" operation on the entire data array
    as whole, when m = N.
    """
    N = len(x)
    W = [_np.exp(sign * 2j * _np.pi * i / N)
          for i in range(N)]          # exp(-j...) is default
    x = bitrev(x)
    m = 2
    while m <= N:
        for s in range(0, N, m):
            for i in range(m/2):
                n = i * N / m
                a, b = s + i, s + i + m/2
                x[a], x[b] = x[a] + W[n % N] * x[b], x[a] - W[n % N] * x[b]
        m *= 2
    return x
# end def fft


def ifft(X):
    """
    Inverse FFT with normalization by N, so that x == ifft(fft(x)) within
    round-off errors.
    """
    N, x = len(X), fft(X, sign=1)       # e^{j2\pi/N}
    for i in range(N):
        x[i] /= float(N)
    return x
# end def ifft



# ======================================================================== #
# ======================================================================== #

if __name__ == "__main__":
    print("hello,world.")
    pi = 3.1415926
    x = []
    n = 64
    for i in range(0, n):
        p = math.sin(2 * pi * i / n)
        x.append(p)

    xr = x[:]
    xi = x[:]
    rex, imx = complex_dft(xr, xi, n)
    print("complet_dft(): n=%i"%(n,))
    print("rex: ")
    print_list([int(e) for e in rex])
    print("imx: ")
    print_list([int(e) for e in imx])

    fr = x[:]
    fi = x[:]

    fft_basic(fr, fi, n)
    print("fft_basic(): n=%i"%(n,))
    print("rex: ")
    print_list([int(e) for e in fr])
    print("imx: ")
    print_list([int(e) for e in fi])


# ======================================================================== #
# ======================================================================== #