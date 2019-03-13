# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:15:38 2019

@author: gawe
"""

#
# Discrete Fourier transform (Python)
# by Project Nayuki, 2017. Public domain.
# https://www.nayuki.io/page/how-to-implement-the-discrete-fourier-transform
#


#
# Computes the discrete Fourier transform (DFT) of the given complex vector.
# 'input' is a sequence of numbers (integer, float, or complex).
# Returns a list of complex numbers as output, having the same length.
#
import cmath
def compute_dft_complex(input):
	n = len(input)
	output = []
	for k in range(n):  # For each output element
		s = complex(0)
		for t in range(n):  # For each input element
			angle = 2j * cmath.pi * t * k / n
			s += input[t] * cmath.exp(-angle)
		output.append(s)
	return output


#
# (Alternate implementation using only real numbers.)
# Computes the discrete Fourier transform (DFT) of the given complex vector.
# 'inreal' and 'inimag' are each a sequence of n floating-point numbers.
# Returns a tuple of two lists of floats - outreal and outimag, each of length n.
#
import math
def compute_dft_real_pair(inreal, inimag):
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


 # =========================== #

# Alternative implementation
 # Discrete Fourier Transform (DFT)
# FB - 20141227
import random
import math
import cmath
pi2 = cmath.pi * 2.0
def DFT(fnList):
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

# TEST
print "Input Sine Wave Signal:"
N = 360 # degrees (Number of samples)
a = float(random.randint(1, 100))
f = float(random.randint(1, 100))
p = float(random.randint(0, 360))
print "frequency = " + str(f)
print "amplitude = " + str(a)
print "phase ang = " + str(p)
print
fnList = []
for n in range(N):
    t = float(n) / N * pi2
    fn = a * math.sin(f * t + p / 360 * pi2)
    fnList.append(fn)

print "DFT Calculation Results:"
FmList = DFT(fnList)
threshold = 0.001
for (i, Fm) in enumerate(FmList):
    if abs(Fm) > threshold:
        print "frequency = " + str(i)
        print "amplitude = " + str(abs(Fm) * 2.0)
        p = int(((cmath.phase(Fm) + pi2 + pi2 / 4.0) % pi2) / pi2 * 360 + 0.5)
        print "phase ang = " + str(p)
        print

### Recreate input signal from DFT results and compare to input signal
##fnList2 = InverseDFT(FmList)
##for n in range(N):
##    print fnList[n], fnList2[n].real



# ============================================ #
# An implementation of the FFT

import math

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
	print "[%d]: {"%(n)
	for i in xrange(0, n):
		print l[i],
	print "}"


if __name__ == "__main__":
	print "hello,world."
	pi = 3.1415926
	x = []
	n = 64
	for i in range(0, n):
		p = math.sin(2 * pi * i / n)
		x.append(p)

	xr = x[:]
	xi = x[:]
	rex, imx = complex_dft(xr, xi, n)
	print "complet_dft(): n=", n
	print "rex: "
	print_list([int(e) for e in rex])
	print "imx: "
	print_list([int(e) for e in imx])

	fr = x[:]
	fi = x[:]

	fft_basic(fr, fi, n)
	print "fft_basic(): n=", n
	print "rex: "
	print_list([int(e) for e in fr])
	print "imx: "
	print_list([int(e) for e in fi])
