#! /usr/bin/python

import os
import numpy as np
from numpy import arange
from numpy import zeros
from numpy import absolute as abs
from numpy import square
from numpy import real 
from numpy import sqrt 
from numpy import exp 
from numpy import concatenate as cat
from numpy import conjugate as conj
from numpy.fft import fft
from numpy.fft import ifft
from math import pi

###############################################################
def lewis_ccor(navigt, templt, N, Q, M, P):

  cc  = zeros(P)  # normalized cross-correlation

  ns  = zeros(N+1)  # navigator sum
  ns2 = zeros(N+1)  # navigator sum of squares

  for i in range(N):
    a = navigt[i];
    ns[i+1] = a + ns[i];
    ns2[i+1] = a*a + ns2[i];

  q = Q-1

  template = templt[q:q+M]

  ts  = sum(template)         # template sum
  ts2 = sum(square(template)) # template sum of squares
  tm  = ts/M                  # template mean
  tv  = ts2 - square(ts)/M    # template variance

  v1 = template - tm;

  for i in range(P):
    k = i+M
    A = ns[k] - ns[i]
    C = ns2[k] - ns2[i]

    nm = A/M
    nv = C - A*A/M

    v2 = navigator[i:k] - nm
    numerator = sum(v1*v2)

    denominator = sqrt(tv*nv)
    cc[i] = numerator/denominator

  return cc

###############################################################
def template_functions(templt, kernel, N, Q, M, P):

  templt2 = square(abs(templt))

  tmp = ifft(fft(templt)*kernel)
  gc = tmp[range(P)]

  tmp = ifft(fft(templt2)*kernel)
  gg = real(tmp[range(P)])

  templt_padded = cat((templt[Q-1:Q+M-1],zeros(N-M)))
  FTpg = fft(templt_padded)/M

  return gc, gg, FTpg 

###############################################################
def complex_ccor(navigt, gc, gg, kernel, FTpg, N, Q, M, P):

  navigt2 = square(abs(navigt))

  tmp = ifft(fft(navigt)*kernel)
  fc = tmp[range(P)]

  tmp = ifft(fft(navigt2)*kernel)
  ff = real(tmp[range(P)])

  FTnv = fft(navigt)

  tmp = fft(conj(FTnv)*FTpg)/N
  fgc = tmp[range(P)]

  q = Q-1

  gcq = gc[q]
  ggq = gg[q]

  numerator = real(fgc - conj(fc)*gcq)

  denominator = sqrt((ff - square(abs(fc)))*
                     (ggq - square(abs(gcq))))

  return numerator/denominator

###################################################################
if __name__ == '__main__':

  tx1 = 80
  tx2 = 106

  n = 128
  q = tx1
  m = tx2-tx1+1
  p = n-m+1

  A = np.fromfile("navigators.dat",sep="\t").reshape(n,3)

  template = []
  navigator = []

  for i in range(n):
    template = template + [A[i][1]]
    navigator = navigator + [A[i][2]]

  k = arange(1,n)
  kernel = (1.0/m)*((exp(1j*2*pi*m*k/n) - 1)/(exp(1j*2*pi*k/n) - 1))
  kernel = cat(([1+1j*0.0], kernel))

  gc, gg, FTpg = \
    template_functions(template, kernel, n, q, m, p)

  cc = \
    complex_ccor(navigator, gc, gg, kernel, FTpg, n, q, m, p)

  lewis_cc = \
    lewis_ccor(navigator, template, n, q, m, p)

  for i in range(n-m+1):
    print("%3d % 16.14f % 16.14f %16.14f" % \
          (i+1, lewis_cc[i], cc[i], abs(lewis_cc[i]-cc[i])))
