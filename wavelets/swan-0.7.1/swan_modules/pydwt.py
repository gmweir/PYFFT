# Discrete wavelet transform

import numpy as np
from functools import reduce

try:
    from scipy.ndimage import convolve1d
    _scipy_loaded = True
except :
    _scipy_loaded = False

def upsample(v):
    out = np.zeros(len(v)*2-1)
    out[::2] = v
    return out

def next_pow2(x):
    return 2.**np.ceil(np.log2(x))

def pad_func(ppd):
    func = None
    if ppd == 'zpd':
        func = lambda x,a:  x*0.0
    elif ppd == 'cpd':
        func = lambda x,a:  x*a
    return func
    
def evenp(n):
    return not n%2

def mirrorpd(k, L):
    if 0 <= k < L : return k
    else: return -(k)%L


# Redundant, modwt, a' trous
def dec_atrous_numpy(sig, lev, phi=np.array([1./16, 1./4, 3./8, 1./4, 1./16])):
    L = len(sig) 
    padlen = len(phi)
    assert L > padlen
    if _scipy_loaded:
        apprx = convolve1d(sig, phi, mode='mirror')
    else:
        indices = [mirrorpd(i, L) for i in list(range(-padlen, 0)) + list(range(0,L)) + list(range(L, L+padlen))]
        padded_sig = sig[indices]
        apprx = np.convolve(padded_sig, phi, mode='same')[padlen:padlen+L]
    w = (sig - apprx) # wavelet coefs
    if lev <= 0: return sig
    elif lev == 1 or L < len(upsample(phi)): return [w, apprx]
    else: return [w] + dec_atrous(apprx, lev-1, upsample(phi))


# version with direct convolution (slow)
def dec_atrous1(v, lev, phi=np.array([1./16, 1./4, 3./8, 1./4, 1./16])):
    coefs = []
    cprev = v.copy()
    cnext = np.zeros(v.shape)
    L,lphi = len(v), len(phi)
    phirange = np.arange(lphi) - int(lphi/2)
    Ll = range(L)
    for j in range(lev):
        phiind = (2**j)*phirange
        cvals = [np.sum(phi*cprev[(l+phiind)%L]) for l in  Ll]
        coefs.append(cprev - cvals)
        cprev = np.array(cvals)
    return coefs + [cprev]


### version with direct convolution (scales better with decomposition level)
import itertools as itt
def decompose1d_direct(v, level,
                       phi=np.array([1./16, 1./4, 3./8, 1./4, 1./16]),
                       dtype= 'float64'):
    cprev = v.copy()
    cnext = np.zeros(v.shape)
    L,lphi = len(v), len(phi)
    phirange = np.arange(lphi) - int(lphi/2)
    Ll = np.arange(L, dtype='int')
    coefs = np.ones((level+1, L),dtype=dtype)
    for j in range(level):
        phiind = (2**j)*phirange
        approx = reduce(np.add,
                        (p*cprev[(Ll+pi)%L] for p,pi in itt.izip(phi, phiind)))
        coefs[j] = cprev - approx
        cprev = approx
    coefs[j+1] = approx
    return coefs


from scipy import weave
def weave_conv_wholes(vout,v,phi,ind):
    code = """
    long L,i,k,ki;
    L = Nv[0];
    for(i=0; i<L; i++){
       VOUT1(i) = 0;
       for(k=0; k<Nphi[0]; k++){
          ki = i+IND1(k);
          if (ki < 0){ki = -ki%L;}
          else if (ki >= L){ki = L-2-ki%L;}
          VOUT1(i) += PHI1(k)*V1(ki);
       }
    }
    """
    weave.inline(code, ['vout','v','phi', 'ind'])


def dec_atrous1_weave(v, level,
                      phi=np.array([1./16, 1./4, 3./8, 1./4, 1./16]),
                      dtype= 'float64'):
    cprev = v.copy()
    cnext = np.zeros(v.shape)
    L,lphi = len(v), len(phi)
    phirange = np.arange(lphi) - int(lphi/2)
    coefs = np.ones((level+1, L),dtype=dtype)
    for j in range(level):
        phiind = (2**j)*phirange
        approx = np.zeros(v.shape)
        weave_conv_wholes(approx, cprev, phi, phiind)
        coefs[j] = cprev - approx
        cprev = approx
    coefs[j+1] = approx
    return coefs


dec_atrous = dec_atrous1_weave
