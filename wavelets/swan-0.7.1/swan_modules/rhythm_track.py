import numpy as np
from scipy.interpolate import splrep,splev

def in_range(v, low, high):
    if low > high:
        low,high = high,low
    return (v > low) * (v <= high)
    

def locextr(x, mode = 1, **kwargs):
   "Finds local maxima when mode = 1, local minima when mode = -1"
   tck = splrep(list(range(len(x))),x, **kwargs)
   res = 0.05
   xfit = np.arange(0,len(x), res)
   dersign = mode*np.sign(splev(xfit, tck, der=1))
   return xfit[dersign[:-1] - dersign[1:] > 1.5]

def all_vert_maxima(arr):
    return list(map(locextr, arr))

def limit_bounds(vec, lbound, ubound):
   out = np.copy(vec)
   if lbound:
      out = out[out >= lbound]
   if ubound:
        out = out[out <= ubound]
   return out
         

def locmax_pos(m):
    mfreqs,mtimes = np.where(np.diff(np.sign(np.diff(m, axis=0)), axis=0) < 0)
    return mtimes, mfreqs+2



def trace(seq, seed, memlen = 2, memweight = 0.75,
          lbound = None, ubound = None):
   out = []
   seeds = seed*np.ones(memlen)
   for k,el in enumerate(seq):
      el = limit_bounds(el, lbound, ubound)
      if len(el) > 0:
         j = np.argmin(abs(el - np.mean(seeds)))
         seeds[:-1] = seeds[1:]
         seeds[-1] = el[j]
         out.append((k,el[j]))
   return np.array(out)

def ind_to_val(ind, vrange, nsteps, dv=None):
    if dv is None: dv = (vrange[1] - vrange[0])/float(nsteps)
    return vrange[0] + ind*dv

def val_to_ind(val, vrange, nsteps, dv=None):
    if dv is None: dv = (vrange[1] - vrange[0])/float(nsteps)
    return (val-vrange[0])/dv


def trace_rhythm(arr, seed, extent, start = None, stop=None,rhsd = 0.1,
                 memlen=8):
    """
    traces horizontal ridge in array and returns time, frequency and energy
    vectors
    """
    if start is None:
        start = extent[0]
    if stop is None:
        stop = extent[1]
    
    trange, frange = extent[:2], extent[2:]
    nfreqs, ntimes = arr.shape
    df = (frange[1] - frange[0])/nfreqs
    rhsd_i = rhsd/df


    start_i = val_to_ind(start, trange, ntimes)
    stop_i = val_to_ind(stop, trange, ntimes)
    seed_i = val_to_ind(seed, frange, nfreqs)

    xtimes, xfreqs = locmax_pos(arr[:,start_i:stop_i])
    xu = np.unique(xtimes)

    #arr_tr = map(locextr, arr[:,start_i:stop_i].T)
    arr_tr = (xfreqs[xtimes==i] for i in xu)
    
    rhythm = trace(arr_tr, seed_i,
                   memlen=memlen,
                   lbound = seed_i - 2*rhsd_i,
                   ubound = seed_i + 2*rhsd_i)
    rhythm[:,0] += start_i

    tck = splrep(rhythm[:,0], rhythm[:,1])
    rhfreqs = np.asarray(list(map(int,
                             np.round(splev(np.arange(start_i, ntimes),
                                            tck)))))

    fvec = ind_to_val(rhythm[:,1], frange, nfreqs)
    tvec = ind_to_val(rhythm[:,0], trange, ntimes)
    evec = np.asarray([np.sum(arr[rhfreqs[i]-rhsd:rhfreqs[i]+rhsd,i+start_i])
                       for i in np.arange(len(tvec))])
    return tvec, fvec, evec
                  












