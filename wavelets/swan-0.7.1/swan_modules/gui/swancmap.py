# Colormaps for Swan

#from numpy import *
from pylab import *
import matplotlib as mpl


def hsv2rgb(ctuple):
    """Convert hsv tuple to rgb tuple
    based on wikipedia page"""
    h,s,v = ctuple
    h = (h*360.0)%360. # hue is an angle

    hi = floor(h/60.)

    f = h/60. - hi
    if not hi%2 : f= 1-f
    p = v*(1 - s)
    q = v*(1 - s*f)
    
    if (hi == 0): r,g,b = v,q,p
    elif (hi == 1): r,g,b = q,v,p
    elif (hi == 2): r,g,b = p,v,q
    elif (hi == 3): r,g,b = p,q,v
    elif (hi == 4): r,g,b = q,p,v
    elif (hi == 5): r,g,b = v,p,q
    
    return r,g,b


def get_rgbswan_data2():
    sat,val = 0.99,0.99
    colors = {'red':0,'green':1,'blue':2}
    step = 0.05
    nsteps = 256

    #hues = linspace(0.2,1, nsteps)
    hues = [0.7, 0.6, 0.5, 0.45, 0.3, 0.2, 0.15, 0.1, 0.0]
    hue_ind = linspace(0, 1, len(hues))

    rgb_vect = [ (i, hsv2rgb((hue, sat, val))) 
                 for i,hue in zip(hue_ind, hues)]
    res = {}

    for color,ind in list(colors.items()):
        res[color] = [(x[0], x[1][ind], x[1][ind]) for x in rgb_vect]    
    return res

def get_rgswan_data():
    colors = {'red':0,'green':1,'blue':2}
    nhues = 25
    hue_ind = linspace(0.0, 1.0, 2*nhues)
    reds,greens = np.ones(2*nhues), np.ones(2*nhues)
    reds[:nhues] = linspace(0,1.0,nhues)
    greens[nhues:] = linspace(0,1.0,nhues)[::-1]
    rgb_vect = [(i, (r, g, 0.0)) for i,r,g in zip(hue_ind,reds,greens)]
    res = {}
    for color,ind in list(colors.items()):
        res[color] = [(x[0], x[1][ind], x[1][ind]) for x in rgb_vect]
    return res

def set_rgbswan_cm():
    LUTSIZE = mpl.rcParams['image.lut']
    
    _rgbswan_data =  get_rgbswan_data2() 

    cm.datad['rgbswan'] = _rgbswan_data
    cm.rgbswan  = mpl.colors.LinearSegmentedColormap('rgbswan',
                                                            _rgbswan_data, LUTSIZE)
    return cm.rgbswan

def _make_swanrg():
    LUTSIZE = mpl.rcParams['image.lut']
    _rgswan_data =  get_rgswan_data()
    cmap = mpl.colors.LinearSegmentedColormap('rgswan',
                                              _rgswan_data, LUTSIZE)
    return cmap

def _make_swanrgb():
    LUTSIZE = mpl.rcParams['image.lut']
    _rgbswan_data =  get_rgbswan_data2()
    cmap = mpl.colors.LinearSegmentedColormap('rgbswan',
                                              _rgbswan_data, LUTSIZE)
    return cmap

swanrgb = _make_swanrgb()
swansummer =  _make_swanrg()


def _swanrgb():
    '''
    set the default colormap to rgbswan and apply to current image if any.
    See help(colormaps) for more information
    '''
    rc('image', cmap='rgbswan')
    im = gci()

    if im is not None:
        im.set_cmap(cm.rgbswan)
    draw_if_interactive()
