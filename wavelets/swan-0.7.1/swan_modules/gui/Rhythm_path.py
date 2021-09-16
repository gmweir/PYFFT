import gtk
from pylab import setp, arange, array

def best (scoref, lst):
    n,winner = 0, lst[0]
    for i, item in enumerate(lst):
        if  scoref(item, winner): n, winner = i, item
    return n,winner

def min1(scoref, lst):
    return best(lambda x,y: x < y, list(map(scoref, lst)))

def in_range(n, region):
    return (n > region[0]) and (n < region[1])

def nearest_item_ind(items, x, fn = lambda a: a):
    return min1(lambda p: abs(fn(p) - x), items)[0]

class banded_point:
    """2D point with a +- band attached to it"""
    def __init__(self, x, y, band):
        self.x = x
        self.y = y
        self.b = band
        

class Rhythm_path:
    """ Add rhythm path to the canvas """
    def __init__(self, canvas, i=0, band=0.5):
        self.x = []   # x of the skeleton line
        self.y = []   # y of the skeleton line
        self.v = []   # indices of the eds
        self.px = []  # xpoints in path
        self.py = []  # ypoints in path
        self.bpoints  = [] # banded points in path
        #---------------------------------
        self.band = band      # Frequency band (herz)
        self.plhndl = []     # Plot handle
        #---------------------------------
        self.canvas = canvas
        self.id = i
        self.butcb = [None,None,None] # Buttons callbacks
        self.box = gtk.HBox(False,0)
        self.b_adj = gtk.Adjustment(value=band,
                                    lower=0.0,
                                    upper=1000,
                                    step_incr=0.01,
                                    page_incr=0.04)
        band_spin = gtk.SpinButton(self.b_adj,climb_rate=0.2,digits=3)
        band_spin.set_numeric(True)
        band_spin.connect('value-changed',self.cb_band)
        self.buts = [gtk.ToggleButton('%d' %int(i)),
                     gtk.CheckButton('P'), gtk.CheckButton('R'),
                     band_spin]
        for b in self.buts:
            self.box.pack_start(b,False,False,0)

    def __del__(self):
        print('Destructor of RhythmPath')
        for b in self.buts:
            self.box.remove(b)
        self.clear()
        self.rclear()
        for i in range(4):
            setp(self.plhndl[i], 'data', (self.px, self.py), 'visible', False)
        self.canvas.draw()
        pass
    
    def push_point(self, x, y):
        self.bpoints.append(banded_point(x, y, self.band))
        if len(self.bpoints) > 1:
            self.bpoints.sort(lambda p1,p2: cmp(p1.x, p2.x))

    def pop_point(self, x, y):
        n = nearest_item_ind(self.bpoints, x, lambda p: p.x)
        self.bpoints.pop(n)

    def clear(self):
        self.bpoints = []
    
    def rclear(self):
        """ Rhythm clear """
        while len(self.x) > 0:
            self.x.pop()
            self.y.pop()
            self.v.pop()

    def plot_setup0(self,ax,L):
        self.plhndl.append(ax.plot(self.px,self.py,'or-'))
        self.plhndl.append(ax.plot(self.px,array(self.py)+self.band,'y-'))
        self.plhndl.append(ax.plot(self.px,array(self.py)-self.band,'y-'))
        self.plhndl.append(ax.plot(self.x, self.y,'-',color='w',linewidth=5.0, alpha=0.8))
        setp(ax,'xlim',L[0], 'ylim',L[1])

    def vectors_from_bpoints(self):
        px = array([p.x for p in self.bpoints])
        py = array([p.y for p in self.bpoints])
        pl = py - self.band
        pu = py + self.band
        return px,py,pu,pl

    def plot_setup(self,ax,L):
        
        px,py,pu,pl = self.vectors_from_bpoints()
        
        self.plhndl.append(ax.plot(px, py, 'or-'))
        self.plhndl.append(ax.plot(px, pu, 'y-'))
        self.plhndl.append(ax.plot(px, pl, 'y-'))
        self.plhndl.append(ax.plot(self.x, self.y,'-',color='w',linewidth=5.0, alpha=0.5))
        setp(ax,'xlim',L[0], 'ylim',L[1])

    def preplot(self,canvas,vis=True):
        
        px,py,pu,pl = self.vectors_from_bpoints()

        setp(self.plhndl[0],'data', (px, py),'visible',vis)
        setp(self.plhndl[1],'data', (px, pu),'visible',vis)
        setp(self.plhndl[2],'data', (px, pl),'visible',vis)
        canvas.draw()

    def cb_band(self,w,data=0):
        """Callback for changing the frequency band for rhythm selection"""
        self.band = w.get_value()
        self.preplot(self.canvas)

    def rreplot(self,canvas,vis=True):
        setp(self.plhndl[3],'data', (self.x,self.y),'visible',vis)
        canvas.draw()


