# This is the file for backend-independent (only matplotlib) part of the GUI
# Isn't working yet
#
# I think I should redesign it. It should work from normal pylab
# session. So, it's just a stub for future work.

from pylab import * # TODO: import only needed stuff
from swan.utils import *

from matplotlib.figure import Figure

from swan import pycwt

class SwanGUI:
    def __init__(self,fname=None,
                 Fs = 2.0,
                 f0 = 1.5,
                 labelsize = 10.0,
                 verbose = 1):
        
        self.verbose = verbose
        self.deoutbf = True # wether to remove outbursts

        self.data = None
        self.Fs = Fs # sampling frequency

        self.rc = {'label-size': labelsize,
                   'wavelet' : 'Morlet',
                   }

        self.fig = Figure(figsize=(8,6), dpi=100)

        self.f_pars = {'start': 0.1, 'step': 0.025, 'stop': 10.,
                       'log?': False} #frequency parameters #self.set_freqs =

        self.skeletons = []
        self.skel_plots = []
        
        self.rind = -1

        self.cwt_freshly_runf = False
        self.ehndl = None # mouse click event handler
        self.cwt = None
        self.show_how = self.show_surface
        self.wavelet = pycwt.Morlet(f0)
        self.cwt_image = None

        # -- Prepare different parts --
        set_rgbswan_cm()
        self.setup_show_funcs()
        self.setup_axes()

        # -- Try to load data if the file name is provided --
        if defined(fname):
            self.load_data(fname)
            self.plot_data()

        self.window.show_all()

    def setup_show_funcs(self):
        """ Prepare functions for different show options """
        self.show_funcs = {
            'Modulus': lambda x: abs(x),
            'Phase': lambda x: arctan2(x.imag, x.real),
            'Real' : lambda x: x.real,
            'Imaginary': lambda x: x.imag,
            'EDS': self.eds,
            #'EDS': lambda x: (x.real**2. + x.imag**2.)/self.wavelet.fc,
            'EDS-A-1': lambda x: self.eds_a_helper(x,1),
            'EDS-A-2': lambda x: self.eds_a_helper(x,2)
                }
        self.image_logscale = False
        self.show_func = self.show_funcs['EDS'] # default view

    def eds(self, x):
        return pycwt.eds(x)/std(self.data)**2

    def eds_a_helper(self, x, alpha):
        """ Show EDS with ridge amplitude correction """
        #alpha = 1.9798196483 # magic alpha
        coefs = (self.freqs**alpha) * 2.0/ pi
        eds = self.eds(x)
        if len(eds.shape) >1: 
            for i in range(eds.shape[1]):
                eds[:,i] *= coefs
        else:
            eds *= coefs
        #return eds/self.wavelet.fc
        return eds
    
    def load_data(self,fname):
        """ Loads the data """
        try:
            data = numpy.loadtxt(fname)
            s = data.shape
            self.fname = fname
            
            if len(s) < 2:
                if len(data)%2 :
                    data = data[:-1]
                    #self.data = data - mean(data)
                if self.deoutbf:
                    data = deoutburst(data)
                self.data = detrend1(data)

                self.time_from_Fs(self.Fs)
            else:
                if s[0] < s[1]:
                    data = transpose(data)
                    s = data.shape

                time = data[:,0] / 1000.0 # Here I assume that time is in ms
                data = data[:,1]
                
                if len(data)%2 :
                    data = data[:-1]
                    time = time[:-1]
                
                self.data = data
                self.time = time
                self.Fs = 1./mean(time[1:]- time[:-1])
        except:
            print("Error in data loading:")
            print(sys.exc_info()[0], sys.exc_info()[1])
            self.fname = None
            self.data = None

    def setup_axes(self):
        """Prepares axes for further work"""
        rc('xtick', labelsize = self.rc['label-size'])
        rc('ytick', labelsize = self.rc['label-size'])
        self.fig.clear()
        self.ax = [self.fig.add_axes((0.04, 0.2, 0.8, 0.78))]
        self.ax.append(self.fig.add_axes((0.04, 0.04, 0.8, 0.12),
                                         sharex=self.ax[0]))
        self.ax.append(self.fig.add_axes((0.85, 0.2, 0.04, 0.78), 
                                         xticklabels=[], 
                                         yticklabels=[],
                                         visible = False))
        self.ax[0].hold(True)
        self.ax[1].hold(False)


    def image_logscale_toggle(self,w):
        self.image_logscale = w.get_active();
        self.change_display(self['display_combo'])

    @with_data
    def on_apply_cwt_clicked(self, w):
        self.freqs = self.set_freqs()
        self.cwt_calc()
        self.cwt_display()


    def set_freqs(self):
        if self.f_pars['log?']:
            a = log2(self.f_pars['start'])
            b = log2(self.f_pars['stop'])
            if self.verbose:
                print(a,b, 1/self.f_pars['step'])
            return 2.**arange(a, b, self.f_pars['step'])
        else:
            return arange(self.f_pars['start'],
                          self.f_pars['stop'],
                          self.f_pars['step'])

        return 2.**arange(a, b, self.f_pars['step'])

    @with_data
    def cwt_calc(self):
        self.cwt = pycwt.cwt_f(self.data, self.freqs, Fs=self.Fs, \
                wavelet = self.wavelet)
        self.cwt_freshly_runf = True
        pass

    @with_cwt
    def cwt_display(self):
        """Show the results of the wavelet transform"""
        
        self.clear_all_rhythms()
        self.ax[0].clear()
        self.rm_skeletons()
        self['local_maxima_chk'].set_active(False)

        self.show_how()
        self.show_cone()
        setp(self.ax[0],'yscale',('linear','log')[self.f_pars['log?']])
        self.show_colorbar()

    @with_cwt
    def show_cone(self):
        w = self.wavelet
        f = self.freqs
        ax = self.ax[0]
        t = self.time[-1]
        try:
            ax.fill_betweenx(f, 0.0,
                             w.cone(f),
                             alpha=0.5, color='black')
            ax.fill_betweenx(f,
                             t+w.cone(-f),
                             t,
                             alpha=0.5, color='black')
        except:
            print("Can't use fill_betweenx function: update\
            maptlotlib?")


    
    
    @with_cwt
    def show_surface(self):
        ax = self.ax[0]
        self.cwt_image = ax.imshow(self.show_func(self.cwt),
                                   origin='lower',
                                   extent=(self.time[0], self.time[-1],
                                           self.freqs[0], self.freqs[-1]),
                                   interpolation='bilinear',
                                   cmap=self.curr_cmap)



    @with_cwt
    def show_contour(self):
        ax = self.ax[0]
        self.cwt_image = ax.contour(self.show_func(self.cwt),
                                    origin='lower',
                                    extent=(self.time[0], self.time[-1],
                                            self.freqs[0], self.freqs[-1]),
                                    cmap=self.curr_cmap)
    @with_cwt
    def show_colorbar(self):
        self.fig.colorbar(self.cwt_image, cax=self.ax[2])
        setp(self.ax[2], visible=True)

    @with_data
    def time_from_Fs(self,Fs=2.0):
        tfinish = len(self.data)/Fs
        self.time = arange(0,1.2*tfinish,1./Fs)[:len(self.data)]


    @with_data
    def plot_data(self):
        """Plot current data"""
        if self.time is None:
            self.ax[1].plot(self.data,'k-')
        else:
            self.ax[1].plot(self.time,self.data,'k-')
            setp(self.ax[1],'xlim',(self.time[0], self.time[-1]))
        self.canvas.draw()
        return 0

    def renew_skelsp(self):
        return defined(self.data) and \
                (len(self.skeletons) is 0) and\
                self.cwt_freshly_runf

    
    def plot_a_skeleton(self,skel):
        return self.ax[0].plot(skel[:,0], skel[:,1], 'y.', markersize = 0.1)

    def set_ax0_tight(self):
        setp(self.ax[0], 'xlim', (0, self.time[-1]),\
                'ylim', (self.freqs.min(), self.freqs.max()))
    
    def skeletons_shade(self, flag=True):
        for plh in self.skel_plots:
            setp(plh,'visible', not flag)
        self.canvas.draw()

    def calc_skeletons(self):
        """calculate local modulus maxima (skeleton lines)"""
        L =  len(self.data)
        del(self.skeletons[:])
        for i in range(L):
            xe,ye = get_extr_deriv(self.freqs, self.show_func(self.cwt[:,i]))
            ye[:] = self.time[i]
            self.skeletons.append(transpose(array([ye,xe])))
            if not i%10:
                sys.stderr.write (make_progress_str(i,L))
    
#------------------- End of the SwanGUI class ------------

