#!/usr/bin/env python
"""
Swan --- a tool for wavelet analysis of data, 
inspired by WAND, a Wavelet Analyser of Data by D. Postnov

I wanted to write my own version in python/GTK instead of TCL/Tk
and try to advance the ideas behind.
"""
#Copyright 2005-2008 Alexey Brazhe
#brazhe@biophys.msu.ru
#
#-----------------------------------------------------
# This file is part of Swan package.
#
# Swan is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# Swan is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# the software; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307 USA 
#-----------------------------------------------------

# TODO: ...

try:
    import psyco
    psyco.full()
except:
    pass


import os, sys
import gtk
import gtk.glade
#from gzip import GzipFile as gzfile

import numpy

from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigCanv
except ImportError:
    print("No gtkagg module, loading gtk")
    from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigCanv

from matplotlib.backends.backend_gtk import NavigationToolbar2GTK as NToolbar

#from pylab import cm, rc, setp, load
from pylab import *


from swan import pycwt
from swan.gui import Rhythm_path

from swan.utils import *

def defined(obj):
    """ Unifies some tests """
    return (obj is not None)



def with_data(method):
    """Only do something when data have been loaded (decorator)"""
    def _(self,*args,**kwargs):
        if defined(self.data):
            method(self,*args,**kwargs)
        else:
            print("Have no data loaded, nothing to do...")
    return _

def with_cwt(method):
    """Only do something when cwt have been computed (decorator)"""
    def _(self, *args, **kwargs):
        if defined(self.cwt):
            method(self,*args,**kwargs)
        else:
            print("No cwt computed, nothing to do...")
    return _



from .swancmap import set_rgbswan_cm, swanrgb

def determine_path ():
    """Borrowed from wxglade.py"""
    try:
        root = __file__
        if os.path.islink (root):
            root = os.path.realpath (root)
        return os.path.dirname (os.path.abspath (root))
    except:
        print("I'm sorry, but something is wrong.")
        print("There is no __file__ variable. Please contact the author.")
        sys.exit ()


class Swan:
    def __init__(self,fname=None,
                 Fs = 2.0,
                 f0 = 1.5,
                 labelsize = 10.0,
                 verbose = 1):
        
        self.verbose = verbose
        self.deoutbf = True # wether to remove outbursts

        self.data = None
        self.Fs = Fs # sampling frequency

        self.rc = {'glade-file':'/glade/swan.glade',
                   'host-path': determine_path(),
                   'label-size': labelsize,
                   'wavelet' : 'Morlet',
                   }

        self.wTree = gtk.glade.XML(self.rc['host-path'] + self.rc['glade-file'])
        self.window = self["swan_main"]
 
        self.fig = Figure(figsize=(8,6), dpi=100)

        self['morlet_f0'].set_value(f0)
        self['Fs_spin'].set_value(Fs)

        self.f_pars = {'start': 0.1, 'step': 0.025, 'stop': 10., 'log?': False} #frequency parameters
        #self.set_freqs = self.set_freqs_linear

        self.skeletons = []
        self.skel_plots = []
        
        self.rind = -1
        self.rps = []   # Rhythm paths

        self.cwt_freshly_runf = False
        self.ehndl = None # mouse click event handler
        self.cwt = None
        self.show_how = self.show_surface
        self.wavelet = pycwt.Morlet(f0)
        self.cwt_image = None

        # -- Prepare different parts --
        set_rgbswan_cm()
        self.setup_callbacks()
        self.setup_show_funcs()
        self.setup_colormaps()
        self.setup_axes()
        self.setup_canvas()
        self.setup_combos()


        # -- Try to load data if the file name is provided --
        if defined(fname):
            self.load_data(fname)
            self.plot_data()

        self.window.show_all()

    def setup_callbacks(self):
        wdict = {"on_load_data_activate": self.on_load_data_activate,
                 "on_Fs_spin_value_changed": self.on_Fs_changed,
                 "on_apply_cwt_clicked": self.on_apply_cwt_clicked,
                 "on_morlet_f0_value_changed": self.on_f0_changed,
                 "on_add_rhythm_clicked": self.new_rhythm,
                 "on_rem_rhythm_clicked": self.del_rhythm,
                 "on_apply_rhythm_clicked": self.apply_path,
                 "on_save_rhythm_clicked": self.save_rhythm,
                 "on_colormaps_combo_changed" : self.change_cmap,
                 "on_display_combo_changed" : self.change_display,
                 "on_show_how_combo_changed": self.change_show_how,
                 "on_freq_log_scale_toggled": self.on_freq_log_scale_toggled,
                 "on_local_maxima_toggled" : self.on_local_maxima_toggled,
                 "on_logscale_image_toggled" : self.image_logscale_toggle,
                 "on_f_start_value_changed": lambda x : self.on_f_params_changed(x,'start'),
                 "on_f_step_value_changed": lambda x : self.on_f_params_changed(x,'step'),
                 "on_f_stop_value_changed": lambda x : self.on_f_params_changed(x,'stop'),
                 "on_swan_main_destroy":gtk.main_quit,
                 "on_quit1_activate":gtk.main_quit
                }
        self.wdict=wdict
        self.wTree.signal_autoconnect(wdict)

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
        #coefs = self.freqs**alpha * 2.0/ pi
        coefs = (self.freqs**alpha) * 2.0/ pi
        eds = self.eds(x)
        if len(eds.shape) >1: 
            for i in range(eds.shape[1]):
                eds[:,i] *= coefs
        else:
            eds *= coefs
        #return eds/self.wavelet.fc
        return eds


    def setup_colormaps(self):
        """ Sets the avalable colormpas up """
        self.curr_cmap = cm.jet
        self.known_cmaps = ('rgbswan', 'autumn', 'bone', 'binary', 'cool', 'copper', 'flag',\
                'gray', 'hot', 'hsv', 'jet', 'pink', \
                'prism', 'spectral', 'spring', 'summer', 'winter')
        for m in self.known_cmaps:
            self['colormaps_combo'].append_text(m)
        #self['colormaps_combo'].set_active(8)

    def setup_combos(self):
        """ Sets default values in combo boxes"""
        self['display_combo'].set_active(1)
        self['show_how_combo'].set_active(0)
        self['colormaps_combo'].set_active(0)
    
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
                self.data = detrend1(data)
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
                self['Fs_spin'].set_value(self.Fs)

        except:
            print("Error in data loading:")
            print(sys.exc_info()[0], sys.exc_info()[1])
            self.fname = None
            self.data = None

    def setup_canvas(self):
        """Prepares canvas to draw in"""
        self.canvas = FigCanv(self.fig)
        self.toolbar = NToolbar(self.canvas, None)
        self['pylab_box'].pack_start(self.canvas,True,True)
        self['pylab_box'].pack_start(self.toolbar, False, False)

    def setup_axes(self):
        """Prepares axes for further work"""
        rc('xtick', labelsize = self.rc['label-size'])
        rc('ytick', labelsize = self.rc['label-size'])
        self.fig.clear()
        self.ax = [self.fig.add_axes((0.04, 0.2, 0.8, 0.78))]
        self.ax.append(self.fig.add_axes((0.04, 0.04, 0.8, 0.12), sharex=self.ax[0]))
        self.ax.append(self.fig.add_axes((0.85, 0.2, 0.04, 0.78), 
                                         xticklabels=[], 
                                         yticklabels=[],
                                         visible = False))
        self.ax[0].hold(True)
        self.ax[1].hold(False)
        #self.ax[2].hold(False)

    def on_load_data_activate(self, widget):
        """ Callback for the Open menu item """
        fname = self.select_data_file()
        if fname is None:
            if self.verbose:
                print("We didn't choose any file ")
            return
        if self.verbose:
            print("We chose the file %s" % fname)
        self.load_data(fname)
        self.plot_data()


    def on_Fs_changed(self,widget):
        """ Callback to change the sampling frequency """
        self.Fs = widget.get_value()
        self.time_from_Fs(self.Fs)
        self.plot_data()
        return 0

    def on_f_params_changed(self, spinw, key):
        "Update  frequency parameters for cwt"
        if self.verbose >1:
            print('Changing key:', key)
        self.f_pars[key] = spinw.get_value()


    def on_freq_log_scale_toggled(self,w):
        self.f_pars['log?'] = w.get_active()


    def image_logscale_toggle(self,w):
        self.image_logscale = w.get_active();
        self.change_display(self['display_combo'])

    @with_data
    def on_apply_cwt_clicked(self, w):
        self.freqs = self.set_freqs()
        self.cwt_calc()
        self.cwt_display()


    def on_f0_changed(self,w):
        f0 = w.get_value()
        self.wavelet.set_f0(f0)
        self.show_funcs['EDS'] = lambda x: (x.real**2. + x.imag**2.)/self.wavelet.fc

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
        if self.ehndl is None:
            self.ehndl = self.canvas.mpl_connect('button_press_event',self.cont_path)
        self.show_colorbar()
        self.canvas.draw()

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
        self.cwt_image = self.ax[0].imshow(self.show_func(self.cwt),
                                           origin='lower',\
                                           extent=(self.time[0], self.time[-1], \
                                                   self.freqs[0], self.freqs[-1]),\
                                           interpolation='bilinear',\
                                           cmap=self.curr_cmap)



    @with_cwt
    def show_contour(self):
        self.cwt_image = self.ax[0].contour(self.show_func(self.cwt), \
                                                origin='lower',\
                                                extent=(self.time[0], self.time[-1], \
                                                            self.freqs[0], self.freqs[-1]),\
                                                cmap=self.curr_cmap)
    @with_cwt
    def show_colorbar(self):
        self.fig.colorbar(self.cwt_image, cax=self.ax[2])
        setp(self.ax[2], visible=True)

    def change_show_how(self, w):
        """ Callback for the 'show how' combo"""
        val = w.get_active_text()
        if val == 'Surface':
            self.show_how = self.show_surface
        else:
            self.show_how = self.show_contour
        self.cwt_display()
        pass


    def select_data_file(self, data=None):
        dlg = gtk.FileChooserDialog('Select data file',
                                    None,
                                    action=gtk.FILE_CHOOSER_ACTION_OPEN,
                                    buttons=(gtk.STOCK_CANCEL,
                                             gtk.RESPONSE_CANCEL,
                                             gtk.STOCK_OPEN,
                                             gtk.RESPONSE_OK))
        dlg.set_default_response(gtk.RESPONSE_OK)
        flt = gtk.FileFilter()
        flt.set_name('Data files')
        flt.add_pattern('*.*')
        dlg.add_filter(flt)
        resp = dlg.run()
        if resp ==gtk.RESPONSE_OK:
            fname = dlg.get_filename()
        else:
            fname = None
        dlg.destroy()
        return fname

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

    def on_local_maxima_toggled(self,w,data=None):
        """Show on/off the skeletons line"""
        #TODO: re-structure this!
        if w.get_active():
            if self.renew_skelsp():
                self.calc_skeletons()
                self.skel_plots = []
                for skel in self.skeletons:
                    self.skel_plots.append(self.plot_a_skeleton(skel))
                self.set_ax0_tight()
                self.canvas.draw()
            else:
                self.skeletons_shade(False)
        else:
            self.skeletons_shade(True)

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

    def rm_skeletons(self):
        self.skeletons_shade()
        del(self.skeletons[:])
        del(self.skel_plots[:])
        pass

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

    def change_cmap(self,w,udata=None):
        """Change a color map"""
        cmkey = w.get_active_text()
        self.curr_cmap = getattr(cm, cmkey)
        if defined(self.cwt_image):
            setp(self.cwt_image,cmap=self.curr_cmap)
        self.canvas.draw()

    
    def change_display(self,w,udata = None):
        """Change display mode"""
        display_key = w.get_active_text()
        if display_key is None:
            display_key = 'Modulus'
        show_func = self.show_funcs[display_key]
        if self.image_logscale and ((display_key == 'Modulus') or 
                                    (display_key == 'EDS') or
                                    (display_key == 'EDS-A-1') or 
                                    (display_key == 'EDS-A-2')):
            self.show_func = lambda x: log(show_func(x))
        else:
            self.show_func = show_func
        self.cwt_display()


    def toggle_path(self,w,id=0,but=1):
        """Show on/off for the selected path or rhythm"""
        if self.verbose > 1:
            print('Caller: %d, %s' % (id, ('P', 'R')[but-1]))
        if but == 1:
            self.rps[id].preplot(self.canvas,w.get_active())
        elif but ==2:
            self.rps[id].rreplot(self.canvas,w.get_active())
        
    def  toggle_rhythm(self,w,data=0):
        """Select on rhythm-path and deselect others"""
        if self.verbose > 1:
            print('Caller: %d' % data)
        for rp in self.rps:
            if rp.id == data:
                self.rind = (-1,data)[w.get_active()]
                if self.verbose >2 :
                    print('self.rind ', self.rind)
            else:
                rp.buts[0].handler_block(rp.butcb[0])      # fantastiske :)
                rp.buts[0].set_active(False)
                rp.buts[0].handler_unblock(rp.butcb[0])    # fantastiske :)


    def clear_all_rhythms(self):
        while len(self.rps) > 0:
            self.del_rhythm()

    def del_rhythm(self,w=None,data=None):
        x = self.rps.pop()
        x.__del__()
        if self.rind > -1:
            self.rind -= 1
        self.window.show_all()

    def new_rhythm(self,w, data=None):
        """Starts new rhythm path"""
        i = len(self.rps)
        nrp = Rhythm_path.Rhythm_path(self.canvas, i, 0.04)
        nrp.butcb[0] = nrp.buts[0].connect('toggled',self.toggle_rhythm,i)
        nrp.buts[0].set_active(True)
        nrp.buts[1].set_active(True)
        nrp.butcb[1] = nrp.buts[1].connect('toggled',self.toggle_path,i,1)
        nrp.butcb[2] = nrp.buts[2].connect('toggled',self.toggle_path,i,2)
        #self.boxes['ctrls/rh'].pack_start(nrp.box,False,False,0)
        self['rhythms_store'].pack_start(nrp.box,False,False,0)
        nrp.plot_setup(self.ax[0],[(0,self.time[-1]), (self.freqs[0], self.freqs[-1])])
        self.rps.append(nrp)
        self.rind = i
        self.window.show_all()

    def cont_path(self,event):
        """Continues the rhythm path by adding points or removing them"""
        i = self.rind
        if i > -1:
            if event.inaxes == self.ax[0]:
                crps = self.rps[i]
                if event.button == 1:                      # left click
                    crps.push_point(event.xdata,event.ydata)
                elif event.button == 3:                    # right click
                    crps.pop_point(event.xdata,event.ydata)
                elif event.button == 2:                    # middle click
                    crps.clear()
                crps.preplot(self.canvas)

    def apply_path(self,w,udata=None):
        i = self.rind
        if i > -1:
            self.path_2_rhythm0()
            crps = self.rps[i]
            if crps.buts[2].get_active():
                crps.rreplot(self.canvas)
            else:
                crps.buts[2].set_active(True)

    def path_2_rhythm0(self):

        i = self.rind                   # index of the current rhythm path
        crps = self.rps[i]              # current rhythm path
        
        x,y,v = [],[],[]
        px,py,pu,pl = crps.vectors_from_bpoints()
        
        L = self.get_xind(px[-1]) - self.get_xind(px[0]) + 1
        
        for k in range(len(px) - 1):
            p = polyfit(px[k:k+2], py[k:k+2], 1)  # Linear approximation between two path points
            sy = arange(self.cwt.shape[0])            # eds surface y-indices
            jstart = self.get_xind(px[k])
            jstop = self.get_xind(px[k+1])
            self.ax[0].hold(True)
            for j in range(jstart,jstop):
                yslice = self.show_func(self.cwt[:,j])
                ym = p[0]*self.time[j] + p[1]
                yl = min(sy[self.freqs > ym-crps.band]) 
                yu = max(sy[self.freqs <= (ym + crps.band)])+1
                tmp = argsort(yslice[yl:yu])[-1]
                x.append(self.time[j])
                v.append(yslice[yl+tmp])
                y.append(self.freqs[yl+tmp])
                if self.verbose:
                    if not j % 10:
                        sys.stderr.write (make_progress_str(j,L))
        crps.x = x
        crps.y = y
        crps.v = v
        if self.verbose:
            print('\nDone with path to rhythm conversion')
    


    def save_rhythm(self,w,udata=None):
        i = self.rind
        if len(self.rps[i].x) > 0:
            ya = zeros((len(self.rps[i].x)),dtype='d')
            fname = self.select_rhythm()
            if defined(fname):
                out_f = fname + '_f.dat'  
                out_a = fname + '_a.dat'  
                save(out_f,self.rps[i].y)
                save(out_a,self.rps[i].v)
                print(fname)
            

    def select_rhythm(self, data=None):
        dlg = gtk.FileChooserDialog('Where should we save our rhythm?',
                None,
                action=gtk.FILE_CHOOSER_ACTION_SAVE,
                buttons=(gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL,gtk.STOCK_SAVE,gtk.RESPONSE_OK))
        dlg.set_default_response(gtk.RESPONSE_OK)
        flt = gtk.FileFilter()
        flt.set_name('All files')
        flt.add_pattern('*')
        dlg.add_filter(flt)
        resp = dlg.run()
        if resp ==gtk.RESPONSE_OK:
            fname = dlg.get_filename()
        else:
            fname = None
        dlg.destroy()
        return fname


    def get_xind(self,xp):
        return int(around(xp*self.Fs))

                
    def main(self):
        gtk.main()
    
    def __getitem__(self,key):
        return self.wTree.get_widget(key)
#------------------- End of the Swan class ------------

