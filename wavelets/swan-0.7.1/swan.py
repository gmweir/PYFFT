#!/usr/bin/env python

""" Swan -- a Swan wavelet analysis tool """

import getopt
import sys


def help():
    usage = \
    """
    swan -- a tool for wavelet analysis of the data

    Usage:
    swan [options] [data file]

    Options could be:
    -h         print this message and exit
    -v         be more verbose
    -z <num>   zero frequency of the Morlet wavelet
    -s <num>   sampling frequency of the data
    -d <num>   downsample the signal to the given frequency

    Data file should be a column or row vector of values 
    in textual (ASCII) format.  (It can be gzipped too.)
    """
    print(usage)
    sys.exit(0)


if __name__ == '__main__':
    f0 = 1.5
    Fs = 2.0
    verbose = 0
    down_samp_f = None # a flag should be set up if we want to downsample the signal
    opts,args = getopt.getopt(sys.argv[1:],'hvz:s:d:')
    for o,v in opts:
        if o == '-h':
            help()
        elif o == '-v':
            verbose += 1
        elif o == '-z':           # zero frequency for morlet
            f0 = float(v)
        elif o == '-s':           # sampling rate in data file
            Fs = float(v)
        elif o == '-d':           #downsample
            down_samp_f = float(v)
    
    data_file = None
    if len(args) > 0:
        data_file = args[0]

    from  swan.gui import swan_gtk
    swgui = swan_gtk.Swan(data_file, Fs=Fs,
                          f0=f0, verbose=verbose)
    swgui.main()
