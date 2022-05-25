# This section is to improve python compataibilty between py27 and py3
# from __future__ import absolute_import

import sys
#from . import python
#sys.modules[__package__] = pycwt

import kPyWavelet as kpywave
import pycwt


importlib = __import__('importlib')
__swanlib__ = importlib.import_module('swan-0.7.1')

sys.modules['swan'] = __swanlib__

__import__("swan-0.7.1")