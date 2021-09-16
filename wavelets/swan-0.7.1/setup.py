#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup

classifiers=[
      'Development Status :: 4 - Beta',
      'Environment :: X11 Applications :: GTK',
      'Intended Audience :: End Users/Desktop',
      'License :: OSI Approved :: GNU General Public License (GPL)',
      'Operating System :: POSIX :: Linux',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering',
        ]

setup(name='swan',
      version = "0.7.1",
      description = 'Easy continuous wavelet analysis',
      author = 'Alexey Brazhe',
      author_email = "brazhe@gmail.com",
      license = 'GNU GPL',
      url = "http://cell.biophys.msu.ru/static/swan",
      packages = ['swan', 'swan.gui'],
      package_dir = {'swan':'swan_modules'},
      package_data = {'swan.gui': ["glade/swan.glade"]},
      #scripts = ['swan'],
      long_description = \
      """swan is a tool for wavelet data analysis.
      It's meant to be simple in use and easy to extend.""",
      classifiers=classifiers,
      )
