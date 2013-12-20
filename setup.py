#!/usr/bin/python
from setuptools import setup
import distutils.cmd
import sys, os, subprocess, traceback, re

def get_version(): return '0.0.4'

def main():
    setup(name             = 'varsens',
          version          = get_version(),
          description      = 'Python Variance Based Sensitivity Analysis',
          long_description = 'This package is to provide in Python a model independent method of doing Variance Based Sensitivity Analysis via the method of Saltelli.',
          author           = 'Shawn Garbett',
          author_email     = 'shawn@garbett.org',
          url              = 'http://github.com/LoLab-VU/varsens',
          packages         = ['varsens'],
          cmdclass         = {'test': test},
          keywords         = ['sensitivity','mathematics','engineering'],
          classifiers      = [  'Development Status :: 2 - Pre-Alpha',
                                'Environment :: Console',
                                'Intended Audience :: Science/Research',
                                'License :: CC Attribution-NonCommercial 3.0 Unported',
                                'Operating System :: OS Independent',
                                'Programming Language :: Python :: 2',
                                'Topic :: Scientific/Engineering :: Mathematics',
                                'Topic :: Scientific/Engineering :: Statistics'
                             ]
          )

class test(distutils.cmd.Command):
    description = "run tests (requires nose)"
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
          import nose
          from nose.plugins.manager import DefaultPluginManager
          excludes = [r'^examples$', r'^deprecated$']
          config = nose.config.Config(exclude=map(re.compile, excludes),
                                      plugins=DefaultPluginManager(),
                                      env=os.environ)
          nose.run(defaultTest='varsens', config=config, argv=['', '--with-doctest'])

if __name__ == '__main__':
    main()
