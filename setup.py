# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(name='coh_lib',
      version='0.1',
      description='Fringe tracking simulator',
      url='http://github.com/cpannet/coh_pack',
      author='Cyril Pannetier',
      author_email='cyril@cpannetier.fr',
      license='',
      install_requires=['matplotlib','numpy','scipy','astropy'],
      packages=find_packages(),
      zip_safe=False)