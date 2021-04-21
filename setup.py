# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(name='cophasing',
      version='0.1',
      description='Fringe tracking simulator',
      url='https://gitlab.oca.eu/cpannetier/fringetracker.git',
      author='Cyril Pannetier',
      author_email='cyril@cpannetier.fr',
      license='',
      install_requires=['matplotlib','numpy','scipy','astropy','pandas','sympy'],
      packages=find_packages(),
      zip_safe=False)