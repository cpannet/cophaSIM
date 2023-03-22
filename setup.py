# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(name='cophasim',
      version='0.1',
      description='Fringe tracking simulator',
      url='https://gitlab.oca.eu/cpannetier/cophaSIM.git',
      author='Cyril Pannetier',
      author_email='contact@cpannetier.fr',
      license='',
      install_requires=['matplotlib','numpy','scipy','astropy','pandas','sympy',
                        'tabulate','bode_utils','joblib','psutil'],
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True,
      package_data = {'cophasim':['data/interferometers/*.fits',
                                   'data/observations/CHARA/Unresolved/*.fits',
                                   'data/observations/CHARA/1mas/*.fits',
                                   'data/observations/CHARA/2mas/*.fits']})