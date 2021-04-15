# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:48:32 2020

@author: cpannetier
"""
import os

import numpy as np

from coh_lib.coh_tools import generate_spectra
from coh_lib.skeleton import MakeAtmosphereCoherence

# Spectral Sampling
lmbda1 = 1.2 ; lmbda2 = 1.8 ; MW = 5 ; OW = 10
spectra, sigma = generate_spectra(lmbda1, lmbda2, OW=OW, MW=MW, mode='linear_sig')

# Temporal sampling
dt = 1                  # Time of a frame [ms]
NT = 30000

# Disturbance power
# According to Colavita et al and Buscher et al: sig² = 6.88*(B/r0)^{5/3} [rad]
r0 = 0.15 #[m]
t0 = 10 #ms
V = 0.31*r0/t0*1e3
L0 = 25
ampl_from_r0 = np.sqrt(6.88*(L0/r0)**(5/3))*0.55/(2*np.pi)
print(ampl_from_r0)
ampl = ampl_from_r0

tel=0
#%%

datadir = 'C:/Users/cpannetier/Documents/Python_packages/coh_lib/coh_lib/data/'
# Intereferometer
InterferometerFile = datadir+'interferometers/CHARAinterferometerH.fits'

# Saving file
DisturbanceFile = datadir+f'disturbances/random_{tel}_r{int(r0*100)}_tau{t0}_L{L0}_{NT*dt}ms.fits'

# Create disturbance
coher = MakeAtmosphereCoherence(DisturbanceFile, InterferometerFile,
                                overwrite=True,
                                NT=NT,spectra=spectra,
                                dist='random', tel=tel,startframe=50,
                                r0=r0, t0=t0,L0=L0,highFC=True)
