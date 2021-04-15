# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:16:53 2020

@author: cpannetier

Create the FITSfile of an interferometer array.

FITSFILE parameters:
    - NA: Number of telescopes                              - [INT]
    - NIN: Number of baselines                              - [INT]
    - TelNames: Telescopes names                              - LIST[NA]
    - TelCoordinates: Coordinates (X,Y,Z) of the telescopes    - ARRAY[NA,3]
    - BaseNames: Baselines names                               - LIST[NA]
    - BaseCoordinates: Coordinates of the baseline vectors  - ARRAY[NIN,3]
    
"""

# Define the workspace as being the coh_pack main workspace
import os
# os.chdir('C:/Users/cpannetier/Documents/These/FringeTracking/Python/coh_pack/coh_lib')

import matplotlib.pyplot as plt
import numpy as np
plt.close('all')

import coh_lib.coh_tools as ct

ArrayName = 'CHARA'

#official coordinates in [µm]
coords = np.array([[125333989.819,305932632.737,-5909735.735],\
                         [70396607.118,269713282.258,-2796743.645],\
                             [0,0,0],\
                                 [-5746854.437,33580641.636,636719.086],\
                                     [-175073332.211,216320434.499,-10791111.235],\
                                         [-69093582.796,199334733.235,467336.023]])
TelCoordinates=coords*1e-6      # [m]

TelNames = ['E1','E2','S1','S2','W1','W2']
NA = len(TelNames)

band = 'R'
if band == 'H':     
    T_tel = 0.1
    T_inj = 0.65
    T_strehl = 0.8
    T_BS = 1            # No beam splitter in H ? OA?
if band == 'R':
    T_tel = 0.03
    T_inj = 0.5
    T_strehl = 0.8
    T_BS = 0.88         # Transmission beam-splitter before injection

# Same transmissions on all telescopes
TelTransmissions = T_tel*T_inj*T_strehl*T_BS*np.ones(NA)

# Diameter of a telescope and collecting surface [meter²]
surface = 0.74 #MAM thesis - diameter=1m and occultation=0.25m
TelSurfaces = surface*np.ones(NA)


NIN = NA*(NA-1)//2
BaseNames = []
BaseCoordinates = np.zeros([NIN,3])
for ia in range(NA):
    for iap in range(ia+1, NA):
        k = ct.posk(ia,iap,NA)
        BaseNames.append(TelNames[ia]+TelNames[iap])
        BaseCoordinates[k] = TelCoordinates[iap] - TelCoordinates[ia]
           

#%%
from astropy.io import fits
datadir = 'C:/Users/cpannetier/Documents/These/FringeTracking/Python/coh_pack/coh_lib/data/'

filename = 'CHARAinterferometer.fits'
filepath = datadir+'interferometers/'+filename

OVERWRITE = True

fileexists = os.path.exists(filepath)

if fileexists:
    if OVERWRITE:
        os.remove(filepath)
    else:
        raise Exception(f'{filepath} already exists.')


hdr = fits.Header()
hdr['NAME'] = ArrayName
hdr['NA'] = NA
hdr['NIN'] = NIN

primary = fits.PrimaryHDU(header=hdr)

col1 = fits.Column(name='TelNames', format='2A', array=TelNames)
col2 = fits.Column(name='TelCoordinates', format='3D', array=TelCoordinates)
col3 = fits.Column(name='TelTransmissions', format='1E', array=TelTransmissions)
col4 = fits.Column(name='TelSurfaces', format='1E', array=TelSurfaces)

col5 = fits.Column(name='BaseNames', format='4A', array=BaseNames)
col6 = fits.Column(name='BaseCoordinates', format='3D', array=BaseCoordinates)



print(f'Saving file into {filepath}')
# hdu.writeto(filepath)



