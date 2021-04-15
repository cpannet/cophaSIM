# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 21:34:43 2020

@author: cpannetier

This routine simulates:
    - tracking in H band
    - star: Unresolved
    - Calculator: INTEGRATOR
    - Fringe sensor: SPICA-type PERFECT
    - Piston disturbance: Step
    - With or without noise

"""

# Define the workspace as being the coh_pack2 main workspace
import os

datadir = 'C:/Users/cpannetier/Documents/These/FringeTracking/Python/coh_pack/coh_lib/data/'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import coh_lib.skeleton as cl
import coh_lib.coh_tools as ct
from coh_lib import config

plt.close('all')
config.newfig = 0

"""
TRUE FRINGE SENSOR so we need to initialize it before generating star in order to
know the micro and macro spectras.
"""
directory = 'C:/Users/cpannetier/Documents/These/FringeTracking/SPICA-FT/V2PM_SPICA/'
V2PMfilename = 'MIRCX_ABCD_H_PRISM22_V2PM.fits'
FSfitsfile = directory+V2PMfilename

# Initialize the fringe sensor with the gain
# from coh_lib.SPICA_FS import SPICAFS_TRUE
# spectra, spectraM = SPICAFS_TRUE(init=True,fitsfile=FSfitsfile,OS=1)


lmbda1 = 1.4
lmbda2 = 1.7
MW=5
OW=5
# spectra = np.arange(lmbda1, lmbda2, )
spectra, spectraM = ct.generate_spectra(lmbda1, lmbda2, OW=10, MW=5, mode='linear_sig')

from coh_lib.SPICA_FS import SPICAFS_PERFECT
SPICAFS_PERFECT(init=True, spectra=spectra, spectraM=spectraM)

# Configuration parameters
NT = 3000               # Duration of the simulation [frames]
dt = 3                  # Time of a frame [ms]
latencytime = 2*dt     # Latency of the system
TELref = 1              # If 0, all telescope move. If ia!=0, Tel ia static
# Camera charcateristics
QE = np.ones(config.MW)*0.7    # Quantum efficiency [e/ph]
RON = 0.5              # ReadOUt Noise [e/e]

#%%

CHARAfile = datadir+'interferometers/CHARAinterferometerH.fits'

ObservationFile = datadir+'observations/CHARA/Unresolved_mag0.fits'

DisturbanceFile = datadir+'disturbances/random_0_r15_tau10_L25_30000ms.fits'

# Initialize global coh variable
cl.initialize(CHARAfile, ObservationFile, DisturbanceFile, NT=NT, spectra=spectra,\
                      fs='truespica', ft='ftspica',TELref=TELref, dt = dt,\
                          ron = RON, qe=QE,
                          starttracking=5,latencytime=latencytime)   
    
# Add noise
config.noise = False

Ngd = 1 ; GainPD = 0.3 ; GainGD = .7 ; roundGD = True ;Threshold=True #DEFAULT FRINGE-TRACKER parameters

"""
OTHER FRINGE-TRACKER parameters
"""
Ngd = 40
GainPD = .2
GainGD = .35
roundGD=True
Threshold=True


# from SPICA_FS import SPICAFS_PERFECT
# SPICAFS_PERFECT(init=True)

# Initialize the fringe tracker with the gain
cl.SimpleIntegrator(init=True, Ngd=Ngd, GainPD=GainPD, GainGD=GainGD,
                    roundGD=roundGD,Threshold=Threshold)


#%% Launch the simulator and save the data in the coh_file.pkl file

SavingFile = 'demo.fits'
cl.loop()


#%%  Display data
plt.close('all')
cl.display('opd', OPDdetails=True, wl=1.6)


#%% Display performance

WavelengthOfInterest = [0.6,0.7,0.8,1.65]
StartingTime = 20
DIT = 50
cl.ShowPerformance(StartingTime,WavelengthOfInterest,DIT)

#%%

# cl.SpectralAnalysis(OPD=(1,2))


#%%

