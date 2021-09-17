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

datadir = 'C:/Users/cpannetier/Documents/Python_packages/cophasing/cophasing/data/'

import numpy as np
import matplotlib.pyplot as plt
import cophasing.skeleton as sk
import cophasing.coh_tools as ct
from cophasing import config

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
from cophasing.SPICA_FS import SPICAFS_TRUE
OW=10
spectra, spectraM = SPICAFS_TRUE(init=True,fitsfile=FSfitsfile,OW=OW)

# lmbda1 = 1.4
# lmbda2 = 1.7
# MW=5
# OW=20
# # spectra = np.arange(lmbda1, lmbda2, )
# spectra, spectraM = ct.generate_spectra(lmbda1, lmbda2, OW=OW, MW=MW, mode='linear_sig')

# from cophasing.SPICA_FS import SPICAFS_PERFECT
# SPICAFS_PERFECT(init=True, spectra=spectra, spectraM=spectraM)

# Configuration parameters
NT = 300                           # Duration of the simulation [frames]
dt = 3                              # Time of a frame [ms]
latencytime = 2*dt                  # Latency of the system
TELref = 0                          # If 0, all telescope move. If ia!=0, Tel ia static
# Camera charcateristics
QE = np.ones(config.MW)*0.7         # Quantum efficiency [e/ph]
RON = .5                           # ReadOUt Noise [e/e]

#%%

CHARAfile = datadir+'interferometers/CHARAinterferometerH.fits'

ObservationFile = datadir+'observations/CHARA/Unresolved_mag0.fits'

datadir2 = "C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/disturbances/"
DisturbanceFile = datadir2+'WithInjectionLosses/random_pis6_trans23_r10_tau4_L25_100000ms.fits'


# Initialize global coh variable
sk.initialize(CHARAfile, ObservationFile, DisturbanceFile, NT=NT,
              spectra=spectra, spectraM=spectraM,
              TELref=TELref, dt = dt,
              ron = RON, qe=QE,latencytime=latencytime,
              seedph=1,seedron=1,
              start_at_zero=False)   
    
# Add noise
config.noise = True

"""
DEFAULT FRINGE-TRACKER parameters
Optimised for r0=15; Tau0=10ms; L0=25m
"""
Ngd = 40 ; GainPD = 0.23 ; GainGD = .16 ; roundGD=True ; Threshold = True
Ncross=1 ; 
search=True;Sweep0=20 ; Sweep30s=10 ; Slope=0.5
Vfactors=[-8.25, -7.25, -4.25, 1.75, 3.75, 8.75]
CPref=True ; Ncp=300 ; ThresholdGD=30 ; ThresholdPD=1.5
usePDref = True ; useWmatrices=True

"""
SOME OPTIMISATION SETS
"""
# Mag5 & No noise: Ngd = 1 ; GainPD = 0.1 ; GainGD = .7
# Mag5 & Noise: Ngd = 40 ; GainPD = 0.23 ; GainGD = .16 (not tested with State-machine)
# Mag6 & RON=0.5 ; Ngd = 40 ; GainPD = 0.15 ; GainGD = .16 ;ThresholdGD=?
# Some examples of ThresholdGD depending on magnitude for 3 steps in manualsteps_tel0_50µm_30000ms.fits
# All with RON=0.5; Ngd = 40 ; GainPD = 0.15 ; GainGD = .16
# ThresholdGD --> Mag3:30; Mag4:12; Mag5:8; Mag6:4; Mag7:2.5 but don't work well

"""
TRUE FRINGE-TRACKER parameters
"""
# Ngd = 40 ; GainPD = 0.23 ; GainGD = .1 ; roundGD=True ; Threshold = True
# Ncross=1 ; 
# search=True; Sweep0=20 ; Sweep30s=10 ; Slope=.100   # 300nm/DIT -> 12µm/40DIT
# Vfactors=[-8.25, -7.25, -4.25, 1.75, 3.75, 8.75]
# CPref=True ; Ncp=300 ; ThresholdGD=30 ; ThresholdPD=1.5
# usePDref = True ; useWmatrices=False

# Initialize the fringe tracker with the gain
from cophasing.SPICA_FT import SPICAFT
SPICAFT(init=True, Ngd=Ngd, GainPD=GainPD, GainGD=GainGD, roundGD=roundGD,
        Ncross=Ncross,
        search=search,Sweep0=Sweep0,Sweep30s=Sweep30s,Slope=Slope,Vfactors=Vfactors,
        CPref=CPref,Ncp=Ncp,ThresholdGD=ThresholdGD,ThresholdPD=ThresholdPD,
        Threshold=Threshold,usePDref=usePDref,useWmatrices=useWmatrices,
        usecupy=False)
#%% Launch the simulator and save the data in the coh_file.pkl file

SavingFile = 'demo.fits'
sk.loop()


#%%  Display data
plt.close('all')
sk.display(wl=1.6, OneTelescope=True)


#%% Display performance

WavelengthOfInterest = [0.6,0.7,0.8,1.65]
StartingTime = 20
DIT = 50
# sk.ShowPerformance(StartingTime,WavelengthOfInterest,DIT)

#%%

# sk.SpectralAnalysis(OPD=(1,2))


#%%

