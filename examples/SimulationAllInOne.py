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

import pandas as pd
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
from cophasing.MIRCx_FS import MIRCxFS
lmbda1 = 1.5
lmbda2 = 1.7
MW=10
OW=10
# spectra = np.arange(lmbda1, lmbda2, )
spectra, spectraM = ct.generate_spectra(lmbda1, lmbda2, OW=OW, MW=MW, mode='linear_sig')
NW = len(spectra)

MIRCxFS(init=True, spectra=spectra, spectraM=spectraM)


# lmbda1 = 1.4
# lmbda2 = 1.7
# MW=5
# OW=10
# # spectra = np.arange(lmbda1, lmbda2, )
# spectra, spectraM = ct.generate_spectra(lmbda1, lmbda2, OW=OW, MW=5, mode='linear_sig')

# from coh_lib.SPICA_FS import SPICAFS_PERFECT
# SPICAFS_PERFECT(init=True, spectra=spectra, spectraM=spectraM)

# Configuration parameters
NT = 200               # Duration of the simulation [frames]
dt = 3                  # Time of a frame [ms]
latencytime = 2*dt     # Latency of the system
TELref = 0              # If 0, all telescope move. If ia!=0, Tel ia static
# Camera charcateristics
QE = np.ones(MW)*0.7    # Quantum efficiency [e/ph]
RON = 0.5              # ReadOUt Noise [e/e]


#%%

CHARAfile = datadir+'interferometers/CHARAinterferometerH.fits'

ObservationFile = datadir+'observations/CHARA/Unresolved_mag0.fits'

datadir2 = "C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/disturbances/"

DisturbanceFile = datadir2+'NoDisturbances/NoDisturbances.fits'
# DisturbanceFile = datadir2+'WithInjectionLosses/random_pis6_trans23_r10_tau4_L25_100000ms.fits'

# Initialize global coh variable
sk.initialize(CHARAfile, ObservationFile, DisturbanceFile, NT=NT,
              spectra=spectra, spectraM=spectraM,
              TELref=TELref, dt = dt,
              ron = RON, qe=QE,latencytime=latencytime,
              seedph=1,seedron=1,
              start_at_zero=False)
    
# Add noise
config.noise = True

Ngd = 1 ; GainPD = 0.2 ; GainGD = 2 ; roundGD = True ;Threshold=True #DEFAULT FRINGE-TRACKER parameters

"""
OTHER FRINGE-TRACKER parameters
"""
Ngd = 1
GainPD = 0
GainGD = 0
roundGD=True
Threshold=True
usePDref = True

# Initialize the fringe tracker with the gain
from cophasing.SPICA_FT import SPICAFT
SPICAFT(init=True, GainPD=GainPD, GainGD=GainGD, search=False,Nvar=5, Ngd=40,Ncross=1)

#%% Launch the simulator and save the data in the coh_file.pkl file

# SavingFile = 'demo.fits'
sk.loop()


#%%  Display data
plt.close('all')
sk.display(wl=1.55)


#%% Display performance

ct.studyP2VM()

#%%

