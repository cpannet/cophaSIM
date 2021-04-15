# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:18:00 2020

@author: cpannetier
"""

# Define the workspace as being the coh_pack2 main workspace
import os
# os.chdir('C:/Users/cpannetier/Documents/These/FringeTracking/Python/coh_pack/coh_lib')

datadir = 'C:/Users/cpannetier/Documents/These/FringeTracking/Python/coh_pack/coh_lib/data/'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import coh_lib.skeleton as cl
import coh_lib.coh_tools as ct
from coh_lib import config
#%%

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

#%%
""" Initialise the simulation configurations """

CHARAfile = datadir+'interferometers/CHARAinterferometerH.fits'

ObservationFile = datadir+'observations/CHARA/Unresolved_mag0.fits'

DisturbanceFile = datadir+'disturbances/random_0_r15_tau10_L25_30000ms.fits'
# DisturbanceFile = datadir+'disturbances/step_T2_500nano_3seconds.fits'

# Configuration parameters
NT = 3000               # Duration of the simulation [frames]
dt = 3                  # Time of a frame [ms]
latencytime = 2*dt     # Latency of the system
TELref = 1              # If 0, all telescope move. If ia!=0, Tel ia static
# Camera charcateristics
QE = np.ones(config.MW)*0.7    # Quantum efficiency [e/ph]
RON = 0.5              # ReadOUt Noise [e/e]

# Initialize global coh variable
cl.initialize(CHARAfile, ObservationFile, DisturbanceFile, NT=NT, spectra=spectra,\
                      fs='truespica', ft='ftspica',TELref=TELref, dt = dt,\
                          ron = RON, qe=QE,
                          starttracking=5,latencytime=latencytime)     

# Add noise
config.noise = True

Ngd = 1 ; GainPD = 0.3 ; GainGD = .7 ; roundGD = True ; Threshold=True #DEFAULT FRINGE-TRACKER parameters

"""
OTHER FRINGE-TRACKER parameters
"""
Ngd = 40 ; GainPD = 0.11 ; GainGD = 0.05 ;roundGD = True ;Threshold = True

# Initialize the fringe tracker with the gain
cl.SimpleIntegrator(init=True, Ngd=Ngd, GainPD=GainPD, GainGD=GainGD,
                    roundGD=roundGD,Threshold=Threshold)


#%% Launch the simulator and save the data in the coh_file.pkl file

TestLoop = True
if TestLoop:
    cl.loop()

    plt.close('all')
    cl.display(wl=1.6)


#%% Test OptimGain
from coh_lib.optimisation import OptimGain
from coh_lib import simu

# Initialize the fringe tracker with the gain
cl.SimpleIntegrator(init=True, Ngd=Ngd, GainPD=GainPD, GainGD=GainGD,
                    roundGD=roundGD,Threshold=Threshold)

tel1=1;tel2=2 ; telescopes = (tel1,tel2)
# telescopes = 0

TimeBonds=100; WavelengthOfInterest=1.5; DIT=50
ib = ct.posk(tel1-1,tel2-1, config.NA)

GainsPD=np.arange(0.05,0.8,step=0.05)

bestGain, iOptim, VarOPD, VarCP, FCArray = OptimGain(GainsPD=GainsPD,
                                                     TimeBonds=TimeBonds, 
                                                     WavelengthOfInterest=WavelengthOfInterest, 
                                                     DIT=DIT,telescopes=(tel1,tel2),
                                                     optim='FC')

FC = FCArray[iOptim,ib]
    

#%%

textinfos = """RMS OPD(disturbance) = 14µm \n\
DIT=3ms \n\
RON=3e/e \n\
ENF=1.5 \n\
GD=4 \n\
DIT(GD)=1 frame
Latence=1 frame
Instrument throughput=5%
"""

# GainsGD=np.arange(0,15,step=1)
plt.close(9)
fig = plt.figure(9)
fig.suptitle('Evolution of the fringe contrast with the gain')

ax1 = fig.add_subplot(111)
ax1.plot(GainsPD, FCArray[:,ib])
ax1.set_xlabel('Gain PD')
ax1.set_ylabel('Fringe contrast')
ax1.text(0.15,0.5,textinfos)

ax2 = ax1.twinx()
ax2.set_ylabel('OPD variance [µm]')
ax2.plot(GainsPD, VarOPD[:,ib], linestyle='--')
# ax2.set_yscale('log')
# ax2.set_ylim(0,np.max())
# ax2.set_xticks(ax1.get_xticks())
# ax2.set_xticklabels(bestGainlist)
# ax1.grid()
# ax2.grid()
plt.grid()
plt.show()
