# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:18:00 2020

@author: cpannetier
"""

# Define the workspace as being the coh_pack2 main workspace
import os
os.chdir('C:/Users/cpannetier/Documents/These/FringeTracking/Python/coh_pack/coh_lib')

datadir = 'data/'

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
from coh_lib.SPICA_FS import SPICAFS_TRUE
spectra, spectraM = SPICAFS_TRUE(init=True,fitsfile=FSfitsfile,OS=1)

# MW=5
# lmbda1 = 1.3
# lmbda2 = 1.75
# R = 22
# spectra = ct.generatespectra(lmbda1, lmbda2, 10*R, MW=MW)
# NW = len(spectra)
# OW = int(NW/MW)
# sigma = 1/spectra

from coh_lib.config import Observation

# Camera charcateristics
QE = np.ones(config.MW)*0.7    # Quantum efficiency [e/ph]
RON = 3                 # ReadOUt Noise [e/e]
dt = 3                  # Time of a frame [ms]


#%%

CHARAfile = './data/interferometers/CHARAinterferometerH.fits'

ObservationFile = './data/observations/PerfvsMag/Unresolved_mag0.0.fits'

DisturbanceFile = './data/disturbances/random_T2.fits'

# Initialize global coh variable
cl.initialize(CHARAfile, ObservationFile, DisturbanceFile, MW=5, NT=200, spectra=spectra,\
                      fs='truespica', ft='ftspica',TELref=1, dt = dt,\
                          ron = RON, qe=QE,
                          starttracking=5,latencytime=dt)     

# Add noise
config.noise = True

Ngd = 1 ; GainPD = 0.3 ; GainGD = .7 ; roundGD = True ; Threshold=True #DEFAULT FRINGE-TRACKER parameters

"""
OTHER FRINGE-TRACKER parameters
"""
Ngd = 1
GainPD = 0.32
GainGD = 0#4#1.1
roundGD = True
Threshold = True

# from SPICA_FS import SPICAFS_PERFECT
# SPICAFS_PERFECT(init=True)

# Initialize the fringe tracker with the gain
cl.SimpleIntegrator(init=True, Ngd=Ngd, GainPD=GainPD, GainGD=GainGD,
                    roundGD=roundGD,Threshold=Threshold)

# from SPICA_FT import SPICAFT
# SPICAFT(init=True, Ngd=1, GainPD=.3, GainGD=GainGD)

#%% Launch the simulator and save the data in the coh_file.pkl file
# SavingFile = 'demo.fits'
# cl.loop()


#%%  Display data
# plt.close('all')
# cl.display(wl=1.6)


def OptimGainPD2(GainsPD=[],GainsGD=[],optim='FC',
                TimeBonds=100, WavelengthOfInterest=1.5, DIT=50,
                telescopes=0):
    
    from coh_lib import simu
    from coh_lib import config
    import multiprocessing
    
    pool = multiprocessing.Pool()

    from coh_lib.config import NIN, NC
    
    if len(GainsGD):
        gainstr='GainGD'
        Ngains = len(GainsGD)
        Gains = GainsGD
    elif len(GainsPD):
        gainstr='GainPD'
        Ngains = len(GainsPD)
        Gains = GainsPD
    else:
        raise Exception('Need GainsPD or GainsGD.')
        
    print(f"Start {gainstr} optimisation with sample gains={Gains}")
    
    # SimpleIntegrator(init=True, GainPD=Gains[0], GainGD=0)
    
    
    VarOPD = np.zeros([Ngains,NIN])     # Phase variances
    VarCP = np.zeros([Ngains, NC])      # Cosure Phase variances
    FCArray = np.zeros([Ngains, NIN])   # Contains the fringe contrasts
    
    
    minValue = 10000
    iOptim = -1
    
        

    myresult = pool.starmap(GiveResults, Gains.reshape([Ngains,1]))
    
    print(myresult)


def GiveResults(G):
    
    config.FT['GainPD'] = G
    print(G)
    # print(f'{gainstr}={G}')
    
    # Launch the simulator
    cl.loop()
    # Load the performance observables into simu module
    cl.ShowPerformance(TimeBonds, WavelengthOfInterest,DIT, display=False)
            
    
    return simu.VarOPD, simu.VarCPD, simu.VarCGD, simu.FringeContrast





#%% Test OptimGain
from coh_lib import simu

# Initialize the fringe tracker with the gain
cl.SimpleIntegrator(init=True, Ngd=Ngd, GainPD=GainPD, GainGD=GainGD,
                    roundGD=roundGD,Threshold=Threshold)

tel1=1;tel2=2

TimeBonds=100; WavelengthOfInterest=1.5; DIT=50
ib = ct.posk(tel1-1,tel2-1, config.NA)

GainsPD=np.arange(0,0.5,step=0.05)
config.ObservationFile = './data/observations/CHARA/Unresolved_mag0.fits'
bestGain, iOptim, VarOPD, VarCP, FCArray = OptimGainPD2(GainsPD=GainsPD,
                                                           TimeBonds=TimeBonds, WavelengthOfInterest=WavelengthOfInterest, DIT=DIT,
                                                           telescopes=(tel1,tel2),optim='FC')

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
