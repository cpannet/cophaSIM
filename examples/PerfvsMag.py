# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 21:34:43 2020

@author: cpannetier

Draws the figure of the performance function of the magnitude of the star.
Requirements:
    -Directory with the fitsfiles of all wished magnitudes.

Script:
    - intialise configuration
    - get the list of fitsfiles
    - browse the fitsfiles and optimise gain (PD or GD)
    - gather results (perf and corresponding gain)
    - plot figure Perf vs Magnitude (and gain)

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
Ngd = 40 ; GainPD = 0.2 ; GainGD = 0.35 ;roundGD = True ;Threshold = True

# Initialize the fringe tracker with the gain
cl.SimpleIntegrator(init=True, Ngd=Ngd, GainPD=GainPD, GainGD=GainGD,
                    roundGD=roundGD,Threshold=Threshold)

# from SPICA_FT import SPICAFT
# SPICAFT(init=True, Ngd=1, GainPD=.3, GainGD=GainGD)

#%% Launch the simulator and save the data in the coh_file.pkl file
"""Launch the simulator"""

TestLoop = True
if TestLoop:
    cl.loop()

    plt.close('all')
    cl.display(wl=1.6)

#%% Test OptimGain

from coh_lib.optimisation import OptimGain
import glob


"""
STARTING optimising the gain of the GD
"""

tel1=1;tel2=2 ; telescopes = (tel1,tel2)
# telescopes = 0

TimeBonds=100; WavelengthOfInterest=1.5; DIT=50
ib = ct.posk(tel1-1,tel2-1, config.NA)

GainsGD=np.arange(0.05,0.8,step=0.05)
GainsPD=np.arange(0.05,0.8,step=0.05)

FClist=[]
OPDlist=[]
VarCPlist=[]
bestGainGDlist=[]
ObsFiles = []
ObsFiles = glob.glob(datadir+'observations/PerfvsMag2/*.fits')
mags = []
for file in ObsFiles:
    start = file.find("_mag") + len("_mag")
    end = file.find(".fits")
    mag = file[start:end]
    mags.append(mag)
    print(f'Mag={mag}')
    
    config.ObservationFile = file
    
    # Initialize the fringe tracker with the gain GD
    cl.SimpleIntegrator(init=True, Ngd=Ngd, GainPD=0, GainGD=0,
                    roundGD=False,Threshold=False)
    
    bestGain, iOptim, VarOPD, VarCP, FCArray = OptimGain(GainsGD=GainsGD,
                                                                TimeBonds=TimeBonds, WavelengthOfInterest=WavelengthOfInterest, DIT=DIT,
                                                                telescopes=telescopes,optim='FC')
    
    bestGainGDlist.append(bestGain)
    if not telescopes:
        bestFC = np.mean(FCArray[iOptim,:])
        bestOPD = np.mean(VarOPD[iOptim,:])
    else:
        bestFC = FCArray[iOptim,ib]
        bestOPD = VarOPD[iOptim,ib]
    FClist.append(bestFC)
    OPDlist.append(bestOPD)
    

#%%

"""
Optimising the gain of the PD using the list of optimised gains GD
"""

tel1=1;tel2=2 ; telescopes = (tel1,tel2)
# telescopes = 0

TimeBonds=100; WavelengthOfInterest=0.75; DIT=50
ib = ct.posk(tel1-1,tel2-1, config.NA)

FClist=[]
OPDlist=[]
VarCPlist=[]
bestGainlist=[]
mags = []
ifile=0
for file in ObsFiles:
    start = file.find("_mag") + len("_mag")
    end = file.find(".fits")
    mag = file[start:end]
    mags.append(mag)
    print(f'Mag={mag}')
    
    config.ObservationFile = file
    GainGD = bestGainGDlist[ifile]
    # Initialize the fringe tracker with the gain GD
    cl.SimpleIntegrator(init=True, Ngd=Ngd, GainPD=0, GainGD=GainGD,
                    roundGD=True,Threshold=True)

    bestGain, iOptim, VarOPD, VarCP, FCArray = OptimGain(GainsPD=GainsPD,
                                                               TimeBonds=TimeBonds, WavelengthOfInterest=WavelengthOfInterest, DIT=DIT,
                                                               telescopes=telescopes,optim='FC')
    
    bestGainlist.append(bestGain)
    
    
    if not telescopes:
        bestFC = np.mean(FCArray[iOptim,:])
        bestOPD = np.mean(VarOPD[iOptim,:])
    else:
        bestFC = FCArray[iOptim,ib]
        bestOPD = VarOPD[iOptim,ib]
        
    FClist.append(bestFC)
    OPDlist.append(bestOPD)
    

    ifile+=1


#%% Save data


file1 = open("MyFile.txt","a")

file1.write('Magnitudes:')
file1.write(str(mags))
file1.write('\n Gains GD:')
file1.write(str(bestGainGDlist))
# file1.write('\n Average Fringe contrast at 1.5µm')
# file1.write(str(FClist))
# file1.write('\n Average OPD:')
# file1.write(str(OPDlist))

file1.write('\n')
file1.write('\n Optimisation sur Phase-Delay en utilisant Gain GD=0.32 \n')
file1.write('\n Gains PD:')
file1.write(str(bestGainlist))
file1.write('\n Average Fringe contrast at 1.5µm')
file1.write(str(FClist))
file1.write('\n Average OPD:')
file1.write(str(OPDlist))

file1.close()


#%%
GainPD = bestGain

GainPD=GainPD;GainGD=GainGD;roundGD=True;Threshold=True
cl.SimpleIntegrator(init=True, Ngd=Ngd, GainPD=GainPD, GainGD=GainGD,
                    roundGD=roundGD,Threshold=Threshold)
TestLoop = True
if TestLoop:
    cl.loop()

    plt.close('all')
    cl.display('disturbances','opd','OPDgathered',OPDdetails=True, wl=1.6)

#%%
textinfos = f"""Atmosphere: r\u2080=15cm ; \u03C4\u2080=10ms \n\
DIT={dt}ms \n\
Total readout noise=0.5e \n\
ENF=1.5 \n\
DIT(GD)=40 frames
Latence={latencytime}ms
Instrument throughput=2%
"""

magsArr = np.array([float(mag) for mag in mags])

magsArrRound = np.round(magsArr, 1)

plt.close(9)
fig = plt.figure(9)
ax1 = fig.add_subplot(111)
# fig.suptitle('Evolution of the fringe contrast with target magnitude')
ax1.plot(mags, FClist)
ax1.set_xlabel('Magnitude')
ax1.set_ylabel('Fringe contrast')
# ax1.text(1,0.7,textinfos)
ax1.set_xticklabels(magsArrRound,rotation=90)
# ax2 = ax1.twiny()
# ax2.set_xlabel('Gain PD')
# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticks(ax1.get_xticks())
# ax2.set_xticklabels(np.round(bestGainlist,2),rotation=90)
# ax3 = ax1.twinx()
# ax3.plot(mags, Var)
plt.grid()
plt.show()


#%%

plt.close(10)
fig = plt.figure(10)
ax1 = fig.add_subplot(111)
# fig.suptitle('Evolution of the OPD residues with target magnitude')
ax1.plot(mags, OPDlist)
ax1.set_xlabel('Magnitude')
ax1.set_ylabel('OPD rms [microns]')
# ax1.text(1,0.4,textinfos)
ax1.set_xticklabels(magsArrRound,rotation=90)
# ax2 = ax1.twiny()
# ax2.set_xlabel('Gain PD')
# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticks(ax1.get_xticks())
# ax2.set_xticklabels(np.round(bestGainlist,2),rotation=90)
# ax3 = ax1.twinx()
# ax3.plot(mags, Var)
plt.grid()
plt.show()

#%%

plt.close(11)
fig = plt.figure(11)
ax1 = fig.add_subplot(111)
# fig.suptitle('Evolution of the fringe contrast with target magnitude')
ax1.plot(mags, FClist)
ax1.set_xlabel('Magnitude')
ax1.set_ylabel('Fringe contrast')
# ax1.text(1,0.4,textinfos)

# ax1.set_yscale('log')
# ax2 = ax1.twiny()
# ax2.set_xlabel('Gain PD')
# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticks(ax1.get_xticks())
# ax2.set_xticklabels(np.round(bestGainlist,2),rotation=90)
ax2 = ax1.twinx()
ax2.set_ylabel('OPD variance [µm²]')
ax2.plot(mags, OPDlist, linestyle='--')
# ax2.set_ylim(0,np.max(OPDlist))
# ax2.set_yscale('log')
ax1.set_xticklabels(magsArrRound,rotation=90)
ax1.set_ylim(0,1)
plt.grid()
plt.show()


#%% Calculation of the fringe contrast directly from the OPDlist

FC075 = np.exp(-np.array(OPDlist)*(2*np.pi/0.75)**2/2)

plt.close(12)
fig = plt.figure(12)
ax1 = fig.add_subplot(111)
# fig.suptitle('Evolution of the fringe contrast with target magnitude')
ax1.plot(mags, FC075)
ax1.set_xlabel('Magnitude')
ax1.set_ylabel('Fringe contrast')
# ax1.text(1,0.4,textinfos)

# ax1.set_yscale('log')
# ax2 = ax1.twiny()
# ax2.set_xlabel('Gain PD')
# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticks(ax1.get_xticks())
# ax2.set_xticklabels(np.round(bestGainlist,2),rotation=90)
ax2 = ax1.twinx()
ax2.set_ylabel('OPD variance [µm²]')
ax2.plot(mags, OPDlist, linestyle='--')
# ax2.set_ylim(0,np.max(OPDlist))
# ax2.set_yscale('log')
ax1.set_xticklabels(magsArrRound,rotation=90)
ax1.set_ylim(0,1)
plt.grid()
plt.show()

#%% 
"""
With data from 20/11/2020 with:
Disturbance: 
random
r0 = 15cm
t0 = 10ms
L0 = 25m
Telescopes = all
"""

# mags2 = ['0.0', '0.91', '1.75', '2.5', '3.19', '3.81', '4.38', '4.89', '5.36', '5.78', '6.17', '6.52', '6.84', '7.13', '7.39', '7.63', '7.85', '8.04', '8.22', '8.38', '8.53', '8.67', '8.79', '8.9', '9.0']
# OPDlist2 = [0.0013513985282497146, 0.0013527074609589085, 0.0013792062997363863, 0.0013722689707546076, 0.0013877840219664178, 0.0014444220572181377, 0.001477445398791456, 0.0015936939508816037, 0.0018189246044394449, 0.002032326110755207, 0.002203477749449763, 0.0025506098298845588, 0.009560042391218777, 0.07106733328233174, 0.09644162985621324, 0.06045445693867187, 0.21401848190470585, 0.4273800448404026, 0.729134136723181, 0.6936126278951491, 0.8032109717304782, 0.915945897098773, 0.9306375924856252, 1.181919076363625, 1.2248702567997187]

# magsArr2 = np.array([float(mag) for mag in mags2])
# magsArrRound2 = np.round(magsArr2, 1)

# magsArrInt = [0,1,2,3,5,6,7,8,9]
# tickmags = [0, 1, 2, 4, 7, 10, 13, 17, 24]
# wl = 0.75
# FC075_2 = np.exp(-np.array(OPDlist2)*(2*np.pi/wl)**2/2)

# plt.close(13)
# fig = plt.figure(13)
# ax1 = fig.add_subplot(111)
# # fig.suptitle('Evolution of the fringe contrast with target magnitude')
# ax1.plot(mags2, FC075_2)
# ax1.set_xlabel('Magnitude', fontsize=14)
# ax1.set_ylabel(f'Fringe contrast at {wl}µm', fontsize=14)
# # ax1.text(1,0.4,textinfos)

# # ax1.set_yscale('log')
# # ax2 = ax1.twiny()
# # ax2.set_xlabel('Gain PD')
# # ax2.set_xlim(ax1.get_xlim())
# # ax2.set_xticks(ax1.get_xticks())
# # ax2.set_xticklabels(np.round(bestGainlist,2),rotation=90)
# ax2 = ax1.twinx()
# ax2.set_ylabel('OPD RMS [µm]', fontsize=14)
# ax2.plot(mags2, np.sqrt(OPDlist2), linestyle='--')
# # ax2.set_ylim(0,np.max(OPDlist))
# # ax2.set_yscale('log')
# # ax1.set_xscale('log')
# # ax1.set_xticks(tickmags)
# ax1.set_xticklabels(magsArrRound2, fontsize=14)
# # ax1.set_xscale('log')
# ax1.set_yscale('log')
# ax1.set_ylim(1e-5,1)

# plt.grid()
# plt.show()

#%%

"""
Calculation of the performance for each magnitude, knowing the optimised gains
"""
from coh_lib import simu

tel1=1;tel2=2 ; telescopes = (tel1,tel2)
telescopes = 0

TimeBonds=100; WavelengthOfInterest=0.75; DIT=50
ib = ct.posk(tel1-1,tel2-1, config.NA)

GainsGD = [0.35,0.35, 0.35, 0.35, 0.4, 0.4, 0.4, 0.35, 0.35, 0.35, 0.4, 0.45, 
           0.45, 0.4, 0.45,0.35, 0.55, 0.7, 0.7,0.65,0.35, 0.15 , 0.5, 0.2, 0.3]

GainsPD = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
           0.2,0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15, 0.15]

FClist=[]
OPDlist=[]
VarCPlist=[]
bestGainlist=[]
mags = []
ifile=0
for file in ObsFiles:
    start = file.find("_mag") + len("_mag")
    end = file.find(".fits")
    mag = file[start:end]
    mags.append(mag)
    print(f'Mag={mag}')
    
    config.ObservationFile = file
    # Initialize the fringe tracker with the gain GD
    GainPD=GainsPD[ifile]
    GainGD=GainsGD[ifile]
    cl.SimpleIntegrator(init=True, Ngd=Ngd, GainPD=GainPD, GainGD=GainGD,
                    roundGD=True,Threshold=True)

    cl.loop()
    cl.ShowPerformance(TimeBonds, WavelengthOfInterest,DIT, display=False)
    
    VarOPD = simu.VarOPD
    FCArray = simu.FringeContrast
    
    if not telescopes:
        bestFC = np.mean(FCArray)
        bestOPD = np.mean(VarOPD)
    else:
        bestFC = FCArray[ib]
        bestOPD = VarOPD[ib]
        
    FClist.append(bestFC)
    OPDlist.append(bestOPD)
    

    ifile+=1
    
#%%
magsArr = np.array([float(mag) for mag in mags])
magsArrRound = np.round(magsArr, 1)

plt.close(14)
fig = plt.figure(14)
ax1 = fig.add_subplot(111)
# fig.suptitle('Evolution of the fringe contrast with target magnitude')
ax1.plot(magsArr, FClist)
ax1.set_xlabel('Magnitude', fontsize=14)
ax1.set_ylabel(f'Fringe contrast at {WavelengthOfInterest}µm', fontsize=14)
# ax1.text(1,0.4,textinfos)

# ax1.set_yscale('log')
# ax2 = ax1.twiny()
# ax2.set_xlabel('Gain PD')
# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticks(ax1.get_xticks())
# ax2.set_xticklabels(np.round(bestGainlist,2),rotation=90)
ax2 = ax1.twinx()
ax2.set_ylabel('OPD RMS [µm]', fontsize=14)
ax2.plot(magsArr, np.sqrt(OPDlist), linestyle='--')
# ax2.set_ylim(0,np.max(OPDlist))
# ax2.set_yscale('log')
# ax1.set_xscale('log')
# ax1.set_xticks(tickmags)
# ax1.set_xticklabels(magsArrRound, fontsize=14)
# ax1.set_xscale('log')
# ax1.set_yscale('log')
ax1.set_ylim(0,1)
ax2.set_ylim(0,1)

plt.grid()
plt.show()
