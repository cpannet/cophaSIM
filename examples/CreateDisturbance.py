# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:48:32 2020

@author: cpannetier
"""
import os

import numpy as np

from cophasing.coh_tools import generate_spectra
from cophasing.skeleton import MakeAtmosphereCoherence

from mypackage.vrac import remove_char

# Spectral Sampling
lmbda1 = 1.4 ; lmbda2 = 1.8 ; MW = 5 ; OW = 10
spectra, sigma = generate_spectra(lmbda1, lmbda2, OW=OW, MW=MW, mode='linear_sig')

# Temporal sampling
dt = 1                  # Time of a frame [ms]
NT = 100000

# Disturbance power
# According to Colavita et al and Buscher et al: sig² = 6.88*(B/r0)^{5/3} [rad]
r0 = 0.15 #[m]
t0 = 10 #ms
V = 0.31*r0/t0*1e3
L0 = 25
ampl_from_r0 = np.sqrt(6.88*(L0/r0)**(5/3))*0.55/(2*np.pi)
dist = 'manualsteps' # step, random, manysteps
print(ampl_from_r0)
Levents = [(0,100,10000,50), (1,500,10000,50), (2,800,10000,50)]

ampl = 50
tel=6
#%%

datadir = 'C:/Users/cpannetier/Documents/Python_packages/coh_lib/coh_lib/data/'
datadir2 = "C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/"
# Intereferometer
InterferometerFile = datadir+'interferometers/CHARAinterferometerH.fits'



# =============================================================================
# NO PISTON NOR PHOTOMETRIC DISTURBANCES
# =============================================================================
# datadir = 'C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/disturbances/NoDisturbances/' 
# DisturbanceFile = datadir+"NoDisturbances.fits"

# # Create disturbance
# coher = MakeAtmosphereCoherence(DisturbanceFile, InterferometerFile,
#                                 overwrite=True,
#                                 NT=NT,spectra=spectra,
#                                 dist='coherent')


# =============================================================================
# 1 RANDOM PISTON DISTURBANCE
# =============================================================================

# atmcondition = 'med'

# # # Values from IDL code of MAM
# # if atmcondition=='bad':
# #     r0 = 0.067 ; t0 = 2.9      # Bad seeing --> V=0.31*r_0/tau0 = 7,2m/s
# # elif atmcondition == 'med':
# #     r0 = 0.088 ; t0 = 3        # Medium seeing --> V=9m/s
# # elif atmcondition == 'good':
# #     r0 = 0.108 ; t0 = 4.7      # Good seeing --> 7,1m/s


# # Values from MA thesis (Che 2013 and CHARA meeting)
# if atmcondition=='bad':
#     r0 = 0.06 ; t0 = 2.3      # Bad seeing (20th percentile = 20% below) --> V=0.31*r_0/tau0 = 8m/s
# elif atmcondition == 'med':
#     r0 = 0.10 ; t0 = 4        # Medium seeing (50th percentile) --> V=7.8m/s
# elif atmcondition == 'good':
#     r0 = 0.12 ; t0 = 10      # Good seeing (80th percentile) --> 3,7m/s


# ampl_from_r0 = np.sqrt(6.88*(L0/r0)**(5/3))*0.55/(2*np.pi)
# print(f"OPD rms (30s) computed from L0 and r0: {ampl_from_r0}µm")


# # Saving file
# DisturbanceFile = datadir2+f"disturbances/random_{tel}_{atmcondition}_r{int(round(r0*100))}_tau{int(round(t0))}_L{L0}_{NT*dt}ms.fits"

# # DisturbanceFile = datadir+f"perturb{atmcondition}_{round(NT*1e-3)}.fits"
 
# pows = (-2/3,-8/3,-17/3)
# # Create disturbance
# coher = MakeAtmosphereCoherence(DisturbanceFile, InterferometerFile,
#                                 overwrite=True,
#                                 NT=NT,spectra=spectra,
#                                 dist='random', tel=tel,startframe=50,
#                                 r0=r0, t0=t0,L0=L0,highFC=True,pows=pows,new=True)#, ampl=15/np.sqrt(2))


# =============================================================================
# NO PISTON DISTURBANCE AND REGULAR TOTAL INJECTION LOSSES
# =============================================================================

# datadir = 'C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/disturbances/' 

# # Add a transmission loss at 100ms during 200 milliseconds on telescop 2.
# TransDisturb = {'type':'manual','TEL2':[[100,200,0]]}
# AffectedTels = "".join([string[3:] for string in list(TransDisturb.keys()) if 'TEL' in string])

# # Saving file
# DisturbanceFile = datadir+f"WithInjectionLosses/coherent_trans_tel{AffectedTels}.fits"

# # Create disturbance
# coher = MakeAtmosphereCoherence(DisturbanceFile, InterferometerFile,
#                                 overwrite=True,
#                                 NT=NT,spectra=spectra,
#                                 dist='coherent',
#                                 TransDisturb=TransDisturb,
#                                 r0=r0, t0=t0,L0=L0,highCF=True)



# =============================================================================
# NO PISTON DISTURBANCE AND REGULAR TOTAL AND PARTIAL INJECTION LOSSES
# =============================================================================

# datadir = 'C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/disturbances/' 

# # On telescope 2: 
# #   - Total transmission loss at 100ms during 200 ms
# #   - Partial transmission loss (50%) at 200ms during 100ms
# # On telescope 3:
# #   - Total transmission loss at 300ms during 100ms
# TransDisturb = {'type':'manual','TEL2':[[100,200,0],[200,100,0.5]], 'TEL3':[[300,100,0.7]]}
# AffectedTels = "".join([string[3:] for string in list(TransDisturb.keys()) if 'TEL' in string])

# # Saving file
# DisturbanceFile = datadir+f"WithInjectionLosses/coherent_trans_tel{AffectedTels}.fits"

# # Create disturbance
# coher = MakeAtmosphereCoherence(DisturbanceFile, InterferometerFile,
#                                 overwrite=True,
#                                 NT=NT,spectra=spectra,
#                                 dist='coherent',
#                                 TransDisturb=TransDisturb)

# =============================================================================
# RANDOM PISTON DISTURBANCE AND REGULAR TOTAL INJECTION LOSSES
# =============================================================================

# atmcondition = 'med'

# # Values from MA thesis (Che 2013 and CHARA meeting)
# L0 = 25
# if atmcondition=='bad':
#     r0 = 0.06 ; t0 = 2.3      # Bad seeing (20th percentile = 20% below) --> V=0.31*r_0/tau0 = 8m/s
# elif atmcondition == 'med':
#     r0 = 0.10 ; t0 = 4        # Medium seeing (50th percentile) --> V=7.8m/s
# elif atmcondition == 'good':
#     r0 = 0.12 ; t0 = 10      # Good seeing (80th percentile) --> 3,7m/s


# datadir = 'C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/disturbances/' 

# # Add a total transmission loss at 200ms during 500 milliseconds on the telescope 2.
# TransDisturb = {'type':'manual','TEL2':[[100,200,0],[200,100,0.5]], 'TEL3':[[300,100,0.7]]}
# AffectedTels = "".join([string[3:] for string in list(TransDisturb.keys()) if 'TEL' in string])

# # Saving file
# DisturbanceFile = datadir+f"random_pis{tel}_trans{AffectedTels}_r{int(r0*100)}_tau{t0}_L{L0}_{NT*dt}ms.fits"

# # Create disturbance
# coher = MakeAtmosphereCoherence(DisturbanceFile, InterferometerFile,
#                                 overwrite=True,
#                                 NT=NT,spectra=spectra,
#                                 dist='random', tel=tel,startframe=50,
#                                 seed=101,
#                                 TransDisturb=TransDisturb,
#                                 r0=r0, t0=t0,L0=L0,highCF=True)


# =============================================================================
# NO PISTON DISTURBANCE BUT INJECTION LOSSES (from MIRCx file)
# =============================================================================
# datadir = 'C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/disturbances/tests/' 
# DisturbanceFile = datadir+"photometries/MIRCxInjections.fits"
# TransDisturb = {'type':'fileMIRCx', 'file': 'C:/Users/cpannetier/Documents/These/FringeTracking/SPICA-FT/InjectionPerturbations/mircx01780_datarts.fits'}


# # Create disturbance
# coher = MakeAtmosphereCoherence(DisturbanceFile, InterferometerFile,
#                                 overwrite=True,
#                                 NT=NT,spectra=spectra,
#                                 dist='coherent',
#                                 TransDisturb=TransDisturb)


# =============================================================================
# RANDOM PISTON DISTURBANCE AND INJECTION LOSSES (from MIRCx file)
# =============================================================================

# atmcondition = 'med'

# # Values from MA thesis (Che 2013 and CHARA meeting)
# L0 = 25
# if atmcondition=='bad':
#     r0 = 0.06 ; t0 = 2.3      # Bad seeing (20th percentile = 20% below) --> V=0.31*r_0/tau0 = 8m/s
# elif atmcondition == 'med':
#     r0 = 0.10 ; t0 = 4        # Medium seeing (50th percentile) --> V=7.8m/s
# elif atmcondition == 'good':
#     r0 = 0.12 ; t0 = 10      # Good seeing (80th percentile) --> 3,7m/s


# datadir = 'C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/disturbances/' 

# # MIRCx injection losses
# TransDisturb = {'type':'fileMIRCx', 'file': 'C:/Users/cpannetier/Documents/These/FringeTracking/SPICA-FT/InjectionPerturbations/mircx01780_datarts.fits'}

# # Saving file
# DisturbanceFile = datadir+f"WithInjectionLosses/random_pis{tel}_transMIRCx_r{int(r0*100)}_tau{t0}_L{L0}_{NT*dt}ms.fits"

# # Create disturbance
# coher = MakeAtmosphereCoherence(DisturbanceFile, InterferometerFile,
#                                 overwrite=True,
#                                 NT=NT,spectra=spectra,
#                                 dist='random', tel=tel,startframe=50,
#                                 seed=101,
#                                 TransDisturb=TransDisturb,
#                                 r0=r0, t0=t0,L0=L0,highCF=True)


# =============================================================================
# RANDOM PISTON DISTURBANCE+INJECTION LOSSES (from MIRCx file)+REGULAR TOTAL LOSSES
# =============================================================================

# atmcondition = 'med'

# # Values from MA thesis (Che 2013 and CHARA meeting)
# L0 = 25
# if atmcondition=='bad':
#     r0 = 0.06 ; t0 = 2.3      # Bad seeing (20th percentile = 20% below) --> V=0.31*r_0/tau0 = 8m/s
# elif atmcondition == 'med':
#     r0 = 0.10 ; t0 = 4        # Medium seeing (50th percentile) --> V=7.8m/s
# elif atmcondition == 'good':
#     r0 = 0.12 ; t0 = 10      # Good seeing (80th percentile) --> 3,7m/s


# datadir = 'C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/disturbances/' 

# # MIRCx injection losses + some total losses during 200ms
# TransDisturb = {'type':'both', 'TEL2':[[500,200,0],[1500,100,0]], 'TEL3':[[1000,200,0]],
#                 'file': 'C:/Users/cpannetier/Documents/These/FringeTracking/SPICA-FT/InjectionPerturbations/mircx01780_datarts.fits'}
# AffectedTels = "".join([string[3:] for string in list(TransDisturb.keys()) if 'TEL' in string])

# # Saving file
# DisturbanceFile = datadir+f"WithInjectionLosses/random_pis{tel}_transMIRCxand{AffectedTels}_r{int(r0*100)}_tau{t0}_L{L0}_{NT*dt}ms.fits"

# # Create disturbance
# coher = MakeAtmosphereCoherence(DisturbanceFile, InterferometerFile,
#                                 overwrite=True,
#                                 NT=NT,spectra=spectra,
#                                 dist='random', tel=tel,startframe=50,
#                                 seed=101,
#                                 TransDisturb=TransDisturb,
#                                 r0=r0, t0=t0,L0=L0,highCF=True)



# =============================================================================
# PHASE JUMPS AT REGULAR INTERVALS
# =============================================================================
# datadir = 'C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/disturbances/' 

# # Addition of phase jumps to simulate mecanical problems or vibrations
# PhaseJumps = {'TEL2':[[100,0,-5],[200,0,10]]}#, 'TEL3':[[300,100,0.7]]}
# AffectedTels = "".join([string[3:] for string in list(PhaseJumps.keys()) if 'TEL' in string])

# # Saving file
# DisturbanceFile = datadir+f"WithPhaseJumps/phasejumps_tel{AffectedTels}.fits"

# coher = MakeAtmosphereCoherence(DisturbanceFile, InterferometerFile,
#                                 overwrite=True,
#                                 NT=NT,spectra=spectra,
#                                 dist="coherent",
#                                 PhaseJumps=PhaseJumps)


# =============================================================================
# PHASE JUMPS AT REGULAR INTERVALS
# =============================================================================
# datadir = 'C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/disturbances/' 

# # Addition of phase jumps to simulate mecanical problems or vibrations
# PhaseJumps = {'TEL2':[[100,0,-5],[200,200,10]]}#, 'TEL3':[[300,100,0.7]]}
# AffectedTels = "".join([string[3:] for string in list(PhaseJumps.keys()) if 'TEL' in string])

# # Saving file
# DisturbanceFile = datadir+f"WithPhaseJumps/phasejumps_tel{AffectedTels}.fits"

# coher = MakeAtmosphereCoherence(DisturbanceFile, InterferometerFile,
#                                 overwrite=True,
#                                 NT=NT,spectra=spectra,
#                                 dist="coherent",
#                                 PhaseJumps=PhaseJumps)

# =============================================================================
# RANDOM PISTON DISTURBANCE+PHASE JUMPS+INJECTION LOSSES (from MIRCx file)+REGULAR TOTAL LOSSES
# =============================================================================

atmcondition = 'med'

# Values from MA thesis (Che 2013 and CHARA meeting)
L0 = 25
if atmcondition=='bad':
    r0 = 0.06 ; t0 = 2.3      # Bad seeing (20th percentile = 20% below) --> V=0.31*r_0/tau0 = 8m/s
elif atmcondition == 'med':
    r0 = 0.10 ; t0 = 4        # Medium seeing (50th percentile) --> V=7.8m/s
elif atmcondition == 'good':
    r0 = 0.12 ; t0 = 10      # Good seeing (80th percentile) --> 3,7m/s


datadir = 'C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/disturbances/' 

# MIRCx injection losses + some total losses during 200ms
TransDisturb = {'type':'both', 'TEL2':[[500,200,0],[1500,100,0]], 'TEL3':[[1000,200,0]],
                'file': 'C:/Users/cpannetier/Documents/These/FringeTracking/SPICA-FT/InjectionPerturbations/mircx01780_datarts.fits'}
TransTels = "".join([string[3:] for string in list(TransDisturb.keys()) if 'TEL' in string])

# Addition of phase jumps to simulate mecanical problems or vibrations
PhaseJumps = {'TEL2':[[100,0,-5],[200,200,10]]}#, 'TEL3':[[300,100,0.7]]}
PisTels = "".join([string[3:] for string in list(PhaseJumps.keys()) if 'TEL' in string])


# Saving file
DisturbanceFile = datadir+f"AllKindOfDisturbances/random_pis{tel}andphasejumps{PisTels}_transMIRCxand{TransTels}_r{int(r0*100)}_tau{t0}_L{L0}_{NT*dt}ms.fits"

# Create disturbance
coher = MakeAtmosphereCoherence(DisturbanceFile, InterferometerFile,
                                overwrite=True,
                                NT=NT,spectra=spectra,
                                dist='random', tel=tel,startframe=50,
                                seed=101,
                                TransDisturb=TransDisturb, PhaseJumps=PhaseJumps,
                                r0=r0, t0=t0,L0=L0,highCF=True)




# =============================================================================
# 10 RANDOM PISTON DISTURBANCES FOR OPTIMISATION
# =============================================================================
# 
# tel=0
# atmcondition = 'bad'

# # Values from MA thesis (Che 2013 and CHARA meeting)
# if atmcondition=='bad':
#     r0 = 0.06 ; t0 = 2.3      # Bad seeing (20th percentile = 20% below) --> V=0.31*r_0/tau0 = 8m/s
# elif atmcondition == 'med':
#     r0 = 0.10 ; t0 = 4        # Medium seeing (50th percentile) --> V=7.8m/s
# elif atmcondition == 'good':
#     r0 = 0.12 ; t0 = 10      # Good seeing (80th percentile) --> 3,7m/s

# Nfiles = 10
# for ifile in range(Nfiles):
#     seed=(ifile+1)*10       # Nécessaire de mulitplier par 10 pour éviter que deux mêmes seed reviennent entre deux T de deux fichiers différents
#     # Saving file
#     DisturbanceFile = f'C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/disturbances/for_optim/{atmcondition}/random_{tel}_r{int(r0*100)}_tau{int(round(t0))}_L{L0}_{NT*dt}ms_{ifile}.fits'
    
#     # Create disturbance
#     coher = MakeAtmosphereCoherence(DisturbanceFile, InterferometerFile,
#                                     overwrite=True,
#                                     NT=NT,spectra=spectra,
#                                     dist='random', tel=tel,startframe=50,
#                                     seed=seed,
#                                     r0=r0, t0=t0,L0=L0,highFC=True)


# ===================================================
# RAMP FOR SCANNING THE SNR OF THE COHERENCE ENVELOP
# ===================================================

# # Saving file
# DisturbanceFile = datadir2+'disturbances/EtudeThreshold/scan240micron_tel6_faster.fits'
# NT = 1500
# # Create disturbance
# coher = MakeAtmosphereCoherence(DisturbanceFile, InterferometerFile,
#                                 overwrite=True,
#                                 NT=NT,spectra=spectra,
#                                 dist='slope', tel=tel, value_start=-120, value_end=120, ampl=2)


# ===================================================
# RAMP FOR SCANNING ONE WAVELENGTH
# ===================================================

# # Saving file
# DisturbanceFile = datadir2+'disturbances/EtudeThreshold/ramp_tel6_over1wl.fits'
# NT = 1000
# tel=6
# # Create disturbance
# coher = MakeAtmosphereCoherence(DisturbanceFile, InterferometerFile,
#                                 overwrite=True,
#                                 NT=NT,spectra=spectra,
#                                 dist='slope', tel=tel, value_start=-1.5, value_end=1.5)

# ===================================================
# STEP FOR TESTING THE RESPONSE TO A SIMPLE DISTURBANCE
# ===================================================

# # Saving file
# DisturbanceFile = datadir2+'disturbances/EtudeThreshold/step0.fits'
# NT = 30000 ; tel=1
# # Create disturbance
# coher = MakeAtmosphereCoherence(DisturbanceFile, InterferometerFile,
#                                 overwrite=True,
#                                 NT=NT,spectra=spectra,
#                                 dist='step', tel=tel, ampl=0, startframe=50)


#%% See created disturbance

import matplotlib.pyplot as plt
from cophasing.coh_tools import get_infos


timestamps,lambdas, piston, transmission, FreqSampling, PSD, PSDfilter, params = get_infos(DisturbanceFile)

NA = np.shape(piston)[1]

PSDexists = (PSD is np.zeros_like(PSD))

if PSDexists:
    plt.close('Disturbances1')
    fig=plt.figure('Disturbances1')
    fig.suptitle(f'Profile disturbances for r0={r0}cm, t0={t0}ms and L0={L0}m.')
    ax1,ax2 = fig.subplots(nrows=2)
    for ia in range(NA):
        ax1.plot(timestamps*1e-3, piston[:,ia], label=f"T{ia+1}")
    ax2.loglog(FreqSampling, PSD, color='b')
    ax2.loglog(FreqSampling, PSDfilter, color='k')
    ax1.set_ylabel('Piston [µm]')
    ax1.set_xlabel('Time [s]')
    ax2.set_ylabel('Piston [µm²/Hz]')
    ax2.set_xlabel('Frequency [Hz]')
    # ax2.legend()
    plt.show()
    
else:
    plt.close('Disturbances1')
    fig=plt.figure('Disturbances1')
    fig.suptitle(f'Profile disturbances for r0={r0}cm, t0={t0}ms and L0={L0}m.')
    ax = fig.subplots()
    for ia in range(NA):
        ax.plot(timestamps*1e-3, piston[:,ia], label=f"T{ia+1}")
    ax.set_ylabel('Piston [µm]')
    ax.set_xlabel('Time [s]')
    # ax2.legend()
    plt.show()



fig = plt.figure("Transmission disturbance for the six telescopes", clear=True)
for ia in range(NA):
    axes = fig.subplots(NA,1,sharex=True,sharey=True)
    for ia in range(NA):
        axes[ia].plot(timestamps, transmission[:,4,ia])
    axes[-1].set_ylim(0,1)
fig.show()


fig = plt.figure("Average Transmission disturbance over the spectra for the six telescopes", clear=True)
for ia in range(NA):
    axes = fig.subplots(NA,1,sharex=True,sharey=True)
    for ia in range(NA):
        axes[ia].plot(timestamps, np.mean(transmission[:,:,ia],axis=1))
    axes[-1].set_ylim(0,1)
fig.show()

#%%
fig = plt.figure("Average Transmission disturbance over the telescopes for the six telescopes", clear=True)
for ia in range(NA):
    plt.plot(timestamps, np.mean(transmission[:,:,1],axis=1))
    plt.ylim(0,1)
fig.show()