# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:41:21 2020

@author: cpannetier

SIMU MODULE

Contains all the data processed by the simulator to be read, modified and/or
saved during and after the simulation.

"""

import numpy as np

try:
    from .config import NA,NB,NC,NT,dt,MW,NW,NIN,FS,latency
except:
    raise Exception("The module 'config' must be imported and initialised \
                    before importing simu")

print(NT)
# Time
it = 0
timestamps = np.arange(NT)*dt


# Coherent flux [NB] - Most general formalism: contains photometries information
CoherentFluxObject = np.zeros([NW, NB])*1j      # Object coherent flux
CfDisturbance = np.zeros([NT, NW, NB])*1j       # Disturbance coherent flux
CfTrue = np.zeros([NT, NW, NB])*1j              # True coherent flux
CfEstimated = np.zeros([NT, MW, NB])*1j         # Estimated coherent flux


# Coherent flux [NIN] - Only purely coherent flux
CfPD = np.zeros([NT, MW, NIN])*1j              # Dispersion corrected coherent flux
CfGD = np.zeros([NT, MW, NIN])*1j              # GD coherent flux
SquaredCoherenceDegree = np.zeros([NT, MW, NIN])          # Estimated coherence degree \    
VisibilityEstimated = np.zeros([NT, MW, NIN])*1j         # Estimated fringe visibility \    
VisibilityTrue = np.zeros([NT, MW, NIN])*1j         # True expected fringe visibility \    
    

# Closure Phases [NC]
ClosurePhaseObject = np.zeros([NW,NC])          # Closure Phase Object
ClosurePhasePD = np.zeros([NT,NC])              # PD closure phase        
ClosurePhaseGD = np.zeros([NT,NC])              # GD closure phase


# OPD-space observables [NIN]
OPDTrue = np.zeros([NT,NIN])                    # True Optical Path Delay
OPDDisturbance = np.zeros([NT,NIN])             # OPD-space disturbance
PDEstimated = np.zeros([NT,NIN])                # Estimated baselines PD [rad]
varPD = np.zeros([NT,NIN])                      # Estimated "PD variance" = 1/SNR²
SquaredSNRMovingAverage = np.zeros([NT,NIN])    # Estimated SNR² averaged over N dit
TemporalVariancePD = np.zeros([NT,NIN])         # Temporal Variance PD estimator
TemporalVarianceGD = np.zeros([NT,NIN])         # Temporal Variance GD estimator
GDEstimated = np.zeros([NT,NIN])                # Estimated baselines GD [rad]
OPDCommand = np.zeros([NT+latency,NIN])         # OPD-space command ODL
PDCommand = np.zeros([NT+latency,NIN])          # OPD-space PD command
GDCommand = np.zeros([NT+latency,NIN])          # OPD-space GD command
PDResidual = np.zeros([NT,NIN])                 # Estimated residual PD = PD-PDref
GDResidual = np.zeros([NT,NIN])                 # Estimated residual GD = GD-GDref


# Piston-space observables [NA]
PistonTrue = np.zeros([NT,NA])                  # True Pistons
PistonDisturbance = np.zeros([NT,NA])           # Piston disturbances
PistonPDCommand = np.zeros([NT+latency,NA])              # Piston-space PD command
PistonGDCommand = np.zeros([NT+latency,NA])              # Piston-space GD command
SearchCommand = np.zeros([NT+latency,NA])              # Piston-space Search command
CommandODL = np.zeros([NT+latency,NA])          # Delay lines positionnal command

TransmissionDisturbance = np.ones([NT,NW,NA])   # Transmission disturbance of the telescopes
PhotometryDisturbance = np.zeros([NT,NW,NA])    # Resulting photometry disturbances (scaled by the object photometry)
PhotometryEstimated = np.zeros([NT,MW,NA])      # Estimated photometries

# Other observables
# DisturbancePSD = np.zeros([NT])                 # Power-Spectral Distribution of
#                                                 # one disturbance
# FreqSampling = np.zeros([])                     # Frequency sampling           
# DisturbanceFilter = np.zeros([])                # Disturbance PSD filter

MacroImages = np.zeros([NT,MW,FS['NP']])        # Contains microtime images sumed up on macro-wl 
CovarianceReal = np.zeros([NT,MW,NB])           # Covariances of the real part of the coherent flux
CovarianceImag = np.zeros([NT,MW,NB])           # Covariances of the imaginary part of the coherent flux
Covariance = np.zeros([NT,MW,NB,NB])            # Covariances of the real and imaginary parts of the coherent flux
varFlux = np.zeros([NT,MW,FS['NP']])            # Variance Flux

PDref = np.zeros([NT,NIN])                        # PD reference vector
GDref = np.zeros([NT,NIN])                        # GD reference vector

FTmode = np.ones([NT])                          # Save mode of the Fringe Tracker
                                                # 0: off, 1: Search, 2: Track

Ipd = np.ones([NT,NIN,NIN])                         # Weighting matrix PD command
Igd = np.ones([NT,NIN,NIN])                         # Weighting matrix GD command
IgdRank =np.ones([NT])                      # Rank of the weighting matrix GD
time_since_loss=np.zeros(NT)                    # Time since the loss of one telescope
"""
Performance Observables will be introduced and processed when the function 
"SeePerf" is called

    -VarOPD                                     # Temporal Variance OPD [µm]
    -TempVarPD                                  # Temporal Variance PD [rad]
    -TempVarGD                                  # Temporal Variance of GD [rad]
    -VarCPD                                     # Temporal Variance of CPD [rad]
    -VarCGD                                     # Temporal Variance of CGD [rad]
    -FringeContrast                             # Fringe Contrast at given wavelengths [0,1]
"""

VarOPD = np.zeros([NIN])
TempVarPD = np.zeros([NIN])
TempVarGD = np.zeros([NIN])
VarCPD = np.zeros([NC])
VarCGD =np.zeros([NC])


# Investigation Variables

SquaredSNRMovingAverage2 = np.zeros([NT,NIN])           # Estimated SNR
varPDnum = np.zeros([NT,NIN])
varPDnum2 = np.zeros([NT,NIN])
varPDdenom = np.zeros([NT,NIN])
varPD2 = np.zeros([NT,NIN])
varNum2 = np.zeros([NT,MW,NIN])

del NA,NB,NC,NT,dt,MW,NW,NIN,FS


