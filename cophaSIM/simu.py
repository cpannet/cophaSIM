# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:41:21 2020

@author: cpannetier

SIMU MODULE

Contains all the data processed by the simulator to be read, modified and/or
saved during and after the simulation.

"""

import numpy as np

from .config import NA,NB,NC,NT,dt,NW,NIN,FS,latency
from . import config

MW = config.FS['MW']
NINmes = config.FS['NINmes']
NBmes = config.FS['NBmes']
# Time
it = 0
timestamps = np.arange(NT)*dt


# Coherent flux [NB] - Most general formalism: contains photometries information
CoherentFluxObject = np.zeros([NW, NB])*1j      # Object coherent flux
VisibilityObject = np.zeros([NW, NB])*1j        # Object Visibility
CfDisturbance = np.zeros([NT, NW, NB])*1j       # Disturbance coherent flux
CfTrue = np.zeros([NT, NW, NB])*1j              # True coherent flux
CfTrue_r = np.zeros([NT, NW, NBmes])*1j              # True coherent flux reduced to measured baselines
CfEstimated = np.zeros([NT, MW, NBmes])*1j         # Estimated coherent flux


# Coherent flux [NIN] - Only purely coherent flux
CfPD = np.zeros([NT, MW, NINmes])*1j              # Dispersion corrected coherent flux
CfGD = np.zeros([NT, MW, NINmes])*1j              # GD coherent flux
SquaredCoherenceDegree = np.zeros([NT, MW, NINmes])          # Estimated coherence degree \    
VisibilityEstimated = np.zeros([NT, MW, NINmes])*1j         # Estimated fringe visibility \    
VisibilityTrue = np.zeros([NT, MW, NIN])*1j         # True expected fringe visibility \    
    

# Closure Phases [NC]
ClosurePhaseObject = np.zeros([NW,NC])          # Closure Phase Object
BispectrumGD = np.ones([NT,NC])+0j              # GD bispectrum
BispectrumPD = np.ones([NT,NC])+0j              # PD bispectrum
ClosurePhasePD = np.zeros([NT,NC])              # PD closure phase        
ClosurePhaseGD = np.zeros([NT,NC])              # GD closure phase


# OPD-space observables [NIN]
OPDTrue = np.zeros([NT,NIN])                    # True Optical Path Delay
OPDDisturbance = np.zeros([NT,NIN])             # OPD-space disturbance
PDEstimated = np.zeros([NT,NINmes])                # Estimated baselines PD [rad]
PDEstimated2 = np.zeros([NT,NINmes])                # Estimated baselines PD after patch [rad]
varPD = np.zeros([NT,NINmes])                      # Estimated "PD variance" = 1/SNR²
varGD = np.zeros([NT,NINmes])                      # Estimated "GD variance" = 1/SNR²
SquaredSNRMovingAverage = np.zeros([NT,NINmes])    # Estimated SNR² averaged over N dit
TrackedBaselines = np.zeros([NT,NINmes])
TemporalVariancePD = np.zeros([NT,NINmes])         # Temporal Variance PD estimator
TemporalVarianceGD = np.zeros([NT,NINmes])         # Temporal Variance GD estimator
GDEstimated = np.zeros([NT,NINmes])                # Estimated baselines GD [rad]
GDEstimated2 = np.zeros([NT,NINmes])                # Estimated baselines GD after patch [rad]
OPDCommand = np.zeros([NT+1,NIN])               # OPD-space command ODL
PDCommand = np.zeros([NT+1,NINmes])                # OPD-space PD command
GDCommand = np.zeros([NT+1,NINmes])                # OPD-space GD command
EffectiveOPDMove = np.zeros([NT+latency,NIN])   # Effective move of the delay lines in OPD-space

PDResidual = np.zeros([NT,NINmes])                 # Estimated residual PD = PD-PDref
PDResidual2 = np.zeros([NT,NINmes])                 # Estimated residual PD = PD-PDref after Ipd
GDResidual = np.zeros([NT,NINmes])                 # Estimated residual GD = GD-GDref
GDResidual2 = np.zeros([NT,NINmes])                 # Estimated residual GD = GD-GDref after Igd
GDErr = np.zeros([NT,NINmes])                      # Error that integrates GD integrator
OPDSearchCommand = np.zeros([NT+1,NIN])           # Search command projected in the OPD-space

# Piston-space observables [NA]
PistonTrue = np.zeros([NT,NA])                  # True Pistons
PistonDisturbance = np.zeros([NT,NA])           # Piston disturbances
PistonPDCommand = np.zeros([NT+1,NA])           # Piston-space PD command
PistonGDCommand = np.zeros([NT+1,NA])           # Piston-space GD command
PistonGDCommand_beforeround = np.zeros([NT+1,NA])
GDPistonResidual = np.zeros([NT,NA])            # GD residuals on telescopes
PDPistonResidual = np.zeros([NT,NA])            # PD residuals on telescopes
SearchCommand = np.zeros([NT+1,NA])             # Piston-space Search command
CommandODL = np.zeros([NT+1,NA])                # Delay lines positionnal command calculated at time it
EffectiveMoveODL = np.zeros([NT+latency,NA])    # Effective move of the delay lines

TransmissionDisturbance = np.ones([NT,NW,NA])   # Transmission disturbance of the telescopes
PhotometryDisturbance = np.zeros([NT,NW,NA])    # Resulting photometry disturbances (scaled by the object photometry)
PhotometryEstimated = np.zeros([NT,MW,NA])      # Estimated photometries

# Other observables
# DisturbancePSD = np.zeros([NT])                 # Power-Spectral Distribution of
#                                                 # one disturbance
# FreqSampling = np.zeros([])                     # Frequency sampling           
# DisturbanceFilter = np.zeros([])                # Disturbance PSD filter

MacroImages = np.zeros([NT,MW,FS['NP']])        # Contains microtime images sumed up on macro-wl 
CovarianceReal = np.zeros([NT,MW,NBmes])           # Covariances of the real part of the coherent flux
CovarianceImag = np.zeros([NT,MW,NBmes])           # Covariances of the imaginary part of the coherent flux
Covariance = np.zeros([NT,MW,NBmes,NBmes])            # Covariances of the real and imaginary parts of the coherent flux
BiasModCf = np.zeros([NT,MW,NINmes])               # Bias on the estimator of the module of the coherent flux.
varFlux = np.zeros([NT,MW,FS['NP']])            # Variance Flux
SNRPhotometry = np.zeros([NT,NA])               # SNR of the photometry estimation

PDref = np.zeros([NT,NINmes])                      # PD reference vector
GDref = np.zeros([NT,NINmes])                      # GD reference vector
OPDrefObject = np.zeros([NIN])                  # PD reference of the Object (only for analysis)

CfPDref = np.ones([NT,NINmes])+0j
CfGDref = np.ones([NT,NINmes])+0j

FTmode = np.ones([NT])                          # Save mode of the Fringe Tracker
                                                # 0: off, 1: Search, 2: Track

Ipd = np.ones([NT,NINmes,NINmes])                         # Weighting matrix PD command
Igd = np.ones([NT,NINmes,NINmes])                         # Weighting matrix GD command
Igdna = np.ones([NT,NA,NA])
IgdRank =np.ones([NT])                      # Rank of the weighting matrix GD
time_since_loss=np.zeros(NT)                    # Time since the loss of one telescope
NoPhotometryFiltration = np.zeros([NT,NA,NA])  # Matrix that filters telescopes which have no photometry
LostTelescopes = np.zeros([NT,NA])
noSignal_on_T = np.zeros([NT,NA])

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
VarGDRes = np.zeros([NINmes])  # Temporal variance estimator GD after Igd weighting
VarPDRes = np.zeros([NINmes])  # Temporal variance estimator PD after Ipd weighting
VarGDEst = np.zeros([NINmes])  # Temporal variance estimator GD
VarPDEst = np.zeros([NINmes])  # Temporal variance estimator PD
VarCPD = np.zeros([NC])
VarCGD =np.zeros([NC])
SNRSI=np.zeros([NIN])
VarPiston = np.zeros([NA])

# Investigation Variables

SquaredSNRMovingAveragePD = np.zeros([NT,NINmes])           # Estimated SNR
SquaredSNRMovingAverageGD = np.zeros([NT,NINmes])           # Estimated SNR
SquaredSNRMovingAverageGDUnbiased = np.zeros([NT,NINmes])
varPDnum = np.zeros([NT,NINmes])
varPDnum2 = np.zeros([NT,NINmes])
varPDdenom = np.zeros([NT,NINmes])
varGDdenom = np.zeros([NT,NINmes])
varGDdenomUnbiased = np.zeros([NT,NINmes])
varPD = np.zeros([NT,NINmes])
varGDUnbiased = np.zeros([NT,NINmes])
varNum2 = np.zeros([NT,MW,NINmes])
LossDueToInjection = np.zeros(NT)
eps = np.zeros([NT,NA])
it_last = np.zeros([NT,NA])
last_usaw = np.zeros([NT,NA])

del NA,NB,NC,NT,dt,MW,NW,NIN,FS

