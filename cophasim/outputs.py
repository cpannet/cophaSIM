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

"""Basic parameters"""
MW = config.FS['MW']
NINmes = config.FS['NINmes']
NBmes = config.FS['NBmes']
# Time
it = 0
TimeID="notime"
timestamps = np.arange(NT)*config.dt    # ms
outputsFile = ""
simulatedTelemetries = True         # False = telemetries from spica-ft instrument

"""Coherent flux [NB]""" # Most general formalism: contains photometries information
CoherentFluxObject = np.zeros([NW, NB])*1j          # Object coherent flux
VisibilityObject = np.zeros([NW, NB])*1j            # Object Visibility
CfDisturbance = np.zeros([NT, NW, NB])*1j           # Disturbance coherent flux
CfTrue = np.zeros([NT, NW, NB])*1j                  # True coherent flux
CfTrue_r = np.zeros([NT, NW, NBmes])*1j             # True coherent flux reduced to measured baselines
CfEstimated = np.zeros([NT, MW, NBmes])*1j          # Estimated coherent flux


"""Coherent flux [NIN]""" # Only purely coherent flux
CfPD = np.zeros([NT, MW, NINmes])*1j                # Dispersion corrected coherent flux
CfGD = np.zeros([NT, MW, NINmes])*1j                # Coherent flux for GD calculation
CfGDMeanOverLmbda = np.zeros([NT, NINmes])*1j       # GD coherent flux
SquaredCoherenceDegree = np.zeros([NT, MW, NINmes]) # Estimated coherence degree \    
VisibilityEstimated = np.zeros([NT, MW, NINmes])*1j # Estimated fringe visibility \    
VisibilityTrue = np.zeros([NT, MW, NIN])*1j         # True expected fringe visibility \    
    

"""Closure Phases [NC]"""
ClosurePhaseObject = np.zeros([NW,NC])              # Closure Phase Object
BispectrumGD = np.ones([NT,NC])+0j                  # GD bispectrum
BispectrumPD = np.ones([NT,NC])+0j                  # PD bispectrum
ClosurePhasePD = np.zeros([NT,NC])                  # PD closure phase        
ClosurePhaseGD = np.zeros([NT,NC])                  # GD closure phase


"""OPD-space observables [NIN]""" # All baselines, even non-measured ones.
# The three true OPD quantities (signal, correction and residues)
OPDDisturbance = np.zeros([NT,NIN])                 # OPD-space disturbance
EffectiveOPDMove = np.zeros([NT+latency,NIN])       # Effective move of the delay lines in OPD-space
OPDTrue = np.zeros([NT,NIN])                        # True Optical Path Delay
reconstructedOPD = np.zeros([NT,NIN])               # OPD estimated from gd and pd measurement (for open-loop estimation)

# Commands
OPDCommand = np.zeros([NT+1,NIN])                   # OPD-space command ODL
OPDCommandRelock = np.zeros([NT+1,NIN])             # RELOCK command projected in the OPD-space (verify NIN or NINmes ?)

# Astrophysical Object
OPDrefObject = np.zeros([NIN])                      # PD reference of the Object (only for analysis)

"""OPD-space observables [NINmes]""" # Only baselines where FS makes measurement
# Estimations
PDEstimated = np.zeros([NT,NINmes])                 # Estimated baselines PD [rad]
PDEstimated2 = np.zeros([NT,NINmes])                # Estimated baselines PD after patch [rad]
GDEstimated = np.zeros([NT,NINmes])                 # Estimated baselines GD [rad]
GDEstimated2 = np.zeros([NT,NINmes])                # Estimated baselines GD after patch [rad]
PDResidual = np.zeros([NT,NINmes])                  # Estimated residual PD = PD-PDref after Ipd (eq.35)
PDResidual2 = np.zeros([NT,NINmes])                 # Estimated residual PD = PD-PDref
GDResidual = np.zeros([NT,NINmes])                  # Estimated residual GD = GD-GDref after Igd (eq.35)
GDResidual2 = np.zeros([NT,NINmes])                 # Estimated residual GD = GD-GDref
GDErr = np.zeros([NT,NINmes])                       # Error that integrates GD integrator
PDref = np.zeros([NT,NINmes])                       # PD reference vector
GDref = np.zeros([NT,NINmes])                       # GD reference vector
CfPDref = np.ones([NT,NINmes])+0j                   # Phasor of the PDref vector, not used currently
CfGDref = np.ones([NT,NINmes])+0j                   # Phasor of the GDref vector, not used currently

# Noise estimations
varPD = np.zeros([NT,NINmes])                       # Estimated "PD variance" = 1/SNR²
varGD = np.zeros([NT,NINmes])                       # Estimated "GD variance" = 1/SNR²
varGDnew = np.zeros([NT,NINmes])                    # Estimated "GD variance" (squared numerator)

SquaredSNRMovingAverage = np.zeros([NT,NINmes])     # Estimated SNR² averaged over N dit
SquaredSNR = np.zeros([NT,NINmes])                  # Estimated instantaneous SNR²
TrackedBaselines = np.zeros([NT,NINmes])
TemporalVariancePD = np.zeros([NT,NINmes])          # Temporal Variance PD estimator
TemporalVarianceGD = np.zeros([NT,NINmes])          # Temporal Variance GD estimator

# Commands 
PDCommand = np.zeros([NT+1,NINmes])                 # OPD-space PD command
GDCommand = np.zeros([NT+1,NINmes])                 # OPD-space GD command
GDCommandIntegrator = np.zeros([NT+1,NINmes])       # OPD-space GD command before round

# Search state
diffOffsets = np.zeros([NT,NINmes])                 # Differential offsets (p1-p2) where the fringes are found
diffOffsets_best = np.zeros([NT,NINmes])
SearchSNR = np.zeros([NT,NINmes])                   # Differential offsets (p1-p2) where the fringes are found
snrEvolution = [[]]*NINmes                          # Will be reshaped (increased) by SEARCH state
offsetsEvolution = [[]]*NINmes                      # Will be reshaped (increased) by SEARCH state
globalMaximumSnr = np.zeros([NT,NINmes])
globalMaximumOffset = np.zeros([NT,NINmes])
secondMaximumSnr = np.zeros([NT,NINmes])

"""Piston-space observables [NA]"""
# The three true piston quantities (signal, correction and residues)
PistonDisturbance = np.zeros([NT,NA])               # Piston disturbances
EffectiveMoveODL = np.zeros([NT+latency,NA])        # Effective move of the delay lines
PistonTrue = np.zeros([NT,NA])                      # True Pistons

# Estimations
GDPistonResidual = np.zeros([NT,NA])                # GD residuals on telescopes
PDPistonResidual = np.zeros([NT,NA])                # PD residuals on telescopes

# Noise
SNRPhotometry = np.zeros([NT,NA])                   # SNR of the photometry estimation

# Commands
PistonGDCommand_beforeround = np.zeros([NT+1,NA])
PistonPDCommand = np.zeros([NT+1,NA])               # Piston-space PD command
PistonGDCommand = np.zeros([NT+1,NA])               # Piston-space GD command
CommandRelock = np.zeros([NT+1,NA])                 # Piston-space RELOCK command
CommandSearch = np.zeros([NT+1,NA])                 # Piston-space SEARCH command
CommandODL = np.zeros([NT+1,NA])                    # Delay lines positionnal command calculated at time it
SearchCommand = np.zeros([NT+1,NA])
PistonGDcorr = np.zeros([NT,NA])

# Photometries
TransmissionDisturbance = np.ones([NT,NW,NA])       # Transmission disturbance of the telescopes
PhotometryDisturbance = np.zeros([NT,NW,NA])        # Resulting photometry disturbances (scaled by the object photometry)
PhotometryEstimated = np.zeros([NT,MW,NA])          # Estimated photometries

# Other observables
# DisturbancePSD = np.zeros([NT])                   # Power-Spectral Distribution of
#                                                   # one disturbance
# FreqSampling = np.zeros([])                       # Frequency sampling           
# DisturbanceFilter = np.zeros([])                  # Disturbance PSD filter

"""Detector"""
MacroImages = np.zeros([NT,MW,FS['NP']])            # Contains microtime images sumed up on macro-wl 


""" FT state-machine quantities"""
FTmode = np.ones([NT])                              # State mode of the Fringe Tracker
                                                    # 0: off, 1: RELOCK, 2: TRACK, 3: SEARCH
GainPD = np.zeros([NT])                             # Gain PD in real time, usually constant
GainGD = np.zeros([NT])                             # Gain GD in real time, usually constant
Ipd = np.ones([NT,NINmes,NINmes])                   # Weighting matrix PD command
Igd = np.ones([NT,NINmes,NINmes])                   # Weighting matrix GD command
Igdna = np.ones([NT,NA,NA])                         # Matrice with ones everywhere
singularValuesSqrt = np.zeros([NT,NA-1])            # Squared Root of Singular values of Igd
IgdRank =np.ones([NT])                              # Rank of the weighting matrix GD
time_since_loss=np.zeros(NT)                        # Time since the loss of one telescope
NoPhotometryFiltration = np.zeros([NT,NA,NA])       # Matrix that filters telescopes which have no photometry
LostTelescopes = np.zeros([NT,NA])                  # Lost telescopes
LostBaselines = np.zeros([NT,NINmes])               # Lost baselines
noFlux = np.zeros([NT,NA])                          # No flux on telescopes
noSignalOnTel = np.zeros([NT,NA])                   # No flux on telescopes since Nframes
PhotometryAverage = np.zeros([NT,NA])               # Moving Average photometry (for photometry loss assumption)
Is = np.ones([NT,NA,NA])
rankIs = np.zeros(NT)
Ws = np.zeros([NT,NINmes])



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

"""Additional quantities for debugging"""
CovarianceReal = np.zeros([NT,MW,NBmes])            # Covariances of the real part of the coherent flux
CovarianceImag = np.zeros([NT,MW,NBmes])            # Covariances of the imaginary part of the coherent flux
Covariance = np.zeros([NT,MW,NBmes,NBmes])          # Covariances of the real and imaginary parts of the coherent flux
BiasModCf = np.zeros([NT,MW,NINmes])                # Bias on the estimator of the module of the coherent flux.
varFlux = np.zeros([NT,MW,FS['NP']])                # Variance Flux

SquaredSnrPD = np.zeros([NT,NINmes])                # Estimated instantaneous SNR PD
SquaredSnrGD = np.zeros([NT,NINmes])                # Estimated instantaneous SNR GD
SquaredSnrGDUnbiased = np.zeros([NT,NINmes])                # Estimated instantaneous SNR GD unbiased

SquaredSNRMovingAveragePD = np.zeros([NT,NINmes])           # Averaged estimated SNR PD
SquaredSNRMovingAverageGD = np.zeros([NT,NINmes])           # Averaged estimated SNR GD

SquaredSNRMovingAverageGDUnbiased = np.zeros([NT,NINmes])
SquaredSNRMovingAverageGDnew = np.zeros([NT,NINmes])
varPDnum = np.zeros([NT,NINmes])
varPDnum2 = np.zeros([NT,NINmes])
varPDdenom = np.zeros([NT,NINmes])
varGDdenom = np.zeros([NT,NINmes])
varGDdenomUnbiased = np.zeros([NT,NINmes])
varPD = np.zeros([NT,NINmes])
varGDUnbiased = np.zeros([NT,NINmes])
varNum2 = np.zeros([NT,MW,NINmes])
LossDueToInjection = np.zeros(NT)
TelescopesThatNeedARestart = np.zeros([NT,NA])
eps = np.zeros([NT,NA])
it_last = np.zeros([NT,NA])
last_usaw = np.zeros([NT,NA])

del NA,NB,NC,NT,dt,MW,NW,NIN,FS

    
    
