# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:07:00 2020

@author: cpannetier

CONFIG MODULE

This module contains all parameters of the configuration, initialised with default
parameters and set by the user.    

So far it contains two classes I want to use in my simulation.
Defined classes so far:
    - Observation: Informations about the observation we want to simulate.
    - ScienceObject: Store every parameters relative to the object.

Classes truly used classes in the algorithm:
    - Observation
    - ScienceObject
    
"""

import numpy as np


# class FringeTracker:
#     """
#     Fringe Tracker configuration:
#         - FringeSensor class: defines the configuration of the Fringe Sensor
#         and so the resolution, number of pixels (image format) and calibrated 
#         P2VM (image demodulation).
#         - PistonCalculator class: defines the configuration of the Piston 
#         Calculator. It is the system core, containing for example:
#             - the State Machine variable: state
#             - the PD, GD and ClosurePhase integration times
#     """
    
#     class FringeSensor:
#         """
#         Fringe Sensor parameters:
#             - Name: [string] Name of the fringe sensor method corresponding 
#             simulating the Fringe Sensor.
#             - V2PM: [NWxNPxNB array] Matrix Visibility to Pixel. Initialized to ones
#             array but will be automatically redefined during coh_init method.
#             - V2PM: [NWxNBxNP array] Matrix Pixel to Visibility. Initialized to ones
#             array but will be automatically redefined during coh_init method.
#             - MacroP2VM: [MWxNBxNP array] Macro Matrix Pixel to Visibility. Initialized to ones
#             array but will be automatically redefined during coh_init method.
#             - NP: [int] Number of pixels per spectral channel on the detector
#             - DIT: [int] Integration time of the Fringe Sensor in [ms]
#         """
    
#         def __init__(self):
#             self.name='default'                   # Default fringe sensor (one coherence = 2 pixels)
#             self.V2PM=np.ones([NW,NP,NB])       # Matrix Visibility to Pixel, will be created by the FS chosen above
#             self.P2VM=np.ones([NW,NB,NP])       # Matrix Pixel to Visibility, will be created by the FS chosen above
#             self.MacroP2VM=np.ones([MW,NB,NP])
#             self.NP = 4                         # Number of pixels
#             self.DIT = 3                        # Integration Time

    
#     class PistonCalculator:
#         """
#         Piston Calculator parameters:
#             - Name: [string] defines the calculator used by the simulator
#             - state: [integer] Statemachine starting mode (0=idle,1=search,2=track)
#             - GainPD: [float] PD gain
#             - GainGD: [float] GD gain
#             - Piston2OPD: [NBxNA array] Transfert Matrix from Pistons to OPDs
#             - OPD2Piston: [NBxNA array] Transfert Matrix from OPDs to Pistons
#             - Ngd: [int] Number of frames accumulation of the Group-Delay
#             - Ncp: [int] # Number of frames accumulation of the Closure Phase
            
#         """
        
#         def __init__(self,**kwargs):
            
#             self.Name = 'SPICA'
#             self.State = 0
#             self.GainGD = 0.7
#             self.GainPD = 0.3
#             self.Piston2OPD = np.zeros([NIN,NA])    # Piston to OPD matrix
#             for ia in range(NA):
#                 for iap in range(ia+1,NA):
#                     k = int(ia*NA-ia*(ia+3)/2+iap-1)
#                     self.Piston2OPD[k,ia] = 1
#                     self.Piston2OPD[k,iap] = -1    
#             self.OPD2Piston = np.linalg.pinv(self.Piston2OPD)   # OPD to pistons matrix
#             self.Ngd = 40       # Number of frames accumulation of the Group-Delay
#             self.Ncp = 300     # Number of frames accumulation of the Closure Phase
#             self.Sweep = 100    # Time before changing direction
#             self.Slope = 100*1e-3*dt        # Searching velocity [µm/frame]
#             self.Vfactors = np.array([0, -10,-9,-6,2,7])/10            # Non redundant SPICA
#                     # Fringe sensor parameters (should be defined by Fringe sensor itself)
#             self.Ncross = 1             # Distance between spectral channels for GD calc
#             self.R = np.abs((MW-1)*PDspectra/(spectraM[-1] - spectraM[0]))
            
#             for key in kwargs.keys():
#                 setattr(self, key, kwargs[key])



# class Interferometer:
#     """
#     Interferometer parameters
#     """
    
#     def __init__(self):
#         self.Name = 'CHARA'
#         self.NA = 2
#         self.NB=NA**2                # total number of baselines
#         self.NIN=int(NA*(NA-1)*2)    # Number of independant interferometric coherences
#         self.NC = int((NA-2)*(NA-1))   # Number of independant closure phases


# class Source:
#     """
#     Source parameters
#     """
    
#     def __init__(self):
#         self.spectra=np.linspace(1.5,1.75,NW)      # Micro-sampling wavelength
#         self.spectraM=spectra                    # Macro-sampling wavelength
#         self.PDspectra=np.mean(spectra)          # Mean wavelength
#         self.spectrum=np.ones_like(spectra)      # Source distribution spectral power
#         self.CfObj=np.ones(4)


class Observation():
    """
    Informations about the observation we want to simulate.
    Attributes:
        - ArrayName: STRING
            Array name
            Init: 'CHARA'
        - Date:
            Date of the observation
            Init: '2020-01-01 00:00:00'
        - AltAz: TUPLE
            Star Coordinates at the observation date.
            Init: (90,0) -> Zenith
        - UsedTelescopes:
            Names of the used telescopes during the observation.
            Init: ['E1','E2','S1','S2','W1','W2']
        
    """
    def __init__(self,Filepath='', ArrayName='CHARA', Date='2020-01-01 00:00:00',
                 AltAz=(90,0),
                 UsedTelescopes = 'all',
                 **kwargs):
        
        self.Filepath = Filepath
        self.ArrayName = ArrayName
        self.AltAz = AltAz
        self.Date = Date
        self.UsedTelescopes = UsedTelescopes
        
        
class ScienceObject():
    """
    Store every parameters relative to the object.
    Attributes:
        - Name: STRING
            If existing star, the name of the star.
            If not existing star, the name of the type (for information):
                Binary
                Simple
        - Filename: STRING
            If a filename already gets the visibilities, it saves time.
        - Coordinates: Astropy.SkyCoord
            In degrees, store the right ascension and declination of the object.
        - Star1: DICTIONARY
            - Angular diameter: of the star in arcsec
            - Position: of the star in the (alpha,beta) plane
            - Hmag: FLOAT
                Magnitude in H band.
            - SpectralType: STRING
            - T: FLOAT
                Temperature of the star
            Init:{'AngDiameter':0.01, 'Position':[0,0],'Hmag':0}
                
        - Star2: DICTIONARY
            Possibility to define a second star on the same model as StarOne
            
    """
    
    def __init__(self, Name='Simple'):
        
        self.Name = Name
        self.Star1 = {'AngDiameter':0.01, 'Position':(0,0),'Hmag':0}


# Interferometer
Name = 'CHARA'
NA=6                    # number of apertures
NB=NA**2                # total number of baselines
NIN=int((NA-2)*(NA+1)/2+1)    # Number of independant interferometric coherences
NC = int((NA-2)*(NA-1))   # Number of independant closure phases
NT=512
MT=NT
OT=1
NW=5
MW=5
OW=1

NY=0
ND=0

dyn=0

# Source
spectra=np.linspace(1.5,1.75,NW)      # Micro-sampling wavelength
spectraM=spectra                    # Macro-sampling wavelength
PDspectra=np.mean(spectra)          # Mean wavelength
spectrum=np.ones_like(spectra)      # Source distribution spectral power
CfObj=np.ones(4)


# Fringe sensor
fs='default'                # Default fringe sensor (one coherence = 2 pixels)
NP = 4                      # Number of pixels
V2PM=np.ones([NW,NP,NB])   # Matrix Visibility to Pixel, will be created by the FS chosen above
P2VM=np.ones([NW,NB,NP])   # Matrix Pixel to Visibility, will be created by the FS chosen above
MacroP2VM=np.ones([MW,NB,NP])
dt = 3                      # Frame time
R = np.abs((MW-1)*PDspectra/(spectraM[-1] - spectraM[0]))

# Create the dictionnary of the Fringe Sensor: will be completed by the fringesensor
# Expected keys: NP, V2PM, P2VM, MacroP2VM, R
FS = {'NP':NP}                 


# Noises
noise=True   # No noise by default [0 if noise]
qe = 0.7*np.ones(MW)    # Quantum efficiency
ron = 2     # Readout noise
phnoise=0       # No photon noise by default [1 if ph noise]
enf=1.5     # Excess Noise Factor defined as enf=<M²>/<M>² where M is the avalanche gain distrib
M = 1           # Average Amplification factor eAPD

# Random state Seeds
seedph=42   
seedron=43
seeddist=44
latency = 1                 # Latency in number of frames

# Piston Calculator
# pc='integrator'     # Default command calculator (integrator PD and GD without statemachine)
state=0              # Statemachine starting mode (0=idle,1=search,2=track)

GainGD = 0.7
GainPD = 0.3
Ngd = 40
Ncp = 150
Sweep = 100
Slope = Sweep*dt*1e-3
Vfactor = np.array([0, -10,-9,-6,2,7])/10    # Non redundant SPICA
Ncross = 1
usePDref = True

# Piston Calculator dictionnary
FT = {'Name':'integrator',
      'GainPD':0.5}              # Distance between spectral channels for GD calc
    

# Simulation parameters
filename='cohdefault'
ich=np.array([NP,2])        # For display only: bases correspondances on pixels
TELref=0            # For display only: reference delay line
newfig=0
timestamp = np.arange(NT)*dt       # Timestamps in [ms]



# Tests with classes
# PC = FringeTracker.PistonCalculator()
# FS = FringeTracker.FringeSensor()

        