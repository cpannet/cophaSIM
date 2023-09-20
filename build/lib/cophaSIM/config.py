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

import matplotlib.pyplot as plt
import numpy as np
import os,pkg_resources
from scipy.special import binom
from astropy.io import fits
from . import coh_tools as ct
from . import tol_colors as tc

colors=tc.tol_cset('muted')

"""
ALL THESE VALUES ARE BY DEFAULT AND WILL BE UPDATED WITH THE FUNCTION INITIALISE
"""

"""Basic parameters"""
Name = 'CHARA'
NA=6                            # number of apertures
NB=NA**2                        # Number of unknown photometries and phases 
NIN=int(NA*(NA-1)/2)            # Number of independant OPDs
NC = int(binom(NA,3))           # Number of closure phases
ND = int((NA-1)*(NA-2)/2)       # Number of independant closure phases
NT=512
MT=NT
NW=5
OW=1
dt = 3                              # Frame time
timestamps = np.arange(NT)*dt       # Timestamps in [ms]
starttracking = 0

"""Source"""
spectra=np.linspace(1.45,1.75,NW)       # Micro-sampling wavelength
spectraM=spectra                        # Macro-sampling wavelength
wlOfTrack=np.mean(spectra)              # Mean wavelength
spectrum=np.ones_like(spectra)          # Source distribution spectral power


"""Fringe Sensor dictionnary"""
# will be completed by the fringesensor method.
FS = {'NP':4*NIN, 'MW':NW, 'NINmes':NIN,'NBmes':NB, 'NCmes':NC}

"""Noises"""
noise=True      # No noise by default [0 if noise]
qe = 0.7        # Quantum efficiency
ron = 2         # Readout noise
phnoise=0       # No photon noise by default [1 if ph noise]
enf=1.5         # Excess Noise Factor defined as enf=<M²>/<M>² where M is the avalanche gain distrib
M = 1           # Average Amplification factor eAPD

"""Random state Seeds"""
seedph=42   
seedron=43
seeddist=44
latency = 1                 # Latency in number of frames

"""Fringe-tracker"""
# will be completed by the fringe-tracker method.
FT = {'ThresholdGD':np.zeros(NIN), 'ThresholdPD':0}
    

"""Other parameters"""
TELref=0            # For display only: reference delay line
newfig=0

verbose,vebose2=False,False


class Interferometer():
    """
    Interferometer parameters
    """
    
    def __init__(self,name='chara'):
        
        self.get_array(name=name)
        
    def get_array(self,name=''):
        
        if "fits" in name:
            filepath = name
            if not os.path.exists(filepath):
                try:
                    if verbose:
                        print("Looking for the interferometer file into the package's data")
                    filepath = pkg_resources.resource_stream(__name__,filepath)
                except:
                    raise Exception(f"{filepath} doesn't exist.")
        
        elif name == 'chara':
            if verbose:
                print("Take CHARA 6T information")
            
            filepath = pkg_resources.resource_stream(__name__,'data/interferometers/CHARA_6T.fits')
                       
        else:
            raise Exception("For defining the array, you must give a file \
    or a name (currently only the name CHARA is available).")

        with fits.open(filepath) as hdu:
            ArrayParams = hdu[0].header
            NA, NIN = ArrayParams['NA'], ArrayParams['NIN']
            self.name = ArrayParams['NAME']
            self.NA = NA
            self.NIN = NIN
            TelData = hdu[1].data
            BaseData = hdu[2].data
        
            self.TelNames = TelData['TelNames']
            self.telNameLength = 2
            self.TelCoordinates = TelData['TelCoordinates']
            self.TelTransmissions = TelData['TelTransmissions']
            self.TelSurfaces = TelData['TelSurfaces']
            self.BaseNames = BaseData['BaseNames']
            self.BaseCoordinates = BaseData['BaseCoordinates']
            
        self.BaseNorms = np.linalg.norm(self.BaseCoordinates[:,:2],axis=1)
        
        
    def plot(self):
        
        plt.rcParams['figure.figsize']=(8,8)
        font = {'family' : 'DejaVu Sans',
                'weight' : 'normal',
                'size'   : 15}
        
        rcParamsFS = {"axes.grid":False,
                        "figure.constrained_layout.use": True,
                        'figure.subplot.hspace': 0,
                        'figure.subplot.wspace': 0,
                        'figure.subplot.left':0,
                        'figure.subplot.right':1
                        }
        plt.rcParams.update(rcParamsFS)
        plt.rc('font', **font)
        
        title=self.name
        fig=plt.figure(title, clear=True)
        ax=fig.subplots()
        for ia in range(NA):
            name1,(x1,y1) = self.TelNames[ia],self.TelCoordinates[ia,:2]
            for iap in range(ia+1,NA):
                ib=ct.posk(ia,iap,NA)
                x2,y2 = self.TelCoordinates[iap,:2]
                ax.plot([x1,(x2+x1)/2],[y1,(y2+y1)/2],color=colors[0],linestyle='-',linewidth=10,zorder=-1)
                ax.plot([(x2+x1)/2,x2],[(y2+y1)/2,y2],color=colors[0],linestyle='-',linewidth=10,zorder=-1)
                ax.annotate(f"{round(self.BaseNorms[ib])}", ((x1+x2)/2-3,(y1+y2)/2-3),color=colors[1])
        
        for ia in range(NA):
            name1,(x1,y1) = self.TelNames[ia],self.TelCoordinates[ia,:2]
            ax.scatter(x1,y1,marker='o',edgecolor="k",facecolor='None',linewidth=15,s=8)
            ax.annotate(name1, (x1-20,y1-15),color="k")
            ax.annotate(f"({ia+1})", (x1+5,y1-10),color=colors[3])
            
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        
        xwidth = np.ptp(self.TelCoordinates[:,0])*1.2
        minX = np.min(self.TelCoordinates[:,0])
        xmin = minX - 0.1*xwidth
        
        ywidth = np.ptp(self.TelCoordinates[:,1])*1.2
        minY = np.min(self.TelCoordinates[:,1])
        ymin = minY - 0.1*ywidth
        
        xmax,ymax  = xmin+xwidth , ymin+ywidth
        ax.set_xlim([xmin,xmax]) ; ax.set_ylim([ymin,ymax])
        
        #ax.text(xmin+50,ymax-50,name,fontsize=60,color='w')
        #ax.text(0.5*xmin,0,"Planet Formation Imager",fontsize=50,color='w')
        
        # if len(savedir):
        #     if not os.path.exists(savedir):
        #         os.makedirs(savedir, exist_ok=True)
            
        #     ax.axis("off")
        #     if not len(self.name):
        #         name = "test"

        #     if isinstance(ext, list):
        #         for exttemp in ext:
        #             if exttemp == 'png':
        #                 fig.savefig(f"{savedir}{name}.{exttemp}",transparent=True)
        #             else:
        #                 fig.savefig(f"{savedir}{name}.{exttemp}")
        #     else:
        #         if ext == 'png':
        #             fig.savefig(f"{savedir}{name}.{ext}",transparent=True)
        #         else:
        #             fig.savefig(f"{savedir}{name}.{ext}")
    
        plt.rcParams.update(plt.rcParamsDefault)

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
            Init: ['S1','S2','E1','E2','W1','W2']
        
    """
    def __init__(self,Filepath='', ArrayName='CHARA', Date='2020-01-01 00:00:00',
                 AltAz=(90,0),
                 UsedTelescopes = ['S1','S2','E1','E2','W1','W2'],
                 **kwargs):
        
        self.Filepath = Filepath
        self.ArrayName = ArrayName
        if AltAz != 'no':
            self.AltAz = AltAz
        self.DATE = Date
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
#             self.R = np.abs((MW-1)*wlOfTrack/(spectraM[-1] - spectraM[0]))
            
#             for key in kwargs.keys():
#                 setattr(self, key, kwargs[key])



        
# class Source:
#     """
#     Source parameters
#     """
    
#     def __init__(self):
#         self.spectra=np.linspace(1.5,1.75,NW)      # Micro-sampling wavelength
#         self.spectraM=spectra                    # Macro-sampling wavelength
#         self.wlOfTrack=np.mean(spectra)          # Mean wavelength
#         self.spectrum=np.ones_like(spectra)      # Source distribution spectral power
#         self.CfObj=np.ones(4)




