# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:21:16 2020

@author: cpannetier

GENERATES AN OBJECT IRRADIANCE ACCORDING TO PARAMETERS:
    - 
In this routine, you define the parameters of the interferometer and object.
It calculates the resulting complex visibilities on all baselines for a given 
spectra.

create_star function stores the visibility and relative informations in a FITS file.
Then, the routine reads it so that we can see what has been stored.


OBSERVATION DEFINITION
Define the interferometer and the observation parameters:
    - Interferometer File
    - Target Alt-Azimut [deg]
    - Date ['YYYY-MM-DD HH:mm:ss']

TARGET DEFINITION
Define the target geometry and photometry. If "Name"=Unresolved, the visibility
is automatically put to 1.
For that, define as many attribute "Star" as necessary. "Star" must be a 
dictionnary with the following keys: 
    - Hmag: H magnitude
    - AngDiameter - [mas]
    - Position on the optical axis (if two object) - [mas]
    - T: [K] Star temperature. OPTIONAL. It fits a Black-Body of the 
    temperature T


"""

# Define the workspace as being the coh_pack main workspace
import os

import numpy as np

from coh_lib.coh_tools import generate_spectra, create_obsfile
from coh_lib.config import ScienceObject, Observation


# Spectral Sampling
lmbda1 = 1.2 ; lmbda2 = 1.8 ; MW = 5 ; OW = 10
spectra, sigma = generate_spectra(lmbda1, lmbda2, OW=OW, MW=MW, mode='linear_sig')

datadir = 'C:/Users/cpannetier/Documents/Python_packages/coh_lib/coh_lib/data/'

Obs = Observation(ArrayName='CHARA',
                  Filepath = datadir+'interferometers/CHARAinterferometerH.fits',
                  AltAz=(90,0))

Target = ScienceObject('Binary')

mags = [0]#range(10)#np.round(9-np.logspace(0,np.log10(9), num=20),2)
for mag in mags:
    Target.Star1 = {'Position':(0,0),'AngDiameter':0.05,'Hmag':mag}
    Target.Name = 'Manual'
    
    import random

    # list of random float between a range 50.50 to 500.50
    randomFloatList = []
    # Set a length of the list to 10
    for i in range(15):
        # any random float between 50.50 to 500.50
        # don't use round() if you need number as it is
        x = round(random.uniform(0, np.pi), 2)
        randomFloatList.append(x)
    
    print("Printing list of 10 random float numbers")
    print(randomFloatList)
    
    Target.Phases = np.array(randomFloatList)
    # If second object:
    # Target.Star2 = {'Position':(0,-0.3),'AngDiameter':0.05,'Hmag':1}
    
    ObservationFile = datadir+f'observations/CHARA/ManualHigherCP_mag{mag}.fits'
    
    CohIrr, UncohIrr, VisObj,_,_ = create_obsfile(spectra, Obs,Target,
                                                     savingfilepath=ObservationFile,
                                                     display=True,
                                                     overwrite=True) 


