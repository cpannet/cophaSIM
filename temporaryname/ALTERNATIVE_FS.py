# -*- coding: utf-8 -*-
"""
The SPICA fringe sensor measures the coherent flux after simulating the image
given by the real oversampled coherence flux and adding noises on it.

INPUT: oversampled true coherent flux [NW,NB]

OUTPUT: macrosampled measured coherent flux [MW,NB]

Calculated and stored observables:
    - Photometries: simu.PhotometryEstimated [MW,NA]
    - Visibilities: simu.Visibility [MW,NIN]

"""

import numpy as np
import matplotlib.pyplot as plt
import os

from . import coh_tools as ct
from . import config
from .FS_DEFAULT import ABCDmod, realisticABCDmod
from scipy.special import binom

from . import tol_colors as tc
colors=tc.tol_cset('muted')

"""
CHARA - 6 telescopes - S1S2E1E2W1W2
"""

NA=6
d0=(np.ones([NA,NA])-np.identity(NA))/5

d1 = np.array([[0,1/2,0,1/2,0,0],
                  [1/6,0,0,1/3,1/6,1/3],
                  [0,0,0,1/2,0,1/2],
                  [1/6,1/3,1/6,0,0,1/3],
                  [0,1/2,0,0,0,1/2],
                  [0,1/3,1/6,1/3,1/6,0]])

d2 = np.array([[0,1/2,0,1/2,0,0],
                  [1/4,0,0,1/4,1/4,1/4],
                  [0,0,0,1/2,0,1/2],
                  [1/4,1/4,1/4,0,0,1/4],
                  [0,1/2,0,0,0,1/2],
                  [0,1/4,1/4,1/4,1/4,0]])

d3 = np.array([[0,1/3,1/3,0,1/3,0],
                  [1/3,0,0,1/3,0,1/3],
                  [1/3,0,0,1/3,1/3,0],
                  [0,1/3,1/3,0,0,1/3],
                  [1/3,0,1/3,0,0,1/3],
                  [0,1/3,0,1/3,1/3,0]])

d4 = np.array([[0,1/3,0,1/3,0,1/3],
               [1/3,0,1/3,0,1/3,0],
               [0,1/3,0,1/3,0,1/3],
               [1/3,0,1/3,0,1/3,0],
               [0,1/3,0,1/3,0,1/3],
               [1/3,0,1/3,0,1/3,0]])

d5 = np.array([[0,0,1/3,1/3,1/3,0],
              [0,0,0,1/3,1/3,1/3],
              [1/3,0,0,0,1/3,1/3],
              [1/3,1/3,0,0,0,1/3],
              [1/3,1/3,1/3,0,0,0],
              [0,1/3,1/3,1/3,0,0]])

d6 = np.array([[0,1,0,0,0,0],
              [1/3,0,0,1/3,0,1/3],
              [0,0,0,1,0,0],
              [0,1/3,1/3,0,0,1/3],
              [0,0,0,0,0,1],
              [0,1/3,0,1/3,1/3,0]])

d7 = np.array([[0,1,0,0,0,1],
              [1,0,1,0,0,0],
              [0,1,0,1,0,0],
              [0,0,1,0,1,0],
              [0,0,0,1,0,1],
              [1,0,0,0,1,0]])/2

d8 = np.array([[0,1,0,0,0,0],
              [1/2,0,0,0,0,1/2],
              [0,0,0,1,0,0],
              [0,0,1/2,0,1/2,0],
              [0,0,0,1/2,0,1/2],
              [0,1/2,0,0,1/2,0]])

d9 = np.array([[0,1/2,1/2,0,0,0],
              [1/2,0,0,0,0,1/2],
              [1/2,0,0,1/2,0,0],
              [0,0,1/2,0,1/2,0],
              [0,0,0,1,0,0],
              [0,1,0,0,0,0]])

"""
CHARA 7T - 7 télescopes S1S2E1E2W1W2B0
"""

NA=7
pw7_21_35 = np.ones([NA,NA])/(NA-1)

pw7_6_0 = np.array([[0,1,0,0,0,0,0],
                    [1/2,0,0,0,0,0,1/2],
                    [0,0,0,1,0,0,0],
                    [0,0,1/2,0,0,0,1/2],
                    [0,0,0,0,0,1,0],
                    [0,0,0,0,1/2,0,1/2],
                    [0,1/3,0,1/3,0,1/3,0]])

pw7_7_0 = np.array([[0,1,0,0,0,0,0],
                    [1/3,0,0,0,0,1/3,1/3],
                    [0,0,0,1,0,0,0],
                    [0,0,1/3,0,0,1/3,1/3],
                    [0,0,0,0,0,1,0],
                    [0,1/3,0,1/3,1/3,0,0],
                    [0,1/2,0,1/2,0,0,0]])


pw7_8_2 = np.array([[0,1,0,0,0,0,0],
                    [1/3,0,0,0,0,1/3,1/3],
                    [0,0,0,1,0,0,0],
                    [0,0,1/3,0,0,1/3,1/3],
                    [0,0,0,0,0,1,0],
                    [0,1/4,0,1/4,1/4,0,1/4],
                    [0,1/3,0,1/3,0,1/3,0]])


pw7_9_3 = np.array([[0,1/2,0,0,0,0,1/2],
                    [1/2,0,0,0,0,0,1/2],
                    [0,0,0,1/2,0,0,1/2],
                    [0,0,1/2,0,0,0,1/2],
                    [0,0,0,0,0,1/2,1/2],
                    [0,0,0,0,1/2,0,1/2],
                    [1/6,1/6,1/6,1/6,1/6,1/6,0]])

# suitable for CHARA_7T_balanced - S1S2E1E2W1W2B0
pw7_11_2 = np.array([[0,1/3,0,0,1/3,0,1/3],
                    [1/3,0,0,0,0,1/3,1/3],
                    [0,0,0,1/3,1/3,0,1/3],
                    [0,0,1/3,0,0,1/3,1/3],
                    [1/3,0,1/3,0,0,1/3,0],
                    [0,1/3,0,1/3,1/3,0,0],
                    [1/4,1/4,1/4,1/4,0,0,0]])

pw7_12_6 = np.array([[0,1/2,0,0,0,0,1/2],
                    [1/4,0,0,0,1/4,1/4,1/4],
                    [0,0,0,1/2,0,0,1/2],
                    [0,0,1/4,0,1/4,1/4,1/4],
                    [0,1/3,0,1/3,0,1/3,0],
                    [0,1/4,0,1/4,1/4,0,1/4],
                    [1/5,1/5,1/5,1/5,0,1/5,0]])

pw7_12_7 = np.array([[0,1/2,0,0,0,0,1/2],
                    [1/4,0,0,1/4,0,1/4,1/4],
                    [0,0,0,1/2,0,0,1/2],
                    [0,1/4,1/4,0,0,1/4,1/4],
                    [0,0,0,0,0,1/2,1/2],
                    [0,1/4,0,1/4,1/4,0,1/4],
                    [1/6,1/6,1/6,1/6,1/6,1/6,0]])

pw7_13_6 = np.array([[0,1/3,0,0,0,1/3,1/3],
                    [1/4,0,0,0,1/4,1/4,1/4],
                    [0,0,0,1/3,0,1/3,1/3],
                    [0,0,1/4,0,1/4,1/4,1/4],
                    [0,1/3,0,1/3,0,1/3,0],
                    [1/5,1/5,1/5,1/5,1/5,0,0],
                    [1/4,1/4,1/4,1/4,0,0,0]])

pw7_14_10 = np.array([[0,1/3,0,0,0,1/3,1/3],
                    [1/4,0,0,0,1/4,1/4,1/4],
                    [0,0,0,1/3,0,1/3,1/3],
                    [0,0,1/4,0,1/4,1/4,1/4],
                    [0,1/3,0,1/3,0,1/3,0],
                    [1/6,1/6,1/6,1/6,1/6,0,1/6],
                    [1/5,1/5,1/5,1/5,0,1/5,0]])

pw7_16_14 = np.array([[0,1/4,0,0,1/4,1/4,1/4],
                    [1/4,0,0,0,1/4,1/4,1/4],
                    [0,0,0,1/4,1/4,1/4,1/4],
                    [0,0,1/4,0,1/4,1/4,1/4],
                    [1/5,1/5,1/5,1/5,0,1/5,0],
                    [1/6,1/6,1/6,1/6,1/6,0,1/6],
                    [1/5,1/5,1/5,1/5,0,1/5,0]])

"""
MROI - 10 telescopes - W0W1W2W3S1S2S3N1N2N3
"""
pw10_9_0 = np.array([[0,1/3,0,0,1/3,0,0,1/3,0,0],
               [1/2,0,1/2,0,0,0,0,0,0,0],
               [0,1/2,0,1/2,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0,0,0],
               [1/2,0,0,0,0,1/2,0,0,0,0],
               [0,0,0,0,1/2,0,1/2,0,0,0],
               [0,0,0,0,0,1,0,0,0,0],
               [1/2,0,0,0,0,0,0,0,1/2,0],
               [0,0,0,0,0,0,0,1/2,0,1/2],
               [0,0,0,0,0,0,0,0,1,0]])

# For homogeneous MROI
pw10_12_4 = np.array([[0,1/3,0,0,1/3,0,0,1/3,0,0],
               [1/4,0,1/4,0,1/4,0,0,1/4,0,0],
               [0,1/2,0,1/2,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0,0,0],
               [1/4,1/4,0,0,0,1/4,0,1/4,0,0],
               [0,0,0,0,1/2,0,1/2,0,0,0],
               [0,0,0,0,0,1,0,0,0,0],
               [1/4,1/4,0,0,1/4,0,0,0,1/4,0],
               [0,0,0,0,0,0,0,1/2,0,1/2],
               [0,0,0,0,0,0,0,0,1,0]])

# For dense MROI
pw10_15_0 = np.array([[0,1/3,0,0,1/3,0,0,1/3,0,0],
               [1/4,0,1/4,0,0,1/4,0,0,1/4,0],
               [0,1/4,0,1/4,1/4,0,0,1/4,0,0],
               [0,0,1,0,0,0,0,0,0,0],
               [1/4,0,1/4,0,0,1/4,0,0,1/4,0],
               [0,1/4,0,0,1/4,0,1/4,1/4,0,0],
               [0,0,0,0,0,1,0,0,0,0],
               [1/4,0,1/4,0,0,1/4,0,0,1/4,0],
               [0,1/4,0,0,1/4,0,0,1/4,0,1/4],
               [0,0,0,0,0,0,0,0,1,0]])

pw10_18_13 = np.array([[0,1/3,0,0,1/3,0,0,1/3,0,0],
               [1/6,0,1/6,0,1/6,1/6,0,1/6,1/6,0],
               [0,1/4,0,1/4,1/4,0,0,1/4,0,0],
               [0,0,1,0,0,0,0,0,0,0],
               [1/6,1/6,1/6,0,0,1/6,0,1/6,1/6,0],
               [0,1/4,0,0,1/4,0,1/4,1/4,0,0],
               [0,0,0,0,0,1,0,0,0,0],
               [1/6,1/6,1/6,0,1/6,1/6,0,0,1/6,0],
               [0,1/4,0,0,1/4,0,0,1/4,0,1/4],
               [0,0,0,0,0,0,0,0,1,0]])

pw10_15_5 = np.array([[0,1/3,0,0,1/3,0,0,1/3,0,0],
               [1/4,0,1/4,0,1/4,0,0,1/4,0,0],
               [0,1/4,0,1/4,0,1/4,0,0,1/4,0],
               [0,0,1,0,0,0,0,0,0,0],
               [1/4,1/4,0,0,0,1/4,0,1/4,0,0],
               [0,0,1/4,0,1/4,0,1/4,0,1/4,0],
               [0,0,0,0,0,1,0,0,0,0],
               [1/4,1/4,0,0,1/4,0,0,0,1/4,0],
               [0,0,1/4,0,0,1/4,0,1/4,0,1/4],
               [0,0,0,0,0,0,0,0,1,0]])

pw10_18_11 = np.array([[0,1/6,1/6,0,1/6,1/6,0,1/6,1/6,0],
               [1/4,0,1/4,0,1/4,0,0,1/4,0,0],
               [1/5,1/5,0,1/5,0,1/5,0,0,1/5,0],
               [0,0,1,0,0,0,0,0,0,0],
               [1/4,1/4,0,0,0,1/4,0,1/4,0,0],
               [1/5,0,1/5,0,1/5,0,1/5,0,1/5,0],
               [0,0,0,0,0,1,0,0,0,0],
               [1/4,1/4,0,0,1/4,0,0,0,1/4,0],
               [1/5,0,1/5,0,0,1/5,0,1/5,0,1/5],
               [0,0,0,0,0,0,0,0,1,0]])

pw10_18_6 = np.array([[0,1/3,0,0,1/3,0,0,1/3,0,0],
               [1/4,0,1/4,0,1/4,0,0,1/4,0,0],
               [0,1/4,0,1/4,0,1/4,0,0,1/4,0],
               [0,0,1/3,0,0,0,1/3,0,0,1/3],
               [1/4,1/4,0,0,0,1/4,0,1/4,0,0],
               [0,0,1/4,0,1/4,0,1/4,0,1/4,0],
               [0,0,0,1/3,0,1/3,0,0,0,1/3],
               [1/4,1/4,0,0,1/4,0,0,0,1/4,0],
               [0,0,1/4,0,0,1/4,0,1/4,0,1/4],
               [0,0,0,1/3,0,0,1/3,0,1/3,0]])

pw10_21_14 = np.array([[0,1/6,1/6,0,1/6,1/6,0,1/6,1/6,0],
               [1/5,0,1/5,1/5,1/5,0,0,1/5,0,0],
               [1/5,1/5,0,1/5,0,1/5,0,0,1/5,0],
               [0,1/2,1/2,0,0,0,0,0,0,0],
               [1/5,1/5,0,0,0,1/5,1/5,1/5,0,0],
               [1/5,0,1/5,0,1/5,0,1/5,0,1/5,0],
               [0,0,0,0,1/2,1/2,0,0,0,0],
               [1/5,1/5,0,0,1/5,0,0,0,1/5,1/5],
               [1/5,0,1/5,0,0,1/5,0,1/5,0,1/5],
               [0,0,0,0,0,0,0,1/2,1/2,0]])

pw10_21_12 = np.array([[0,1/6,1/6,0,1/6,1/6,0,1/6,1/6,0],
               [1/4,0,1/4,0,1/4,0,0,1/4,0,0],
               [1/5,1/5,0,1/5,0,1/5,0,0,1/5,0],
               [0,0,1/3,0,0,0,1/3,0,0,1/3],
               [1/4,1/4,0,0,0,1/4,0,1/4,0,0],
               [1/5,0,1/5,0,1/5,0,1/5,0,1/5,0],
               [0,0,0,1/3,0,1/3,0,0,0,1/3],
               [1/4,1/4,0,0,1/4,0,0,0,1/4,0],
               [1/5,0,1/5,0,0,1/5,0,1/5,0,1/5],
               [0,0,0,1/3,0,0,1/3,0,1/3,0]])


NA=10
pw10_45_36 = (np.ones([NA,NA])-np.identity(NA))/(NA-1)


NA=20 # Planet Finder Imager

# Each telescope linked to its closest neighbour
pw20_20 = np.zeros([NA,NA])
for ia in range(NA-1):
    pw20_20[ia,ia+1] = 1/2
    pw20_20[ia,ia-1] = 1/2

pw20_20[NA-1,0] = 1/2
pw20_20[NA-1,NA-2] = 1/2

# Each telescope linked to its four closest neighbours
pw20_40 = np.zeros([NA,NA])
for ia in range(NA-2):
    pw20_40[ia,ia-1] = 1/4
    pw20_40[ia,ia-2] = 1/4
    pw20_40[ia,ia+1] = 1/4
    pw20_40[ia,ia+2] = 1/4

pw20_40[NA-2,NA-3] = 1/4
pw20_40[NA-2,NA-4] = 1/4
pw20_40[NA-2,NA-1] = 1/4
pw20_40[NA-2,0] = 1/4

pw20_40[NA-1,NA-2] = 1/4
pw20_40[NA-1,NA-3] = 1/4
pw20_40[NA-1,0] = 1/4
pw20_40[NA-1,1] = 1/4


descriptions = {"PW6-15-10":d0, "PW6-9-4-1":d1,"PW6-9-4-2":d2,"PW6-9-2":d3,"PW6-9-0":d4, "PW6-9-2-b":d5,
                "PW6-6-1":d6,"PW6-6-0":d7,  "PW6-5-0":d8, "PW6-5-0-0":d9,
                "PW7-6-0":pw7_6_0,"PW7-7-0":pw7_7_0,"PW7-8-2":pw7_8_2,
                "PW7-9-3":pw7_9_3,"PW7-11-2":pw7_11_2,"PW7-12-6":pw7_12_6,
                "PW7-12-7":pw7_12_7,"PW7-13-6":pw7_13_6,"PW7-14-10":pw7_14_10,
                "PW7-16-14":pw7_16_14,"PW7-21-35":pw7_21_35,
                "PW10-9-0":pw10_9_0,"PW10-12-4":pw10_12_4,"PW10-15-0":pw10_15_0,
                "PW10-15-5":pw10_15_5,"PW10-18-13":pw10_18_13, "PW10-18-11":pw10_18_11,
                "PW10-18-6":pw10_18_6,"PW10-21-14":pw10_21_14,"PW10-21-12":pw10_21_12,
                "PW10-45-36":pw10_45_36,
                "PW20-20":pw20_20, "PW20-40":pw20_40}


def PAIRWISE(*args, init=False, spectra=[], spectraM=[], T=1, name='', 
             description='PW6-15-10', modulation='ABCD',
             display=False, savedir='',ext='pdf',ArrayDetails=0,DisplayBaselengths=True):
    """
    
    Receive coherent fluxes and returns estimated coherent fluxes.
    When initialised, creates the V2PM of any desired pairwise fringe-sensor.
    
    Parameters
    ----------
    *args : optional arguments
        Must be the true coherent flux that receives the FS.
    init : BOOLEAN, optional
        DESCRIPTION. The default is False.
    spectra : ARRAY [NW], optional
        Micro spectral sampling. The default is [].
    spectraM : ARRAY [MW], optional
        Macro spectral sampling. The default is [].
    T : FLOAT, optional
        Transmission of the FS. The default is 1.
    name : STRING, optional
        Name of the FS. The default is ''.
    description : ARRAY OR STRING, optional
        Description of the FS. The default is 'PW6-15-10'.
        This paramter can be the name of a type of fringe-sensor among:
            'PW6-15-10', 'PW6-9-4-1', 'PW6-9-4-2', 'PW6-9-2', 'PW6-9-0',
            'PW6-9-2-b', 'PW6-6-1', 'PW6-6-0', 'PW6-5-0', 'PW6-5-0-0',
            'PW10-9-0', 'PW10-12-3-1', 'PW10-12-3-2', 'PW10-15-6', 'PW10-18-9'
        Or be an array dimensions [NAxNA] that gives the connexions between 
        telescopes on this logic: the element at the position (i,j) gives
        the ratio of flux of the telescope i that is shared with the telescope j.
    modulation : STRING, optional
        'ABCD' or 'AC. The default is 'ABCD'.
    display : BOOLEAN, optional
        Display the array with its connexions. The default is False.
    savedir : STRING, optional
        If a string is given, it saves the figure at this string. The default is ''.
    ext : STRING, optional
        Extension of the figure file. The default is 'pdf'.
    ArrayDetails : STRING, optional
        Fitsfile that contains information on the interferometer, necessary
        for telescope positions. The default is 0.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    currCfEstimated : ARRAY [MW,NINmes]
        Estimated coherent fluxes for the measured baselines.

    """
    
    if init:
        if isinstance(description,str):
            if not len(name):
                name=description
            description = descriptions[description]
        
        NA=np.shape(description)[0] ; NIN=NA*(NA-1)//2
        NC = int(binom(NA,3))
        
        d=np.zeros([NIN,2])     # Transmissions basewise in amplitude
        for ia in range(NA):
            for iap in range(ia+1,NA):
                ib=ct.posk(ia,iap,NA)
                d[ib,0]=np.sqrt(description[ia,iap])
                d[ib,1]=np.sqrt(description[iap,ia])
        
        if modulation == 'ABCD':
            modulator=ABCDmod()  # Classic balanced ABCD modulation in the right order (matrix [4x2])
            ModulationIndices = [0,1,2,3]
            
        if modulation == 'AC':
            from .FS_DEFAULT import ACmod
            modulator=ACmod()  # Balanced AC modulation in the right order (matrix [2x2])
            ModulationIndices = [0,1]
            
        NMod = len(modulation)
        
        A2P,ichdetails,active_ich=ct.makeA2P(d, modulator, reducedmatrix=True)
        
        A2P = A2P * np.sqrt(T)          # Add the transmission loss into the matrix elements.
        
        ct.check_nrj(A2P)               # Check if A2P is the matrix of a physical system.
      
        config.FS['A2P'] = A2P
          
        ich = [ichdetails[NMod*k][0] for k in range(len(ichdetails)//NMod)]
        NINmes = len(ich)
      
        NP, NA = np.shape(A2P)
        
        config.FS['name'] = name
        config.FS['func'] = PAIRWISE
        config.FS['ich'] = ich
        config.FS['active_ich'] = active_ich
        
        validcp=[]; active_cp = np.zeros([NC])
        for ia in range(NA):
            for iap in range(ia+1,NA):
                for iapp in range(iap+1,NA):
                    ib = ct.posk(ia,iap,NA)      # coherent flux (ia,iap)  
                    valid1=(active_ich[ib]>=0)
                    ib = ct.posk(iap,iapp,NA) # coherent flux (iap,iapp)    
                    valid2=(active_ich[ib]>=0)
                    ib = ct.posk(ia,iapp,NA) # coherent flux (iapp,ia)    
                    valid3=(active_ich[ib]>=0)
        
                    if valid1*valid2*valid3:
                        validcp.append((ia+1,iap+1,iapp+1))
                        ic = ct.poskfai(ia,iap,iapp,NA)
                        active_cp[ic] = 1
        
        config.FS['validcp'] = validcp
        config.FS['NCmes'] = len(validcp)
        config.FS['active_cp'] = active_cp
        
        config.FS['description'] = description
        
        if not ArrayDetails:
            raise Exception("No interferometric array has been given so we can't display the combination architecture.")
        
        InterfArray=ct.get_array(name=ArrayDetails)
        PhotometricBalance = np.ones(NIN)
        for ia in range(NA):
            for iap in range(ia+1,NA):
                ib=ct.posk(ia,iap,NA)
                x2,y2 = InterfArray.TelCoordinates[iap,:2]
                T1=d[ib][0] ; T2=d[ib][1]
                if T1 or T2:
                    PhotometricCoherence = 2*np.sqrt(T1*T2)/(T1+T2)
                else:
                    PhotometricCoherence=0
                UncoherentPhotometry = (T1+T2)*(NA-1) # Normalised by the maximal photometry
                # PhotometricBalance = N.rho où N=I1*I2 et rho = 2sqrt(I1I2)/(I1+I2)
                # Et on écrit I1 = T1*(N-1) pour que si T1=1/(N-1) (cas équilibré) I1=1
                # et donc SNR=1
                PhotometricBalance[ib] = (T1*T2)
        
        config.FS['PhotometricBalance'] = PhotometricBalance  # TV of the baselines normalised by its value in case of equal repartition on all baselines.
        config.FS['Modulation'] = modulation
        config.FS['ABCDind'] = ModulationIndices
        config.FS['NMod'] = NMod
        config.FS['NP'] = NP
        config.FS['T'] = T
        config.FS['ichdetails'] = ichdetails
        config.FS['NINmes'] = NINmes            # Number of measured baselines
        config.FS['NBmes'] = NA+2*NINmes        # phot + cos + sin
        
        
        V2PM,V2PMgrav,V2PM_r = ct.MakeV2PfromA2P(A2P)
        
        P2VM = np.linalg.pinv(V2PM)
        P2VM[np.abs(P2VM)<1e-10]=0
        
        P2VMgrav = np.linalg.pinv(V2PMgrav)
        P2VMgrav[np.abs(P2VMgrav)<1e-10]=0
        
        P2VM_r = np.linalg.pinv(V2PM_r)
        P2VM_r[np.abs(P2VM_r)<1e-10]=0
        
        NW, MW = len(spectra), len(spectraM)
        
        # Noise maps
        config.FS['imsky']=np.zeros([MW,NP])                # Sky background (bias)
        config.FS['sigmap']=np.zeros([MW,NP])               # Dark noise
        
        # Resolution of the fringe sensor
        midlmbda = np.mean(spectra)
        deltalmbda = (np.max(spectra) - np.min(spectra))/MW
        config.FS['R'] = midlmbda/deltalmbda
        config.FS['MW'] = MW
        
        config.FS['V2PM'] = np.repeat(V2PM[np.newaxis,:,:],NW,0)
        config.FS['P2VM'] = np.repeat(P2VM[np.newaxis,:,:],NW,0)
        config.FS['MacroP2VM'] = np.repeat(P2VM[np.newaxis,:,:],MW,0)
        
        # GRAVITY format
        config.FS['V2PMgrav1'] = np.repeat(V2PMgrav[np.newaxis,:,:],NW,0)
        config.FS['P2VMgrav1'] = np.repeat(P2VMgrav[np.newaxis,:,:],NW,0)
        config.FS['MacroP2VMgrav1'] = np.repeat(P2VMgrav[np.newaxis,:,:],MW,0)
        
        
        # REDUCED GRAVITY format
        config.FS['V2PM_r'] = np.repeat(V2PM_r[np.newaxis,:,:],NW,0)
        config.FS['P2VM_r'] = np.repeat(P2VM_r[np.newaxis,:,:],NW,0)
        config.FS['MacroP2VM_r'] = np.repeat(P2VM_r[np.newaxis,:,:],MW,0)
        
        
        
        # The matrix of the elements norm only for the calculation of the bias of |Cf|².
        # /!\ To save time, it's in [NIN,NP]
        config.FS['ElementsNormDemod'] = np.zeros([MW,NIN,NP])
        for imw in range(MW):
            ElementsNorm = config.FS['MacroP2VM'][imw]*np.conj(config.FS['MacroP2VM'][imw])
            config.FS['ElementsNormDemod'][imw] = np.real(ct.NB2NIN(ElementsNorm.T).T)
    
        config.FS['V2PMgrav'] = ct.simu2GRAV(config.FS['V2PM'])
        config.FS['P2VMgrav'] = ct.simu2GRAV(config.FS['P2VM'], direction='p2vm')
        config.FS['MacroP2VMgrav'] = ct.simu2GRAV(config.FS['MacroP2VM'], direction='p2vm')
        
        config.FS['Piston2OPD'] = np.zeros([NIN,NA])    # Piston to OPD matrix
        config.FS['OPD2Piston'] = np.zeros([NA,NIN])    # OPD to Pistons matrix
        Piston2OPD_forInv = np.zeros([NIN,NA])
        
        for ia in range(NA):
            for iap in range(ia+1,NA):
                ib = ct.posk(ia,iap,NA)
                config.FS['Piston2OPD'][ib,ia] = 1
                config.FS['Piston2OPD'][ib,iap] = -1
                if active_ich[ib]>=0:
                    Piston2OPD_forInv[ib,ia] = 1
                    Piston2OPD_forInv[ib,iap] = -1
            
        config.FS['OPD2Piston'] = np.linalg.pinv(Piston2OPD_forInv)   # OPD to pistons matrix
        config.FS['OPD2Piston'][np.abs(config.FS['OPD2Piston'])<1e-8]=0
        
        config.FS['OPD2Piston_moy'] = np.copy(config.FS['OPD2Piston'])
        if config.TELref:
            iTELref = config.TELref - 1
            L_ref = config.FS['OPD2Piston'][iTELref,:]
            config.FS['OPD2Piston'] = config.FS['OPD2Piston'] - L_ref
            
            
        config.FS['OPD2Piston_r'] = np.zeros([NA,NINmes])
        config.FS['OPD2Piston_moy_r'] = np.zeros([NA,NINmes])
        config.FS['Piston2OPD_r'] = np.zeros([NINmes,NA])
        
        k=0
        for ia in range(NA):
            for iap in range(ia+1,NA):
                ib = ct.posk(ia,iap,NA)
                if active_ich[ib]>=0:
                    config.FS['Piston2OPD_r'][k] = config.FS['Piston2OPD'][ib]
                    config.FS['OPD2Piston_r'][:,k] = config.FS['OPD2Piston'][:,ib]
                    config.FS['OPD2Piston_moy_r'][:,k] = config.FS['OPD2Piston_moy'][:,ib]
                    k+=1
            
        
        """
        FOR DARK BACKGROUND POWERPOINT --> WHITE FONT COLORS AND TRANSPARENT PNG BACKGROUND
        """
        if display:
            
            if len(savedir):
                plt.rcParams['figure.figsize']=(16,16)
                font = {'family' : 'DejaVu Sans',
                        'weight' : 'normal',
                        'size'   : 22}
                
                rcParamsFS = {"axes.grid":False,
                                "figure.constrained_layout.use": True,
                                'figure.subplot.hspace': 0,
                                'figure.subplot.wspace': 0,
                                'figure.subplot.left':0,
                                'figure.subplot.right':1
                                }
                plt.rcParams.update(rcParamsFS)
                plt.rc('font', **font)
            
            title=name
            fig=plt.figure(title, clear=True)
            ax=fig.subplots()
            for ia in range(NA):
                name1,(x1,y1) = InterfArray.TelNames[ia],InterfArray.TelCoordinates[ia,:2]
                for iap in range(ia+1,NA):
                    ib=ct.posk(ia,iap,NA)
                    x2,y2 = InterfArray.TelCoordinates[iap,:2]
                    T1=d[ib][0] ; T2=d[ib][1]
                    if T1 or T2:
                        PhotometricCoherence = 2*np.sqrt(T1*T2)/(T1+T2)
                    else:
                        PhotometricCoherence=0
                    UncoherentPhotometry = T1*T2*(NA-1) # Normalised by the maximal photometry
                    PhotometricBalance = UncoherentPhotometry * PhotometricCoherence
                    if T1*T2:
                        ax.plot([x1,(x2+x1)/2],[y1,(y2+y1)/2],color=colors[0],linestyle='-',linewidth=15*T1**2,zorder=-1)
                        ax.plot([(x2+x1)/2,x2],[(y2+y1)/2,y2],color=colors[0],linestyle='-',linewidth=15*T2**2,zorder=-1)
                        if DisplayBaselengths:
                            ax.annotate(f"{round(InterfArray.BaseNorms[ib])}", ((x1+x2)/2-3,(y1+y2)/2-3),color='w')
            
            for ia in range(NA):
                name1,(x1,y1) = InterfArray.TelNames[ia],InterfArray.TelCoordinates[ia,:2]
                ax.scatter(x1,y1,marker='o',edgecolor="w",facecolor='None',linewidth=15,s=8)
                if ia not in [1,4,5,6] or (NA!=10):
                    ax.annotate(name1, (x1-20,y1+5),color="w")
                    ax.annotate(f"({ia+1})", (x1-5,y1+5),color=colors[3],fontsize=35)
                else:
                    ax.annotate(name1, (x1-20,y1-15),color="w")
                    ax.annotate(f"({ia+1})", (x1-5,y1-15),color=colors[3],fontsize=35)
                
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            
            xwidth = np.ptp(InterfArray.TelCoordinates[:,0])*1.2
            minX = np.min(InterfArray.TelCoordinates[:,0])
            xmin = minX - 0.2*np.abs(minX)
            
            ywidth = np.ptp(InterfArray.TelCoordinates[:,1])*1.2
            minY = np.min(InterfArray.TelCoordinates[:,1])
            ymin = minY - 0.2*np.abs(minY)
            
            xmax,ymax  = xmin+xwidth , ymin+ywidth
            ax.set_xlim([xmin,xmax]) ; ax.set_ylim([ymin,ymax])
            
            #ax.text(xmin+50,ymax-50,name,fontsize=60,color='w')
            #ax.text(0.5*xmin,0,"Planet Formation Imager",fontsize=50,color='w')
            
            if len(savedir):
                if not os.path.exists(savedir):
                    os.makedirs(savedir, exist_ok=True)
                
                ax.axis("off")
                if not len(name):
                    name = "test"

                if isinstance(ext, list):
                    for exttemp in ext:
                        if exttemp == 'png':
                            fig.savefig(f"{savedir}{name}.{exttemp}",transparent=True)
                        else:
                            fig.savefig(f"{savedir}{name}.{exttemp}")
                else:
                    if ext == 'png':
                        fig.savefig(f"{savedir}{name}.{ext}",transparent=True)
                    else:
                        fig.savefig(f"{savedir}{name}.{ext}")
        
            plt.rcParams.update(plt.rcParamsDefault)
        
        
        """
        
        
        if display:
            
            if len(savedir):
                plt.rcParams['figure.figsize']=(16,16)
                font = {'family' : 'DejaVu Sans',
                        'weight' : 'normal',
                        'size'   : 22}
                
                rcParamsFS = {"axes.grid":False,
                               "figure.constrained_layout.use": True,
                               'figure.subplot.hspace': 0,
                               'figure.subplot.wspace': 0,
                               'figure.subplot.left':0,
                               'figure.subplot.right':1
                               }
                plt.rcParams.update(rcParamsFS)
                plt.rc('font', **font)
            
            title=name
            fig=plt.figure(title, clear=True)
            ax=fig.subplots()
            for ia in range(NA):
                name1,(x1,y1) = InterfArray.TelNames[ia],InterfArray.TelCoordinates[ia,:2]
                for iap in range(ia+1,NA):
                    ib=ct.posk(ia,iap,NA)
                    x2,y2 = InterfArray.TelCoordinates[iap,:2]
                    T1=d[ib][0] ; T2=d[ib][1]
                    if T1 or T2:
                        PhotometricCoherence = 2*np.sqrt(T1*T2)/(T1+T2)
                    else:
                        PhotometricCoherence=0
                    UncoherentPhotometry = T1*T2*(NA-1) # Normalised by the maximal photometry
                    PhotometricBalance = UncoherentPhotometry * PhotometricCoherence
                    if T1*T2:
                        ax.plot([x1,(x2+x1)/2],[y1,(y2+y1)/2],color=colors[0],linestyle='-',linewidth=15*T1**2,zorder=-1)
                        ax.plot([(x2+x1)/2,x2],[(y2+y1)/2,y2],color=colors[0],linestyle='-',linewidth=15*T2**2,zorder=-1)
                        if DisplayBaselengths:
                            ax.annotate(f"{round(InterfArray.BaseNorms[ib])}", ((x1+x2)/2-3,(y1+y2)/2-3),color=colors[1])
            
            for ia in range(NA):
                name1,(x1,y1) = InterfArray.TelNames[ia],InterfArray.TelCoordinates[ia,:2]
                ax.scatter(x1,y1,marker='o',edgecolor="k",facecolor='None',linewidth=15,s=8)
                if ia not in [1,4,5,6] or (NA!=10):
                    #ax.annotate(name1, (x1-20,y1+5),color="k")
                    ax.annotate(f"({ia+1})", (x1+10,y1+5),color=colors[3])
                else:
                    #ax.annotate(name1, (x1-20,y1-15),color="k")
                    ax.annotate(f"({ia+1})", (x1+10,y1-15),color=colors[3])
                
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            
            xwidth = np.ptp(InterfArray.TelCoordinates[:,0])*1.2
            minX = np.min(InterfArray.TelCoordinates[:,0])
            xmin = minX - 0.2*np.abs(minX)
            
            ywidth = np.ptp(InterfArray.TelCoordinates[:,1])*1.2
            minY = np.min(InterfArray.TelCoordinates[:,1])
            ymin = minY - 0.2*np.abs(minY)
            
            xmax,ymax  = xmin+xwidth , ymin+ywidth
            ax.set_xlim([xmin,xmax]) ; ax.set_ylim([ymin,ymax])
            
            ax.text(xmin+20,ymax-20,name,fontsize='large')
            
            
            if len(savedir):
                if not os.path.exists(savedir):
                    os.makedirs(savedir, exist_ok=True)
                
                ax.axis("off")
                if not len(name):
                    name = "test"
                    
                if isinstance(ext, list):
                    for extension in ext:
                        fig.savefig(f"{savedir}{name}.{extension}")
                else:
                    fig.savefig(f"{savedir}{name}.{extension}")
        
            plt.rcParams.update(plt.rcParamsDefault)
        """
        return


    from .config import NA, NW, OW, FS
    from . import simu
    
    NBmes, NINmes = config.FS['NBmes'], config.FS['NINmes']
    active_ich = config.FS['active_ich']
    
    it = simu.it
    
    iow = 0
    imw=0
    image_iw = np.zeros(FS['NP'])
    
    currCfTrue = args[0]
    currCfTrue_r = np.zeros([NW,NBmes])
    
    for iw in range(config.NW):
        currCfTrue_r[iw] = ct.ReducedVector(currCfTrue[iw],active_ich,NA,form='NBcomplex')
        
        Modulation = FS['V2PM_r'][iw,:,:]
        image_iw = np.real(np.dot(Modulation,currCfTrue_r[iw,:]))*config.FS['T']
        
        simu.MacroImages[it,imw,:] += image_iw
        
        iow += 1
        if iow == OW:
            imw+=1
            iow = 0      

    
    if config.noise:
        from .skeleton import addnoise
        if np.min(simu.MacroImages[it,:,:])<0:
            print(f"Negative value on image at t={it}, before noise.\nI take absolue value.")
            simu.MacroImages[it,:,:] = np.abs(simu.MacroImages[it,:,:])
            
        simu.MacroImages[it,:,:] = addnoise(simu.MacroImages[it,:,:])
    
    # if np.min(simu.MacroImages[it]) < 0:
    #     print(f'Negative image value at t={it}')
        
    if config.FS['Modulation']=='ABCD':
        # estimates coherences
        currCfEstimated = np.zeros([FS['MW'],NBmes])*1j
        for imw in range(FS['MW']):
            Demodulation = config.FS['MacroP2VM_r'][imw,:,:]
            currCfEstimated[imw,:] = np.dot(Demodulation,simu.MacroImages[it,imw,:])
            
        
    elif config.FS['Modulation']=='AC':  # Necessary patch for AC demodulation
        # estimates coherences
        specialCf = np.zeros([FS['MW'],NBmes])*1j
        for imw in range(FS['MW']):
            Demodulation = config.FS['MacroP2VM_r'][imw,:,:]
            specialCf[imw,:] = np.dot(Demodulation,simu.MacroImages[it,imw,:])
            
        currCfEstimated = np.zeros([FS['MW'],NBmes])*1j
        for ia in range(NA):
            currCfEstimated[:,ia*(NA+1)] = specialCf[:,ia*(NA+1)]
            for iap in range(ia+1,NA):
                ib=ia*NA+iap
                Module = np.sqrt(specialCf[:,ia*(NA+1)]*specialCf[:,iap*(NA+1)])
                phase = np.imag(specialCf[:,ib]/Module)
                currCfEstimated[:,ib] = Module*np.exp(1j*phase)
                currCfEstimated[:,iap*NA+ia] = Module*np.exp(-1j*phase)
                
    return currCfEstimated






def ALLINONE(*args,init=False, T=1, spectra=[], spectraM=[], NA=2, 
             posi=[-0.25, 0.25], MFD=0.254, posi_center=0.05, posp=[-0.456, -0.38],F=40, p=0.024, 
             Dsize=(320,256), Dc=0.396, PSDwindow=0.396, Tphot=0.1, Tint=0.9):
    """
    From the oversampled coherent fluxes, simulates the noisy image on the detector 
    and estimates the macrosampled coherent fluxes.
    
    INPUT:
        - If init: all the below parameters.
        - If not init: currCfTrue - Oversampled True Coherent Flux   [NW,NB]

    OUTPUT: 
        - currCfEstimated - Macrosampled measured coherent flux [MW,NB]

    USED OBSERVABLES/PARAMETERS:
        - config.FS
    UPDATED OBSERVABLES/PARAMETERS:
        - simu.MacroImages: [NT,MW,NIN] Estimated PD before subtraction of the reference
        
    SUBROUTINES:
        - skeleton.add_camera_noise

    Parameters
    ----------
    *args : ARRAY [NW,NB]
        Expect oversampled coherent flux currCfTrue.
    init : BOOLEAN, optional
        If True, initialize the parameters of the fringe sensor.
        Needs spectra, spectraM
        All this parameters are stored in the dictionnary config.FS.
        Needs to be called before starting the simulation.
        The default is False.
    spectra : ARRAY [NW], necessary if INIT
        Spectral microsampling. The default is [].
    spectraM : ARRAY [MW], necessary if INIT
        Spectral macrosampling. The default is [].
    posi : LIST [NA], optional
        Positions [mm] of the fibers output on the V-groove 
        (it defines the spatial frequencies)
    MFD : FLOAT, optional
        Mode-field diameter in the microlenses plane of the V-groove.
    posi_center : FLOAT, optional
        Position of the center of the interferogram.
    posp : LIST [NA], optional
        Positions [mm] of the photometric beams on the detector.
        It has an impact on the SNR of the photometric signal if no field stop 
        is used (Dc=0)
    F : FLOAT, optional
        Focal length [mm] of the imaging lens
    p : FLOAT, optional
        Pixel size [mm] of the camera
    Dsize : TUPLE, optional
        Size [H,L] in number of pixels of the detector.
    Dc : FLOAT, optional
        Semi-Diameter [mm] of field stop. If zero, no field stop.
    PSDwindow : FLOAT, optional
        Semi-diameter [mm] of the window used for the calculation of the
        interferogram PSD.
    Tphot : FLOAT, optional
        Transmission in the photometric channel.
    Tint : FLOAT, optional
        Transmission in the interferometric channel.
    
    
    Returns
    -------
    currCfEstimated : ARRAY [MW,NB]
        Macrosampled measured coherent flux.

    """
    
    if init:
        
        if NA!=2:
            
            if not posi:    # Positions of the fibers on the V-groove.
                # These positions give the most compact non redondant positions (from Lacour thesis)
                if NA==3:
                    posi = [-0.75, -0.25, 0.75]
                elif NA==4:
                    posi = [-1.5, -1, 0.5, 1.5]
                elif NA==5:
                    posi = [-2.5, -2, 0.5, 1.5, 2.5]
                elif NA==6:
                    posi = [-4.5, -4, -2.5, 0.5, 1.5, 4]
                elif NA==7:
                    posi = [-6, -5.5, -4, -1, 3, 5.5, 6.5]
                    
            if not posp:    # Positions of the photometric beams on the detector.
                # We put at the extremity of the detector
                Bout=2*p ; Bin = np.min(np.abs(posi[:-1]-posi[1:]))
                alpha = Bout/Bin
                posp = (np.array(posi) - posi[0])*alpha - Dsize[1]/2*p +2*p
                    
        if len(posp) != NA:
            raise Exception(f"The FS takes a different number of beams ({len(posp)}) than the one\
given in config ({NA}).")
        
        if not PSDwindow:
            if Dc:
                PSDwindow = Dc
            else:   # Minimum between the detector available space and the minimal separation between the 2 channels.
                PSDwindow = np.min([posi_center - np.max(posp), Dsize[0]*24-posi_center]) 
        
        NIN = int(NA*(NA-1)/2) ; NB = NA**2
        NW = len(spectra) ; MW = len(spectraM)
        NP = int(NA + 2*PSDwindow//p)  # 6 photometric beams + the interferogram
        
        Baselines = np.zeros(NIN)
        pixel_positions = np.linspace(-PSDwindow,PSDwindow,NP-NA)
        
        # Approximation
        detectorMFD = 2*np.mean(spectra)*F/MFD
        
        ich = np.array([[1,2]])
        
        active_ich = list(np.arange(NIN))
        ichorder = np.arange(NIN)
        
        config.FS['func'] = ALLINONE
        config.FS['ich'] = ich
        config.FS['ichorder'] = ichorder
        config.FS['NP'] = NP
        config.FS['MW'] = MW
        config.FS['posi'] = posi
        config.FS['posi_center'] = posi_center
        config.FS['MFD'] = MFD
        config.FS['detectorMFD'] = detectorMFD
        config.FS['posp'] = posp
        config.FS['F'] = F
        config.FS['p'] = p
        config.FS['Dc'] = Dc
        config.FS['PSDwindow'] = PSDwindow
        config.FS['Tphot'] = Tphot ; config.FS['Tint'] = Tint
        config.FS['description'] = (np.ones([NA,NA]) - np.identity(NA))/(NA-1)
        config.FS['active_ich'] = active_ich
        config.FS['PhotometricBalance'] = np.ones(NIN)   # TV² of the baselines normalised by its value for equal repartition on all baselines.
        
        # Noise maps
        config.FS['imsky']=np.zeros([MW,NP])                # Sky background (bias)
        config.FS['sigmap']=np.zeros([MW,NP])               # Dark noise
        
        # Resolution of the fringe sensor
        midlmbda = np.mean(spectra)
        deltalmbda = (np.max(spectra) - np.min(spectra))/MW
        config.FS['R'] = midlmbda/deltalmbda        
        
        # Hard coding of the P2VM
        V2PM = np.zeros([NW,NP,NB])*1j; MacroV2PM = np.zeros([MW,NP,NB])*1j
        
        GaussianEnvelop = np.exp(-2*(2*pixel_positions/detectorMFD)**2)
        EnergyDistribution = GaussianEnvelop/np.sum(GaussianEnvelop)*Tint
        
        # Creation of the oversampled V2PM
        iow=0 ; imw=0; OW = NW/MW
        for iw in range(NW):
            wl = spectra[iw]
            for ia in range(NA):
                V2PM[iw,ia,ia*(NA+1)] = Tphot               # Photometric beams
                V2PM[iw,NA:,ia*(NA+1)] = np.ones(NP-NA)*EnergyDistribution     # Interferometric beams
                for iap in range(ia+1,NA):
                    ib = ct.posk(ia,iap,NA)
                    Baselines[ib] = np.abs(posi[iap]-posi[ia])
                    
                    OPD = Baselines[ib]/F * pixel_positions*1e3
                    PhaseDelays = 2*np.pi/spectra[iw] * OPD
                    PhaseDelaysM = 2*np.pi/spectra[imw] * OPD
                    
                    V2PM[iw,NA:,ia*NA+iap] = np.exp(PhaseDelays*1j)*EnergyDistribution
                    V2PM[iw,NA:,iap*NA+ia] = np.exp(-PhaseDelays*1j)*EnergyDistribution
            
            MacroV2PM[imw] += V2PM[iw]/OW
                    
            iow+=1
            if iow==OW:
                imw+=1
                iow=0
        
        # Oversampled Pixel-to-Visibility matrix
        P2VM = np.linalg.pinv(V2PM)
        
        # Undersampled Pixel-to-Visibility matrix
        MacroP2VM = np.linalg.pinv(MacroV2PM)
        
        config.FS['V2PM'] = V2PM
        config.FS['P2VM'] = P2VM
        config.FS['MacroP2VM'] = MacroP2VM

        # The matrix of the elements norm only for the calculation of the bias of |Cf|².
        # /!\ To save time, it's in [NIN,NP]
        config.FS['ElementsNormDemod'] = np.zeros([MW,NIN,NP])
        for imw in range(MW):
            config.FS['ElementsNormDemod'][imw] = np.real(ct.NB2NIN(config.FS['MacroP2VM'][imw]*np.conj(config.FS['MacroP2VM'][imw])))


        config.FS['V2PMgrav'] = ct.simu2GRAV(config.FS['V2PM'])
        config.FS['P2VMgrav'] = ct.simu2GRAV(config.FS['P2VM'], direction='p2vm')
        config.FS['MacroP2VMgrav'] = ct.simu2GRAV(config.FS['MacroP2VM'], direction='p2vm')
        
        config.FS['Piston2OPD'] = np.zeros([NIN,NA])    # Piston to OPD matrix
        config.FS['OPD2Piston'] = np.zeros([NA,NIN])    # OPD to Pistons matrix
        Piston2OPD_forInv = np.zeros([NIN,NA])
        
        for ia in range(NA):
            for iap in range(ia+1,NA):
                ib = ct.posk(ia,iap,NA)
                config.FS['Piston2OPD'][ib,ia] = 1
                config.FS['Piston2OPD'][ib,iap] = -1
                if active_ich[ib]>=0:
                    Piston2OPD_forInv[ib,ia] = 1
                    Piston2OPD_forInv[ib,iap] = -1
            
        config.FS['OPD2Piston'] = np.linalg.pinv(Piston2OPD_forInv)   # OPD to pistons matrix
        config.FS['OPD2Piston'][np.abs(config.FS['OPD2Piston'])<1e-8]=0
        
        config.FS['OPD2Piston_moy'] = np.copy(config.FS['OPD2Piston'])
        if config.TELref:
            iTELref = config.TELref - 1
            L_ref = config.FS['OPD2Piston'][iTELref,:]
            config.FS['OPD2Piston'] = config.FS['OPD2Piston'] - L_ref
        
        
        return
    
    from .config import NA, NB
    from . import simu
    
    it = simu.it
    
    iow = 0
    imw=0
    image_iw = np.zeros(config.FS['NP'])
    
    currCfTrue = args[0]
               
    for iw in range(config.NW):
        
        Modulation = config.FS['V2PM'][iw,:,:]
        image_iw = np.real(np.dot(Modulation,currCfTrue[iw,:]))
        
        simu.MacroImages[it,imw,:] += image_iw
        
        iow += 1
        if iow == config.OW:
            imw+=1
            iow = 0      

    
    if config.noise:
        from .skeleton import addnoise
        if np.min(simu.MacroImages[it,:,:])<0:
            print(f"Negative value on image at t={it}, before noise.\nI take absolue value.")
            simu.MacroImages[it,:,:] = np.abs(simu.MacroImages[it,:,:])
            
        simu.MacroImages[it,:,:] = addnoise(simu.MacroImages[it,:,:])
    
    # estimates coherences
    currCfEstimated = np.zeros([config.FS['MW'],NB])*1j
    for imw in range(config.FS['MW']):
        Demodulation = config.FS['MacroP2VM'][imw,:,:]
        currCfEstimated[imw,:] = np.dot(Demodulation,simu.MacroImages[it,imw,:])
    
    return currCfEstimated





def SPICAFS_PERFECT(*args,T=1, init=False, spectra=[], spectraM=[]):
    """
    Measures the coherent flux after simulating the image given by the real 
    oversampled coherence flux and adding noises on it.
    
    INPUT:
        - If init: all the below parameters.
        - If not init: currCfTrue - Oversampled True Coherent Flux   [NW,NB]

    OUTPUT: 
        - currCfEstimated - Macrosampled measured coherent flux [MW,NB]

    USED OBSERVABLES/PARAMETERS:
        - config.FS
    UPDATED OBSERVABLES/PARAMETERS:
        - simu.MacroImages: [NT,MW,NIN] Estimated PD before subtraction of the reference
        - simu.GD_: [NT,MW,NIN] Estimated GD before subtraction of the reference
        - simu.CommandODL: Piston Command to send       [NT,NA]
        
    SUBROUTINES:
        - skeleton.add_camera_noise

    Parameters
    ----------
    *args : ARRAY [NW,NB]
        Expect oversampled coherent flux currCfTrue.
    init : BOOLEAN, optional
        If True, initialize the parameters of the fringe sensor. 
        Needs spectra, spectraM
        All this parameters are stored in the dictionnary config.FS.
        Needs to be called before starting the simulation.
        The default is False.
    spectra : ARRAY [NW], necessary if INIT
        Spectral microsampling. The default is [].
    spectraM : ARRAY [MW], necessary if INIT
        Spectral macrosampling. The default is [].
        
    Returns
    -------
    currCfEstimated : ARRAY [MW,NB]
        Macrosampled measured coherent flux.

    """


    from . import config
    
    if init:
        
        from .config import NA,NB
        
        # Created by the user here
        ich = np.array([[1,2], [1,3], [2,3], [2,4], [1,4], [1,5], [2,5], [1,6],[2,6],\
                  [3,6],[3,4],[3,5],[4,5],[4,6],[5,6]])
        
        ichorder = [0,1,4,5,7,2,3,6,8,10,11,9,12,13,14]
            
        config.FS['func'] = SPICAFS_PERFECT
        config.FS['ich'] = ich
        config.FS['ichorder'] = ichorder
        NG = np.shape(ich)[0]       # should always be equal to NIN
        
        # Classic balanced ABCD modulation of each baseline
        
        M_ABCD = ABCDmod()          # A2P ABCD modulation
        NMod = len(M_ABCD)          # Number of modulations for each baseline
        config.FS['Modulation'] = 'ABCD'
        ABCDind = [0,1,2,3]
        config.FS['ABCDind'] = ABCDind
        NP = NMod*NG
        
        config.FS['NMod'] = NMod
        config.FS['NP'] = NP
        
        NIN = NP//NMod
        OrderingIndex = np.zeros(NP,dtype=np.int8)
        for ib in range(NIN):
            for k in range(NMod):
                OrderingIndex[ib*NMod+k] = ichorder[ib]*NMod+ABCDind[k]
                
        config.FS['T'] = T
        
        # Build the A2P of SPICA
        
        M_spica = np.zeros([NP,NA])*1j
        for ig in range(NG):
            for ia in range(2):
                M_spica[NMod*ig:NMod*(ig+1),ich[ig,ia]-1] = M_ABCD[:,ia]
        
        # Build the V2P and P2V matrices
        
        V2PM = np.zeros([NP,NB])*1j
        for ip in range(NP):
            for ia in range(NA):
                for iap in range(NA):
                    k = ia*NA+iap
                    V2PM[ip, k] = M_spica[ip,ia]*np.transpose(np.conjugate(M_spica[ip,iap]))/(NA-1)
        
        P2VM = np.linalg.pinv(V2PM)    
        
        NW, MW = len(spectra), len(spectraM)
        
        # Noise maps
        config.FS['imsky']=np.zeros([MW,NP])                # Sky background (bias)
        config.FS['sigmap']=np.zeros([MW,NP])               # Dark noise
        
        # Resolution of the fringe sensor
        midlmbda = np.mean(spectra)
        deltalmbda = (np.max(spectra) - np.min(spectra))/MW
        config.FS['R'] = midlmbda/deltalmbda
        
        config.FS['V2PM'] = np.repeat(V2PM[np.newaxis,:,:],NW,0)
        config.FS['P2VM'] = np.repeat(P2VM[np.newaxis,:,:],NW,0)
        config.FS['MacroP2VM'] = np.repeat(P2VM[np.newaxis,:,:],MW,0)
    
    
        config.FS['V2PMgrav'] = ct.simu2GRAV(config.FS['V2PM'])
        config.FS['P2VMgrav'] = ct.simu2GRAV(config.FS['P2VM'], direction='p2vm')
        config.FS['MacroP2VMgrav'] = ct.simu2GRAV(config.FS['MacroP2VM'], direction='p2vm')
        
        return
    
    from .config import NA, NB, NW, MW, OW
    from . import simu
    
    it = simu.it
    
    iow = 0
    imw=0
    image_iw = np.zeros(config.FS['NP'])
    
    currCfTrue = args[0]*config.FS['T']               # Transmission of the CHIP
               
    for iw in range(config.NW):
        
        Modulation = config.FS['V2PM'][iw,:,:]
        image_iw = np.real(np.dot(Modulation,currCfTrue[iw,:]))
        
        simu.MacroImages[it,imw,:] += image_iw
        
        iow += 1
        if iow == OW:
            imw+=1
            iow = 0      

    
    if config.noise:
        from .skeleton import addnoise
        simu.MacroImages[it,:,:] = addnoise(simu.MacroImages[it,:,:])
    
    # if np.min(simu.MacroImages[it]) < 0:
    #     print(f'Negative image value at t={it}')
    
    # estimates coherences
    currCfEstimated = np.zeros([MW,NB])*1j
    for imw in range(MW):
        Demodulation = config.FS['MacroP2VM'][imw,:,:]
        currCfEstimated[imw,:] = np.dot(Demodulation,simu.MacroImages[it,imw,:])
    
    return currCfEstimated


if __name__ == "__main__":
    
    SPICAFS_PERFECT(init=True)
    
    import matplotlib.pyplot as plt
    import config
    plt.figure()
    plt.imshow(np.angle(config.FS['P2VM'][0]))
    plt.show()
    
    directory = 'C:/Users/cpannetier/Documents/These/FringeTracking/SPICA-FT/V2PM_SPICA/'
    V2PMfilename = 'MIRCX_ABCD_H_PRISM22_V2PM.fits'
    fitsfile = directory+V2PMfilename
    SPICAFS_TRUE2(fitsfile=fitsfile,init=True,OW=10)
    import matplotlib.pyplot as plt
    import config
    plt.figure()
    plt.imshow(np.angle(config.FS['P2VM'][0]))
    plt.show()
    
    
