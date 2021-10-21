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
from astropy.io import fits


from . import coh_tools as ct
from . import config
from .FS_DEFAULT import ABCDmod, realisticABCDmod



def PAIRWISE(*args, init=False, spectra=[], spectraM=[], T=1, description=[[1,1],[1,1],[1,1]]/np.sqrt(2), modulation='ABCD', clean_up=False):
    
    if init:
        
        if modulation == 'ABCD':
            modulator=ABCDmod()  # Classic balanced ABCD modulation in the right order (matrix [4x2])
        
        modulations = [char for char in modulation]
        NMod = len(modulation)
        
        A2P,ichdetails=ct.makeA2P(description, modulator)
        
        A2P = A2P * np.sqrt(T)          # Add the transmission loss into the matrix elements.
        
        ct.check_nrj(A2P)               # Check if A2P is the matrix of a physical system.
      
        ich = [ichdetails[4*k][0] for k in range(len(ichdetails)//NMod)]
        NINmes = len(ich)  
      
        NP, NA = np.shape(A2P)
        
        ABCDind = [0,1,2,3]
        
        config.FS['func'] = PAIRWISE
        config.FS['ich'] = ich
        config.FS['Modulations'] = modulations
        config.FS['ABCDind'] = ABCDind
        config.FS['NMod'] = NMod
        config.FS['NP'] = NP
        config.FS['T'] = T
        config.FS['ichdetails'] = ichdetails
        config.FS['NINeff'] = NINmes
        
        V2PM = ct.MakeV2PfromA2P(A2P)
        
        P2VM = np.linalg.pinv(V2PM)
        
        
            
        NW, MW = len(spectra), len(spectraM)
        
        # Noise maps
        config.FS['imsky']=np.zeros([MW,NP])                # Sky background (bias)
        config.FS['sigsky']=np.zeros([MW,NP])               # Dark noise
        
        # Resolution of the fringe sensor
        midlmbda = np.mean(spectra)
        deltalmbda = (np.max(spectra) - np.min(spectra))/MW
        config.FS['R'] = midlmbda/deltalmbda
        config.FS['MW'] = MW
        
        config.FS['V2PM'] = np.repeat(V2PM[np.newaxis,:,:],NW,0)
        config.FS['P2VM'] = np.repeat(P2VM[np.newaxis,:,:],NW,0)
        config.FS['MacroP2VM'] = np.repeat(P2VM[np.newaxis,:,:],MW,0)
    
    
        config.FS['V2PMgrav'] = ct.simu2GRAV(config.FS['V2PM'])
        config.FS['P2VMgrav'] = ct.simu2GRAV(config.FS['P2VM'], direction='p2vm')
        config.FS['MacroP2VMgrav'] = ct.simu2GRAV(config.FS['MacroP2VM'], direction='p2vm')
        
        return


    from .config import NA, NB, NW, OW, FS
    from . import simu
    
    it = simu.it
    
    iow = 0
    imw=0
    image_iw = np.zeros(FS['NP'])
    
    currCfTrue = args[0]               # Transmission of the CHIP
               
    for iw in range(config.NW):
        
        Modulation = FS['V2PM'][iw,:,:]
        image_iw = np.abs(np.dot(Modulation,currCfTrue[iw,:]))
        
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
    currCfEstimated = np.zeros([FS['MW'],NB])*1j
    for imw in range(FS['MW']):
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
        config.FS['Modulations'] = ['A','B','C','D']
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
        config.FS['sigsky']=np.zeros([MW,NP])               # Dark noise
        
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
        image_iw = np.abs(np.dot(Modulation,currCfTrue[iw,:]))
        
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


# def PAIRWISE_INTERMEDIARY(*args, init=False,T=1, spectra=[], spectraM=[], phaseshifts=[-1,0,1,2],transmissions=[1,1,1,1]):
#     """
#     Pairwise fringe-sensor that estimates the coherent fluxes of all baselines
#     by measuring only a part of the available baseline coherent fluxes.
    
#     INPUT:
#         - If init: all the below parameters.
#         - If not init: currCfTrue - Oversampled True Coherent Flux   [NW,NB]

#     OUTPUT: 
#         - currCfEstimated - Macrosampled measured coherent flux [MW,NB]

#     USED OBSERVABLES/PARAMETERS:
#         - config.FS
#     UPDATED OBSERVABLES/PARAMETERS:
#         - simu.MacroImages: [NT,MW,NIN] Estimated PD before subtraction of the reference
#         - simu.GD_: [NT,MW,NIN] Estimated GD before subtraction of the reference
#         - simu.CommandODL: Piston Command to send       [NT,NA]
        
#     SUBROUTINES:
#         - skeleton.add_camera_noise

#     Parameters
#     ----------
#     *args : ARRAY [NW,NB]
#         Expect oversampled coherent flux currCfTrue.
#     init : BOOLEAN, optional
#         If True, initialize the parameters of the fringe sensor. 
#         Needs spectra, spectraM
#         All this parameters are stored in the dictionnary config.FS.
#         Needs to be called before starting the simulation.
#         The default is False.
#     spectra : ARRAY [NW], necessary if INIT
#         Spectral microsampling. The default is [].
#     spectraM : ARRAY [MW], necessary if INIT
#         Spectral macrosampling. The default is [].
        
#     Returns
#     -------
#     currCfEstimated : ARRAY [MW,NB]
#         Macrosampled measured coherent flux.

#     """

#     from . import config
    
#     if init:
        
#         from .config import NA,NB
        
#         # Created by the user here
#         ich = np.array([[1,2], [1,3], [2,3], [2,4], [1,4], [1,5], [2,5], [1,6],[2,6],\
#                   [3,6],[3,4],[3,5],[4,5],[4,6],[5,6]])
#         ichorder = [0,1,4,5,7,2,3,6,8,10,11,9,12,13,14]

#         config.FS['func'] = SPICAFS_REALISTIC
#         config.FS['ich'] = ich
#         config.FS['ichorder'] = ichorder
        
#         NG = np.shape(ich)[0]       # should always be equal to NIN
        
#         # Classic balanced ABCD modulation of each baseline
        
#         M_ABCD = realisticABCDmod(phaseshifts, transmissions)          # A2P ABCD modulation
#         NMod = len(M_ABCD)          # Number of modulations for each baseline
#         config.FS['Modulations'] = ['A','B','C','D']
#         ABCDind = [0,1,2,3]
#         config.FS['ABCDind'] = ABCDind
#         config.FS['Phaseshifts'] = [k*np.pi/2 for k in phaseshifts]
        
#         NP = NMod*NG
        
#         config.FS['NMod'] = NMod
#         config.FS['NP'] = NP
        
#         NIN = NP//NMod
#         OrderingIndex = np.zeros(NP,dtype=np.int8)
#         for ib in range(NIN):
#             for k in range(NMod):
#                 OrderingIndex[ib*NMod+k] = ichorder[ib]*NMod+ABCDind[k]
        
#         config.FS['orderingindex'] = OrderingIndex
        
        
#         config.FS['T'] = T
        
#         # Build the A2P of SPICA
        
#         M_spica = np.zeros([NP,NA])*1j
#         for ig in range(NG):
#             for ia in range(2):
#                 M_spica[NMod*ig:NMod*(ig+1),ich[ig,ia]-1] = M_ABCD[:,ia]
        
#         # Build the V2P and P2V matrices
        
#         V2PM = np.zeros([NP,NB])*1j
#         for ip in range(NP):
#             for ia in range(NA):
#                 for iap in range(NA):
#                     k = ia*NA+iap
#                     V2PM[ip, k] = M_spica[ip,ia]*np.transpose(np.conjugate(M_spica[ip,iap]))/(NA-1)
        
#         P2VM = np.linalg.pinv(V2PM)    
        
#         NW, MW = len(spectra), len(spectraM)
        
#         # Noise maps
#         config.FS['imsky']=np.zeros([MW,NP])                # Sky background (bias)
#         config.FS['sigsky']=np.zeros([MW,NP])               # Dark noise
        
#         # Resolution of the fringe sensor
#         midlmbda = np.mean(spectra)
#         deltalmbda = (np.max(spectra) - np.min(spectra))/MW
#         config.FS['R'] = midlmbda/deltalmbda
        
#         config.FS['V2PM'] = np.repeat(V2PM[np.newaxis,:,:],NW,0)
#         config.FS['P2VM'] = np.repeat(P2VM[np.newaxis,:,:],NW,0)
#         config.FS['MacroP2VM'] = np.repeat(P2VM[np.newaxis,:,:],MW,0)
    
    
#         config.FS['V2PMgrav'] = ct.simu2GRAV(config.FS['V2PM'])
#         config.FS['P2VMgrav'] = ct.simu2GRAV(config.FS['P2VM'], direction='p2vm')
#         config.FS['MacroP2VMgrav'] = ct.simu2GRAV(config.FS['MacroP2VM'], direction='p2vm')
        
#         return
    
#     from .config import NA, NB, NW, MW, OW
#     from . import simu
    
#     it = simu.it
    
#     iow = 0
#     imw=0
#     image_iw = np.zeros(config.FS['NP'])
    
#     currCfTrue = args[0]*config.FS['T']               # Transmission of the CHIP
               
#     for iw in range(config.NW):
        
#         Modulation = config.FS['V2PM'][iw,:,:]
#         image_iw = np.abs(np.dot(Modulation,currCfTrue[iw,:]))
        
#         simu.MacroImages[it,imw,:] += image_iw
        
#         iow += 1
#         if iow == OW:
#             imw+=1
#             iow = 0      

    
#     if config.noise:
#         from .skeleton import addnoise
#         simu.MacroImages[it,:,:] = addnoise(simu.MacroImages[it,:,:])
    
#     # if np.min(simu.MacroImages[it]) < 0:
#     #     print(f'Negative image value at t={it}')
    
#     # estimates coherences
#     currCfEstimated = np.zeros([MW,NB])*1j
#     for imw in range(MW):
#         Demodulation = config.FS['MacroP2VM'][imw,:,:]
#         currCfEstimated[imw,:] = np.dot(Demodulation,simu.MacroImages[it,imw,:])
    
#     return currCfEstimated




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
    
    
