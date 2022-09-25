# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 17:19:42 2021

The MIRCx fringe sensor measures the coherent flux after simulating the image
given by the real oversampled coherence flux and adding noises on it.

INPUT: oversampled true coherent flux [NW,NB]

OUTPUT: macrosampled measured coherent flux [MW,NB]

Calculated and stored observables:
    - Photometries: simu.PhotometryEstimated [MW,NA]
    - Visibilities: simu.Visibility [MW,NIN]

"""

import numpy as np
from astropy.io import fits
from scipy.special import binom

from . import coh_tools as ct
from . import config
from .FS_DEFAULT import ABCDmod, realisticABCDmod

def MIRCxFS(*args,init=False, T=1, spectra=[], spectraM=[], posi=[], MFD=0.254,
            posi_center=4.68, posp=[],F=250, p=0.024, Dsize=(320,256), Dc=3, 
            PSDwindow=3, Tphot=0.2, Tint=0.8):
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
        
        if not posi:    # Positions of the fibers on the V-groove.
            posi = [-2.75,-2.25,-0.5,0.75,2.25,3.25]
        
        if not posp:    # Positions of the photometric beams on the detector.
            # posp = [3.84,3.74,3.52,3.41,3.19,3.12]
            posp = [0.84,0.94, 1.16, 1.27, 1.49, 1.56]
        
        if not PSDwindow:
            if Dc:
                PSDwindow = Dc
            else:   # Minimum between the detector available space and the minimal separation between the 2 channels.
                PSDwindow = np.min([posi_center - np.max(posp), Dsize[0]*24-posi_center]) 
        
        NA=len(posp)
        NIN = int(NA*(NA-1)/2) ; NB=NA**2 ; NC = int(binom(NA,3))
        NW = len(spectra) ; MW = len(spectraM)
        NP = int(NA + 2*PSDwindow//p)  # 6 photometric beams + the interferogram
        
        Baselines = np.zeros(NIN)
        pixel_positions = np.linspace(-PSDwindow,PSDwindow,NP-NA)
        
        # The FWHM on the detector is given in Anugu et al: 2.1 (certainly
        # at lmbda=1.55).
        # I calculate the MFD by the formula MFD = FWHM/0.59
        detectorMFD = 3.56
        
        ich = np.array([[1,2], [1,3], [1,4], [1,5], [1,6], [2,3],
                        [2,4], [2,5], [2,6], [3,4],[3,5],[3,6],
                        [4,5],[4,6],[5,6]])
        ich = ['12','13','14','15','16','23','24','25','26','34','35','36','45','46','56']
        
        ichorder = np.arange(NIN)
        active_ich = list(np.ones(NIN))
        
        config.FS['name'] = 'MIRCxFS'
        config.FS['func'] = MIRCxFS
        config.FS['ich'] = ich
        config.FS['ichorder'] = ichorder
        config.FS['active_ich'] = active_ich
        config.FS['NINmes'] = NIN
        config.FS['NBmes'] = NB
        config.FS['NCmes'] = NC
        config.FS['PhotometricBalance'] = np.ones(NIN)
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
        
        
        # Noise maps
        config.FS['imsky']=np.zeros([MW,NP])                # Sky background (bias)
        config.FS['sigmap']=np.zeros([MW,NP])               # Dark noise
        
        # Resolution of the fringe sensor
        midlmbda = np.mean(spectra)
        deltalmbda = (np.max(spectra) - np.min(spectra))/MW
        config.FS['R'] = midlmbda/deltalmbda        
        
        # Hard coding of the P2VM
        V2PM = np.zeros([NW,NP,NB])*1j; MacroV2PM = np.zeros([MW,NP,NB])*1j
        
        GaussianEnvelop = np.exp(-2*(2*pixel_positions/3.56)**2)
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
            ElementsNorm = config.FS['MacroP2VM'][imw]*np.conj(config.FS['MacroP2VM'][imw])
            config.FS['ElementsNormDemod'][imw] = np.real(ct.NB2NIN(ElementsNorm.T).T)
            
        config.FS['V2PMgrav'] = ct.simu2GRAV(config.FS['V2PM'])
        config.FS['P2VMgrav'] = ct.simu2GRAV(config.FS['P2VM'], direction='p2vm')
        config.FS['MacroP2VMgrav'] = ct.simu2GRAV(config.FS['MacroP2VM'], direction='p2vm')
        
        config.FS['V2PM_r'] = config.FS['V2PMgrav']
        config.FS['P2VM_r'] = config.FS['P2VMgrav']
        config.FS['MacroP2VM_r'] = config.FS['MacroP2VMgrav']
        
        
        config.FS['Piston2OPD'] = np.zeros([NIN,NA])    # Piston to OPD matrix
        config.FS['OPD2Piston'] = np.zeros([NA,NIN])    # OPD to Pistons matrix
        Piston2OPD_forInv = np.zeros([NIN,NA])
        
        for ia in range(NA):
            for iap in range(ia+1,NA):
                ib = ct.posk(ia,iap,NA)
                config.FS['Piston2OPD'][ib,ia] = 1
                config.FS['Piston2OPD'][ib,iap] = -1
                if active_ich[ib]:
                    Piston2OPD_forInv[ib,ia] = 1
                    Piston2OPD_forInv[ib,iap] = -1
            
        config.FS['OPD2Piston'] = np.linalg.pinv(Piston2OPD_forInv)   # OPD to pistons matrix
        config.FS['OPD2Piston'][np.abs(config.FS['OPD2Piston'])<1e-8]=0
        
        config.FS['OPD2Piston_moy'] = np.copy(config.FS['OPD2Piston'])
        if config.TELref:
            iTELref = config.TELref - 1
            L_ref = config.FS['OPD2Piston'][iTELref,:]
            config.FS['OPD2Piston'] = config.FS['OPD2Piston'] - L_ref
        
        config.FS['Piston2OPD_r'] = config.FS['Piston2OPD']
        config.FS['OPD2Piston_r'] = config.FS['OPD2Piston']
        config.FS['OPD2Piston_moy_r'] = config.FS['OPD2Piston_moy']
        
        
        
        
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



