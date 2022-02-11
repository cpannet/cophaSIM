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
        # ich = np.array([12,13,23,24,14,15,25,16,26,36,34,35,45,46,56])
        ich = np.array([[1,2], [1,3], [2,3], [2,4], [1,4], [1,5], [2,5], [1,6],[2,6],\
                  [3,6],[3,4],[3,5],[4,5],[4,6],[5,6]])
        
        ichorder = [0,1,4,5,7,2,3,6,8,10,11,9,12,13,14] ; NIN=15
        active_ich = np.ones(NIN)
        
        config.FS['name'] = 'PW6-15-10_perfect'
        config.FS['func'] = SPICAFS_PERFECT
        config.FS['ich'] = ich
        config.FS['ichorder'] = ichorder
        config.FS['active_ich'] = active_ich
        config.FS['PhotometricSNR'] = np.ones(NIN)   # TV² of the baselines normalised by its value for equal repartition on all baselines.
        
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
        config.FS['MW']=MW
        
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
        
        # The matrix of the elements norm only for the calculation of the bias of |Cf|².
        # /!\ To save time, it's in [NIN,NP]
        config.FS['ElementsNormDemod'] = np.zeros([MW,NIN,NP])
        for imw in range(MW):
            ElementsNorm = config.FS['MacroP2VM'][imw]*np.conj(config.FS['MacroP2VM'][imw])
            config.FS['ElementsNormDemod'][imw] = ct.NB2NIN(ElementsNorm.T).T
        
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
        
        if config.TELref:
            iTELref = config.TELref - 1
            L_ref = config.FS['OPD2Piston'][iTELref,:]
            config.FS['OPD2Piston'] = config.FS['OPD2Piston'] - L_ref
        
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


def SPICAFS_REALISTIC(*args,T=1, init=False, spectra=[], spectraM=[], phaseshifts=[-1,0,1,2],transmissions=[1,1,1,1]):
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
        ichorder = [0,1,4,5,7,2,3,6,8,10,11,9,12,13,14] ; NIN=15
        active_ich = np.ones(NIN)
        
        config.FS['name'] = 'PW6-15-10_realistic'
        config.FS['func'] = SPICAFS_REALISTIC
        config.FS['ich'] = ich
        config.FS['ichorder'] = ichorder
        config.FS['active_ich'] = active_ich
        
        NG = np.shape(ich)[0]       # should always be equal to NIN
        
        # Classic balanced ABCD modulation of each baseline
        
        M_ABCD = realisticABCDmod(phaseshifts, transmissions)          # A2P ABCD modulation
        NMod = len(M_ABCD)          # Number of modulations for each baseline
        config.FS['Modulation'] = ['A','B','C','D']
        ABCDind = [0,1,2,3]
        config.FS['ABCDind'] = ABCDind
        config.FS['Phaseshifts'] = [k*np.pi/2 for k in phaseshifts]
        
        NP = NMod*NG
        
        config.FS['NMod'] = NMod
        config.FS['NP'] = NP
        
        NIN = NP//NMod
        OrderingIndex = np.zeros(NP,dtype=np.int8)
        for ib in range(NIN):
            for k in range(NMod):
                OrderingIndex[ib*NMod+k] = ichorder[ib]*NMod+ABCDind[k]
        
        config.FS['orderingindex'] = OrderingIndex
        
        
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
        config.FS['MW']=MW
        
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
        config.FS['active_ich'] = np.ones(NIN)
        config.FS['PhotometricSNR'] = np.ones(NIN)   # TV² of the baselines normalised by its value for equal repartition on all baselines.
        
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
        
        if config.TELref:
            iTELref = config.TELref - 1
            L_ref = config.FS['OPD2Piston'][iTELref,:]
            config.FS['OPD2Piston'] = config.FS['OPD2Piston'] - L_ref
            
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



def SPICAFS_TRUE(*args, init=False, T=0.5, wlinfo=False, **kwargs):
    """
    Init: Reads the fitsfile and load the different parameters NP, ich, T, 
    Modulation, spectra into the config module.
    Run: Takes true current Coherent Flux, calculates the image, add noise 
    and estimates noisy Coherent Flux.

    Parameters
    ----------
    *args : TYPE
        - if init: expect fitsfile: string
        FITS file of the SPICA's fringe sensor V2PM
        - if run: expect currCfTrue: ARRAY [NW,NB]
        
    init : BOOLEAN, optional
        If True, initialising mode. The default is False.
        If False, running mode.
    T : float, optional
        Transmission of the CHIP. The default is 1.
    wlinfo : float, optional
        If True, returns the extremal wavelength. The default is False.
    **kwargs : TYPE
        if OW given: oversample the macro spectra with the OW factor.

    Raises
    ------
    Exception
        The oversampling must be integer. Given: {OW}.

    Returns
    -------
    If Init: empty
    If wlinfo: TUPLE
        Extremal wavelengths.
    If Run: ARRAY [MW,NB]
        Estimated macro coherent flux.

    """

    # print(config.FS)
    # config?
    
    if wlinfo:
        
        fitsfile = kwargs['fitsfile']
        
        hdul = fits.open(fitsfile)
        
        # detectordico=hdul[1].data       # Base and ABCD positions information
        wldico = hdul[2].data           # Wavelength information
    
        # We pick up the calibrated wavelengths and corresponding wavebands
        spectraM = wldico['EFF_WAVE']*1e6       # Convert to [µm]
        wavebandv2pm = np.abs(wldico['EFF_BAND'])*1e6   # Convert to [µm]
    
        minspectra = spectraM[0]-wavebandv2pm[0]/2
        maxspectra = spectraM[-1]+wavebandv2pm[-1]/2
        
        print(f'The sensor passband is from {minspectra}µm to {maxspectra}µm.')
        
        return minspectra, maxspectra
    
    if init:
        
        from .config import NW,spectra
        fitsfile = kwargs['fitsfile']
        
        hdul = fits.open(fitsfile)
        
        detectordico=hdul[1].data       # Base and ABCD positions information
        wldico = hdul[2].data           # Wavelength information
        v2pmdico = hdul[3].data         # contains the V2PM
            
        config.FS['name'] = 'SPICAFS_PW_real'
        config.FS['func'] = SPICAFS_TRUE
        # We read the interferometric channels indices and modulation patterns
        ichraw = detectordico['BEAM_INDEX']
        # Modulation = list(detectordico['ABCD_INDEX'][:4])
        Modulation = 'BDAC'
        ABCDind = [2,0,3,1]
        NMod = len(Modulation)
        NP = len(ichraw)
        
        config.FS['Modulation'] = Modulation
        config.FS['ABCDind'] = ABCDind
        config.FS['NMod'] = NMod
        config.FS['NP'] = NP
        config.FS['ich'] = np.array([(ichraw[i]) for i in range(0,NP,NMod)])
        
        NIN=NP//NMod ; NA=np.max(ichraw)
        ichorder=np.zeros(NIN,dtype=np.int)
        for ib in range(NIN):
            ia,iap=ichraw[ib]
            ibconventional=ct.posk(ichraw[ib*NMod,0]-1,ichraw[ib*NMod,1]-1,NA)
            ichorder[ibconventional] = ib
            
        config.FS['ichorder'] = ichorder
        active_ich = np.ones(NIN)
        config.FS['active_ich'] = active_ich
        config.FS['PhotometricSNR'] = np.ones(NIN)   # TV² of the baselines normalised by its value for equal repartition on all baselines.
        
        OrderingIndex = np.zeros(NP,dtype=np.int8)
        for ib in range(NIN):
            for k in range(NMod):
                OrderingIndex[ib*NMod+k] = ichorder[ib]*NMod+ABCDind[k]
        
        config.FS['orderingindex'] = OrderingIndex
        config.FS['T'] = T
        
        # We pick up the calibrated wavelengths and corresponding wavebands
        spectraM = wldico['EFF_WAVE']*1e6       # Convert to [µm]
        wavebandv2pm = np.abs(wldico['EFF_BAND'])*1e6   # Convert to [µm]
        
        # Spectral resolution of the first spectral channel
        config.FS['R'] = spectraM[0]/wavebandv2pm[0]
        
        # Number of wavelengths in V2PM and in the reference spectra
        MW = len(spectraM)
        config.FS['MW'] = len(spectraM)
        
        # Noise maps
        config.FS['imsky']=np.zeros([MW,NP])             # Sky background (bias)
        np.random.seed(config.seedron)
        config.FS['sigsky']=np.random.randn(MW,NP)       # Dark Noise [ron=1rms]

        # Now we pike up the P2VM and sort it according to wavelengths
        V2PM = v2pmdico['V2PM']
        
        if spectraM[0] > spectraM[-1]:      # First wavelength higher than last wl
            V2PM = np.flip(V2PM,0)
            spectraM = np.flip(spectraM)
            
        
        # NP = np.shape(V2PM)[1]
        NB = np.shape(V2PM)[2]
        
        # print(config.FS)
        # ich=[]
        # NQ = 2
        # NIN = 2*NQ      
        # for ip in range(NP//4):      # We assume ABCD are logically sorted (ABCD)
        #     ich.append(list(detectordico['BEAM_INDEX'][4*ip]-1))
        # ich = np.array(ich)+1
        
        # From the undersampled V2PM we calculates the corresponding P2VM and build 
        # the oversampled V2PM and P2VM
        
        if 'OW' in kwargs.keys():
            OW = kwargs['OW']
            if not isinstance(kwargs['OW'],int):
                raise Exception(f'The oversampling must be integer. Given: {OW}')
            MacroV2PMgrav = V2PM
            # for imw in range(MW):
                
            spectra,spectraM = ct.oversample_wv(spectraM,OW,
                                             spectraband=wavebandv2pm,
                                             mode='linear_wl')
            NW = len(spectra)
            MicroV2PMgrav = np.zeros([NW,NP,NB])
            MicroP2VMgrav = np.zeros([NW,NB,NP])
            # MicroP2VM = np.zeros([NW,NB,NP])*1j
            iot,imw = 0,0
            for iw in range(NW):
                if iot==OW:
                    imw+=1
                    iot=0
                MicroV2PMgrav[iw] = MacroV2PMgrav[imw]
                MicroP2VMgrav[iw] = np.linalg.pinv(MicroV2PMgrav[iw])
            
            MicroV2PM, MicroP2VM = ct.coh__GRAV2simu(MicroV2PMgrav)
            
            config.MW = len(spectraM)

            config.FS['V2PM'] = MicroV2PM
            config.FS['P2VM'] = MicroP2VM
            
            config.FS['V2PMgrav'] = MicroV2PMgrav
            config.FS['P2VMgrav'] = MicroP2VMgrav
            
            MacroP2VM = np.ones([MW,NB,NP])*1j
            MacroP2VMgrav = np.ones([MW,NB,NP])
            for imw in range(MW):
                MacroP2VM[imw] = MicroP2VM[imw*OW]
                MacroP2VMgrav[imw] = MicroP2VMgrav[imw*OW]
                
            config.FS['MacroP2VM'] = MacroP2VM
            config.FS['MacroP2VMgrav'] = MacroP2VMgrav
            
            # Changes the oversampled spectra and initializes the macro spectra
            config.spectra = spectra
            config.NW = NW
            config.spectraM = spectraM
            config.MW = MW

            return spectra, spectraM
            
        else:
            MicroV2PMgrav = np.zeros([NW,NP,NB])
            MicroP2VMgrav = np.zeros([NW,NB,NP])
            newspectra=np.zeros(NW)
            bands=[[]]                  # will stock the wavelength of each band
            for imw in range(1,MW):
                bands.append([])
            
            k=0    
            for iw in range(NW):
                # Test if the reference wavelength is in one of the FS spectral channels
                for imw in range(MW):
                    is_between = spectraM[imw]-wavebandv2pm[imw]/2 <= spectra[iw] <= \
                        spectraM[imw]+wavebandv2pm[imw]/2
                    if is_between:      # If it is, we add it to the new p2vm
                        MicroV2PMgrav[k] = V2PM[imw]
                        MicroP2VMgrav[k] = np.linalg.pinv(MicroV2PMgrav[k])
                        newspectra[k] = spectra[iw]          # The wavelength is added to the final spectra
                        bands[imw].append(k)
                        k+=1
                        break
            
            # We adapt it to the simulation formalism and generate pseudo-inverse matrix
            MicroV2PM, MicroP2VM = ct.coh__GRAV2simu(MicroV2PMgrav[:k])
            
            newspectra = newspectra[:k]
            NW = k
            MicroV2PM = MicroV2PM[:k]
            MicroP2VM = MicroP2VM[:k]
            
            r = NW%MW
            if r!=0:       # in order to have an integer oversampling
                to_delete = []          # Stock the indices of elements to remove
                for i in range(r):
                    bandwidths = [len(band) for band in bands]
                    maxband=np.argmax(bandwidths)
                    to_delete.append(bands[maxband].pop(-1))
                    
                newspectra = np.delete(newspectra,to_delete)
                MicroV2PM = np.delete(MicroV2PM, to_delete,axis=0)
                MicroP2VM = np.delete(MicroP2VM, to_delete,axis=0)
                
                NW = len(newspectra)
                spectra = newspectra
                
            MacroP2VM = np.zeros([MW,NB,NP])*1j
            MacroP2VMgrav = np.zeros([MW,NB,NP])
            for imw in range(MW):
                iw = bands[imw][0]
                MacroP2VM[imw] = MicroP2VM[iw]
                MacroP2VMgrav[imw] = MicroP2VMgrav[iw]
    
            config.FS['V2PM'] = MicroV2PM
            config.FS['P2VM'] = MicroP2VM
            config.FS['MacroP2VM'] = MacroP2VM
            
            config.FS['V2PMgrav'] = MicroV2PMgrav
            config.FS['P2VMgrav'] = MicroP2VMgrav
            config.FS['MacroP2VMgrav'] = MacroP2VMgrav
        
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
            
            if config.TELref:
                iTELref = config.TELref - 1
                L_ref = config.FS['OPD2Piston'][iTELref,:]
                config.FS['OPD2Piston'] = config.FS['OPD2Piston'] - L_ref
        
        
            # Changes the oversampled spectra and initializes the macro spectra
            config.spectra = newspectra
            config.NW = NW
            config.spectraM = spectraM
            config.MW = MW
        
            return
    
    from .config import NB, NW, MW, OW
    from . import simu
    
    it = simu.it
    
    iow = 0
    imw=0
    image_iw = np.zeros(config.FS['NP'])
    
    currCfTrue = args[0]*config.FS['T']               # Transmission of the CHIP
 
    for iw in range(config.NW):
        
        Modulation = config.FS['V2PM'][iw,:,:]
        image_iw = np.real(np.dot(Modulation,currCfTrue[iw,:]))
        
        simu.MacroImages[it,imw,:] += image_iw      # Integrate flux into spectral channel
        
        iow += 1
        if iow == OW:
            imw+=1
            iow = 0
            
    if config.noise:
        from .skeleton import addnoise
        simu.MacroImages[it,:,:] = addnoise(simu.MacroImages[it,:,:])
    
    # if np.min(simu.MacroImages[it]) < 0:
    #         print(f'Negative image value at t={it}')
            
    # estimates coherences
    currCfEstimated = np.zeros([MW,NB])*1j
    for imw in range(MW):
        Demodulation = config.FS['MacroP2VM'][imw,:,:]
        currCfEstimated[imw,:] = np.dot(Demodulation,simu.MacroImages[it,imw,:])/config.FS['T']
    
    
    return currCfEstimated


def SPICAFS_TRUE2(*args, init=False, OW=10, wlinfo=False, **kwargs):
    
    """
    Read the fits file of SPICA's fringe sensor and returns its V2PM, P2VM and 
    interferometric channels.

    Parameters
    ----------
    fitsfile: string
        FITS file of the SPICA's fringe sensor V2PM

    Returns
    -------
    V2PM : [NW,NP,NB] floats
        Visibility to Pixels matrix
    P2VM : [NW, NB, NP] floats
        Pixel to Visibility matrix
    ich : [NP, 2] int  (it might change later for ABCD sorting info -> [NP,3])
        Interferometric channel sorting
    """
    
    if wlinfo:
        
        fitsfile = kwargs['fitsfile']
        
        hdul = fits.open(fitsfile)
        
        # detectordico=hdul[1].data       # Base and ABCD positions information
        wldico = hdul[2].data           # Wavelength information
    
        # We pick up the calibrated wavelengths and corresponding wavebands
        spectraM = wldico['EFF_WAVE']*1e6       # Convert to [µm]
        wavebandv2pm = np.abs(wldico['EFF_BAND'])*1e6   # Convert to [µm]
    
        minspectra = spectraM[0]-wavebandv2pm[0]/2
        maxspectra = spectraM[-1]+wavebandv2pm[-1]/2
        
        print(f'The sensor passband is from {minspectra}µm to {maxspectra}µm.')
        
        return minspectra, maxspectra
    
    if init:
        
        # from .config import NW,spectra
        fitsfile = kwargs['fitsfile']
        
        hdul = fits.open(fitsfile)
        
        detectordico=hdul[1].data       # Base and ABCD positions information
        wldico = hdul[2].data           # Wavelength information
        v2pmdico = hdul[3].data         # contains the V2PM
            
        # hdul.close()
        
        # We read the interferometric channels indices and modulation patterns
        ichraw = detectordico['BEAM_INDEX']
        Modulation = ''.join(list(detectordico['ABCD_INDEX'][:4]))
        NMod = len(Modulation)
        NP = len(ichraw)
        
        config.FS['Modulation'] = Modulation
        config.FS['NMod'] = NMod
        config.FS['NP'] = NP
        config.FS['ich'] = np.array([(ichraw[i]) for i in range(0,NP,NMod)])
        
        # We pick up the calibrated wavelengths and corresponding wavebands
        spectraM = wldico['EFF_WAVE']*1e6               # Convert to [µm]
        wavebandv2pm = np.abs(wldico['EFF_BAND'])*1e6   # Convert to [µm]
            
        # Spectral resolution of the first spectral channel
        config.FS['R'] = spectraM[0]/wavebandv2pm[0]
        
        # Number of wavelengths in V2PM and in the reference spectra
        MW = len(spectraM)

        # Now we pike up the P2VM and sort it according to wavelengths
        V2PM = v2pmdico['V2PM']
        
        if spectraM[0] > spectraM[-1]:      # First wavelength higher than last wl
            V2PM = np.flip(V2PM,0)
            spectraM = np.flip(spectraM)
            
        
        # NP = np.shape(V2PM)[1]
        NB = np.shape(V2PM)[2]
        
        
        # ich=[]
        # NQ = 2
        # NIN = 2*NQ      
        # for ip in range(NP//4):      # We assume ABCD are logically sorted (ABCD)
        #     ich.append(list(detectordico['BEAM_INDEX'][4*ip]-1))
        # ich = np.array(ich)+1
        
        # From the undersampled V2PM we calculates the corresponding P2VM and build 
        # the oversampled V2PM and P2VM
        
        
        if not isinstance(OW,int):
            raise f'The oversampling must be integer. Given: {OW}'
        
        MacroV2PM = V2PM
        spectra,spectraM = ct.oversample_wv(spectraM,OW,
                                             spectraband=wavebandv2pm,
                                             mode='linear_wl')
        
        NW = len(spectra)

        MicroV2PM = np.zeros([NW,NP,NB])*1j
        iot,imw = 0,0
        for iw in range(NW):
            if iot==OW:
                imw+=1
                iot=0
            MicroV2PM[iw] = MacroV2PM[imw]
        
        MicroV2PM, MicroP2VM = ct.coh__GRAV2simu(MicroV2PM)
        
        config.FS['V2PM'] = MicroV2PM
        config.FS['P2VM'] = MicroP2VM 
        config.spectra = spectra
        config.NW = NW
        config.spectraM = spectraM
        config.FS['MW'] = MW
        config.OW = OW
        
        return
    
    
    from .config import NB, NW, MW, OW
    from . import simu
    
    it = simu.it
    
    iow = 0
    imw=0
    image_iw = np.zeros(config.FS['NP'])
    
    currCfTrue = args[0]     # declared in argument
                          
    for iw in range(config.NW):
        
        Modulation = config.FS['V2PM'][iw,:,:]
        image_iw = np.abs(np.dot(Modulation,currCfTrue[iw,:]))
        
        simu.MacroImages[it,imw,:] += image_iw/OW
        
        iow += 1
        if iow == OW:
            imw+=1
            iow = 0
            
    if config.noise:
        from . import add_camera_noise
        simu.MacroImages[it,:,:] = add_camera_noise(simu.MacroImages[it,:,:])
    
    # if np.min(simu.MacroImages[it]) < 0:
    #         print(f'Negative image value at t={it}')
            
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
    
    
