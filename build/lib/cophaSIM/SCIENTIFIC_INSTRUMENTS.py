# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 09:35:20 2022

@author: cpannetier

THIS SCRIPT GATHERS THE NOISE PROPAGATION OF THE SCIENTIFIC INSTRUMENTS
USED IN PARALLEL WITH THE FRINGE-TRACKER.
THEY TAKE AS INPUT:
    - the sequences of 
        - the coherent flux (mutual intensities)
        - the incoherent flux (photometries)
    - integration time
    - observing time

"""


import numpy as np
from . import coh_tools as ct



def SPICAVIS_severalDITs(CoherentFluxObject,ResidualOPDs, spectra,
             IntegrationTimes=np.array([20,100,500,1e3,5e3,1e4]), 
             ObservingTime=10*60*1e3):
    """
    Model of noise of SPICA-VIS. (Marc-Antoine thesis)

    Parameters
    ----------
    CoherentFluxObject : ARRAY [MW,NB]
        Mutual intensities and photometries of the object, sorted as 
        [photometries, R(Gamma), I(Gamma)]
        The spectral sampling must be the one of the science instrument.
        
    ResidualOPDs : ARRAY [NT,NIN]
        Residual OPDs after fringe-tracking.
        The sequences must be only the sequences of interest 
        (in the permanent regime)
    
    IntegrationTimes : INTEGERS ARRAY
        Array or list of the integration time tested (in ms).
        
    ObservingTime : INT
        Total observing time in ms.
        Must be an integer number of IntegrationTime, or will be modified 
        in consequence.

    Returns
    -------
    None.

    """
    
    from .config import dt
    
    MW,NB=np.shape(CoherentFluxObject)
    NA=int(np.sqrt(NB))
    NT,NIN = np.shape(ResidualOPDs)
    
    """ CAMERA CHARACTERISTICS """
    # All these performances are for a camera gain of 1000 and for low spectral
    # resolution.
    Nlambda = 2 # number of pixels recording one spectral channel.
    Ninterf = 400 # number of pixels on which the interferogram is modulated.
    Npix = Nlambda*Ninterf # Total number of pixels
    
    ron = 0.08              # e/pix for 20MHz
    cic = 0.005             # event/pix
    darkcur = 0.00011       # e/pix/sec
    F = np.sqrt(2)                   # Excess noise factor
    
    """ DESIGN CHARACTERISTICS"""
    
    Kappa = 1/6 # Interf/Photom - waiting for correct number.
    
    # The following numbers comes from the image used in 
    # IntegrationSPICA/ImagesCameras/PlotProfiles
    # It accounts for a FFT window of 400pix=5.2mm, sigma=65pix=0.845mm
    # and expansion of the peak on 6sigma in total.
    # This gives sig_f = 1/0.845=1.18mm^(-1), delta_f=1/5.2=0.19mm^(-1)
    # And finally Npicfrange = 6*sig_f / delta_f
    Npicfrange = 37
    
    # Accounts for the interferogram spread on 6sigma in total.
    Gab = 3/np.sqrt(np.pi)/Npix
    Gab_tilde = 3/np.sqrt(np.pi)
    
    Photometries = CoherentFluxObject[:,:NA]
    ObjectVsquare = CoherentFluxObject[:,NA:NA+NIN]**2+CoherentFluxObject[:,NA+NIN:]**2

    MutualIntensities = np.zeros([NT,MW,NIN])
    for iw in range(MW):
        wl = spectra[iw]
        for ib in range(NIN):
            MutualIntensities[:,iw,ib] = ObjectVsquare[iw,ib] * np.exp(2j*np.pi/wl * ResidualOPDs[:,ib])

    IntegrationTimes = np.array(IntegrationTimes)//dt * dt
    Ndit = len(IntegrationTimes)
    
    VarSquaredVis = np.zeros([Ndit,MW,NIN])
    SNR_E = np.zeros([Ndit,NIN])
    SNR_E_perSC = np.zeros([Ndit,MW,NIN])
    
    # START OF THE LOOP
    for idit in range(Ndit):
        DIT = IntegrationTimes[idit] ; OT=DIT/dt
        
        Nframes = ObservingTime//OT
                               
        """ COMPUTATION OF MEAN MUTUAL INTENSITIES"""
        MeanMutualIntensities=np.zeros([MW,NIN])
        for iframe in range(Nframes):
            timerange=range(int(iframe*OT),int((iframe+1)*OT))
            FrameMutualIntensities = MutualIntensities[timerange]
            MeanMutualIntensities += 1/Nframes*np.abs(np.sum(FrameMutualIntensities,axis=0))
    

        """ COMPUTATION OF MEAN PHOTOMETRIES"""
        MeanPhotometries = Photometries
        # MeanPhotometries=np.zeros([MW,NA])
        # for iframe in range(Nframes):
        #     FramePhotometries = Photometries[iframe*OT:(iframe+1)*OT]
        #     MeanPhotometries += 1/Nframes*np.abs(np.sum(FramePhotometries,axis=0))


        """ COMPUTATION OF SNR(|V|²) """
        for ia in range(NA):
            for iap in range(ia+1,NA):
                ib = ct.posk(ia,iap,NA)
                P1 = MeanPhotometries[:,ia]
                P2 = MeanPhotometries[:,iap]
                Pi = (P1+P2)/2*Kappa     # Mean uncoherent energy in interferogram of base ib
                
                EnergyPicFrange = MeanMutualIntensities[:,ib]**2
                
                var_mod = F**2*(Pi + (darkcur*DIT*1e-3*Npix)**2+(cic*Npix)**2)
                var_pi = var_mod + Npix*ron**2
                
                """PhotonNoise [MW,NIN] FLOAT"""
                PhotonNoise = 2*var_mod*EnergyPicFrange*Gab_tilde + var_mod**2
                
                """ReadNoise FLOAT"""
                ReadNoise = Npix*ron**2 + (Npix*ron**2)**2
                
                """CoupledTerms [MW,NIN] FLOAT""" 
                CoupledTerms = 2*Npix*ron**2*EnergyPicFrange*Gab_tilde + 2*var_mod*Npix*ron**2
                
                """Variance Coherent Flux [MW,NIN] FLOAT"""
                var_cf = (PhotonNoise+ReadNoise+CoupledTerms)/Nframes*Npicfrange
                
                """ Denominator D """
                D = P1*P2 * Kappa**2 * Gab
                
                """ Variance D """
                var_D = 2 * var_pi * Pi**2 * (Kappa**2*Gab)**2
                
                """Variance Squared Visibility"""
                VarSquaredVis[idit,:,ib] = var_cf/D**2 + var_D*(EnergyPicFrange/D**2)**2
    
                """SNR(|V|²) as computed in Mourard 2017"""
                SNR_E[idit,ib] = np.sum(EnergyPicFrange,axis=0) / np.sqrt(np.sum(var_cf))
                SNR_E_perSC[idit,:,ib] = EnergyPicFrange / np.sqrt(var_cf)
    
    return IntegrationTimes, VarSquaredVis, SNR_E, SNR_E_perSC


def SPICAVIS(CoherentFluxObject,ResidualOPDs, spectra,DIT=100,R=140):
    """
    Model of noise of SPICA-VIS. (Marc-Antoine thesis)

    Parameters
    ----------
    CoherentFluxObject : ARRAY [MW,NB]
        Mutual intensities and photometries of the object, sorted as 
        [photometries, R(Gamma), I(Gamma)]
        The spectral sampling must be the one of the science instrument.
        
    ResidualOPDs : ARRAY [NT,NIN]
        Residual OPDs after fringe-tracking.
        The sequences must be only the sequences of interest 
        (in the permanent regime)
    
    DIT : FLOAT
        Integration time tested (in ms).
    
    R : FLOAT
        Spectral resolution of the instrument.
        Necessary if spectra is a unique wavelength.

    Returns
    -------
    None.

    """
    
    from .config import dt
    
    MW,NB=np.shape(CoherentFluxObject)
    NA=int(np.sqrt(NB))
    NT,NIN = np.shape(ResidualOPDs)
    
    """ CAMERA CHARACTERISTICS """
    # All these performances are for a camera gain of 1000 and for low spectral
    # resolution.
    Nlambda = 2 # number of pixels recording one spectral channel.
    Ninterf = 400 # number of pixels on which the interferogram is modulated.
    Npix = Nlambda*Ninterf # Total number of pixels
    
    ron = 0.08              # e/pix for 20MHz
    cic = 0.005             # event/pix
    darkcur = 0.00011       # e/pix/sec
    F = np.sqrt(2)                   # Excess noise factor
    
    """ DESIGN CHARACTERISTICS"""
    
    Kappa = 1/6 # Interf/Photom - waiting for correct number.
    
    # The following numbers comes from the image used in 
    # IntegrationSPICA/ImagesCameras/PlotProfiles
    # It accounts for a FFT window of 400pix=5.2mm, sigma=65pix=0.845mm
    # and expansion of the peak on 6sigma in total.
    # This gives sig_f = 1/0.845=1.18mm^(-1), delta_f=1/5.2=0.19mm^(-1)
    # And finally Npicfrange = 6*sig_f / delta_f
    Npicfrange = 37
    
    # Accounts for the interferogram spread on 6sigma in total.
    Gab = 3/np.sqrt(np.pi)/Npix
    Gab_tilde = 3/np.sqrt(np.pi)
    
    Photometries = CoherentFluxObject[:,:NA]
    ObjectVsquare = CoherentFluxObject[:,NA:NA+NIN]**2+CoherentFluxObject[:,NA+NIN:]**2

    Lc = R*spectra

    MutualIntensities = np.zeros([NT,MW,NIN])+0j
    for iw in range(MW):
        wl = spectra[iw]
        CoherenceEnvelopModulation = np.sinc(ResidualOPDs/Lc[iw])
        for ib in range(NIN):
            MutualIntensities[:,iw,ib] = ObjectVsquare[iw,ib] * np.exp(2j*np.pi/wl * ResidualOPDs[:,ib])*CoherenceEnvelopModulation[:,ib]

    IntegrationTime = DIT//dt * dt
    
    VarSquaredVis = np.empty([MW,NIN])*np.nan
    SNR_E = np.empty([NIN])*np.nan
    SNR_E_perSC = np.empty([MW,NIN])*np.nan
    
    try:
        from . import simu
        simu.SNRnum=np.zeros([MW,NIN])*np.nan
        simu.PhNoise=np.zeros([MW,NIN])*np.nan
        simu.RNoise=np.zeros([NIN])*np.nan
        simu.CTerms=np.zeros([MW,NIN])*np.nan
        simu.var_cf=np.zeros([MW,NIN])*np.nan
    
    except:
        pass
    
    OT=DIT/dt
    Nframes = int(NT//OT)
                           
    """ COMPUTATION OF MEAN MUTUAL INTENSITIES"""
    MeanMutualIntensities=np.zeros([MW,NIN])+0j
    for iframe in range(Nframes):
        InFrame = int(iframe*OT) ; OutFrame = int((iframe+1)*OT)
        timerange=range(InFrame,OutFrame)
        FrameMutualIntensities = MutualIntensities[timerange]
        MeanMutualIntensities += 1/Nframes*np.abs(np.sum(FrameMutualIntensities,axis=0))


    """ COMPUTATION OF MEAN PHOTOMETRIES"""
    MeanPhotometries = Photometries*OT
    # MeanPhotometries=np.zeros([MW,NA])
    # for iframe in range(Nframes):
    #     FramePhotometries = Photometries[iframe*OT:(iframe+1)*OT]
    #     MeanPhotometries += 1/Nframes*np.abs(np.sum(FramePhotometries,axis=0))

    
    """ COMPUTATION OF SNR(|V|²) """
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib = ct.posk(ia,iap,NA)
            P1 = MeanPhotometries[:,ia]
            P2 = MeanPhotometries[:,iap]
            Pi = (P1+P2)/2*Kappa     # Mean uncoherent energy in interferogram of base ib

            
            EnergyPicFrange = np.abs(MeanMutualIntensities[:,ib]*np.conj(MeanMutualIntensities[:,ib]))
            
            var_mod = F**2*(Pi + (darkcur*DIT*1e-3*Npix)**2+(cic*Npix)**2)
            var_pi = var_mod + Npix*ron**2
            
            """PhotonNoise [MW,NIN] FLOAT"""
            PhotonNoise = 2*var_mod*EnergyPicFrange*Gab_tilde + var_mod**2
            
            """ReadNoise FLOAT"""
            ReadNoise = Npix*ron**2 + (Npix*ron**2)**2
            
            """CoupledTerms [MW,NIN] FLOAT""" 
            CoupledTerms = 2*Npix*ron**2*EnergyPicFrange*Gab_tilde + 2*var_mod*Npix*ron**2
            
            """Variance Coherent Flux [MW,NIN] FLOAT"""
            var_cf = (PhotonNoise+ReadNoise+CoupledTerms)/Nframes*Npicfrange
            
            """ Denominator D """
            D = P1*P2 * Kappa**2 * Gab
            
            """ Variance D """
            var_D = 2 * var_pi * Pi**2 * (Kappa**2*Gab)**2
            
            """Variance Squared Visibility"""
            VarSquaredVis[:,ib] = var_cf/D**2 + var_D*(EnergyPicFrange/D**2)**2

            """SNR(|V|²) as computed in Mourard 2017 - equation 6.14 thèse MAM"""
            SNR_E[ib] = np.mean(EnergyPicFrange*Gab_tilde,axis=0) / np.sqrt(np.mean(var_cf))
            SNR_E_perSC[:,ib] = EnergyPicFrange / np.sqrt(var_cf)
            
            # simu.VisiSquared[:,ib] = EnergyPicFrange/(P1*P2)
            try:
                simu.SNRnum[:,ib] = EnergyPicFrange*Gab_tilde
                simu.PhNoise[:,ib]=PhotonNoise
                simu.var_cf[:,ib] = var_cf
                simu.RNoise[ib] = ReadNoise
                simu.CTerms[:,ib] = CoupledTerms
            except:
                pass

    
    return IntegrationTime, VarSquaredVis, SNR_E, SNR_E_perSC





