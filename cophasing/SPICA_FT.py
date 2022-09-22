# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:44:22 2020

@author: cpannetier

The SPICA Fringe Tracker calculates the commands to send to the telescopes after 
reading the coherent flux and filtering the most noisy measurements.

INPUT: Coherent flux [MW,NBmes]

OUTPUT: Piston commands [NA]

Calculated and stored observables:


"""

import numpy as np

from .coh_tools import posk, poskfai,NB2NIN

from . import config


def updateFTparams(verbose=False,**kwargs):
    
    if verbose:
        print("Update fringe-tracker parameters:")
    for key, value in zip(list(kwargs.keys()),list(kwargs.values())):
        oldval=config.FT[key]
        if (key=='ThresholdGD') and (isinstance(value,(float,int))):
            config.FT['ThresholdGD'] = np.ones(config.FS['NINmes'])*value
        else:
            config.FT[key] = value
        if verbose:
            if isinstance(value,str):
                if "/" in value:
                    oldval = oldval.split("/")[-1]
                    value = value.split("/")[-1]
            print(f' - Parameter "{key}" changed from {oldval} to {value}')
    

def SPICAFT(*args, init=False, update=False, GainPD=0, GainGD=0, Ngd=50, roundGD='round', Ncross=1,
            search=True,SMdelay=1e3,Sweep0=20, Sweep30s=10, maxVelocity=0.300, Vfactors = [], 
            CPref=True, BestTel=2, Ncp = 300, Nvar = 5, stdPD=0.07,stdGD=0.14,stdCP=0.07,
            cmdOPD=True, switch=1, continu=True,whichSNR='gd',
            ThresholdGD=2, ThresholdPD = 1.5, ThresholdPhot = 2,ThresholdRELOCK=2,
            Threshold=True, useWmatrices=True,
            latencytime=1,usecupy=False, verbose=False,
            **kwargs_for_update):
    """
    Uses the measured coherent flux to calculate the new positions to send 
    to the delay lines. It filters the most noisy baselines in order to 
    improve the fringe tracking.
    
    INPUT:
        - If init: all the below parameters.
        - If not init: CfEstimated - Measured Coherent Flux   [MW,NB]
    
    OUTPUT:
        - currCmd: Piston Command to send to the ODL     [NA]
    
    USED OBSERVABLES:
        - config.FT
    UPDATED OBSERVABLES:
        - simu.PDEstimated: [NT,MW,NINmes] Estimated PD before subtraction of the reference
        - simu.GDEstimated: [NT,MW,NINmes] Estimated GD before subtraction of the reference
        - simu.CommandODL: Piston Command to send       [NT,NA]
        
    SUBROUTINES:
        - ReadCf
        - CommandCalc

    Parameters
    ----------
    *args : TYPE
        Expect CfEstimated.
    init : BOOLEAN, optional
        If True, initialize the below parameters.
        Needs to be called before starting the simulation.
        The default is False.
    GainPD : FLOAT, optional
        Gain PD. The default is 0.
    GainGD : FLOAT, optional
        Gain GD. The default is 0.
    Ngd : INT, optional
        Frame integration GD. The default is 1.
    roundGD : BOOLEAN, optional
        If True, the GD command is rounded to wavelength integers. 
    Ncross : INT, optional
        Separation between two spectral channels for GD calculation. 
    search : TYPE, optional
        DESCRIPTION. The default is True.
    SMdelay: FLOAT, optional
        Time to wait after losing a telescope for triggering the SEARCH command.
    Sweep0 : TYPE, optional
        DESCRIPTION. The default is 20.
    Sweep30s : TYPE, optional
        DESCRIPTION. The default is 10.
    maxVelocity : TYPE, optional
        DESCRIPTION. The default is 6.
    Vfactors : TYPE, optional
        DESCRIPTION. The default is [].
    CPref : BOOLEAN, optional
        If False, the Closure Phase is not subtracted for reference. 
        The default is True.
    Ncp : INT, optional
        Frame integration CP. The default is 1.
    Nvar : TYPE, optional
        DESCRIPTION. The default is 5.
    ThresholdGD : FLOAT, optional
        If the SNR of the estimated GD of a given baseline is lower than 
        ThresholdGD, then this baseline is weighted down.
        The default is 2.
    ThresholdPD : FLOAT, optional
        If the SNR of the estimated GD of a given baseline is lower than 
        ThresholdGD, then this baseline is weighted down.
        DESCRIPTION. The default is 1.5.
    Threshold : BOOLEAN, optional
        If False, the GD works also within a frange. Essentially for debugging and optimisation of the GD gain.
    useWmatrices: BOOLEAN, optional
        Wmatrices means Weighting matrices. If True, the weighting of the commands
        using the SNR of the measurements is used.
    latencytime : TYPE, optional
        DESCRIPTION. The default is 1.
    usecupy : BOOLEAN, optional
        If True, use the cupy module. The default is False.

    Returns
    -------
    currCmd : ARRAY [NA]
        Piston Command to send to the ODL.

    """
    
    if init:
        from .config import NA,NT
        NINmes = config.FS['NINmes']
        
        # config.R = np.abs((config.MW-1)*config.PDspectra/(config.spectraM[-1] - config.spectraM[0]))
        config.FT['Name'] = 'SPICAfromGRAVITY'
        config.FT['func'] = SPICAFT
        config.FT['Ngd'] = Ngd
        config.FT['GainGD'] = GainGD
        config.FT['GainPD'] = GainPD
        config.FT['state'] = np.zeros(NT)
        config.FT['Ncross'] = Ncross
        config.FT['Ncp'] = Ncp
        config.FT['Nvar'] = Nvar
        if isinstance(ThresholdGD,(float,int)):
            config.FT['ThresholdGD'] = np.ones(NINmes)*ThresholdGD
        else:
            if len(ThresholdGD)==NINmes:
                config.FT['ThresholdGD'] = ThresholdGD
            else:
                print(Exception(f"Length of 'ThresholdGD' ({len(ThresholdGD)}) \
parameter does'nt fit the number of measured baselines ({NINmes})\n\
I set ThresholdGD to the {NINmes} first values."))
                config.FT['ThresholdGD'] = ThresholdGD[:NINmes]
            
        if isinstance(ThresholdRELOCK,(float,int)):
            config.FT['ThresholdRELOCK'] = np.ones(NINmes)*ThresholdRELOCK
        else:
            if len(ThresholdRELOCK)==NINmes:
                config.FT['ThresholdRELOCK'] = ThresholdRELOCK
            else:
                print(Exception(f"Length of 'ThresholdRELOCK' ({len(ThresholdRELOCK)}) \
parameter does'nt fit the number of measured baselines ({NINmes})\n\
I set ThresholdRELOCK to the {NINmes} first values."))
                config.FT['ThresholdRELOCK'] = ThresholdRELOCK[:NINmes]
            
        config.FT['ThresholdPD'] = ThresholdPD
        config.FT['stdPD'] = stdPD
        config.FT['stdGD'] = stdGD
        config.FT['stdCP'] = stdCP
        config.FT['CPref'] = CPref
        config.FT['BestTel'] = BestTel
        config.FT['roundGD'] = roundGD
        config.FT['Threshold'] = Threshold
        config.FT['switch'] = switch
        config.FT['continu'] = continu
        config.FT['cmdOPD'] = cmdOPD
        config.FT['useWmatrices'] = useWmatrices
        config.FT['usecupy'] = usecupy
        config.FT['whichSNR'] = whichSNR
        
        # Search command parameters
        config.FT['search'] = search
        config.FT['SMdelay'] = SMdelay          # Waiting time before launching search
        config.FT['Sweep0'] = Sweep0            # Starting sweep in seconds
        config.FT['Sweep30s'] = Sweep30s        # Sweep at 30s in seconds
        config.FT['maxVelocity'] = maxVelocity  # Maximal slope given in µm/frame
        
        # Version usaw vector
        config.FT['usaw'] = np.zeros([NT,NA])
        config.FT['LastPosition'] = np.zeros(NA)
        config.FT['it_last'] = np.zeros(NA)
        config.FT['it0'] = np.zeros(NA)
        config.FT['eps'] = np.ones(NA)
        
        # Version usaw float
        # config.FT['usaw'] = np.zeros([NT])
        # config.FT['usearch'] = np.zeros([NT,NA])
        # config.FT['LastPosition'] = np.zeros([NT+1,NA])
        # config.FT['it_last'] = 0
        # config.FT['it0'] = 0
        # config.FT['eps'] = 1
        
        
        config.FT['ThresholdPhot'] = ThresholdPhot      # Minimal photometry SNR for launching search

        if len(Vfactors) != 0:
            config.FT['Vfactors'] = np.array(Vfactors)
        elif NA==10:
            config.FT['Vfactors'] = np.array([-24.9, -23.9, -18.9, -14.9,
                                              -1.9,   1.1,   9.1,  16.1,
                                              28.1, 30.1])
        elif NA==6:
            config.FT['Vfactors'] = np.array([-8.25, -7.25, -4.25, 1.75, 3.75, 8.75])/8.75
            
        elif NA==7: # Fake values
            config.FT['Vfactors'] = np.array([-8.25, -7.25, -4.25, 1.75, 3.75, 8.75, 10])
            
        config.FT['Velocities'] = config.FT['Vfactors']/np.ptp(config.FT['Vfactors'])*maxVelocity     # The maximal OPD velocity is equal to slope/frame
        
        return

    elif update:
        if verbose:
            print("Update fringe-tracker parameters with:")
        for key, value in zip(list(kwargs_for_update.keys()),list(kwargs_for_update.values())):
            setattr(config.FT, key, value)
            if verbose:
                print(f" - {key}: {getattr(config.FT, key, value)}")

        return
    
    from . import simu

    it = simu.it
    
    currCfEstimated = args[0]

    CfPD, CfGD = ReadCf(currCfEstimated)
    
    currCmd = CommandCalc(CfPD, CfGD)
    
    return currCmd


def ReadCf(currCfEstimated):
    """
    From measured coherent flux, estimates GD, PD, CP, Photometry, Visibilities
    
    NAME: 
        COH_ALGO - Calculates the group-delay, phase-delay, closure phase and 
        visibility from the fringe sensor image.
    
        
    INPUT: CfEstimated [MW,NB]
    
    OUTPUT: 
        
    UPDATE:
        - simu.CfEstimated_             
        - simu.CfPD: Coherent Flux Phase-Delay     [NT,MW,NINmes]
        - simu.CfGD: Coherent Flux GD              [NT,MW,NINmes]
        - simu.ClosurePhasePD                       [NT,MW,NC]
        - simu.ClosurePhaseGD                       [NT,MW,NC]
        - simu.PhotometryEstimated                  [NT,MW,NA]
        - simu.VisibilityEstimated                    [NT,MW,NIN]*1j
        - simu.CoherenceDegree                      [NT,MW,NIN]
    """
    
    from . import simu
    
    from .config import NA,NC
    from .config import MW
    
    it = simu.it            # Time
     
    NINmes = config.FS['NINmes']
    """
    Photometries and CfNIN extraction
    [NT,MW,NA]
    """
    
    PhotEst = np.abs(currCfEstimated[:,:NA])
    
    currCfEstimatedNIN = np.zeros([MW, NINmes])*1j
    for ib in range(NINmes):
        currCfEstimatedNIN[:,ib] = currCfEstimated[:,NA+ib] + 1j*currCfEstimated[:,NA+NINmes+ib]
            
    # Save coherent flux and photometries in stack
    simu.PhotometryEstimated[it] = PhotEst

        
    """
    Visibilities extraction
    [NT,MW,NIN]
    """
    
    for ib in range(NINmes):
        ia = int(config.FS['ich'][ib][0])-1
        iap = int(config.FS['ich'][ib][1])-1
        Iaap = currCfEstimatedNIN[:,ib]         # Interferometric intensity (a,a')
        Ia = PhotEst[:,ia]                      # Photometry pupil a
        Iap = PhotEst[:,iap]                    # Photometry pupil a'
        simu.VisibilityEstimated[it,:,ib] = 2*Iaap/(Ia+Iap)          # Estimated Fringe Visibility of the base (a,a')
        simu.SquaredCoherenceDegree[it,:,ib] = np.abs(Iaap)**2/(Ia*Iap)      # Spatial coherence of the source on the base (a,a')
 
 
    """
    Phase-delays extraction
    PD_ is a global stack variable [NT, NIN]
    Eq. 10 & 11
    """
    D = 0   # Dispersion correction factor: so far, put to zero because it's a 
            # calibration term (to define in coh_fs?)
            
    LmbdaTrack = config.PDspectra
    
    # Coherent flux corrected from dispersion
    for imw in range(MW):
        simu.CfPD[it,imw,:] = currCfEstimatedNIN[imw,:]*np.exp(1j*D*(1-LmbdaTrack/config.spectraM[imw])**2)
        
        # If ClosurePhase correction before wrapping
        # simu.CfPD[it,imw] = simu.CfPD[it,imw]*np.exp(-1j*simu.PDref[it])

    """
    Group-Delays extration
    GD_ is a global stack variable [NT, NIN]
    Eq. 15 & 16
    """
    
    if MW <= 1:
        raise ValueError('Tracking mode = GD but no more than one wavelength. \
                         Need several wavelengths for group delay')              # Group-delay calculation
    
    Ngd = config.FT['Ngd']                 # Group-delay DIT
    Ncross = config.FT['Ncross']           # Distance between wavelengths channels for GD calculation
    
    if it < Ngd:
        Ngd = it+1
    
    # Integrates GD with Ngd last frames (from it-Ngd to it)
    timerange = range(it+1-Ngd,it+1)
    for iot in timerange:
        cfgd = simu.CfPD[iot]*np.exp(-1j*simu.PDEstimated[iot])/Ngd
        
        # If ClosurePhase correction before wrapping
        # cfgd = cfgd*np.exp(-1j*simu.GDref[it])
        
        simu.CfGD[it,:,:] += cfgd
    
    CfPD = simu.CfPD[it]
    CfGD = simu.CfGD[it]

    return CfPD, CfGD



def CommandCalc(CfPD,CfGD):
    """
    Generates the command to send to the optical delay line according to the
    group-delay and phase-delay reduced from the OPDCalculator.

    Parameters
    ----------
    currPD : TYPE
        DESCRIPTION.
    currGD : TYPE
        DESCRIPTION.

    Returns
    -------
    cmd_odl : TYPE
        DESCRIPTION.
    """
    
    from . import simu
    
    """
    INIT MODE
    """
    
    from .config import NA,NC
    from .config import FT,FS
    
    it = simu.it            # Frame number
    
    NINmes = config.FS['NINmes']
    
    """
    Signal-to-noise ratio of the fringes ("Phase variance")
    The function getvar saves the inverse of the squared SNR ("Phase variance")
    in the global stack variable varPD [NT, MW, NIN]
    Eq. 12, 13 & 14
    """
            
    Ngd = FT['Ngd']
    if it < FT['Ngd']:
        Ngd = it+1
    
    Ncross = config.FT['Ncross']  # Distance between wavelengths channels for GD calculation
    
    R = config.FS['R']
    
    """
    WEIGHTING MATRIX
    """

    if config.FT['useWmatrices']:
        
        varcurrPD, varcurrGD = getvar()
        
        # Raw Weighting matrix in the OPD-space
        
        timerange = range(it+1-Ngd, it+1)
        simu.SquaredSNRMovingAveragePD[it] = np.nan_to_num(1/np.mean(simu.varPD[timerange], axis=0))
        simu.SquaredSNRMovingAverageGD[it] = np.nan_to_num(1/np.mean(simu.varGD[timerange], axis=0))
        simu.SquaredSNRMovingAverageGDUnbiased[it] = np.nan_to_num(1/np.mean(simu.varGDUnbiased[timerange], axis=0))
        
        simu.TemporalVariancePD[it] = np.var(simu.PDEstimated[timerange], axis=0)
        simu.TemporalVarianceGD[it] = np.var(simu.GDEstimated[timerange], axis=0)
        
        if config.FT['whichSNR'] == 'pd':
            simu.SquaredSNRMovingAverage[it] = simu.SquaredSNRMovingAveragePD[it]
        else:
            simu.SquaredSNRMovingAverage[it] = simu.SquaredSNRMovingAverageGD[it]
            
        reliablebaselines = (simu.SquaredSNRMovingAverage[it] >= FT['ThresholdGD']**2)
        
        simu.TrackedBaselines[it] = reliablebaselines
        
        Wdiag=np.zeros(NINmes)
        if config.FT['whichSNR'] == 'pd':
            Wdiag[reliablebaselines] = 1/varcurrPD[reliablebaselines]
        else:
            Wdiag[reliablebaselines] = 1/varcurrGD[reliablebaselines]
            
        W = np.diag(Wdiag)
        # Transpose the W matrix in the Piston-space
        MtWM = np.dot(FS['OPD2Piston_r'], np.dot(W,FS['Piston2OPD_r']))
        
        # Singular-Value-Decomposition of the W matrix
        U, S, Vt = np.linalg.svd(MtWM)
        
        Ut = np.transpose(U)
        V = np.transpose(Vt)
        
        """
        GD weighting matrix
        """
        
        reliablepistons = (S>1e-4)  #True at the positions of S verifying the condition
        Sdag = np.zeros([NA,NA])
        Sdag[reliablepistons,reliablepistons] = 1/S[reliablepistons]
        
        # Come back to the OPD-space        
        VSdagUt = np.dot(V, np.dot(Sdag,Ut))
        
        # Calculates the weighting matrix
        currIgd = np.dot(FS['Piston2OPD_r'],np.dot(VSdagUt,np.dot(FS['OPD2Piston_r'], W)))
    
    
        """
        PD Weighting matrix
        """

        Sdag = np.zeros([NA,NA])
        reliablepistons = (S >= config.FT['ThresholdPD']**2)
        notreliable = (reliablepistons==False)
        
        diagS = np.zeros([NA])
        diagS[reliablepistons] = 1/S[reliablepistons]
        diagS[notreliable] = 0#S[notreliable]/FT['ThresholdPD']**4
        Sdag = np.diag(diagS)
        
        # Come back to the OPD-space
        VSdagUt = np.dot(V, np.dot(Sdag,Ut))
        
        # Calculates the weighting matrix
        currIpd = np.dot(FS['Piston2OPD_r'],np.dot(VSdagUt,np.dot(FS['OPD2Piston_r'], W)))
            
    else:
        currIgd = np.identity(NINmes)
        currIpd = np.identity(NINmes)
    
    simu.Igd[it,:,:] = currIgd
    simu.Ipd[it,:,:] = currIpd
        
    """
    Closure phase calculation
    cpPD_ is a global stack variable [NT, NC]
    cpGD_ is a global stack variable [NT, NC]
    Eq. 17 & 18
    """
    
    # Ncp = config.FT['Ncp']
    
    # if it < Ncp:
    #     Ncp = it+1
        
    # bispectrumPD = np.zeros([NC])*1j
    # bispectrumGD = np.zeros([NC])*1j
    
    # timerange = range(it+1-Ncp,it+1) ; validcp=np.zeros(NC)
    # for ia in range(NA):
    #     for iap in range(ia+1,NA):
    #         ib = posk(ia,iap,NA)      # coherent flux (ia,iap)  
    #         valid1=(config.FS['active_ich'][ib] and simu.TrackedBaselines[it,ib])
    #         cs1 = np.sum(simu.CfPD[timerange,:,ib], axis=1)     # Sum of coherent flux (ia,iap)
    #         cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
    #         cfGDmoy1 = np.sum(cfGDlmbdas,axis=1)     # Sum of coherent flux (ia,iap)  
    #         for iapp in range(iap+1,NA):
    #             ib = posk(iap,iapp,NA) # coherent flux (iap,iapp)    
    #             valid2=(config.FS['active_ich'][ib] and simu.TrackedBaselines[it,ib])
    #             cs2 = np.sum(simu.CfPD[timerange,:,ib], axis=1) # Sum of coherent flux (iap,iapp)    
    #             cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
    #             cfGDmoy2 = np.sum(cfGDlmbdas,axis=1)
                
    #             ib = posk(ia,iapp,NA) # coherent flux (iapp,ia)    
    #             valid3=(config.FS['active_ich'][ib] and simu.TrackedBaselines[it,ib])
    #             cs3 = np.sum(np.conjugate(simu.CfPD[timerange,:,ib]),axis=1) # Sum of 
    #             cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
    #             cfGDmoy3 = np.sum(cfGDlmbdas,axis=1)
                
    #             # The bispectrum of one time and one triangle adds up to
    #             # the Ncp last times
    #             ic = poskfai(ia,iap,iapp,NA)        # 0<=ic<NC=(NA-2)(NA-1) 
    #             validcp[ic]=valid1*valid2*valid3
    #             bispectrumPD[ic]=np.sum(cs1*cs2*cs3)
    #             bispectrumGD[ic]=np.sum(cfGDmoy1*cfGDmoy2*np.conjugate(cfGDmoy3))
    
                
    # simu.BispectrumPD[it] = bispectrumPD*validcp+simu.BispectrumPD[it-1]*(1-validcp)
    # simu.BispectrumGD[it] = bispectrumGD*validcp+simu.BispectrumGD[it-1]*(1-validcp)
    
    # cpPD = np.angle(simu.BispectrumPD[it])
    # cpGD = np.angle(simu.BispectrumGD[it])
    
    # cpPD[cpPD<-np.pi+config.FT['stdCP']]=np.pi
    # cpGD[cpGD<-np.pi+config.FT['stdCP']]=np.pi
    
    # simu.ClosurePhasePD[it] = cpPD
    # simu.ClosurePhaseGD[it] = cpPD/config.FS['R']
    
    # BestTel=config.FT['BestTel'] ; itelbest=BestTel-1
    # if config.FT['CPref'] and (it>10):                     # At time 0, we create the reference vectors
    #     for ia in range(NA-1):
    #         for iap in range(ia+1,NA):
    #             if not(ia==itelbest or iap==itelbest):
    #                 ib = posk(ia,iap,NA)
    #                 if itelbest>iap:
    #                     ic = poskfai(ia,iap,itelbest,NA)   # Position of the triangle (0,ia,iap)
    #                 elif itelbest>ia:
    #                     ic = poskfai(ia,itelbest,iap,NA)   # Position of the triangle (0,ia,iap)
    #                 else:
    #                     ic = poskfai(itelbest,ia,iap,NA)
                
    #                 simu.PDref[it,ib] = simu.ClosurePhasePD[it,ic]
    #                 simu.GDref[it,ib] = simu.ClosurePhaseGD[it,ic]   
    
    #                 simu.CfPDref[it,ib] = simu.BispectrumPD[it,ic]#/np.abs(simu.BispectrumPD[it,ic])
    #                 simu.CfGDref[it,ib] = simu.BispectrumGD[it,ic]#/np.abs(simu.BispectrumGD[it,ic])
    
    
    """ NOT WORKING BECAUSE OF NINmes
    Ncp = config.FT['Ncp']
    
    if it < Ncp:
        Ncp = it+1
        
    bispectrumPD = np.zeros([NC])*1j
    bispectrumGD = np.zeros([NC])*1j
    
    timerange = range(it+1-Ncp,it+1) ; validcp=np.zeros(NC); ic=0
    for ia in range(NA):
        for iap in range(ia+1,NA):
            for iapp in range(iap+1,NA):
                
                ib = posk(ia,iap,NA)      # coherent flux (ia,iap)  
                valid1=(config.FS['active_ich'][ib] and simu.TrackedBaselines[it,ib])
                ib = posk(iap,iapp,NA) # coherent flux (iap,iapp)    
                valid2=(config.FS['active_ich'][ib] and simu.TrackedBaselines[it,ib])
                ib = posk(ia,iapp,NA) # coherent flux (iapp,ia)    
                valid3=(config.FS['active_ich'][ib] and simu.TrackedBaselines[it,ib])
                
                if valid1*valid2*valid3:
                    cs1 = np.sum(simu.CfPD[timerange,:,ib], axis=1)     # Sum of coherent flux (ia,iap)
                    cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
                    cfGDmoy1 = np.sum(cfGDlmbdas,axis=1)     # Sum of coherent flux (ia,iap)  
                    
                    cs2 = np.sum(simu.CfPD[timerange,:,ib], axis=1) # Sum of coherent flux (iap,iapp)    
                    cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
                    cfGDmoy2 = np.sum(cfGDlmbdas,axis=1)
                    
                    cs3 = np.sum(np.conjugate(simu.CfPD[timerange,:,ib]),axis=1) # Sum of 
                    cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
                    cfGDmoy3 = np.sum(cfGDlmbdas,axis=1)
                
                # The bispectrum of one time and one triangle adds up to
                # the Ncp last times
                # ic = poskfai(ia,iap,iapp,NA)        # 0<=ic<NC=(NA-2)(NA-1) 
                validcp[ic]=valid1*valid2*valid3
                bispectrumPD[ic]=np.sum(cs1*cs2*cs3)
                bispectrumGD[ic]=np.sum(cfGDmoy1*cfGDmoy2*np.conjugate(cfGDmoy3))
                ic+=1
    
                
    simu.BispectrumPD[it] = bispectrumPD*validcp+simu.BispectrumPD[it-1]*(1-validcp)
    simu.BispectrumGD[it] = bispectrumGD*validcp+simu.BispectrumGD[it-1]*(1-validcp)
    
    cpPD = np.angle(simu.BispectrumPD[it])
    cpGD = np.angle(simu.BispectrumGD[it])
    
    cpPD[cpPD<-np.pi+config.FT['stdCP']]=np.pi
    cpGD[cpGD<-np.pi+config.FT['stdCP']]=np.pi
    
    simu.ClosurePhasePD[it] = cpPD
    simu.ClosurePhaseGD[it] = cpPD/config.FS['R']
    
    BestTel=config.FT['BestTel'] ; itelbest=BestTel-1
    if config.FT['CPref'] and (it>10):                     # At time 0, we create the reference vectors
        for ia in range(NA-1):
            for iap in range(ia+1,NA):
                if not(ia==itelbest or iap==itelbest):
                    ib = posk(ia,iap,NA)
                    if itelbest>iap:
                        ic = poskfai(ia,iap,itelbest,NA)   # Position of the triangle (0,ia,iap)
                    elif itelbest>ia:
                        ic = poskfai(ia,itelbest,iap,NA)   # Position of the triangle (0,ia,iap)
                    else:
                        ic = poskfai(itelbest,ia,iap,NA)
                
                    simu.PDref[it,ib] = simu.ClosurePhasePD[it,ic]
                    simu.GDref[it,ib] = simu.ClosurePhaseGD[it,ic]   
    
                    simu.CfPDref[it,ib] = simu.BispectrumPD[it,ic]#/np.abs(simu.BispectrumPD[it,ic])
                    simu.CfGDref[it,ib] = simu.BispectrumGD[it,ic]#/np.abs(simu.BispectrumGD[it,ic])
 """
    
    """
    GD and PD errors calculation
    """
        
    # Current Phase-Delay
    currPD = np.angle(np.sum(simu.CfPD[it,:,:], axis=0)*np.exp(-1j*simu.PDref[it]))
    # currPD = np.angle(np.sum(simu.CfPD[it,:,:], axis=0)*np.conj(simu.CfPDref[it]))*config.FS['active_ich']
    
    # Current Group-Delay
    currGD = np.zeros(NINmes)
    for ib in range(NINmes):
        cfGDlmbdas = simu.CfGD[it,:-Ncross,ib]*np.conjugate(simu.CfGD[it,Ncross:,ib])
        cfGDmoy = np.sum(cfGDlmbdas)
        
        currGD[ib] = np.angle(cfGDmoy*np.exp(-1j*simu.GDref[it,ib]))
        # currGD[ib] = np.angle(cfGDmoy*np.conj(simu.CfGDref[it,ib])*np.conj(simu.CfPDref[it,ib]**(1/config.FS['R'])))*config.FS['active_ich'][ib]

    simu.PDEstimated[it] = currPD
    simu.GDEstimated[it] = currGD
    
    # Patch to stabilize the PD and GD when too close to the Pi/-Pi shift.
    # --> force it to Pi.
    
    currPD[(currPD+np.pi)<config.FT['stdPD']]=np.pi
    currGD[(currGD+np.pi)<config.FT['stdGD']]=np.pi
    
    simu.PDEstimated2[it] = currPD
    simu.GDEstimated2[it] = currGD
    
    """
    FRINGE SEARCHING command
    """
    
    
    """ Implementation de la fonction SEARCH telle que décrite pas Lacour """
    """ La fonction sawtooth est commune a tous les télescopes """
    
    # IgdRank = np.linalg.matrix_rank(simu.Igd[it])
    # NotCophased = (IgdRank < NA-1)
    # simu.IgdRank[it] = IgdRank
    
    # if NotCophased:
    #     simu.time_since_loss[it]=simu.time_since_loss[it-1]+config.dt
        
    #     # FringeLost = (NotCophased and (IgdRank<np.linalg.matrix_rank(simu.Igd[it-1]))
    #     # This situation could pose a problem but we don't manage it yet        
    #     if (simu.time_since_loss[it] > config.FT['SMdelay']):
            
    #         Igdna = np.dot(config.FS['OPD2Piston'],
    #                         np.dot(simu.Igd[it],config.FS['Piston2OPD']))
            
    #         # Fringe loss
    #         simu.LostTelescopes[it] = (np.diag(Igdna) == 0)*1      # The positions of the lost telescopes get 1.
            
    #         # Photometry loss
    #         simu.noSignal_on_T[it] = 1*(simu.SNRPhotometry[it] < config.FT['ThresholdPhot'])
                
    #         comparison = (simu.noSignal_on_T[it] == simu.LostTelescopes[it])
    #         simu.LossDueToInjection[it] = comparison.all()       # Evaluates if the two arrays are the same
            
    #         if not simu.LossDueToInjection[it]:     # The fringe loss is not due to an injection loss
    #             config.FT['state'][it] = 1
                
    #             if (config.FT['state'][it-1] == 0):         # Last frame, all telescopes were tracked
    #                 config.FT['it0'] = it ; config.FT['it_last'] = it
    #                 config.FT['LastPosition'] = 0#np.copy(config.FT['usaw'][it-1])
        
    #             config.FT['usaw'][it] = searchfunction_basical(config.FT['usaw'][it-1], it)
            
    #             Kernel = np.identity(NA) - Igdna
    #             simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
    #             Kernel = np.dot(simu.NoPhotometryFiltration[it],Kernel)
            
    #             config.FT['usearch'][it] = np.dot(Kernel,config.FT['usaw'][it]*config.FT['Velocities'])
                
    #             # Patch pour commander en incrément comme l'algo réel
    #             SearchIncrement = config.FT['usearch'][it] - config.FT['usearch'][it-1]
                
    #         else:
    #             config.FT['state'][it] = 0
    #             SearchIncrement = 0
    #             print(it, "Loss due to injection")
    #     else:
    #         config.FT['state'][it] = 0
    #         SearchIncrement = 0
    #         print(it, "Delay short")
            
    # else:
    #     simu.time_since_loss[it] = 0
    #     config.FT['state'][it] = 0
    #     config.FT['eps'] = 1
        
    #     SearchIncrement = 0
    #     print(it, "Cophased")

    # # if config.TELref:
    # #     iTel = config.TELref-1
    # #     SearchIncrement = SearchIncrement - SearchIncrement[iTel]
    
    # SearchIncrement = config.FT['search']*SearchIncrement
    
    # # The command is sent at the next time, that's why we note it+1
    # usearch = simu.SearchCommand[it] + SearchIncrement
    
    # simu.SearchCommand[it+1] = usearch
    
    """ Implementation avec la fonction sawtooth spécifique à chaque télescope """
    
    # IgdRank = np.linalg.matrix_rank(simu.Igd[it])
    # NotCophased = (IgdRank < NA-1)
    # simu.IgdRank[it] = IgdRank
    
    # if NotCophased:
    #     simu.time_since_loss[it]=simu.time_since_loss[it-1]+config.dt
        
    #     # FringeLost = (NotCophased and (IgdRank<np.linalg.matrix_rank(simu.Igd[it-1]))
    #     # This situation could pose a problem but we don't manage it yet        
    #     if (simu.time_since_loss[it] > config.FT['SMdelay']):
            
    #         Igdna = np.dot(config.FS['OPD2Piston'],
    #                        np.dot(simu.Igd[it],config.FS['Piston2OPD']))
            
    #         # Fringe loss
    #         simu.LostTelescopes[it] = (np.diag(Igdna) == 0)*1      # The positions of the lost telescopes get 1.
    #         # WeLostANewTelescope = (sum(newLostTelescopes) > 0)
            
    #         # Photometry loss
    #         simu.noSignal_on_T[it] = 1*(simu.SNRPhotometry[it] < config.FT['ThresholdPhot'])
                
    #         comparison = (simu.noSignal_on_T[it] == simu.LostTelescopes[it])
    #         simu.LossDueToInjection[it] = comparison.all()       # Evaluates if the two arrays are the same
            
    #         if not simu.LossDueToInjection[it]:     # The fringe loss is not due to an injection loss
    #             config.FT['state'][it] = 1
                
    #             newLostTelescopes = (simu.LostTelescopes[it] - simu.LostTelescopes[it-1] == 1)
    #             TelescopesThatGotBackPhotometry = (simu.noSignal_on_T[it-1] - simu.noSignal_on_T[it] == 1)
    #             # WeGotBackPhotometry = (sum(TelescopesThatGotBackPhotometry) > 0)
                
    #             TelescopesThatNeedARestart = np.argwhere(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)
                
    #             # if (config.FT['state'][it-1] == 0):         # Last frame, all telescopes were tracked
    #             #     config.FT['it0'] = np.ones(NA)*it ; config.FT['it_last'] = np.ones(NA)*it
    #             #     config.FT['LastPosition'] = np.copy(config.FT['usaw'][it-1])
                    
    #             # elif sum(TelescopesThatNeedARestart) > 0:
                    
    #             #     # Version "Restart only concerned telescopes" (06-10-2021)
    #             #     # --> doesn't work because it avoids some OPDs.
    #             #     # for ia in TelescopesThatNeedARestart:
    #             #     #     config.FT['it0'][ia] = it ; config.FT['it_last'][ia] = it
    #             #     #     config.FT['LastPosition'][ia] = 0
                
    #             #     # Version "Restart all" (06-10-2021)
    #             #     # Restart all telescope from their current position.
    #             #     config.FT['it0'] = np.ones(NA)*it
    #             #     config.FT['it_last'] = np.ones(NA)*it
    #             #     config.FT['LastPosition'] = np.copy(config.FT['usaw'][it-1])
                    
    #             # config.FT['usaw'][it] = searchfunction(config.FT['usaw'][it-1])         # Fonction search de vitesse 1µm/frame par piston
                    
    #             # Kernel = np.identity(NA) - Igdna
    #             # simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
    #             # Kernel = np.dot(simu.NoPhotometryFiltration[it],Kernel)                 
                
    #             # # After multiplication by Kernel, the OPD velocities can only be lower or equal than before
                
    #             # usearch = np.dot(Kernel,config.FT['usaw'][it]*config.FT['Velocities'])
            
    #             if (config.FT['state'][it-1] == 0):# or (sum(TelescopesThatNeedARestart) >0) :
    #                 config.FT['it0'] = it ; config.FT['it_last'] = it
    #                 config.FT['LastPosition'] = config.FT['usaw'][it-1]
            
    #             usaw = np.copy(config.FT['usaw'][it-1])
    #             config.FT['usaw'][it] = searchfunction2(usaw,it)      # In this version, usaw is a float
            
    #             Kernel = np.identity(NA) - Igdna
    #             simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
    #             Kernel = np.dot(simu.NoPhotometryFiltration[it],Kernel)
            
    #             usearch = np.dot(Kernel,config.FT['usaw'][it]*config.FT['Velocities'])
                
    #         else:
    #             config.FT['state'][it] = 0
    #             usearch = simu.SearchCommand[it]
    #     else:
    #         config.FT['state'][it] = 0
    #         usearch = simu.SearchCommand[it]
            
    # else:
    #     simu.time_since_loss[it] = 0
    #     config.FT['state'][it] = 0
    #     # Version usaw vector
    #     # config.FT['eps'] = np.ones(NA)
        
    #     # Version usaw float
    #     config.FT['eps'] = 1
        
    #     usearch = simu.SearchCommand[it]
        
        
    # # if config.TELref:
    # #     iTel = config.TELref-1
    # #     usearch = usearch - usearch[iTel]
    
    # usearch = config.FT['search']*usearch
    # # The command is sent at the next time, that's why we note it+1
    # simu.SearchCommand[it+1] = usearch
    
    
    
    """ New implementation RELOCK command 08/02/2022 """
    
    # IgdRank = np.linalg.matrix_rank(simu.Igd[it])
    # NotCophased = (IgdRank < NA-1)
    # simu.IgdRank[it] = IgdRank
    
    # if NotCophased:
    #     simu.time_since_loss[it]=simu.time_since_loss[it-1]+config.dt
        
    #     # FringeLost = (NotCophased and (IgdRank<np.linalg.matrix_rank(simu.Igd[it-1]))
    #     # This situation could pose a problem but we don't manage it yet        
    #     if (simu.time_since_loss[it] > config.FT['SMdelay']):
            
    #         Igdna = np.dot(config.FS['OPD2Piston'],
    #                         np.dot(simu.Igd[it],config.FS['Piston2OPD']))
            
    #         CophasedBaselines=np.where(np.diag(simu.Igd[it])>0.5)[0]
    #         CophasedPairs=[]
    #         for ib in CophasedBaselines:
    #             ia,iap = config.FS['ich'][ib][0], config.FS['ich'][ib][1]
    #             CophasedPairs.append([ia,iap])
                
    #         CophasedGroups = JoinOnCommonElements(CophasedPairs)
            
    #         # Fringe loss
    #         simu.LostTelescopes[it] = (np.diag(Igdna) == 0)*1      # The positions of the lost telescopes get 1.
    #         # WeLostANewTelescope = (sum(newLostTelescopes) > 0)
            
    #         # Photometry loss
    #         simu.noSignal_on_T[it] = 1*(simu.SNRPhotometry[it] < config.FT['ThresholdPhot'])
                
    #         comparison = (simu.noSignal_on_T[it] == simu.LostTelescopes[it])
    #         simu.LossDueToInjection[it] = (comparison.all() and sum(simu.noSignal_on_T[it])>1)       # Evaluates if the two arrays are the same
            
    #         if not simu.LossDueToInjection[it]:     # The fringe loss is not due to an injection loss
    #             config.FT['state'][it] = 1

    #             # If it=0, initialize LastPosition to 0. 
    #             # Else, it will remain the last value of SearchCommand, which has
    #             # not change since last RELOCK state.
                
    #             LastPosition = config.FT['LastPosition'][it]
                
    #             newLostTelescopes = (simu.LostTelescopes[it] - simu.LostTelescopes[it-1] == 1)
    #             TelescopesThatGotBackPhotometry = (simu.noSignal_on_T[it-1] - simu.noSignal_on_T[it] == 1)
    #             # WeGotBackPhotometry = (sum(TelescopesThatGotBackPhotometry) > 0)
                
    #             TelescopesThatNeedARestart = np.argwhere(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)

    #             if sum(TelescopesThatNeedARestart)>0:
    #                 config.FT['it_last']=it ; #Ldico[ia]['eps']=1 #; Ldico[ia]['it0']=it ;   

    #             usaw, change = searchfunction_inc_basical(it)
    #             config.FT['usaw'][it]= usaw

    #             Kernel = np.identity(NA) - Igdna
    #             Increment = np.dot(Kernel,config.FT['usaw'][it]*config.FT['Velocities'])
    #             Increment = Increment/np.ptp(Increment) * config.FT['maxVelocity']
                
    #             if change:  # Change direction of scan
    #                 # Fais en sorte que les sauts de pistons de télescopes cophasés 
    #                 # entre eux maintiennent l'OPD constante: si 1 et 2 sont cophasés
    #                 # avec OPD=p2-p1, au prochain saut le télescope 2 va à la position
    #                 # du T1 + OPD et pas à la position qu'il avait avant le précédent saut.
    #                 for group in CophasedGroups:    
    #                     for ig in range(1,len(group)):
    #                         ia = int(float(group[ig])-1) ; i0 = int(float(group[0])-1)
    #                         LastPosition[ia] = LastPosition[i0] + simu.SearchCommand[it,ia]-simu.SearchCommand[it,i0]
    #                 usearch = LastPosition + Increment
    #                 LastPosition = simu.SearchCommand[it]
                    
    #             else:
    #                 usearch = simu.SearchCommand[it]+Increment
                    
    #             config.FT['LastPosition'][it+1] = LastPosition
                
    #             # You should send command only on telescope with flux
    #             simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
    #             usearch = np.dot(simu.NoPhotometryFiltration[it],usearch)
            
            
    #         else:
    #             usearch = simu.SearchCommand[it]
        
    #     else:
    #         usearch = simu.SearchCommand[it]
            
    # else:
    #     simu.time_since_loss[it] = 0
    #     usearch = simu.SearchCommand[it]
        
        
    # usearch = config.FT['search']*usearch
    # # The command is sent at the next time, that's why we note it+1
    # simu.SearchCommand[it+1] = usearch
    
    """ Implementation comme Sylvain sans réinitialisation """
    
    
    # IgdRank = np.linalg.matrix_rank(simu.Igd[it])
    # NotCophased = (IgdRank < NA-1)
    # simu.IgdRank[it] = IgdRank
    
    
    # if NotCophased:
    #     simu.time_since_loss[it]=simu.time_since_loss[it-1]+config.dt
        
    #     # FringeLost = (NotCophased and (IgdRank<np.linalg.matrix_rank(simu.Igd[it-1]))
    #     # This situation could pose a problem but we don't manage it yet        
    #     if (simu.time_since_loss[it] > config.FT['SMdelay']):
            
    #         Igdna = np.dot(config.FS['OPD2Piston'],
    #                         np.dot(simu.Igd[it],config.FS['Piston2OPD']))
            
    #         Kernel = np.identity(NA) - Igdna
            
    #         # Fringe loss
    #         simu.LostTelescopes[it] = (np.diag(Igdna) == 0)*1      # The positions of the lost telescopes get 1.
            
    #         # Photometry loss
    #         simu.noSignal_on_T[it] = 1*(simu.SNRPhotometry[it] < config.FT['ThresholdPhot'])
                
    #         comparison = (simu.noSignal_on_T[it] == simu.LostTelescopes[it])
    #         simu.LossDueToInjection[it] = (comparison.all() and sum(simu.noSignal_on_T[it])>1)       # Evaluates if the two arrays are the same
            
    #         if not simu.LossDueToInjection[it]:     # The fringe loss is not due to an injection loss
    #             config.FT['state'][it] = 1
                
    #             newLostTelescopes = (simu.LostTelescopes[it] - simu.LostTelescopes[it-1] == 1)
    #             TelescopesThatGotBackPhotometry = (simu.noSignal_on_T[it-1] - simu.noSignal_on_T[it] == 1)
                
    #             TelescopesThatNeedARestart = np.argwhere(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)

    #             if config.FT['state'][it-1] != 1:
    #                 config.FT['eps'] = np.ones(NA)
    #                 config.FT['it0'] = np.ones(NA)*it
    #                 config.FT['it_last'] = np.ones(NA)*it
                
    #             Velocities = np.dot(Kernel,config.FT['Velocities'])
    #             Increment = searchfunction_inc_sylvain(it, Velocities)    
            
    #             #You should send command only on telescope with flux
    #             simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
    #             Increment = np.dot(simu.NoPhotometryFiltration[it],Increment)
                
    #             usearch = simu.SearchCommand[it] + Increment
                
    #         else:
    #                 Increment = np.zeros(NA)
            
    #     else:
    #         Increment = np.zeros(NA)
            
    # else:
    #     simu.time_since_loss[it] = 0
    #     Increment = np.zeros(NA)
            
    # Increment = config.FT['search']*Increment
    
    # usearch = simu.SearchCommand[it] + Increment
    # # The command is sent at the next time, that's why we note it+1
    # simu.SearchCommand[it+1] = usearch

    
    
    """ Implementation comme Sylvain avec réinitialisation """
    """ PROBLEME: beaucoup d'OPD sont sautées """
    
    
    # IgdRank = np.linalg.matrix_rank(simu.Igd[it])
    # NotCophased = (IgdRank < NA-1)
    # simu.IgdRank[it] = IgdRank
    
    
    # if NotCophased:
    #     simu.time_since_loss[it]=simu.time_since_loss[it-1]+config.dt
        
    #     # FringeLost = (NotCophased and (IgdRank<np.linalg.matrix_rank(simu.Igd[it-1]))
    #     # This situation could pose a problem but we don't manage it yet        
    #     if (simu.time_since_loss[it] > config.FT['SMdelay']):
            
    #         Igdna = np.dot(config.FS['OPD2Piston'],
    #                         np.dot(simu.Igd[it],config.FS['Piston2OPD']))
            
    #         Kernel = np.identity(NA) - Igdna
            
    #         # Fringe loss
    #         simu.LostTelescopes[it] = (np.diag(Igdna) == 0)*1      # The positions of the lost telescopes get 1.
            
    #         # Photometry loss
    #         simu.noSignal_on_T[it] = 1*(simu.SNRPhotometry[it] < config.FT['ThresholdPhot'])
                
    #         comparison = (simu.noSignal_on_T[it] == simu.LostTelescopes[it])
    #         simu.LossDueToInjection[it] = (comparison.all() and sum(simu.noSignal_on_T[it])>1)       # Evaluates if the two arrays are the same
            
    #         if not simu.LossDueToInjection[it]:     # The fringe loss is not due to an injection loss
    #             config.FT['state'][it] = 1
                
    #             newLostTelescopes = (simu.LostTelescopes[it] - simu.LostTelescopes[it-1] == 1)
    #             TelescopesThatGotBackPhotometry = (simu.noSignal_on_T[it-1] - simu.noSignal_on_T[it] == 1)
    #             # WeGotBackPhotometry = (sum(TelescopesThatGotBackPhotometry) > 0)
                
    #             TelescopesThatNeedARestart = np.argwhere(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)

    #             if config.FT['state'][it-1] != 1:
    #                 config.FT['eps'] = np.ones(NA)
    #                 config.FT['it0'] = np.ones(NA)*it
    #                 config.FT['it_last'] = np.ones(NA)*it


    #             # print(TelescopesThatNeedARestart)
    #             # print(config.FT['it_last'])
    #             # print(config.FT['it0'])
    #             for ia in range(NA):
    #                 if ia in TelescopesThatNeedARestart:
    #                     config.FT['eps'][ia] = 1
    #                     config.FT['it_last'][ia] = it
    #                     config.FT['it0'][ia] = it
    #                     config.FT['LastPosition'][ia] = 0
                
    #             Velocities = np.dot(Kernel,config.FT['Velocities'])
    #             Increment = searchfunction_inc_sylvain(it, Velocities)    
            
    #             #You should send command only on telescope with flux
    #             simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
    #             Increment = np.dot(simu.NoPhotometryFiltration[it],Increment)
                
    #             usearch = simu.SearchCommand[it] + Increment
                
    #         else:
    #                 Increment = np.zeros(NA)
            
    #     else:
    #         Increment = np.zeros(NA)
            
    # else:
    #     simu.time_since_loss[it] = 0
    #     Increment = np.zeros(NA)
            
    # Increment = config.FT['search']*Increment
    
    # usearch = simu.SearchCommand[it] + Increment
    # # The command is sent at the next time, that's why we note it+1
    # simu.SearchCommand[it+1] = usearch
    
    
    """ Implementation comme Sylvain:
            - sans réinitialisation
            - avec patch pour garder groupes cophasés lors des sauts
            """
    
    
    IgdRank = np.linalg.matrix_rank(simu.Igd[it])
    NotCophased = (IgdRank < NA-1)
    simu.IgdRank[it] = IgdRank
    
    
    if NotCophased:
        simu.time_since_loss[it]=simu.time_since_loss[it-1]+config.dt
        
        # FringeLost = (NotCophased and (IgdRank<np.linalg.matrix_rank(simu.Igd[it-1]))
        # This situation could pose a problem but we don't manage it yet        
        if (simu.time_since_loss[it] > config.FT['SMdelay']):
            
            Igdna = np.dot(config.FS['OPD2Piston_moy_r'],
                            np.dot(simu.Igd[it],config.FS['Piston2OPD_r']))
            
            Kernel = np.identity(NA) - Igdna
            
            CophasedBaselines=np.where(np.diag(simu.Igd[it])!=0)[0]
            CophasedPairs=[]
            for ib in CophasedBaselines:
                ia,iap = config.FS['ich'][ib][0], config.FS['ich'][ib][1]
                CophasedPairs.append([ia,iap])
                
            CophasedGroups = JoinOnCommonElements(CophasedPairs)
            
            # Fringe loss
            simu.LostTelescopes[it] = (np.diag(Igdna) == 0)*1      # The positions of the lost telescopes get 1.
            # WeLostANewTelescope = (sum(newLostTelescopes) > 0)
            
            # Photometry loss
            simu.noSignal_on_T[it] = 1*(simu.SNRPhotometry[it] < config.FT['ThresholdPhot'])
                
            comparison = (simu.noSignal_on_T[it] == simu.LostTelescopes[it])
            simu.LossDueToInjection[it] = (comparison.all() and sum(simu.noSignal_on_T[it])>1)       # Evaluates if the two arrays are the same
            
            if not simu.LossDueToInjection[it]:     # The fringe loss is not due to an injection loss
                config.FT['state'][it] = 1
                
                newLostTelescopes = (simu.LostTelescopes[it] - simu.LostTelescopes[it-1] == 1)
                TelescopesThatGotBackPhotometry = (simu.noSignal_on_T[it-1] - simu.noSignal_on_T[it] == 1)
                # WeGotBackPhotometry = (sum(TelescopesThatGotBackPhotometry) > 0)
                
                TelescopesThatNeedARestart = np.argwhere(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)

                if config.FT['state'][it-1] != 1:
                    config.FT['eps'] = np.ones(NA)
                    config.FT['it0'] = np.ones(NA)*it
                    config.FT['it_last'] = np.ones(NA)*it
                
                Velocities = np.dot(Kernel,config.FT['Velocities'])
                Increment = searchfunction_inc_sylvain_gestioncophased(it, Velocities, CophasedGroups)
            
                #You should send command only on telescope with flux
                simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
                Increment = np.dot(simu.NoPhotometryFiltration[it],Increment)
                
                usearch = simu.SearchCommand[it] + Increment
                
            else:
                    Increment = np.zeros(NA)
            
        else:
            Increment = np.zeros(NA)
            
    else:
        simu.time_since_loss[it] = 0
        Increment = np.zeros(NA)
            
    Increment = config.FT['search']*Increment
    
    usearch = simu.SearchCommand[it] + Increment
    # The command is sent at the next time, that's why we note it+1
    simu.SearchCommand[it+1] = usearch
    
    
    
    """ Implementation RELOCK incremental 11/05/2022"""
    """ usearch est maintenant un delta à ajouter à la position actuelle des LAR """
    
    # IgdRank = np.linalg.matrix_rank(simu.Igd[it])
    # NotCophased = (IgdRank < NA-1)
    # simu.IgdRank[it] = IgdRank
    
    # if NotCophased:
    #     simu.time_since_loss[it]=simu.time_since_loss[it-1]+config.dt
        
    #     # FringeLost = (NotCophased and (IgdRank<np.linalg.matrix_rank(simu.Igd[it-1]))
    #     # This situation could pose a problem but we don't manage it yet        
    #     if (simu.time_since_loss[it] > config.FT['SMdelay']):
            
    #         Igdna = np.dot(config.FS['OPD2Piston'],
    #                        np.dot(simu.Igd[it],config.FS['Piston2OPD']))
            
    #         CophasedBaselines=np.where(np.diag(simu.Igd[it])>0.5)[0]
    #         CophasedPairs=[]
    #         for ib in CophasedBaselines:
    #             ia,iap = config.FS['ich'][ib][0], config.FS['ich'][ib][1]
    #             CophasedPairs.append([ia,iap])
                
    #         CophasedGroups = JoinOnCommonElements(CophasedPairs)
            
    #         # Fringe loss
    #         simu.LostTelescopes[it] = (np.diag(Igdna) == 0)*1      # The positions of the lost telescopes get 1.
    #         # WeLostANewTelescope = (sum(newLostTelescopes) > 0)
            
    #         # Photometry loss
    #         simu.noSignal_on_T[it] = 1*(simu.SNRPhotometry[it] < config.FT['ThresholdPhot'])
                
    #         comparison = (simu.noSignal_on_T[it] == simu.LostTelescopes[it])
    #         simu.LossDueToInjection[it] = (comparison.all() and sum(simu.noSignal_on_T[it])>1)       # Evaluates if the two arrays are the same
            
    #         if not simu.LossDueToInjection[it]:     # The fringe loss is not due to an injection loss
    #             ### On entre dans le mode RELOCK
                
    #             config.FT['state'][it] = 1          # Variable de suivi d'état du FT

    #             ### On regarde si de nouveaux télescopes viennent juste d'être perdus.
    #             newLostTelescopes = (simu.LostTelescopes[it] - simu.LostTelescopes[it-1] == 1)
    #             TelescopesThatGotBackPhotometry = (simu.noSignal_on_T[it-1] - simu.noSignal_on_T[it] == 1)
    #             # WeGotBackPhotometry = (sum(TelescopesThatGotBackPhotometry) > 0)
                
    #             TelescopesThatNeedARestart = np.argwhere(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)
    #             print(TelescopesThatNeedARestart)
            
    #             ### Pour chaque télescope nouvellement perdu, on réinitialise la fonction usaw
    #             for ia in range(NA):
    #                 if (ia in TelescopesThatNeedARestart) or (config.FT['state'][it-1]!=1):
    #                     config.FT['it0'] = it; config.FT['it_last'][ia]=it;
    #                     config.FT['eps'] = 1
    #                     config.FT['LastPosition'][ia] = 0

    #             usaw,change = searchfunction_inc_basical(it)

    #             #config.FT['usaw'][it]= usaw

    #             Kernel = np.identity(NA) - Igdna
    #             usearch = np.dot(Kernel,usaw*config.FT['Velocities'])
                

    #             #usearch = usearch/np.ptp(usearch) * config.FT['maxVelocity']
                
    #             # if change:  # Change direction of scan
    #             #     # Fais en sorte que les sauts de pistons de télescopes cophasés 
    #             #     # entre eux maintiennent l'OPD constante: si 1 et 2 sont cophasés
    #             #     # avec OPD=p2-p1, au prochain saut le télescope 2 va à la position
    #             #     # du T1 + OPD et pas à la position qu'il avait avant le précédent saut.
    #             #     for group in CophasedGroups:    
    #             #         for ig in range(1,len(group)):
    #             #             ia = int(float(group[ig])-1) ; i0 = int(float(group[0])-1)
    #             #             LastPosition[ia] = LastPosition[i0] + simu.SearchCommand[it,ia]-simu.SearchCommand[it,i0]
    #             #     usearch = LastPosition + Increment
    #             #     LastPosition = simu.SearchCommand[it]
                    
    #             # else:
    #             #     usearch = simu.SearchCommand[it]+Increment
                    
    #             # config.FT['LastPosition'][it+1] = LastPosition
                
    #             # You should send command only on telescope with flux
    #             simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
    #             usearch = np.dot(simu.NoPhotometryFiltration[it],usearch)
            
            
    #         else:
    #             usearch = 0#simu.SearchCommand[it]
        
    #     else:
    #         usearch = 0#simu.SearchCommand[it]
            
    # else:
    #     simu.time_since_loss[it] = 0
    #     usearch = 0#simu.SearchCommand[it]
        
        
    # usearch = config.FT['search']*usearch
    # # The command is sent at the next time, that's why we note it+1
    # simu.SearchCommand[it+1] = simu.SearchCommand[it]+usearch
    
    
        
    """
    Group-Delay tracking
    """
    
    currGDerr = currGD #- simu.GDref[it]
    
    # Keep the GD between [-Pi, Pi]
    # Eq. 35
    # Array elements verifying the condition
    higher_than_pi = (currGDerr > np.pi)
    lower_than_mpi = (currGDerr < -np.pi)
    
    currGDerr[higher_than_pi] -= 2*np.pi
    currGDerr[lower_than_mpi] += 2*np.pi
    
    # Store residual GD for display only [radians]
    simu.GDResidual[it] = currGDerr
    
    # Weights the GD (Eq.35)
    currGDerr = np.dot(currIgd,currGDerr)
     
    simu.GDResidual2[it] = currGDerr
    simu.GDPistonResidual[it] = np.dot(FS['OPD2Piston_r'], currGDerr*R*config.PDspectra/(2*np.pi))
    
    # Threshold function (eq.36)
    if FT['Threshold']:
    
        # Array elements verifying the condition
        higher_than_pi = (currGDerr > np.pi/R*FT['switch'])
        lower_than_mpi = (currGDerr < -np.pi/R*FT['switch'])
        within_pi = (np.abs(currGDerr) <= np.pi/R*FT['switch'])
        
        currGDerr[within_pi] = 0
        if FT['continu']:
            currGDerr[higher_than_pi] -= np.pi/R*FT['switch']
            currGDerr[lower_than_mpi] += np.pi/R*FT['switch']
    
    simu.GDErr[it] = currGDerr
    
    # Integrator (Eq.37)
    if FT['cmdOPD']:     # integrator on OPD
        # Integrator
        simu.GDCommand[it+1] = simu.GDCommand[it] + FT['GainGD']*currGDerr*config.PDspectra*R/2/np.pi
        
        # From OPD to Pistons
        uGD = np.dot(FS['OPD2Piston_r'], simu.GDCommand[it+1])
        
    else:                       # integrator on Pistons
        # From OPD to Piston
        currPistonGD = np.dot(FS['OPD2Piston_r'], currGDerr)*config.PDspectra*R/2/np.pi
        # Integrator
        uGD = simu.PistonGDCommand[it+1] + FT['GainGD']*currPistonGD
        
    # simu.GDCommand[it+1] = simu.GDCommand[it] + FT['GainGD']*currGDerr
    
    # From OPD to Piston
    # uGD = np.dot(FS['OPD2Piston'], simu.GDCommand[it+1])
    
    simu.PistonGDCommand_beforeround[it+1] = uGD
    
    if config.FT['roundGD']=='round':
        jumps = np.round(uGD/config.PDspectra)
        uGD = jumps*config.PDspectra
    elif config.FT['roundGD']=='int':
        for ia in range(NA):
            jumps = int(uGD[ia]/config.PDspectra)
            uGD[ia] = jumps*config.PDspectra
    elif config.FT['roundGD']=='no':
        pass
    else:
        raise ValueError("The roundGD parameter of the fringe-tracker must be 'round', 'int' or 'no'.")
        
    if config.TELref:
        iTel = config.TELref-1
        uGD = uGD - uGD[iTel]
        
        
    simu.PistonGDCommand[it+1] = uGD

    """
    Phase-Delay command
    """
    
    currPDerr = currPD #- simu.PDref[it]
 
    # Keep the PD between [-Pi, Pi]
    # Eq. 35
    
    # Array elements verifying the condition
    higher_than_pi = (currPDerr > np.pi)
    lower_than_mpi = (currPDerr < -np.pi)
    
    currPDerr[higher_than_pi] -= 2*np.pi
    currPDerr[lower_than_mpi] += 2*np.pi
    
    simu.PDResidual[it] = currPDerr
    
    # Weights the PD (Eq.35)
    currPDerr = np.dot(currIpd,currPDerr)
    
    # Store residual PD and piston for display only
    simu.PDResidual2[it] = currPDerr
    simu.PDPistonResidual[it] = np.dot(FS['OPD2Piston_r'], currPDerr*config.PDspectra/(2*np.pi))
    
    # Integrator (Eq.37)
            
    if FT['cmdOPD']:     # integrator on OPD
        # Integrator
        simu.PDCommand[it+1] = simu.PDCommand[it] + FT['GainPD']*currPDerr*config.PDspectra/2/np.pi
        # From OPD to Pistons
        uPD = np.dot(FS['OPD2Piston_r'], simu.PDCommand[it+1])
        
    else:                       # integrator on Pistons
        # From OPD to Piston
        currPistonPD = np.dot(FS['OPD2Piston_r'], currPDerr)*config.PDspectra/2/np.pi
        # Integrator
        uPD = simu.PistonPDCommand[it] + FT['GainPD']*currPistonPD
    
    
    if config.TELref:
        iTel = config.TELref-1
        uPD = uPD - uPD[iTel]
    
    simu.PistonPDCommand[it+1] = uPD
    
    # if config.mode == 'track':
    #     if np.linalg.matrix_rank(currIgd) < NA-1:
    #         tsearch_ = it
    #         config.mode == 'search'
            
    # elif config.mode == 'search':
    #     if np.linalg.matrix_rank(currIgd) == NA-1:
    #         config.mode == 'track'
    #     else:
    #         usaw = searchfunction(NA,Sweep_,Slope_,it-tsearch_)
    #         usearch = Vfactors_*usaw
        
    
    """
    MODULATION command
    """


    
    """
    ODL command
    It is the addition of the GD, PD, SEARCH and modulation functions
    """
    
    CommandODL = uPD + uGD + usearch
    
    # if config.TELref !=-1:
    #     CommandODL = CommandODL - CommandODL[config.TELref]
    
    return CommandODL


def getvar():
    """
    From the image and calibrated quantities (sky), calculates the "Phase variance"
    which is in reality the inverse of the squared signal-to-noise ratio of the 
    fringes.

    Parameters
    ----------


    Returns
    -------
    varPD : TYPE
        DESCRIPTION.

    GLOBAL
    -------
    simu.CfPD : 
        
    simu.varPD :
        

    """
    
    from . import simu
    
    from .config import NA, NIN,MW
    
    from .simu import it
    # from .coh_tools import simu2GRAV, NB2NIN
    
    NINmes = config.FS['NINmes']
    
    image = simu.MacroImages[it]
    
    M = config.M            # Amplification ratio camera
    
    sigmap = config.FS['sigmap']  # Background noise
    imsky = config.FS['imsky']    # Sky image before observation
    # Demod = config.FS['MacroP2VM_r']    # Macro P2VM matrix used for demodulation
    # ElementsNormDemod = config.FS['ElementsNormDemod']
    
    DemodGRAV = config.FS['MacroP2VM_r']
    # Flux variance calculation (eq. 12)
    varFlux = sigmap**2 + M*(image - imsky)
    simu.varFlux[it] = varFlux
    
    """
    Covariance calculation (eq. 13)
    """
    
    for imw in range(MW):
        # simu.CovarianceReal[it,imw] = np.sum(np.real(Demod[imw])**2*varFlux[imw], axis=1)
        # simu.CovarianceImag[it,imw] = np.sum(np.imag(Demod[imw])**2*varFlux[imw], axis=1)

        simu.Covariance[it,imw] = np.dot(DemodGRAV[imw], np.dot(np.diag(varFlux[imw]),np.transpose(DemodGRAV[imw])))
        #simu.BiasModCf[it,imw] = np.dot(ElementsNormDemod[imw],varFlux[imw])
        
    simu.DemodGRAV = DemodGRAV
    
    # Phase variance calculation (eq. 14)
    Nvar = config.FT['Nvar']                # Integration time for phase variance
    if it < Nvar:
        Nvar = it+1
        
    varNum = np.zeros([MW,NINmes]) ; varNum2 = np.zeros([MW,NINmes])
    varPhot = np.zeros([MW,NA])         # Variance of the photometry measurement
    CohFlux = np.zeros([MW,NINmes])*1j
    
# =============================================================================
#     NOTATIONS:
#       CfPD = sqrt(Fi*Fj)*exp(i\Phi_ij)*ComplexCoherenceDegree = X + i*Y
#       Ex = Esperance of X ; Ey = Esperance of Y
#       VarX = Variance of X ; VarY = Variance of Y
# =============================================================================
    diagCovar = np.diagonal(simu.Covariance, axis1=2, axis2=3)
    varPhot = diagCovar[it,:,:NA]
    timerange = range(it+1-Nvar,it+1)
    varX = np.abs(diagCovar[timerange,:,NA:NA+NINmes])
    varY = np.abs(diagCovar[timerange,:,NA+NINmes:])
    varNum = np.mean(varX+varY,axis=0)
    # for ia in range(NA):
    #     ibp = ia*(NA+1)
    #     varPhot[:,ia] = simu.Covariance[it,:,ibp,ibp]       # Variance of photometry at each frame
    #     for iap in range(ia+1,NA):
    #         ib = posk(ia,iap,NA)
    #         ibr=NA+ib; varX = simu.Covariance[timerange,:,ibr,ibr]
    #         ibi=NA+NINmes+ib; varY = simu.Covariance[timerange,:,ibi,ibi]
    #         covarXY = simu.Covariance[timerange,:,ibr,ibi]
            
    #         varNum[:,ib] = np.mean(varX+varY,axis=0)
    #         varNum2[:,ib] = np.mean(varX+varY + 2*covarXY, axis=0)
            
    CohFlux = np.mean(simu.CfPD[timerange], axis=0)
    CfSumOverLmbda = np.sum(CohFlux,axis=0)
    
    simu.varGDdenom[it] = np.sum(np.real(CohFlux*np.conj(CohFlux)),axis=0)  # Sum over lambdas of |CohFlux|² (modified eq.14)
    simu.varGDdenomUnbiased[it] = np.sum(np.real(CohFlux*np.conj(CohFlux))-simu.BiasModCf[it],axis=0)  # Sum over lambdas of |CohFlux|²
    simu.varPDdenom[it] = np.real(CfSumOverLmbda*np.conj(CfSumOverLmbda))#-np.mean(simu.BiasModCf[it],axis=0)) # Original eq.14    
    #simu.varPDdenom2[it] = np.sum(np.mean(np.abs(simu.CfPD[timerange])**2,axis=0),axis=0)
    simu.varPDnum[it] = np.sum(varNum,axis=0)/2     # Sum over lmbdas of Variance of |CohFlux|
    
    simu.varGDUnbiased[it] = simu.varPDnum[it]/simu.varGDdenomUnbiased[it]      # Var(|CohFlux|)/|CohFlux|²
    simu.varPD[it] = simu.varPDnum[it]/simu.varPDdenom[it]      # Var(|CohFlux|)/|CohFlux|²
    simu.varGD[it] = simu.varPDnum[it]/simu.varGDdenom[it]
    
    simu.SNRPhotometry[it,:] = np.sum(simu.PhotometryEstimated[it,:],axis=0)/np.sqrt(np.sum(varPhot,axis=0))
    
    varPD = simu.varPD[it]
    varGD = simu.varGD[it]
    
    return varPD, varGD


def SetThreshold(TypeDisturbance="CophasedThenForeground",
                 manual=False, scan=False,display=False,
                 verbose=True,scanned_tel=6):
    """
    This function enables to estimate the threshold GD for the state-machine.
    It scans the coherence envelop of the FS and displays the estimated SNR².
    It then asks for the user to choose a smart threshold.

    Returns
    -------
    INPUT
        The user can choose the value to return regarding the SNR evolution.

    """
    from cophasing import skeleton as sk
    from cophasing import config

    NINmes = config.FS['NINmes']

    if scan:

        datadir2 = "data/"
    
        R=config.FS['R']
        
        if R < 50:
            DisturbanceFile = datadir2 + 'EtudeThreshold/scan120micron_tel6.fits'
            NT=500
        else:
            DisturbanceFile = datadir2 + 'EtudeThreshold/scan240micron_tel6.fits'
            NT=1000
            
        InitialDisturbanceFile,InitNT = sk.config.DisturbanceFile, sk.config.NT
        
        sk.update_config(DisturbanceFile=DisturbanceFile, NT = NT,verbose=verbose)
        
        # Initialize the fringe tracker with the gain
        # from cophasing.SPICA_FT_r import SPICAFT, updateFTparams
        #SPICAFT(init=True, GainPD=0, GainGD=0,search=False)
        gainPD,gainGD,search=config.FT['GainPD'],config.FT['GainGD'],config.FT['search']
        updateFTparams(GainPD=0, GainGD=0, search=False, verbose=verbose)
        
        # Launch the scan
        sk.loop(verbose)
        
        if manual:
            sk.display('snr',WLOfTrack=1.6, pause=True)
            test1 = input("Set all threshold to same value? [y/n]")
            if (test1=='y') or (test1=='yes'):    
                newThresholdGD = float(input("Set the Threshold GD: "))
                
            elif (test1=='n') or (test1=='no'):
                newThresholdGD=np.ones(NINmes)
                for ib in range(NINmes):
                    newThresholdGD[ib] = float(input(f"Threshold base {config.FS['ich'][ib]}:"))
            else:
                raise ValueError("Please answer by 'y','yes','n' or 'no'")
            
        else:
            from cophasing import simu,coh_tools
            
            scanned_baselines = [coh_tools.posk(ia,scanned_tel-1,config.NA) for ia in range(config.NA-1)]
            k=0;ib=scanned_baselines[k]
            while not config.FS['active_ich'][ib]:
                k+=1
                ib = scanned_baselines[k]
                
            Lc = R*config.PDspectra
            
            ind=np.argmin(np.abs(simu.OPDTrue[:,4]+Lc*0.7))
            
            newThresholdGD = np.array([np.max([2,x]) for x in np.sqrt(simu.SquaredSNRMovingAverage[ind,:])])
                    
            config.FT['ThresholdGD'] = newThresholdGD
            
            sk.display('snr',WLOfTrack=1.6, pause=True, display=display)
            
        sk.update_config(DisturbanceFile=InitialDisturbanceFile, NT=InitNT,
                         verbose=verbose)
        updateFTparams(GainPD=gainPD, GainGD=gainGD, search=search, 
                       ThresholdGD=newThresholdGD,
                       verbose=verbose)
    
    
    else:
        
        DisturbanceFile = TypeDisturbance
        
        NT=200
            
        InitialDisturbanceFile,InitNT = sk.config.DisturbanceFile, sk.config.NT
        
        sk.update_config(DisturbanceFile=DisturbanceFile, NT = NT, verbose=verbose)
        
        # Initialize the fringe tracker with the gain
        from cophasing.SPICA_FT import SPICAFT, updateFTparams
        #SPICAFT(init=True, GainPD=0, GainGD=0,search=False)
        gainPD,gainGD,search=config.FT['GainPD'],config.FT['GainGD'],config.FT['search']
        updateFTparams(GainPD=0, GainGD=0, search=False, verbose=verbose)
        
        # Launch the scan
        sk.loop(verbose=verbose)
        
        if manual:
            sk.display('snr','detector',WLOfTrack=1.6, pause=True)
            test1 = input("Set all threshold to same value? [y/n]")
            if (test1=='y') or (test1=='yes'):    
                newThresholdGD = float(input("Set the Threshold GD: "))
            elif (test1=='n') or (test1=='no'):
                newThresholdGD=np.ones(NINmes)
                for ib in range(NINmes):
                    newThresholdGD[ib] = float(input(f"Threshold base {config.FS['ich'][ib]}:"))
            else:
                raise ValueError("Please answer by 'y','yes','n' or 'no'")
                
                
        else:
            from cophasing import simu,coh_tools
            
            if TypeDisturbance=='NoDisturbance':
                ind=100
                newThresholdGD = np.array([np.max([2,x*0.2]) for x in np.sqrt(simu.SquaredSNRMovingAverage[ind,:])])
 
            elif TypeDisturbance == 'CophasedThenForeground':
                CophasedInd = 50 ; ForegroundInd = 180
                CophasedRange = range(50,100)
                ForegroundRange = range(160,200)
                newThresholdGD = np.ones(NINmes)

                for ib in range(NINmes):
                    SNRcophased = np.mean(np.sqrt(simu.SquaredSNRMovingAverage[CophasedRange,ib]))
                    SNRfg = np.mean(np.sqrt(simu.SquaredSNRMovingAverage[ForegroundRange,ib]))
                    fgstd = np.std(np.sqrt(simu.SquaredSNRMovingAverage[ForegroundRange,ib]))
                    cophasedstd = np.std(np.sqrt(simu.SquaredSNRMovingAverage[CophasedRange,ib]))
                    
                    # Set threshold to a value between max and foreground with a lower limit defined by the std of foreground.
                    newThresholdGD[ib] = np.max([1.5,SNRfg + 5*fgstd,SNRfg+0.2*(SNRcophased-SNRfg)])
                    print()
                    if newThresholdGD[ib] ==0:
                        newThresholdGD[ib] = 10
                        
            newThresholdPD = 1e-3#np.min(newThresholdGD)/2
            
            config.FT['ThresholdGD'] = newThresholdGD
            config.FT['ThresholdPD'] = newThresholdPD

            sk.display('opd','snr','detector',WLOfTrack=1.6, pause=True,display=display)
            
        sk.update_config(DisturbanceFile=InitialDisturbanceFile, NT=InitNT,
                         verbose=verbose)
        updateFTparams(GainPD=gainPD, GainGD=gainGD, search=search, 
                       ThresholdGD=newThresholdGD,ThresholdPD=newThresholdPD,
                       verbose=verbose)
    
    return newThresholdGD


def searchfunction(usaw):
    """
    Calculates a search function for NA telescopes using the last search command.

    Parameters
    ----------
    usaw : TYPE
        DESCRIPTION.

    Returns
    -------
    usaw : TYPE
        DESCRIPTION.

    """
    
    from . import simu
    from .config import NA,dt
    from .simu import it
    
    for ia in range(NA):
        it0 = config.FT['it0'][ia] ; it_last = config.FT['it_last'][ia]
        
        a = config.FT['Sweep30s']/30
        sweep = config.FT['Sweep0'] + a*(it-it0)*config.dt
        
        time_since_last_change = (it-it_last)*config.dt     # depends on ia
        
        if time_since_last_change < sweep:          # this depends on ia
            usaw[ia] = usaw[ia] + config.FT['eps'][ia]
            # return config.FT['eps']*config.FT['Vfactors']*time_since_last_change
        else:
            utemp=usaw[ia]
            config.FT['eps'][ia] = -config.FT['eps'][ia]
            config.FT['it_last'][ia] = it
            usaw[ia] = config.FT['LastPosition'][ia] + config.FT['eps'][ia]
            config.FT['LastPosition'][ia] = utemp
        
    
    simu.eps[it] = config.FT['eps']
    simu.it_last[it] = config.FT['it_last']
    simu.LastPosition[it] = config.FT['LastPosition']
        
    # Investigation data
    
    return usaw
  


def searchfunction_basical(usaw,it):
    
    a = config.FT['Sweep30s']/30000
    sweep = config.FT['Sweep0'] + a*(it-config.FT['it0'])*config.dt
    
    time_since_last_change = (it-config.FT['it_last'])*config.dt
    
    if time_since_last_change < sweep:
        usaw = usaw + config.FT['eps']
        # return config.FT['eps']*config.FT['Vfactors']*time_since_last_change
    else:
        utemp=usaw
        config.FT['eps'] = -config.FT['eps']
        config.FT['it_last'] = it
        usaw = config.FT['LastPosition'] + config.FT['eps']
        config.FT['LastPosition'] = utemp

    config.FT['usaw'][it] = usaw
        
    return usaw

def searchfunction3(usaw,it):
    
    a = config.FT['Sweep30s']/30000
    sweep = config.FT['Sweep0'] + a*(it-config.FT['it0'])*config.dt
    
    time_since_last_change = (it-config.FT['it_last'])*config.dt
    
    if time_since_last_change < sweep:
        usaw = config.FT['eps']
        # return config.FT['eps']*config.FT['Vfactors']*time_since_last_change
    else:
        utemp=usaw
        diff = config.FT['LastPosition'] - usaw
        config.FT['eps'] = -config.FT['eps']
        config.FT['it_last'] = it
        usaw = diff + config.FT['eps']
        config.FT['LastPosition'] = utemp

    config.FT['usaw'][it] = usaw
        
    return usaw

def searchfunction_inc_basical(it):
    """
    Incremental sawtooth function working.
    It returns +1 or -1 depending on the tooth on which it is, and add a delta only at the change of tooth.
    INPUT:
        - dico : DICTIONARY
            Contains relevant informations for the decision of change of tooth.
        - it : FLOAT
            Current time.
            
    OUTPUT:
        - usaw : FLOAT ARRAY
            Incrementation to add to the current ODL position, telescope per telescope.
        - dico : DICTIONNARY
            Updated dico.
    """
    
    it_last=config.FT['it_last']; it0=config.FT['it0'] ; eps=config.FT['eps']
    
    # Coefficient directeur de la fonction d'augmentation du temps avant saut.
    a = config.FT['Sweep30s']/30000
    
    # Temps avant saut de frange
    sweep = config.FT['Sweep0'] + a*(it-it0)*config.dt
    
    # Temps passé depuis dernier saut.
    time_since_last_change = (it-it_last)*config.dt
    
    if time_since_last_change < sweep:  # Pas de saut
        change=False
        usaw = eps
        
    else:   # Saut 
        change=True
        eps = -eps
        it_last = it
        usaw = eps
        config.FT['it_last'] = it_last
        
    config.FT['eps'] = eps
    
    return usaw, change



def searchfunction_inc_sylvain(it, v):
    """
    Incremental sawtooth function working.
    It returns +1 or -1 depending on the tooth on which it is, and add a delta only at the change of tooth.
    INPUT:
        - it : FLOAT
            Current time.
        - v : FLOAT ARRAY
            Velocities of the different ODL
            
    OUTPUT:
        - usaw : FLOAT ARRAY
            Incrementation to add to the current ODL position, telescope per telescope.
        - dico : DICTIONNARY
            Updated dico.
    """
    from .config import NA
    
    move = np.zeros(NA)
    
    for ia in range(NA):
    
        it_last=config.FT['it_last'][ia]; it0=config.FT['it0'][ia] ; eps=config.FT['eps'][ia]
        
        # Coefficient directeur de la fonction d'augmentation du temps avant saut.
        a = config.FT['Sweep30s']/30000
        
        # Temps avant saut de frange
        sweep = config.FT['Sweep0'] + a*(it-it0)*config.dt
        
        # Temps passé depuis dernier saut.
        time_since_last_change = (it-it_last)*config.dt
        
        if time_since_last_change < sweep:  # Pas de saut
            change=False
            move[ia] = eps*v[ia]
            config.FT['LastPosition'][ia]+=move[ia]
            
        else:   # Saut 
            change=True
            config.FT['eps'][ia] = -config.FT['eps'][ia]
            config.FT['it_last'][ia] = it
            move[ia] = -config.FT['LastPosition'][ia]
            config.FT['LastPosition'][ia] = move[ia]
    
    
    return move


def searchfunction_inc_sylvain_gestioncophased(it, v, CophasedGroups):
    """
    Incremental sawtooth function working.
    It returns +1 or -1 depending on the tooth on which it is, and add a delta only at the change of tooth.
    INPUT:
        - it : FLOAT
            Current time.
        - v : FLOAT ARRAY
            Velocities of the different ODL
            
    OUTPUT:
        - usaw : FLOAT ARRAY
            Incrementation to add to the current ODL position, telescope per telescope.
        - dico : DICTIONNARY
            Updated dico.
    """
    from .config import NA
    
    move = np.zeros(NA)
    
    for ia in range(NA):
    
        it_last=config.FT['it_last'][ia]; it0=config.FT['it0'][ia] ; eps=config.FT['eps'][ia]
        
        # Coefficient directeur de la fonction d'augmentation du temps avant saut.
        a = config.FT['Sweep30s']/30000
        
        # Temps avant saut de frange
        sweep = config.FT['Sweep0'] + a*(it-it0)*config.dt
        
        # Temps passé depuis dernier saut.
        time_since_last_change = (it-it_last)*config.dt
        
        if time_since_last_change < sweep:  # Pas de saut
            change=False
            move[ia] = eps*v[ia]
            config.FT['LastPosition'][ia]+=move[ia]
            
        else:   # Saut 
            change=True
            config.FT['eps'][ia] = -config.FT['eps'][ia]
            config.FT['it_last'][ia] = it
            move[ia] = -config.FT['LastPosition'][ia]
            tel=ia+1
            for CophasedGroup in CophasedGroups:
                l = [move[itel-1] for itel in CophasedGroup]
                generalmove = max(set(l), key = l.count)
                
                if tel in CophasedGroup:
                    move[ia] = generalmove
                    
            config.FT['LastPosition'][ia] = move[ia]
    
    return move




def searchfunction_incind(it):
    """
    Incremental sawtooth function working.
    It returns +1 or -1 depending on the tooth on which it is, and add a delta only at the change of tooth.
    INPUT:
        - dico : DICTIONARY
            Contains relevant informations for the decision of change of tooth.
        - it : FLOAT
            Current time.
            
    OUTPUT:
        - usaw : FLOAT ARRAY
            Incrementation to add to the current ODL position, telescope per telescope.
        - dico : DICTIONNARY
            Updated dico.
    """
    from .config import NA
    
    print(config.FT['it_last'],config.FT['it0'],config.FT['eps'])
    
    usaw = [0]*6
    for ia in range(NA):
        it_last=config.FT['it_last'][ia]; it0=config.FT['it0'][ia] ; eps=config.FT['eps'][ia]
        
        a = config.FT['Sweep30s']/30000  # Coefficient directeur de la fonction d'augmentation du temps avant saut.
        sweep = config.FT['Sweep0'] + a*(it-it0)*config.dt
        
        time_since_last_change = (it-it_last)*config.dt
        
        if time_since_last_change < sweep:
            change=False
            usaw[ia] = eps
            config.FT['LastPosition'][ia]+=eps
            
        else:
            change=True
            eps = -eps
            it_last = it
            
            # la fonction usaw prend la valeur qu'elle avait avant le précédent
            # saut.
            usaw[ia] = -config.FT['LastPosition'][ia]
            config.FT['it_last'][ia] = it_last
            config.FT['LastPosition'][ia]=-config.FT['LastPosition'][ia]
            
        config.FT['eps'][ia] = eps
    print(config.FT['LastPosition'])
    print(usaw)
        
    return usaw, change


def JoinOnCommonElements(groups):
    """
    Get a list of groups and join all the groups with common elements.
    INPUT:
        - groups : ARRAY OF LIST or LIST OF LIST
            List of the cophased pairs
    
    OUTPUT:
        - L : LIST OR LIST
            List of the cophased groups
    """

    l = groups

    out = []
    while len(l)>0:
        first, *rest = l
        first = set(first)

        lf = -1
        while len(first)>lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)    
            rest = rest2

        out.append(first)
        l = rest

    L=[]
    for x in out:
        L.append([int(element) for element in list(x)])
            
    return(L)