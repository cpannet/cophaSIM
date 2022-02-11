# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:44:22 2020

@author: cpannetier

The SPICA Fringe Tracker calculates the commands to send to the telescopes after 
reading the coherent flux and filtering the most noisy measurements.

INPUT: Coherent flux [MW,NB]

OUTPUT: Piston commands [NA]

Calculated and stored observables:
    - 
    -
    -
    -
    -
    -
    -

"""

import numpy as np

from .coh_tools import posk, poskfai,NB2NIN

from . import config


def updateFTparams(verbose=True,**kwargs):
    
    print("Update fringe-tracker parameters:")
    for key, value in zip(list(kwargs.keys()),list(kwargs.values())):
        oldval=config.FT[key]
        if (key=='ThresholdGD') and (isinstance(value,(float,int))):
            config.FT['ThresholdGD'] = np.ones(config.NIN)*value
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
            CPref=True, Ncp = 300, Nvar = 5,cmdOPD=True, switch=1,
            ThresholdGD=2, ThresholdPD = 1.5, ThresholdPhot = 2,ThresholdRELOCK=2,
            Threshold=True, useWmatrices=True,
            latencytime=1,usecupy=False, **kwargs_for_update):
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
        - simu.PDEstimated: [NT,MW,NIN] Estimated PD before subtraction of the reference
        - simu.GDEstimated: [NT,MW,NIN] Estimated GD before subtraction of the reference
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
        from .config import NA,NIN,NT
        
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
            config.FT['ThresholdGD'] = np.ones(NIN)*ThresholdGD
        else:
            config.FT['ThresholdGD'] = ThresholdGD
            
        if isinstance(ThresholdGD,(float,int)):
            config.FT['ThresholdRELOCK'] = np.ones(NIN)*ThresholdRELOCK
        else:
            config.FT['ThresholdRELOCK'] = ThresholdRELOCK
            
        config.FT['ThresholdPD'] = ThresholdPD
        config.FT['CPref'] = CPref
        config.FT['roundGD'] = roundGD
        config.FT['Threshold'] = Threshold
        config.FT['switch'] = switch
        config.FT['cmdOPD'] = cmdOPD
        config.FT['useWmatrices'] = useWmatrices
        config.FT['usecupy'] = usecupy
        
        # Search command parameters
        config.FT['search'] = search
        config.FT['SMdelay'] = SMdelay          # Waiting time before launching search
        config.FT['Sweep0'] = Sweep0            # Starting sweep in seconds
        config.FT['Sweep30s'] = Sweep30s        # Sweep at 30s in seconds
        config.FT['maxVelocity'] = maxVelocity  # Maximal slope given in µm/frame
        
        # Version usaw vector
        # config.FT['usaw'] = np.zeros([NT,NA])
        # config.FT['last_usaw'] = np.zeros(NA)
        # config.FT['it_last'] = np.zeros(NA)
        # config.FT['it0'] = np.zeros(NA)
        # config.FT['eps'] = np.ones(NA)
        
        # Version usaw float
        config.FT['usaw'] = np.zeros([NT])
        config.FT['LastPosition'] = np.zeros([NT+1,NA])
        config.FT['it_last'] = 0
        config.FT['it0'] = 0
        config.FT['eps'] = 1
        
        
        config.FT['ThresholdPhot'] = ThresholdPhot      # Minimal photometry SNR for launching search

        if len(Vfactors) != 0:
            config.FT['Vfactors'] = np.array(Vfactors)
        else:
            config.FT['Vfactors'] = np.array([-8.25, -7.25, -4.25, 1.75, 3.75, 8.75])/8.75
        
        config.FT['Velocities'] = config.FT['Vfactors']/np.ptp(config.FT['Vfactors'])*maxVelocity     # The maximal OPD velocity is equal to slope/frame
        

        if usecupy:
            import cupy as cp
            config.FS['sigsky'] = cp.asnumpy(config.FS['sigsky'])  # Background noise
            config.FS['imsky'] = cp.asnumpy(config.FS['imsky'])    # Sky image before observation
        
        return

    elif update:
        print("Update fringe-tracker parameters with:")
        for key, value in zip(list(kwargs_for_update.keys()),list(kwargs_for_update.values())):
            setattr(config.FT, key, value)
            print(f" - {key}: {getattr(config.FT, key, value)}")

        return
    
    from . import simu

    it = simu.it
    
    currCfEstimated = args[0]

    currPD, currGD = ReadCf(currCfEstimated)
    
    simu.PDEstimated[it] = currPD
    simu.GDEstimated[it] = currGD#
    
    currCmd = CommandCalc(currPD, currGD)
    
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
        - simu.CfEstimated_ --> should be coh_turn which do that
        - simu.CfPD: Coherent Flux Phase-Delay     [NT,MW,NIN]
        - simu.CfGD: Coherent Flux GD              [NT,MW,NIN]
        - simu.ClosurePhasePD                       [NT,MW,NC]
        - simu.ClosurePhaseGD                       [NT,MW,NC]
        - simu.PhotometryEstimated                  [NT,MW,NA]
        - simu.VisibilityEstimated                    [NT,MW,NIN]*1j
        - simu.CoherenceDegree                      [NT,MW,NIN]
    """
    
    from . import simu
    
    from .config import NA,NIN,NC
    from .config import MW
    
    it = simu.it            # Time
     
    """
    Photometries and CfNIN extraction
    [NT,MW,NA]
    """
    
    PhotEst = np.zeros([MW,NA])
    currCfEstimatedNIN = np.zeros([MW, NIN])*1j
    for ia in range(NA):
        PhotEst[:,ia] = np.abs(currCfEstimated[:,ia*(NA+1)])
        for iap in range(ia+1,NA):
            ib = posk(ia,iap,NA)
            currCfEstimatedNIN[:,ib] = currCfEstimated[:,ia*NA+iap]
            
    # Save coherent flux and photometries in stack
    simu.PhotometryEstimated[it] = PhotEst

        
    """
    Visibilities extraction
    [NT,MW,NIN]
    """
    
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib = posk(ia,iap,NA)
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
        
    # Current Phase-Delay
    currPD = np.angle(np.sum(simu.CfPD[it,:,:], axis=0))*config.FS['active_ich']
        
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

    currGD = np.zeros(NIN)
    for ib in range(NIN):
        cfGDlmbdas = simu.CfGD[it,:-Ncross,ib]*np.conjugate(simu.CfGD[it,Ncross:,ib])
        cfGDmoy = np.sum(cfGDlmbdas)

        currGD[ib] = np.angle(cfGDmoy)*config.FS['active_ich'][ib]    # Group-delay on baseline (ib).
    
    """
    Closure phase calculation
    cpPD_ is a global stack variable [NT, NC]
    cpGD_ is a global stack variable [NT, NC]
    Eq. 17 & 18
    """
    
    Ncp = config.FT['Ncp']
    
    if it < Ncp:
        Ncp = it+1
        
    bispectrumPD = np.zeros([NC])*1j
    bispectrumGD = np.zeros([NC])*1j
    
    timerange = range(it+1-Ncp,it+1) ; validcp=np.zeros(NC)
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib = posk(ia,iap,NA)      # coherent flux (ia,iap)  
            valid1=config.FS['active_ich'][ib]
            cs1 = np.sum(simu.CfPD[timerange,:,ib], axis=1)     # Sum of coherent flux (ia,iap)
            cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
            cfGDmoy1 = np.sum(cfGDlmbdas,axis=1)     # Sum of coherent flux (ia,iap)  
            for iapp in range(iap+1,NA):
                ib = posk(iap,iapp,NA) # coherent flux (iap,iapp)    
                valid2=config.FS['active_ich'][ib]
                cs2 = np.sum(simu.CfPD[timerange,:,ib], axis=1) # Sum of coherent flux (iap,iapp)    
                cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
                cfGDmoy2 = np.sum(cfGDlmbdas,axis=1)
                
                ib = posk(ia,iapp,NA) # coherent flux (iapp,ia)    
                valid3=config.FS['active_ich'][ib]
                cs3 = np.sum(np.conjugate(simu.CfPD[timerange,:,ib]),axis=1) # Sum of 
                cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
                cfGDmoy3 = np.sum(cfGDlmbdas,axis=1)
                
                # The bispectrum of one time and one triangle adds up to
                # the Ncp last times
                ic = poskfai(ia,iap,iapp,NA)        # 0<=ic<NC=(NA-2)(NA-1) 
                validcp[ic]=valid1*valid2*valid3
                bispectrumPD[ic]=np.sum(cs1*cs2*cs3)
                bispectrumGD[ic]=np.sum(cfGDmoy1*cfGDmoy2*np.conjugate(cfGDmoy3))
    
    simu.BispectrumGD[it] = bispectrumGD
    simu.BispectrumPD[it] = bispectrumPD
    
    bispectrumPD[bispectrumPD<0.05] = 0
    bispectrumGD[bispectrumGD<0.05] = 0
    
    simu.ClosurePhasePD[it] = np.angle(bispectrumPD)*validcp
    simu.ClosurePhaseGD[it] = np.angle(bispectrumGD)*validcp
    
    if config.FT['CPref'] and (it>Ncp):                     # At time 0, we create the reference vectors
        for ia in range(1,NA-1):
            for iap in range(ia+1,NA):
                k = posk(ia,iap,NA)
                ic = poskfai(0,ia,iap,NA)   # Position of the triangle (0,ia,iap)
                simu.PDref[it,k] = simu.ClosurePhasePD[it,ic]
                simu.GDref[it,k] = simu.ClosurePhaseGD[it,ic]

                    
    return currPD, currGD



def CommandCalc(currPD,currGD):
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
    
    from .config import NA,NIN
    from .config import FT,FS
    
    it = simu.it            # Frame number
    
    """
    Signal-to-noise ratio of the fringes ("Phase variance")
    The function getvar saves the inverse of the squared SNR ("Phase variance")
    in the global stack variable varPD [NT, MW, NIN]
    Eq. 12, 13 & 14
    """
            
    Ngd = FT['Ngd']
    if it < FT['Ngd']:
        Ngd = it+1
        
    
    """
    WEIGHTING MATRIX
    """

    if config.FT['useWmatrices']:
        
        varcurrPD, varcurrPD2 = getvar()
        
        # Raw Weighting matrix in the OPD-space
        
        timerange = range(it+1-Ngd, it+1)
        simu.SquaredSNRMovingAverage[it] = np.nan_to_num(1/np.mean(simu.varPD[timerange], axis=0))
        simu.SquaredSNRMovingAverage2[it] = np.nan_to_num(1/np.mean(simu.varPD2[timerange], axis=0))
        simu.SquaredSNRMovingAverageDebiased[it] = np.nan_to_num(1/np.mean(simu.varPDdebiased[timerange], axis=0))
        
        simu.TemporalVariancePD[it] = np.var(simu.PDEstimated[timerange], axis=0)
        simu.TemporalVarianceGD[it] = np.var(simu.GDEstimated[timerange], axis=0)
        
        if config.FT['state'][it-1]:
            SquaredSNRMovingAverage = simu.SquaredSNRMovingAverage2[it]
            reliablebaselines = (SquaredSNRMovingAverage >= FT['ThresholdRELOCK']**2)
        else:
            SquaredSNRMovingAverage = simu.SquaredSNRMovingAverage[it]
            reliablebaselines = (SquaredSNRMovingAverage >= FT['ThresholdGD']**2)
            
        simu.TrackedBaselines[it] = reliablebaselines
        
        Wdiag=np.zeros(NIN)
        Wdiag[reliablebaselines] = 1/varcurrPD[reliablebaselines]
        W = np.diag(Wdiag)
        # Transpose the W matrix in the Piston-space
        MtWM = np.dot(FS['OPD2Piston'], np.dot(W,FS['Piston2OPD']))
        
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
        currIgd = np.dot(FS['Piston2OPD'],np.dot(VSdagUt,np.dot(FS['OPD2Piston'], W)))
    
    
        """
        PD Weighting matrix
        """

        Sdag = np.zeros([NA,NA])
        reliablepistons = (S > config.FT['ThresholdPD']**2)
        notreliable = (reliablepistons==False)
        
        diagS = np.zeros([NA])
        diagS[reliablepistons] = 1/S[reliablepistons]
        diagS[notreliable] = S[notreliable]/FT['ThresholdPD']**4
        Sdag = np.diag(diagS)
        
        # Come back to the OPD-space        
        VSdagUt = np.dot(V, np.dot(Sdag,Ut))
        
        # Calculates the weighting matrix
        currIpd = np.dot(FS['Piston2OPD'],np.dot(VSdagUt,np.dot(FS['OPD2Piston'], W)))
            
    else:
        currIgd = np.identity(NIN)
        currIpd = np.identity(NIN)
    
    simu.Igd[it,:,:] = currIgd
    simu.Ipd[it,:,:] = currIpd
    
    
    """
    FRINGE SEARCHING command
    """
    
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
    #             #     config.FT['last_usaw'] = np.copy(config.FT['usaw'][it-1])
                    
    #             # elif sum(TelescopesThatNeedARestart) > 0:
                    
    #             #     # Version "Restart only concerned telescopes" (06-10-2021)
    #             #     # --> doesn't work because it avoids some OPDs.
    #             #     # for ia in TelescopesThatNeedARestart:
    #             #     #     config.FT['it0'][ia] = it ; config.FT['it_last'][ia] = it
    #             #     #     config.FT['last_usaw'][ia] = 0
                
    #             #     # Version "Restart all" (06-10-2021)
    #             #     # Restart all telescope from their current position.
    #             #     config.FT['it0'] = np.ones(NA)*it
    #             #     config.FT['it_last'] = np.ones(NA)*it
    #             #     config.FT['last_usaw'] = np.copy(config.FT['usaw'][it-1])
                    
    #             # config.FT['usaw'][it] = searchfunction(config.FT['usaw'][it-1])         # Fonction search de vitesse 1µm/frame par piston
                    
    #             # Kernel = np.identity(NA) - Igdna
    #             # simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
    #             # Kernel = np.dot(simu.NoPhotometryFiltration[it],Kernel)                 
                
    #             # # After multiplication by Kernel, the OPD velocities can only be lower or equal than before
                
    #             # usearch = np.dot(Kernel,config.FT['usaw'][it]*config.FT['Velocities'])
            
    #             if (config.FT['state'][it-1] == 0):# or (sum(TelescopesThatNeedARestart) >0) :
    #                 config.FT['it0'] = it ; config.FT['it_last'] = it
    #                 config.FT['last_usaw'] = config.FT['usaw'][it-1]
            
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
    
    IgdRank = np.linalg.matrix_rank(simu.Igd[it])
    NotCophased = (IgdRank < NA-1)
    simu.IgdRank[it] = IgdRank
    
    if NotCophased:
        simu.time_since_loss[it]=simu.time_since_loss[it-1]+config.dt
        
        # FringeLost = (NotCophased and (IgdRank<np.linalg.matrix_rank(simu.Igd[it-1]))
        # This situation could pose a problem but we don't manage it yet        
        if (simu.time_since_loss[it] > config.FT['SMdelay']):
            
            Igdna = np.dot(config.FS['OPD2Piston'],
                           np.dot(simu.Igd[it],config.FS['Piston2OPD']))
            
            CophasedBaselines=np.where(np.diag(simu.Igd[it])>0.5)[0]
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

                # If it=0, initialize LastPosition to 0. 
                # Else, it will remain the last value of SearchCommand, which has
                # not change since last RELOCK state.
                
                LastPosition = config.FT['LastPosition'][it]
                
                newLostTelescopes = (simu.LostTelescopes[it] - simu.LostTelescopes[it-1] == 1)
                TelescopesThatGotBackPhotometry = (simu.noSignal_on_T[it-1] - simu.noSignal_on_T[it] == 1)
                # WeGotBackPhotometry = (sum(TelescopesThatGotBackPhotometry) > 0)
                
                TelescopesThatNeedARestart = np.argwhere(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)

                if sum(TelescopesThatNeedARestart)>0:
                    config.FT['it_last']=it ; #Ldico[ia]['eps']=1 #; Ldico[ia]['it0']=it ;   

                usaw, change = searchfunction_inc(it)
                config.FT['usaw'][it]= usaw

                Kernel = np.identity(NA) - Igdna
                Increment = np.dot(Kernel,config.FT['usaw'][it]*config.FT['Velocities'])
                Increment = Increment/np.ptp(Increment) * config.FT['maxVelocity']
                
                if change:
                    for group in CophasedGroups:
                        for ig in range(1,len(group)):
                            ia = int(float(group[ig])-1) ; i0 = int(float(group[0])-1)
                            LastPosition[ia] = LastPosition[i0] + simu.SearchCommand[it,ia]-simu.SearchCommand[it,i0]
                    usearch = LastPosition + Increment
                    LastPosition = simu.SearchCommand[it]
                    
                else:
                    usearch = simu.SearchCommand[it]+Increment
                    
                config.FT['LastPosition'][it+1] = LastPosition
                
                # You should send command only on telescope with flux
                simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
                usearch = np.dot(simu.NoPhotometryFiltration[it],usearch)
            
            
            else:
                usearch = simu.SearchCommand[it]
        
        else:
            usearch = simu.SearchCommand[it]
            
    else:
        simu.time_since_loss[it] = 0
        usearch = simu.SearchCommand[it]
        
        
    # if config.TELref:
    #     iTel = config.TELref-1
    #     usearch = usearch - usearch[iTel]
    
    usearch = config.FT['search']*usearch
    # The command is sent at the next time, that's why we note it+1
    simu.SearchCommand[it+1] = usearch
    
   
    
    """
    Group-Delay tracking
    """
    
    currGDerr = currGD - simu.GDref[it]
    
    # Keep the GD between [-Pi, Pi]
    # Eq. 35
    # Array elements verifying the condition
    higher_than_pi = (currGDerr > np.pi)
    lower_than_mpi = (currGDerr < -np.pi)
    
    currGDerr[higher_than_pi] -= 2*np.pi
    currGDerr[lower_than_mpi] += 2*np.pi

    
    R = config.FS['R']
    
    # Store residual GD for display only [radians]
    simu.GDResidual[it] = currGDerr*R*config.PDspectra/(2*np.pi)
    
    # Weights the GD (Eq.35)
    currGDerr = np.dot(currIgd,currGDerr)
     
    simu.GDResidual2[it] = currGDerr*R*config.PDspectra/(2*np.pi)
    simu.GDPistonResidual[it] = np.dot(FS['OPD2Piston'], currGDerr*R*config.PDspectra/(2*np.pi))
    
    # Threshold function (eq.36)
    if FT['Threshold']:
    
        # Array elements verifying the condition
        higher_than_pi = (currGDerr > np.pi/R*FT['switch'])
        lower_than_mpi = (currGDerr < -np.pi/R*FT['switch'])
        within_pi = (np.abs(currGDerr) <= np.pi/R*FT['switch'])
        
        # if FT['continu']:
        #     currGDerr[higher_than_pi] -= np.pi/R*FT['switch']
        #     currGDerr[lower_than_mpi] += np.pi/R*FT['switch']
        currGDerr[within_pi] = 0
    
    simu.GDErr[it] = currGDerr
    
    # Integrator (Eq.37)
    if FT['cmdOPD']:     # integrator on OPD
        # Integrator
        simu.GDCommand[it+1] = simu.GDCommand[it] + FT['GainGD']*currGDerr*config.PDspectra*config.FS['R']/2/np.pi
        
        # From OPD to Pistons
        uGD = np.dot(FS['OPD2Piston'], simu.GDCommand[it+1])
        
    else:                       # integrator on Pistons
        # From OPD to Piston
        currPistonGD = np.dot(FS['OPD2Piston'], currGDerr)*config.PDspectra*config.FS['R']/2/np.pi
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
    
    currPDerr = currPD - simu.PDref[it]
 
    # Keep the PD between [-Pi, Pi]
    # Eq. 35
    
    # Array elements verifying the condition
    higher_than_pi = (currPDerr > np.pi)
    lower_than_mpi = (currPDerr < -np.pi)
    
    currPDerr[higher_than_pi] -= 2*np.pi
    currPDerr[lower_than_mpi] += 2*np.pi
    
    simu.PDResidual[it] = currPDerr*config.PDspectra/(2*np.pi)
    
    # Weights the PD (Eq.35)
    currPDerr = np.dot(currIpd,currPDerr)
    
    # Store residual PD and piston for display only
    simu.PDResidual2[it] = currPDerr*config.PDspectra/(2*np.pi)
    simu.PDPistonResidual[it] = np.dot(FS['OPD2Piston'], currPDerr*config.PDspectra/(2*np.pi))
    
    # Integrator (Eq.37)
            
    if FT['cmdOPD']:     # integrator on OPD
        # Integrator
        simu.PDCommand[it+1] = simu.PDCommand[it] + FT['GainPD']*currPDerr*config.PDspectra/2/np.pi
        # From OPD to Pistons
        uPD = np.dot(FS['OPD2Piston'], simu.PDCommand[it+1])
        
    else:                       # integrator on Pistons
        # From OPD to Piston
        currPistonPD = np.dot(FS['OPD2Piston'], currPDerr)*config.PDspectra/2/np.pi
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
    
    from .config import NA, NIN, MW
    
    from .simu import it
    from .coh_tools import simu2GRAV, NB2NIN
    
    image = simu.MacroImages[it]
    
    M = config.M            # Amplification ratio camera
    
    sigsky = config.FS['sigsky']  # Background noise
    imsky = config.FS['imsky']    # Sky image before observation
    Demod = config.FS['MacroP2VM']    # Macro P2VM matrix used for demodulation
    ElementsNormDemod = config.FS['ElementsNormDemod']
    
    DemodGRAV = simu2GRAV(Demod, direction='p2vm')
    
    DemodGRAV = config.FS['MacroP2VMgrav']
    # Flux variance calculation (eq. 12)
    varFlux = sigsky**2 + M*(image - imsky)
    simu.varFlux[it] = varFlux
    
    """
    Covariance calculation (eq. 13)
    """
    
    for imw in range(MW):
        simu.CovarianceReal[it,imw] = np.sum(np.real(Demod[imw])**2*varFlux[imw], axis=1)
        simu.CovarianceImag[it,imw] = np.sum(np.imag(Demod[imw])**2*varFlux[imw], axis=1)

        simu.Covariance[it,imw] = np.dot(DemodGRAV[imw], np.dot(np.diag(varFlux[imw]),np.transpose(DemodGRAV[imw])))
        simu.BiasModCf[it,imw] = np.dot(ElementsNormDemod[imw],varFlux[imw])
        
    simu.DemodGRAV = DemodGRAV
    
    # Phase variance calculation (eq. 14)
    Nvar = config.FT['Nvar']                # Integration time for phase variance
    if it < Nvar:
        Nvar = it+1
        
    varNum = np.zeros([MW,NIN]) ; varNum2 = np.zeros([MW,NIN])
    varPhot = np.zeros([MW,NA])         # Variance of the photometry measurement
    CohFlux = np.zeros([MW,NIN])*1j
    
# =============================================================================
#     NOTATIONS:
#       CfPD = sqrt(Fi*Fj)*exp(i\Phi_ij)*ComplexCoherenceDegree = X + i*Y
#       Ex = Esperance of X ; Ey = Esperance of Y
#       VarX = Variance of X ; VarY = Variance of Y
# =============================================================================
    diagCovar = np.diagonal(simu.Covariance, axis1=2, axis2=3)
    varPhot = diagCovar[it,:,:NA]
    timerange = range(it+1-Nvar,it+1)
    varX = np.abs(diagCovar[timerange,:,NA:NA+NIN])
    varY = np.abs(diagCovar[timerange,:,NA+NIN:])
    varNum = np.mean(varX+varY,axis=0)
    # for ia in range(NA):
    #     ibp = ia*(NA+1)
    #     varPhot[:,ia] = simu.Covariance[it,:,ibp,ibp]       # Variance of photometry at each frame
    #     for iap in range(ia+1,NA):
    #         ib = posk(ia,iap,NA)
    #         ibr=NA+ib; varX = simu.Covariance[timerange,:,ibr,ibr]
    #         ibi=NA+NIN+ib; varY = simu.Covariance[timerange,:,ibi,ibi]
    #         covarXY = simu.Covariance[timerange,:,ibr,ibi]
            
    #         varNum[:,ib] = np.mean(varX+varY,axis=0)
    #         varNum2[:,ib] = np.mean(varX+varY + 2*covarXY, axis=0)
            
    CohFlux = np.mean(simu.CfPD[timerange], axis=0)
    CfSumOverLmbda = np.sum(CohFlux,axis=0)
    
    simu.varPDdenom[it] = np.sum(np.real(CohFlux*np.conj(CohFlux)),axis=0)  # Sum over lambdas of |CohFlux|² (modified eq.14)
    simu.varPDdenomDebiased[it] = np.sum(np.real(CohFlux*np.conj(CohFlux))-simu.BiasModCf[it],axis=0)  # Sum over lambdas of |CohFlux|²
    simu.varPDdenom2[it] = np.real(CfSumOverLmbda*np.conj(CfSumOverLmbda)-np.mean(simu.BiasModCf[it],axis=0)) # Original eq.14
    #simu.varPDdenom2[it] = np.sum(np.mean(np.abs(simu.CfPD[timerange])**2,axis=0),axis=0)
    simu.varPDnum[it] = np.sum(varNum,axis=0)/2     # Sum over lmbdas of Variance of |CohFlux|
    
    simu.varPD[it] = simu.varPDnum[it]/simu.varPDdenom[it]      # Var(|CohFlux|)/|CohFlux|²
    simu.varPD2[it] = simu.varPDnum[it]/simu.varPDdenom2[it]      # Var(|CohFlux|)/|CohFlux|²
    simu.varPDdebiased[it] = simu.varPDnum[it]/simu.varPDdenomDebiased[it]
    
    simu.SNRPhotometry[it,:] = np.sum(simu.PhotometryEstimated[it,:],axis=0)/np.sqrt(np.sum(varPhot,axis=0))
    
    varPD = simu.varPD[it]
    varPD2 = simu.varPD2[it]
    
    return varPD, varPD2


def SetThreshold(manual=False, scan=False,display=False,verbose=True,scanned_tel=6):
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

    if scan:

        datadir2 = "C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/disturbances/"
    
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
        from cophasing.SPICA_FT import SPICAFT, updateFTparams
        #SPICAFT(init=True, GainPD=0, GainGD=0,search=False)
        gainPD,gainGD,search=config.FT['GainPD'],config.FT['GainGD'],config.FT['search']
        updateFTparams(GainPD=0, GainGD=0, search=False, verbose=verbose)
        
        # Launch the scan
        sk.loop()
        
        if manual:
            sk.display('snr',WLOfTrack=1.6, pause=True)
            test1 = input("Set all threshold to same value? [y/n]")
            if (test1=='y') or (test1=='yes'):    
                newThresholdGD = float(input("Set the Threshold GD: "))
                
            elif (test1=='n') or (test1=='no'):
                newThresholdGD=np.ones(config.NIN)
                for ib in range(config.NIN):
                    newThresholdGD[ib] = float(input(f"Threshold base {config.FS['ich'][ib]}:"))
            else:
                raise ValueError("Please answer by 'y','yes','n' or 'no'")
                
            test2 = input("Set also a threshold for RELOCK? [y/n]")
            if (test2 == 'y') or (test2=='yes'):
                
                test3 = input("Set all threshold to same value? [y/n]")
                if (test3 == 'y') or (test3=='yes'):
                    newThresholdRELOCK = float(input("Set the Threshold RELOCK: "))
                
                elif (test3=='n') or (test3=='no'):
                    newThresholdGD=np.ones(config.NIN)
                    newThresholdRELOCK=np.ones(config.NIN)
                    for ib in range(config.NIN):
                        newThresholdGD[ib] = float(input(f"Threshold base {config.FS['ich'][ib]}:"))
                        newThresholdRELOCK[ib] = float(input(f"Threshold RELOCK base {config.FS['ich'][ib]}:"))
                
                else:
                    raise ValueError("Please answer by 'y','yes','n' or 'no'")
            
            elif (test2=='n') or (test2=='no'):
                newThresholdRELOCK=newThresholdGD
                pass
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
            newThresholdRELOCK = np.array([np.max([2,x]) for x in np.sqrt(simu.SquaredSNRMovingAverage2[ind,:])])
                    
            config.FT['ThresholdGD'] = newThresholdGD
            config.FT['ThresholdRELOCK'] = newThresholdRELOCK
            
            sk.display('snr',WLOfTrack=1.6, pause=True,display=display)
            
        sk.update_config(DisturbanceFile=InitialDisturbanceFile, NT=InitNT,
                         verbose=verbose)
        updateFTparams(GainPD=gainPD, GainGD=gainGD, search=search, 
                       ThresholdGD=newThresholdGD, ThresholdRELOCK=newThresholdRELOCK,
                       verbose=verbose)
    
    
    else:
        
        DisturbanceFile = "C:/Users/cpannetier/Documents/Python_packages/cophasing/cophasing/data/disturbances/NoDisturbances/NoDisturbances.fits"
        
        NT=200
            
        InitialDisturbanceFile,InitNT = sk.config.DisturbanceFile, sk.config.NT
        
        sk.update_config(DisturbanceFile=DisturbanceFile, NT = NT, verbose=verbose)
        
        # Initialize the fringe tracker with the gain
        from cophasing.SPICA_FT import SPICAFT, updateFTparams
        #SPICAFT(init=True, GainPD=0, GainGD=0,search=False)
        gainPD,gainGD,search=config.FT['GainPD'],config.FT['GainGD'],config.FT['search']
        updateFTparams(GainPD=0, GainGD=0, search=False, verbose=verbose)
        
        # Launch the scan
        sk.loop()
        
        if manual:
            sk.display('detector',WLOfTrack=1.6, pause=True)
            test1 = input("Set all threshold to same value? [y/n]")
            if (test1=='y') or (test1=='yes'):    
                newThresholdGD = float(input("Set the Threshold GD: "))
            elif (test1=='n') or (test1=='no'):
                newThresholdGD=np.ones(config.NIN)
                for ib in range(config.NIN):
                    newThresholdGD[ib] = float(input(f"Threshold base {config.FS['ich'][ib]}:"))
            else:
                raise ValueError("Please answer by 'y','yes','n' or 'no'")
            
            test2 = input("Set also a threshold for RELOCK? [y/n]")
            if (test2 == 'y') or (test2=='yes'):
                
                test3 = input("Set all threshold to same value? [y/n]")
                if (test3 == 'y') or (test3=='yes'):
                    newThresholdRELOCK = float(input("Set the Threshold RELOCK: "))
                
                elif (test3=='n') or (test3=='no'):
                    newThresholdGD=np.ones(config.NIN)
                    newThresholdRELOCK=np.ones(config.NIN)
                    for ib in range(config.NIN):
                        newThresholdGD[ib] = float(input(f"Threshold base {config.FS['ich'][ib]}:"))
                        newThresholdRELOCK[ib] = float(input(f"Threshold RELOCK base {config.FS['ich'][ib]}:"))
                
                else:
                    raise ValueError("Please answer by 'y','yes','n' or 'no'")
            
            elif (test2=='n') or (test2=='no'):
                newThresholdRELOCK=newThresholdGD
                pass
            else:
                raise ValueError("Please answer by 'y','yes','n' or 'no'")
                
                
        else:
            from cophasing import simu,coh_tools
            
            ind=100
            
            newThresholdGD = np.array([np.max([2,x*0.2]) for x in np.sqrt(simu.SquaredSNRMovingAverage[ind,:])])
            newThresholdRELOCK = np.array([np.max([2,x*0.2]) for x in np.sqrt(simu.SquaredSNRMovingAverage2[ind,:])])
            
            config.FT['ThresholdGD'] = newThresholdGD
            config.FT['ThresholdRELOCK'] = newThresholdRELOCK
            sk.display('detector',WLOfTrack=1.6, pause=True,display=display)
            
        sk.update_config(DisturbanceFile=InitialDisturbanceFile, NT=InitNT,
                         verbose=verbose)
        updateFTparams(GainPD=gainPD, GainGD=gainGD, search=search, 
                       ThresholdGD=newThresholdGD,ThresholdRELOCK=newThresholdRELOCK,
                       verbose=verbose)
    
    return newThresholdGD

    

def getvarcupy():
    """
    From the image and calibrated quantities (sky), calculates the "Phase variance"
    which is in reality the inverse of the squared signal-to-noise ratio of the 
    fringes.

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.

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
    import cupy as cp
    
    from .config import NA, NIN, MW
    from .simu import it
    
    
    image = cp.array(simu.MacroImages[it,:,:])
    
    sigsky = config.FS['sigsky']                    # Background noise
    imsky = config.FS['imsky']                      # Sky image before observation
    Demod = cp.array(config.FS['MacroP2VM'])        # Macro P2VM matrix used for demodulation
    
    # Flux variance calculation (eq. 12)
    varFlux = sigsky**2 + 1*(image - imsky)
    
    """
    Covariance calculation (eq. 13)
    """
    
    for iw in range(MW):
    #     # simu.Covariance[it,iw] = cp.dot(Demod[iw], cp.dot(cp.diag(varFlux[iw]),cp.transpose(Demod[iw])))
        simu.Covariance[it,iw] = cp.sum(Demod[iw]**2*varFlux[iw], axis=1)
        
    # Phase variance calculation (eq. 14)

    Nvar = 5                # Integration time for phase variance
    if it < Nvar:
        Nvar = it+1
        
    vartemp = cp.zeros([MW,NIN])
    vartemp2 = cp.zeros([MW,NIN])*1j
    
    # for iot in range(it+1-Nvar,it+1):
    timerange = np.arange(it+1-Nvar,it+1)
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib = posk(ia,iap,NA)
            ibb = ia*NA+iap
            vartemp[:,ib] = (cp.sum(cp.abs(cp.real(simu.Covariance[timerange,:,ibb])),axis=0)+cp.sum(cp.abs(cp.imag(simu.Covariance[timerange,:,ibb])),axis=0))/Nvar
            vartemp2[:,ib] = cp.sum(cp.array(simu.CfPD[timerange,:,ib]), axis=0)/Nvar
    
    varPD = cp.sum(vartemp,axis=0)/(2*cp.abs(cp.sum(vartemp2,axis=0))**2)

    varPDnumpy = cp.asnumpy(varPD)
    simu.varPD[it] = varPDnumpy
    
    return varPDnumpy


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
            # print(it*dt,"change")                                       # this depends on ia
            utemp=usaw[ia]
            config.FT['eps'][ia] = -config.FT['eps'][ia]
            config.FT['it_last'][ia] = it
            usaw[ia] = config.FT['last_usaw'][ia] + config.FT['eps'][ia]
            config.FT['last_usaw'][ia] = utemp
        
    
    simu.eps[it] = config.FT['eps']
    simu.it_last[it] = config.FT['it_last']
    simu.last_usaw[it] = config.FT['last_usaw']
        
    # Investigation data
    
    return usaw

def searchfunction2(usaw,it):
    
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
        usaw = config.FT['last_usaw'] + config.FT['eps']
        config.FT['last_usaw'] = utemp

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
        diff = config.FT['last_usaw'] - usaw
        config.FT['eps'] = -config.FT['eps']
        config.FT['it_last'] = it
        usaw = diff + config.FT['eps']
        config.FT['last_usaw'] = utemp

    config.FT['usaw'][it] = usaw
        
    return usaw

def searchfunction_inc(it):
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
    
    a = config.FT['Sweep30s']/30000  # Coefficient directeur de la fonction d'augmentation du temps avant saut.
    sweep = config.FT['Sweep0'] + a*(it-it0)*config.dt
    
    time_since_last_change = (it-it_last)*config.dt
    
    if time_since_last_change < sweep:
        change=False
        usaw = eps
        
    else:
        change=True
        eps = -eps
        it_last = it
        usaw = eps
        config.FT['it_last'] = it_last
        
    config.FT['eps'] = eps
    
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
        L.append(list(x))
            
    return(L)