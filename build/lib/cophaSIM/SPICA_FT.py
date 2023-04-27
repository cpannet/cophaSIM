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
from .skeleton import updateFTparams

def SPICAFT(*args, init=False, search=False, update=False, GainPD=0, GainGD=0, Ngd=50, roundGD='round', Ncross=1,
            relock=True,SMdelay=1e3,sweep0=20, sweep30s=10, commonRatio=1.1, covering=10, maxVelocity=0.300, searchMinGD=500,
            relock_vfactors = [], search_vfactors=[],searchThreshGD=3,Nsearch=50,searchSNR='gd',
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
        - outputs.PDEstimated: [NT,MW,NINmes] Estimated PD before subtraction of the reference
        - outputs.GDEstimated: [NT,MW,NINmes] Estimated GD before subtraction of the reference
        - outputs.CommandODL: Piston Command to send       [NT,NA]
        
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
    search : BOOLEAN, optional
        If True, the function triggers the SEARCH state before TRACKING.
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
    relock : TYPE, optional
        DESCRIPTION. The default is True.
    SMdelay: FLOAT, optional
        Time to wait after losing a telescope for triggering the SEARCH command.
    sweep0 : TYPE, optional
        DESCRIPTION. The default is 20.
    sweep30s : TYPE, optional
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
        
        # config.R = np.abs((config.MW-1)*config.wlOfTrack/(config.spectraM[-1] - config.spectraM[0]))
        config.FT['Name'] = 'SPICAfromGRAVITY'
        config.FT['func'] = SPICAFT
        config.FT['Ngd'] = Ngd
        config.FT['GainGD'] = GainGD
        config.FT['GainPD'] = GainPD
        config.FT['state'] = np.zeros(NT+1)
        if search:
            config.FT['state'][0] = 2
            
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
        
        # RELOCK command parameters
        config.FT['relock'] = relock            # If True, the fringe-tracker can eneter in RELOCK state. If False, it never enters this state (for debugging only).
        config.FT['search'] = search            # If True, the simulation begins with SEARCH state. If False, no SEARCH state, directly TRACK state."
        config.FT['SMdelay'] = SMdelay          # Waiting time before launching relock
        config.FT['sweep0'] = sweep0            # Starting sweep in seconds
        config.FT['sweep30s'] = sweep30s        # Sweep at 30s in seconds
        config.FT['commonRatio'] = commonRatio  # Common ratio of the geometrical sequence
        config.FT['covering'] = covering        # Covering of the sawtooth function in microns
        config.FT['maxVelocity'] = maxVelocity  # Maximal slope given in µm/frame
        config.FT['searchThreshGD'] = searchThreshGD*config.wlOfTrack    # Maximal value of GD for considering fringe found.
        config.FT['Nsearch'] = Nsearch          
        config.FT['searchSNR'] = searchSNR      
        config.FT['searchMinGD'] = np.ones(NINmes)*searchMinGD  # Value of minimal reached value of GD during search
        config.FT['diffOffsets_best'] = np.zeros(NINmes)        # Vector that will contain the offsets of the fringes.
        config.FT['Ws'] = np.zeros(NINmes)        # Vector that will contain the offsets of the fringes.
        config.FT['searchMaxSnr'] = np.zeros(NINmes)        # Vector that will contain the offsets of the fringes.
            
        # Version usaw vector
        config.FT['usaw'] = np.zeros([NT,NA])
        config.FT['LastPosition'] = np.zeros(NA)
        config.FT['it_last'] = np.zeros(NA)
        config.FT['it0'] = np.zeros(NA)
        config.FT['eps'] = np.ones(NA)
        config.FT['nbChanges'] = np.ones(NA)

        # Version usaw float
        # config.FT['usaw'] = np.zeros([NT])
        # config.FT['uRelock'] = np.zeros([NT,NA])
        # config.FT['LastPosition'] = np.zeros([NT+1,NA])
        # config.FT['it_last'] = 0
        # config.FT['it0'] = 0
        # config.FT['eps'] = 1
        
        
        config.FT['ThresholdPhot'] = ThresholdPhot      # Minimal photometry SNR for launching relock

        if (len(relock_vfactors) == 0) and (len(relock_vfactors) == NA):
            config.FT['relock_vfactors'] = np.array(relock_vfactors)
        else:
            if verbose:
                print("No or bad relock_vfactors given. I create one.")

            if NA==6:
                config.FT['relock_vfactors'] = np.array([-8.25, -7.25, -4.25, 1.75, 3.75, 8.75])
                
            elif NA==7: # Fake values
                config.FT['relock_vfactors'] = np.array([-8.25, -7.25, -4.25, 1.75, 3.75, 8.75, 10])
                
            elif NA==10:
                config.FT['relock_vfactors'] = np.array([-24.9, -23.9, -18.9, -14.9,
                                                  -1.9,   1.1,   9.1,  16.1,
                                                  28.1, 30.1])
        
        if (len(search_vfactors) == 0) and (len(search_vfactors) == NA):
            config.FT['search_vfactors'] = np.array(search_vfactors)
        else:
            if verbose:
                print("No or bad search_vfactors given. I create one.")
                
            config.FT['search_vfactors'] = np.arange(NA)-NA//2+1
            
        if search==True:
            config.FT['Vfactors'] = config.FT['search_vfactors']
        else:
            config.FT['Vfactors'] = config.FT['relock_vfactors']
            
        config.FT['relock_vfactors'] = config.FT['relock_vfactors']/np.ptp(config.FT['relock_vfactors'])*maxVelocity     # The maximal OPD velocity is equal to slope/frame
        config.FT['search_vfactors'] = config.FT['search_vfactors']/np.ptp(config.FT['search_vfactors'])*maxVelocity     # The maximal OPD velocity is equal to slope/frame
        
        return

    elif update:
        if verbose:
            print("Update fringe-tracker parameters with:")
        for key, value in zip(list(kwargs_for_update.keys()),list(kwargs_for_update.values())):
            setattr(config.FT, key, value)
            if verbose:
                print(f" - {key}: {getattr(config.FT, key, value)}")

        return

    currCfEstimated = args[0]

    CfPD, CfGD = ReadCf(currCfEstimated)
    
    currCmd = CommandCalc(CfPD, CfGD)
    
    return currCmd



def SearchState(CophasedGroups=[]):
    
    from . import outputs,config
    
    from .config import wlOfTrack, NA, FS, FT
    it=outputs.it 
    
    searchThreshGD = FT['searchThreshGD']
    
    Nsearch = FT['Nsearch'] ; Ngd = FT['Ngd']
    timerange = range(it+1-Nsearch, it+1) 
        
    if FT['searchSNR'] == 'pd':
        varSignal = outputs.varPD[timerange]
    elif FT['searchSNR'] == 'gd':
        varSignal = outputs.varGD[timerange]
    elif FT['searchSNR'] == 'pdtemp':
        varSignal = outputs.TemporalVariancePD[timerange]
    elif FT['searchSNR'] == 'gdtemp':
        varSignal = outputs.TemporalVarianceGD[timerange]
    else:
        raise ValueError("Parameter 'searchSNR' must be one of the following: [pd,gd,pdtemp,gdtemp]")
        
    outputs.SearchSNR[it] = np.sqrt(np.nan_to_num(1/np.mean(varSignal,axis=0)))
    
    # SNR_movingaverage = np.sqrt(np.nan_to_num(1/varSignal))
    GradSNR = np.gradient(outputs.SearchSNR[timerange],axis=0)
    
    # Current Group-Delay
    NINmes = FS['NINmes'] ; R = FS['R'] ; Ncross = FT['Ncross']
    currGD = np.zeros(NINmes)
    for ib in range(NINmes):
        cfGDlmbdas = outputs.CfGD[it,:-Ncross,ib]*np.conjugate(outputs.CfGD[it,Ncross:,ib])
        cfGDmoy = np.sum(cfGDlmbdas)
        
        currGD[ib] = np.angle(cfGDmoy*np.exp(-1j*outputs.GDref[it,ib]))
        
    outputs.GDEstimated[it] = currGD
    GDmic = currGD *R*wlOfTrack/(2*np.pi)
    
    MaxSNRCondition = (np.mean(GradSNR[:Nsearch//2],axis=0)*np.mean(GradSNR[Nsearch//2:],axis=0) < 0) # Sign change in first derivative

    #MaxSNRCondition = (np.mean(outputs.SearchSNR[it-Nsearch:it-Nsearch//2],axis=0)-outputs.SearchSNR[it-1]<0)
    # snrHigherThanThreshold = (outputs.SearchSNR[it] > config.FT['ThresholdGD'])
    SNR = np.sqrt(outputs.SquaredSNRMovingAverage[it])
    
    snrHigherThanThreshold = (SNR > FT['ThresholdGD'])
    lowEnoughGD = (np.abs(GDmic) < searchThreshGD)  # C'est pas ouf car latence dans les mesures
    lowerGD = (np.abs(GDmic) < FT['searchMinGD'])
    higherSnr = (SNR > FT['searchMaxSnr'])
    NoRecentChange = (config.FT['it_last'][0] < it-Ngd)
    
    
    # Ws = outputs.SearchSNR[it] * snrHigherThanThreshold * NoRecentChange * lowEnoughGD# * MaxSNRCondition)
    Ws = SNR * snrHigherThanThreshold * NoRecentChange * lowEnoughGD# * GDNullCondition# * GDNullCondition)# * NoRecentChange)# * MaxSNRCondition)
        
    for ib in range(NINmes):
        ia = int(FS['ich'][ib][0])-1
        iap = int(FS['ich'][ib][1])-1
        if Ws[ib] != 0:
            outputs.diffOffsets[it,ib] = outputs.EffectiveMoveODL[it,ia]-outputs.EffectiveMoveODL[it,iap]
    
            if higherSnr[ib]:
                FT['searchMinGD'][ib] = GDmic[ib]
                FT['diffOffsets_best'][ib] = outputs.diffOffsets[it,ib]
                FT['Ws'][ib] = Ws[ib]
        
    outputs.diffOffsets_best[it] = config.FT['diffOffsets_best']
    outputs.Ws[it] = config.FT['Ws']
                
    # Transpose the W matrix in the Piston-space
    Is = np.dot(config.FS['OPD2Piston_r'],np.dot(np.diag(FT['Ws']),config.FS['Piston2OPD_r']))
    outputs.Is[it] = Is
    
    rankIs = np.linalg.matrix_rank(Is)
    outputs.rankIs[it] = rankIs
    
    allTelFound = (rankIs == NA-1)
    
    if allTelFound:
        config.FT['state'][it+1] = 0    # Set SPICA-FT to TRACK state
        
        # Singular-Value-Decomposition of the W matrix
        U, S, Vt = np.linalg.svd(Is)
        
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
        
        Igd = np.dot(FS['Piston2OPD_r'],np.dot(VSdagUt,np.dot(FS['OPD2Piston_r'], np.diag(FT['Ws']))))
        
        uSearchOpd = np.dot(Igd,config.FT['diffOffsets_best'])
        
        uSearch = np.dot(FS['OPD2Piston'],uSearchOpd)
         
        outputs.CommandSearch[it+1] = uSearch
        # outputs.CommandRelock[it+1] = uSearch         # Patch to propagate the command to the fringe-tracker commands
        config.FT['Vfactors'] = config.FT['relock_vfactors']    # Set Vfactors to the RELOCK values since it will never come back to SEARCH state.
        
        # Reinitialise parameters for sawtooth function
        config.FT['eps'] = np.ones(NA)
        config.FT['nbChanges'] = np.ones(NA)
        
    else:
        config.FT['state'][it+1] = 2    # Remain in SEARCH state
        
        # newLostTelescopes = (outputs.LostTelescopes[it] - outputs.LostTelescopes[it-1] == 1)
        # TelescopesThatGotBackPhotometry = (outputs.noSignal_on_T[it-1] - outputs.noSignal_on_T[it] == 1)
        # WeGotBackPhotometry = (sum(TelescopesThatGotBackPhotometry) > 0)
        
        # TelescopesThatNeedARestart = np.argwhere(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)

        if it>=1:
            if config.FT['state'][it-1] != 2:
                config.FT['eps'] = np.ones(NA)
                config.FT['it0'] = np.ones(NA)*it
                config.FT['it_last'] = np.ones(NA)*it
                    
        Increment = relockfunction_inc_sylvain_gestioncophased(it, config.FT['search_vfactors'], config.FT['covering'], CophasedGroups)
    
        #You should send command only on telescope with flux
        #outputs.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(outputs.noSignal_on_T[it])
        #Increment = np.dot(outputs.NoPhotometryFiltration[it],Increment)
        outputs.CommandSearch[it+1] = outputs.CommandSearch[it] + Increment
        
        uSearch = outputs.CommandSearch[it+1]
        
    return uSearch



def ReadCf(currCfEstimated):
    """
    From measured coherent flux, estimates GD, PD, CP, Photometry, Visibilities
    
    NAME: 
        COH_ALGO - Calculates the group-delay, phase-delay, closure phase and 
        visibility from the fringe sensor image.
    
        
    INPUT: CfEstimated [MW,NB]
    
    OUTPUT: 
        
    UPDATE:
        - outputs.CfEstimated_             
        - outputs.CfPD: Coherent Flux Phase-Delay     [NT,MW,NINmes]
        - outputs.CfGD: Coherent Flux GD              [NT,MW,NINmes]
        - outputs.ClosurePhasePD                       [NT,MW,NC]
        - outputs.ClosurePhaseGD                       [NT,MW,NC]
        - outputs.PhotometryEstimated                  [NT,MW,NA]
        - outputs.VisibilityEstimated                    [NT,MW,NIN]*1j
        - outputs.CoherenceDegree                      [NT,MW,NIN]
    """
    
    from . import outputs
    
    from .config import NA,NC
    from .config import MW
    
    it = outputs.it            # Time
     
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
    outputs.PhotometryEstimated[it] = PhotEst

        
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
        outputs.VisibilityEstimated[it,:,ib] = 2*Iaap/(Ia+Iap)          # Estimated Fringe Visibility of the base (a,a')
        outputs.SquaredCoherenceDegree[it,:,ib] = np.abs(Iaap)**2/(Ia*Iap)      # Spatial coherence of the source on the base (a,a')
 
 
    """
    Phase-delays extraction
    PD_ is a global stack variable [NT, NIN]
    Eq. 10 & 11
    """
    D = 0   # Dispersion correction factor: so far, put to zero because it's a 
            # calibration term (to define in coh_fs?)
            
    from .config import wlOfTrack
    
    # Coherent flux corrected from dispersion
    for imw in range(MW):
        outputs.CfPD[it,imw,:] = currCfEstimatedNIN[imw,:]*np.exp(1j*D*(1-wlOfTrack/config.spectraM[imw])**2)
        
        # If ClosurePhase correction before wrapping
        # outputs.CfPD[it,imw] = outputs.CfPD[it,imw]*np.exp(-1j*outputs.PDref[it])

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
        cfgd = outputs.CfPD[iot]*np.exp(-1j*outputs.PDEstimated[iot])/Ngd
        
        # If ClosurePhase correction before wrapping
        # cfgd = cfgd*np.exp(-1j*outputs.GDref[it])
        
        outputs.CfGD[it,:,:] += cfgd
    
    CfPD = outputs.CfPD[it]
    CfGD = outputs.CfGD[it]

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
    
    from . import outputs
    
    """
    INIT MODE
    """
    
    from .config import NA,NC
    from .config import FT,FS
    
    it = outputs.it            # Frame number
    
    NINmes = config.FS['NINmes']
    ich_pos = config.FS['active_ich']
    
                
    Ngd = FT['Ngd']
    if it < FT['Ngd']:
        Ngd = it+1
    
    Ncross = config.FT['Ncross']  # Distance between wavelengths channels for GD calculation
    
    R = config.FS['R']
    
    
    """
    Signal-to-noise ratio of the fringes ("Phase variance")
    The function getvar saves the inverse of the squared SNR ("Phase variance")
    in the global stack variable varPD [NT, MW, NIN]
    Eq. 12, 13 & 14
    """

    varcurrPD, varcurrGD = getvar()
    
    timerange = range(it+1-Ngd, it+1)
    outputs.SquaredSNRMovingAveragePD[it] = np.nan_to_num(1/np.mean(outputs.varPD[timerange], axis=0))
    outputs.SquaredSNRMovingAverageGD[it] = np.nan_to_num(1/np.mean(outputs.varGD[timerange], axis=0))
    outputs.SquaredSNRMovingAverageGDUnbiased[it] = np.nan_to_num(1/np.mean(outputs.varGDUnbiased[timerange], axis=0))
    
    outputs.TemporalVariancePD[it] = np.var(outputs.PDEstimated[timerange], axis=0)
    outputs.TemporalVarianceGD[it] = np.var(outputs.GDEstimated[timerange], axis=0)
    
    if config.FT['whichSNR'] == 'pd':
        outputs.SquaredSNRMovingAverage[it] = outputs.SquaredSNRMovingAveragePD[it]
    else:
        outputs.SquaredSNRMovingAverage[it] = outputs.SquaredSNRMovingAverageGD[it]
        
    
    """
    SEARCH State
    """
    
    if config.FT['state'][it] == 2:
        uSearch = SearchState()
        CommandODL = uSearch
        # CommandODL = np.zeros(NA)
        return CommandODL
    else:
        outputs.CommandSearch[it+1] = outputs.CommandSearch[it]
        uSearch = outputs.CommandSearch[it+1]
    
    """
    WEIGHTING MATRIX
    """

    if config.FT['useWmatrices']:
        
        # Raw Weighting matrix in the OPD-space
        
        reliablebaselines = (outputs.SquaredSNRMovingAverage[it] >= FT['ThresholdGD']**2)
        
        outputs.TrackedBaselines[it] = reliablebaselines
        
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
    
    outputs.Igd[it,:,:] = currIgd
    outputs.Ipd[it,:,:] = currIpd
        
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
    
    timerange = range(it+1-Ncp,it+1) ; validcp=np.zeros(NC); ic=0
    for ia in range(NA):
        for iap in range(ia+1,NA):
            for iapp in range(iap+1,NA):
                
                ib = posk(ia,iap,NA); ib1=ich_pos[ib]      # coherent flux (ia,iap)  
                measured1=(ib1>=0) and outputs.TrackedBaselines[it,ib1]
                ib = posk(iap,iapp,NA); ib2=ich_pos[ib] # coherent flux (iap,iapp)    
                measured2=(ib2>=0) and outputs.TrackedBaselines[it,ib2]
                ib = posk(ia,iapp,NA); ib3=ich_pos[ib] # coherent flux (iapp,ia)    
                measured3=(ib3>=0)
                
                if measured1*measured2*measured3:
                    
                    valid1=outputs.TrackedBaselines[it,ib1]
                    valid2=outputs.TrackedBaselines[it,ib2]
                    valid3=outputs.TrackedBaselines[it,ib3]
                    
                    if valid1*valid2*valid3:
                        cs1 = np.sum(outputs.CfPD[timerange,:,ib1], axis=1)     # Sum of coherent flux (ia,iap)
                        cfGDlmbdas = outputs.CfGD[timerange,Ncross:,ib1]*np.conjugate(outputs.CfGD[timerange,:-Ncross,ib1])
                        cfGDmoy1 = np.sum(cfGDlmbdas,axis=1)     # Sum of coherent flux (ia,iap)  
                        
                        cs2 = np.sum(outputs.CfPD[timerange,:,ib2], axis=1) # Sum of coherent flux (iap,iapp)    
                        cfGDlmbdas = outputs.CfGD[timerange,Ncross:,ib2]*np.conjugate(outputs.CfGD[timerange,:-Ncross,ib2])
                        cfGDmoy2 = np.sum(cfGDlmbdas,axis=1)
                        
                        cs3 = np.sum(np.conjugate(outputs.CfPD[timerange,:,ib3]),axis=1) # Sum of 
                        cfGDlmbdas = outputs.CfGD[timerange,Ncross:,ib3]*np.conjugate(outputs.CfGD[timerange,:-Ncross,ib3])
                        cfGDmoy3 = np.sum(cfGDlmbdas,axis=1)
                        
                        validcp[ic] = 1
                    else:
                        validcp[ic]=0
                else:
                    validcp[ic]=0
                    
                    
                # The bispectrum of one time and one triangle adds up to
                # the Ncp last times
                # ic = poskfai(ia,iap,iapp,NA)        # 0<=ic<NC=(NA-2)(NA-1) 
                
                if validcp[ic]:
                    bispectrumPD[ic]=np.sum(cs1*cs2*cs3)
                    bispectrumGD[ic]=np.sum(cfGDmoy1*cfGDmoy2*np.conjugate(cfGDmoy3))
                else:
                    bispectrumPD[ic]=0
                    bispectrumGD[ic]=0
                ic+=1
    
                
    outputs.BispectrumPD[it] = bispectrumPD*validcp+outputs.BispectrumPD[it-1]*(1-validcp)
    outputs.BispectrumGD[it] = bispectrumGD*validcp+outputs.BispectrumGD[it-1]*(1-validcp)
    
    cpPD = np.angle(outputs.BispectrumPD[it])
    cpGD = np.angle(outputs.BispectrumGD[it])
    
    cpPD[cpPD<-np.pi+config.FT['stdCP']]=np.pi
    cpGD[cpGD<-np.pi+config.FT['stdCP']]=np.pi
    
    outputs.ClosurePhasePD[it] = cpPD
    outputs.ClosurePhaseGD[it] = cpPD/config.FS['R']
    
    BestTel=config.FT['BestTel'] ; itelbest=BestTel-1
    if config.FT['CPref'] and (it>10):                     # At time 0, we create the reference vectors
        for ia in range(NA-1):
            for iap in range(ia+1,NA):
                if not(ia==itelbest or iap==itelbest):
                    ib = posk(ia,iap,NA) ; ibmes = ich_pos[ib]
                    if ibmes >= 0:
                        if itelbest>iap:
                            ic = poskfai(ia,iap,itelbest,NA)   # Position of the triangle (0,ia,iap)
                        elif itelbest>ia:
                            ic = poskfai(ia,itelbest,iap,NA)   # Position of the triangle (0,ia,iap)
                        else:
                            ic = poskfai(itelbest,ia,iap,NA)
                    
                        outputs.PDref[it,ibmes] = outputs.ClosurePhasePD[it,ic]
                        outputs.GDref[it,ibmes] = outputs.ClosurePhaseGD[it,ic]   
        
                        outputs.CfPDref[it,ibmes] = outputs.BispectrumPD[it,ic]#/np.abs(outputs.BispectrumPD[it,ic])
                        outputs.CfGDref[it,ibmes] = outputs.BispectrumGD[it,ic]#/np.abs(outputs.BispectrumGD[it,ic])
    
    """
    GD and PD errors calculation
    """
        
    # Current Phase-Delay
    currPD = np.angle(np.sum(outputs.CfPD[it,:,:], axis=0)*np.exp(-1j*outputs.PDref[it]))
    # currPD = np.angle(np.sum(outputs.CfPD[it,:,:], axis=0)*np.conj(outputs.CfPDref[it]))*ich_pos
    
    # Current Group-Delay
    currGD = np.zeros(NINmes)
    for ib in range(NINmes):
        cfGDlmbdas = outputs.CfGD[it,:-Ncross,ib]*np.conjugate(outputs.CfGD[it,Ncross:,ib])
        cfGDmoy = np.sum(cfGDlmbdas)
        
        currGD[ib] = np.angle(cfGDmoy*np.exp(-1j*outputs.GDref[it,ib]))
        # currGD[ib] = np.angle(cfGDmoy*np.conj(outputs.CfGDref[it,ib])*np.conj(outputs.CfPDref[it,ib]**(1/config.FS['R'])))*ich_pos[ib]

    outputs.PDEstimated[it] = currPD
    outputs.GDEstimated[it] = currGD
    
    # Patch to stabilize the PD and GD when too close to the Pi/-Pi shift.
    # --> force it to Pi.
    
    currPD[(currPD+np.pi)<config.FT['stdPD']]=np.pi
    currGD[(currGD+np.pi)<config.FT['stdGD']]=np.pi
    
    outputs.PDEstimated2[it] = currPD
    outputs.GDEstimated2[it] = currGD
    
    
    
    # """
    # SEARCH state
    # """
    
    # if outputs.SearchState[it]:      
        
    #     Nsearch = FT['Nsearch']
    #     timerange = range(it+1-Nsearch, it+1) 
            
    #     if FT['searchSNR'] == 'pd':
    #         varSignal = outputs.varPD[timerange]
    #     elif FT['searchSNR'] == 'gd':
    #         varSignal = outputs.varGD[timerange]
    #     elif FT['searchSNR'] == 'pdtemp':
    #         varSignal = outputs.TemporalVariancePD[timerange]
    #     elif FT['searchSNR'] == 'gdtemp':
    #         varSignal = outputs.TemporalVarianceGD[timerange]
    #     else:
    #         raise ValueError("Parameter 'searchSNR' must be one of the following: [pd,gd,pdtemp,gdtemp]")
            
    #     outputs.SearchSNR[it] = np.sqrt(np.nan_to_num(1/np.mean(varSignal,axis=0)))
        
    #     # Current Group-Delay
    #     NINmes = FS['NINmes'] ; Ncross = FT['Ncross']
    #     currGD = np.zeros(NINmes)
    #     for ib in range(NINmes):
    #         cfGDlmbdas = outputs.CfGD[it,:-Ncross,ib]*np.conjugate(outputs.CfGD[it,Ncross:,ib])
    #         cfGDmoy = np.sum(cfGDlmbdas)
            
    #         currGD[ib] = np.angle(cfGDmoy*np.exp(-1j*outputs.GDref[it,ib]))
            
    #     outputs.GDEstimated[it] = currGD

    #     Ws = outputs.SearchSNR[it] * ((outputs.SearchSNR[it]-outputs.SearchSNR[it-1]<0)\
    #                                   and (np.abs(currGD) < searchThreshGD*wlOfTrack))
        
    #     zgd = [(0,0)]*NIN
    #     for ia in range(NA):
    #         for iap in range(ia+1,NA):
    #             ib = ct.posk(ia,iap,NA)
    #             if Ws[ib] != 0:
    #                 zgd[ib] = (outputs.EffectiveMoveODL[it,ia],outputs.EffectiveMoveODL[it,iap])
        
    #     Is = np.dot(np.transpose(config.FS['OPD2Piston']),np.dot(np.diag(Ws),config.FS['OPD2piston']))
        
    #     allTelFound = (np.linalg.matrix_rank(Is) < NA-1)
        
    #     if allTelFound:
    #         outputs.SearchState[it+1] = 0
            
            
            
    
    
    
    
    """
    RELOCK state
    """
    
    """ Implementation comme Sylvain:
            - sans réinitialisation
            - avec patch pour garder groupes cophasés lors des sauts
            """
    
    
    IgdRank = np.linalg.matrix_rank(outputs.Igd[it])
    NotCophased = (IgdRank < NA-1)
    outputs.IgdRank[it] = IgdRank
    
    
    if NotCophased and config.FT['relock']:
        outputs.time_since_loss[it]=outputs.time_since_loss[it-1]+config.dt
        
        # FringeLost = (NotCophased and (IgdRank<np.linalg.matrix_rank(outputs.Igd[it-1]))
        # This situation could pose a problem but we don't manage it yet        
        if (outputs.time_since_loss[it] > config.FT['SMdelay']):
            
            Igdna = np.dot(config.FS['OPD2Piston_moy_r'],
                           np.dot(outputs.Igd[it],config.FS['Piston2OPD_r']))
            
            Kernel = np.identity(NA) - Igdna
            
            CophasedBaselines=np.where(np.diag(outputs.Igd[it])!=0)[0]
            CophasedPairs=[]
            for ib in CophasedBaselines:
                ia,iap = config.FS['ich'][ib][0], config.FS['ich'][ib][1]
                CophasedPairs.append([ia,iap])
                
            CophasedGroups = JoinOnCommonElements(CophasedPairs)
            
            # Fringe loss
            outputs.LostTelescopes[it] = (np.diag(Igdna) == 0)*1      # The positions of the lost telescopes get 1.
            # WeLostANewTelescope = (sum(newLostTelescopes) > 0)
            
            # Photometry loss
            outputs.noSignal_on_T[it] = 1*(outputs.SNRPhotometry[it] < config.FT['ThresholdPhot'])
                
            comparison = (outputs.noSignal_on_T[it] == outputs.LostTelescopes[it])
            outputs.LossDueToInjection[it] = (comparison.all() and sum(outputs.noSignal_on_T[it])>1)       # Evaluates if the two arrays are the same
            
            if not outputs.LossDueToInjection[it]:     # The fringe loss is not due to an injection loss
                config.FT['state'][it] = 1
                
                newLostTelescopes = (outputs.LostTelescopes[it] - outputs.LostTelescopes[it-1] == 1)
                TelescopesThatGotBackPhotometry = (outputs.noSignal_on_T[it-1] - outputs.noSignal_on_T[it] == 1)
                # WeGotBackPhotometry = (sum(TelescopesThatGotBackPhotometry) > 0)
                
                TelescopesThatNeedARestart = np.argwhere(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)

                if config.FT['state'][it-1] != 1:
                    config.FT['eps'] = np.ones(NA)
                    config.FT['it0'] = np.ones(NA)*it
                    config.FT['it_last'] = np.ones(NA)*it
                
                Velocities = np.dot(Kernel,config.FT['relock_vfactors'])
                Increment = relockfunction_inc_sylvain_gestioncophased(it, Velocities, config.FT['covering'], CophasedGroups)
            
                #You should send command only on telescope with flux
                outputs.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(outputs.noSignal_on_T[it])
                Increment = np.dot(outputs.NoPhotometryFiltration[it],Increment)
                
                uRelock = outputs.CommandRelock[it] + Increment
                
            else:
                    Increment = np.zeros(NA)
            
        else:
            Increment = np.zeros(NA)
            
    else:
        outputs.time_since_loss[it] = 0
        Increment = np.zeros(NA)
            
    Increment = Increment
    
    outputs.CommandRelock[it+1] = outputs.CommandRelock[it] + Increment
    
    # The command is sent at the next time, that's why we note it+1
    uRelock = outputs.CommandRelock[it+1]
    
    
        
    """
    Group-Delay tracking
    """
    
    if config.FT['CPref']:
        currGDerr = currGD - outputs.GDref[it]
    else:
        currGDerr = currGD
        
    # Keep the GD between [-Pi, Pi]
    # Eq. 35
    # Array elements verifying the condition
    higher_than_pi = (currGDerr > np.pi)
    lower_than_mpi = (currGDerr < -np.pi)
    
    currGDerr[higher_than_pi] -= 2*np.pi
    currGDerr[lower_than_mpi] += 2*np.pi
    
    # Store residual GD for display only [radians]
    outputs.GDResidual[it] = currGDerr
    
    # Weights the GD (Eq.35)
    currGDerr = np.dot(currIgd,currGDerr)
     
    outputs.GDResidual2[it] = currGDerr
    outputs.GDPistonResidual[it] = np.dot(FS['OPD2Piston_r'], currGDerr*R*config.wlOfTrack/(2*np.pi))
    
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
    
    outputs.GDErr[it] = currGDerr
    
    # Integrator (Eq.37)
    if FT['cmdOPD']:     # integrator on OPD
        # Integrator
        outputs.GDCommand[it+1] = outputs.GDCommand[it] + FT['GainGD']*currGDerr*config.wlOfTrack*R/2/np.pi
        
        # From OPD to Pistons
        uGD = np.dot(FS['OPD2Piston_r'], outputs.GDCommand[it+1])
        
    else:                       # integrator on Pistons
        # From OPD to Piston
        currPistonGD = np.dot(FS['OPD2Piston_r'], currGDerr)*config.wlOfTrack*R/2/np.pi
        # Integrator
        uGD = outputs.PistonGDCommand[it+1] + FT['GainGD']*currPistonGD
        
    # outputs.GDCommand[it+1] = outputs.GDCommand[it] + FT['GainGD']*currGDerr
    
    # From OPD to Piston
    # uGD = np.dot(FS['OPD2Piston'], outputs.GDCommand[it+1])
    
    outputs.PistonGDCommand_beforeround[it+1] = uGD
    
    if config.FT['roundGD']=='round':
        jumps = np.round(uGD/config.wlOfTrack)
        uGD = jumps*config.wlOfTrack
    elif config.FT['roundGD']=='int':
        for ia in range(NA):
            jumps = int(uGD[ia]/config.wlOfTrack)
            uGD[ia] = jumps*config.wlOfTrack
    elif config.FT['roundGD']=='no':
        pass
    else:
        raise ValueError("The roundGD parameter of the fringe-tracker must be 'round', 'int' or 'no'.")
        
    if config.TELref:
        iTel = config.TELref-1
        uGD = uGD - uGD[iTel]
        
        
    outputs.PistonGDCommand[it+1] = uGD

    """
    Phase-Delay command
    """
    
    if config.FT['CPref']:
        currPDerr = currPD - outputs.PDref[it]
        
    else:
        currPDerr = currPD
    # Keep the PD between [-Pi, Pi]
    # Eq. 35
    
    # Array elements verifying the condition
    higher_than_pi = (currPDerr > np.pi)
    lower_than_mpi = (currPDerr < -np.pi)
    
    currPDerr[higher_than_pi] -= 2*np.pi
    currPDerr[lower_than_mpi] += 2*np.pi
    
    outputs.PDResidual[it] = currPDerr
    
    # Weights the PD (Eq.35)
    currPDerr = np.dot(currIpd,currPDerr)
    
    # Store residual PD and piston for display only
    outputs.PDResidual2[it] = currPDerr
    outputs.PDPistonResidual[it] = np.dot(FS['OPD2Piston_r'], currPDerr*config.wlOfTrack/(2*np.pi))
    
    # Integrator (Eq.37)
            
    if FT['cmdOPD']:     # integrator on OPD
        # Integrator
        outputs.PDCommand[it+1] = outputs.PDCommand[it] + FT['GainPD']*currPDerr*config.wlOfTrack/2/np.pi
        # From OPD to Pistons
        uPD = np.dot(FS['OPD2Piston_r'], outputs.PDCommand[it+1])
        
    else:                       # integrator on Pistons
        # From OPD to Piston
        currPistonPD = np.dot(FS['OPD2Piston_r'], currPDerr)*config.wlOfTrack/2/np.pi
        # Integrator
        uPD = outputs.PistonPDCommand[it] + FT['GainPD']*currPistonPD
    
    
    if config.TELref:
        iTel = config.TELref-1
        uPD = uPD - uPD[iTel]
    
    outputs.PistonPDCommand[it+1] = uPD
    
    # if config.mode == 'track':
    #     if np.linalg.matrix_rank(currIgd) < NA-1:
    #         trelock_ = it
    #         config.mode == 'relock'
            
    # elif config.mode == 'relock':
    #     if np.linalg.matrix_rank(currIgd) == NA-1:
    #         config.mode == 'track'
    #     else:
    #         usaw = relockfunction(NA,Sweep_,Slope_,it-trelock_)
    #         uRelock = Vfactors_*usaw
        
    
    """
    MODULATION command
    """


    
    """
    ODL command
    It is the addition of the GD, PD, SEARCH and modulation functions
    """
    
    CommandODL = uPD + uGD + uRelock + uSearch
    
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
    outputs.CfPD : 
        
    outputs.varPD :
        

    """
    
    from . import outputs
    
    from .config import NA, NIN,MW
    
    from .outputs import it
    # from .coh_tools import simu2GRAV, NB2NIN
    
    NINmes = config.FS['NINmes']
    
    image = outputs.MacroImages[it]
    
    M = config.M            # Amplification ratio camera
    
    sigmap = config.FS['sigmap']  # Background noise
    imsky = config.FS['imsky']    # Sky image before observation
    # Demod = config.FS['MacroP2VM_r']    # Macro P2VM matrix used for demodulation
    # ElementsNormDemod = config.FS['ElementsNormDemod']
    
    DemodGRAV = config.FS['MacroP2VM_r']
    # Flux variance calculation (eq. 12)
    varFlux = sigmap**2 + M*(image - imsky)
    outputs.varFlux[it] = varFlux
    
    """
    Covariance calculation (eq. 13)
    """
    
    for imw in range(MW):
        # outputs.CovarianceReal[it,imw] = np.sum(np.real(Demod[imw])**2*varFlux[imw], axis=1)
        # outputs.CovarianceImag[it,imw] = np.sum(np.imag(Demod[imw])**2*varFlux[imw], axis=1)

        outputs.Covariance[it,imw] = np.dot(DemodGRAV[imw], np.dot(np.diag(varFlux[imw]),np.transpose(DemodGRAV[imw])))
        #outputs.BiasModCf[it,imw] = np.dot(ElementsNormDemod[imw],varFlux[imw])
        
    outputs.DemodGRAV = DemodGRAV
    
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
    diagCovar = np.diagonal(outputs.Covariance, axis1=2, axis2=3)
    varPhot = diagCovar[it,:,:NA]
    timerange = range(it+1-Nvar,it+1)
    varX = np.abs(diagCovar[timerange,:,NA:NA+NINmes])
    varY = np.abs(diagCovar[timerange,:,NA+NINmes:])
    varNum = np.mean(varX+varY,axis=0)
    # for ia in range(NA):
    #     ibp = ia*(NA+1)
    #     varPhot[:,ia] = outputs.Covariance[it,:,ibp,ibp]       # Variance of photometry at each frame
    #     for iap in range(ia+1,NA):
    #         ib = posk(ia,iap,NA)
    #         ibr=NA+ib; varX = outputs.Covariance[timerange,:,ibr,ibr]
    #         ibi=NA+NINmes+ib; varY = outputs.Covariance[timerange,:,ibi,ibi]
    #         covarXY = outputs.Covariance[timerange,:,ibr,ibi]
            
    #         varNum[:,ib] = np.mean(varX+varY,axis=0)
    #         varNum2[:,ib] = np.mean(varX+varY + 2*covarXY, axis=0)
            
    CohFlux = np.mean(outputs.CfPD[timerange], axis=0)
    CfSumOverLmbda = np.sum(CohFlux,axis=0)
    
    outputs.varGDdenom[it] = np.sum(np.real(CohFlux*np.conj(CohFlux)),axis=0)  # Sum over lambdas of |CohFlux|² (modified eq.14)
    outputs.varGDdenomUnbiased[it] = np.sum(np.real(CohFlux*np.conj(CohFlux))-outputs.BiasModCf[it],axis=0)  # Sum over lambdas of |CohFlux|²
    outputs.varPDdenom[it] = np.real(CfSumOverLmbda*np.conj(CfSumOverLmbda))#-np.mean(outputs.BiasModCf[it],axis=0)) # Original eq.14    
    #outputs.varPDdenom2[it] = np.sum(np.mean(np.abs(outputs.CfPD[timerange])**2,axis=0),axis=0)
    outputs.varPDnum[it] = np.sum(varNum,axis=0)/2     # Sum over lmbdas of Variance of |CohFlux|
    
    outputs.varGDUnbiased[it] = outputs.varPDnum[it]/outputs.varGDdenomUnbiased[it]      # Var(|CohFlux|)/|CohFlux|²
    outputs.varPD[it] = outputs.varPDnum[it]/outputs.varPDdenom[it]      # Var(|CohFlux|)/|CohFlux|²
    outputs.varGD[it] = outputs.varPDnum[it]/outputs.varGDdenom[it]
    
    outputs.SNRPhotometry[it,:] = np.sum(outputs.PhotometryEstimated[it,:],axis=0)/np.sqrt(np.sum(varPhot,axis=0))
    
    varPD = outputs.varPD[it]
    varGD = outputs.varGD[it]
    
    return varPD, varGD


def SetThreshold(TypeDisturbance="CophasedThenForeground",nbSigma=0,
                 manual=False, scan=False,display=False,
                 verbose=True,scanned_tel=6):
    """
    Estimate the threshold GD.
    Different way of doing this, depending on parameters:
        - nbSigma: if a non-null float is given, LAR are sent to foreground \
and ThresholdGD = mean(SNR) + nBsigma * rms(SNR)
        - TypeDisturbance:
            - if CophaseThenForeground: compare SNR in presence and absence \
of fringes and ThresholdGD = SNR_foreground + 0.2 * (SNR_cophased - SNR_foreground)

        - manual: if True, scans the coherence envelop of the FS and displays the estimated SNR².
Then asks for the user to choose a smart threshold.

    Parameters
    ----------
    TypeDisturbance : TYPE, optional
        DESCRIPTION. The default is "CophasedThenForeground".
    nbSigma : TYPE, optional
        DESCRIPTION. The default is 0.
    manual : TYPE, optional
        DESCRIPTION. The default is False.
    scan : TYPE, optional
        DESCRIPTION. The default is False.
    display : TYPE, optional
        DESCRIPTION. The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is True.
    scanned_tel : TYPE, optional
        DESCRIPTION. The default is 6.

    Returns
    -------
    Modify values of ThresholdGD and ThresholdPD.

    """

    from cophasim import skeleton as sk
    from cophasim import config
    from cophasim import outputs

    NINmes = config.FS['NINmes']

    if nbSigma:
        
        NT=200
            
        InitNT = sk.config.NT
        
        foreground = 5*config.FS['R']*config.wlOfTrack*np.arange(config.NA)
        
        sk.update_config(foreground=foreground, NT = NT, verbose=verbose)
        
        # Initialize the fringe tracker with the gain
        from cophasim.SPICA_FT import SPICAFT, updateFTparams
        #SPICAFT(init=True, GainPD=0, GainGD=0,relock=False)
        gainPD,gainGD,relock,state=config.FT['GainPD'],config.FT['GainGD'],config.FT['relock'],config.FT['state']
        updateFTparams(GainPD=0, GainGD=0, relock=False,search=False, verbose=verbose)
        
        config.FT['state'] = np.zeros(NT+1)
        # Launch the scan
        sk.loop(verbose=verbose)
        
        StartComputing = 50
        newThresholdGD = np.ones(NINmes)

        for ib in range(NINmes):
            SNRfg = np.mean(np.sqrt(outputs.SquaredSNRMovingAverage[StartComputing:,ib]))
            fgstd = np.std(np.sqrt(outputs.SquaredSNRMovingAverage[StartComputing:,ib]))
            
            # Set threshold to mean(SNR) + nBsigma * rms(SNR)
            newThresholdGD[ib] = SNRfg + nbSigma*fgstd
            
            if newThresholdGD[ib] ==0:
                newThresholdGD[ib] = 10
                
        
        newThresholdPD = 1e-3#np.min(newThresholdGD)/2
        
        config.FT['ThresholdGD'] = newThresholdGD
        config.FT['ThresholdPD'] = newThresholdPD
        config.FT['state'] = state

        sk.display('opd','snr','detector',wlOfTrack=1.6, pause=True,display=display)
        
        sk.update_config(NT=InitNT,foreground=[],verbose=verbose)
        updateFTparams(GainPD=gainPD, GainGD=gainGD, relock=relock,
                       ThresholdGD=newThresholdGD,ThresholdPD=newThresholdPD,
                       verbose=verbose)


    elif scan:

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
        # from cophasim.SPICA_FT_r import SPICAFT, updateFTparams
        #SPICAFT(init=True, GainPD=0, GainGD=0,relock=False)
        gainPD,gainGD,relock=config.FT['GainPD'],config.FT['GainGD'],config.FT['relock']
        updateFTparams(GainPD=0, GainGD=0, relock=False, verbose=verbose)
        
        # Launch the scan
        sk.loop(verbose)
        
        if manual:
            sk.display('snr',wlOfTrack=1.6, pause=True)
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
            from cophasim import outputs,coh_tools
            
            scanned_baselines = [coh_tools.posk(ia,scanned_tel-1,config.NA) for ia in range(config.NA-1)]
            k=0;ib=scanned_baselines[k]
            while not (config.FS['ich_pos'][ib]>=0):
                k+=1
                ib = scanned_baselines[k]
                
            Lc = R*config.wlOfTrack
            
            ind=np.argmin(np.abs(outputs.OPDTrue[:,4]+Lc*0.7))
            
            newThresholdGD = np.array([np.max([2,x]) for x in np.sqrt(outputs.SquaredSNRMovingAverage[ind,:])])
                    
            config.FT['ThresholdGD'] = newThresholdGD
            
            sk.display('snr',wlOfTrack=1.6, pause=True, display=display)
            
        sk.update_config(DisturbanceFile=InitialDisturbanceFile, NT=InitNT,
                         verbose=verbose)
        updateFTparams(GainPD=gainPD, GainGD=gainGD, relock=relock, 
                       ThresholdGD=newThresholdGD,
                       verbose=verbose)
    
    
    else:
        
        DisturbanceFile = TypeDisturbance
        
        NT=200
            
        InitialDisturbanceFile,InitNT = sk.config.DisturbanceFile, sk.config.NT
        
        sk.update_config(DisturbanceFile=DisturbanceFile, NT = NT, verbose=verbose)
        
        # Initialize the fringe tracker with the gain
        from cophasim.SPICA_FT import SPICAFT, updateFTparams
        #SPICAFT(init=True, GainPD=0, GainGD=0,relock=False)
        gainPD,gainGD,relock,state=config.FT['GainPD'],config.FT['GainGD'],config.FT['relock'],config.FT['state']
        updateFTparams(GainPD=0, GainGD=0, relock=False,search=False, verbose=verbose)
        
        config.FT['state'] = np.zeros(NT+1)
        # Launch the scan
        sk.loop(verbose=verbose)
        
        if manual:
            sk.display('snr','detector',wlOfTrack=1.6, pause=True)
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
            from cophasim import outputs,coh_tools
            
            if TypeDisturbance=='NoDisturbance':
                ind=100
                newThresholdGD = np.array([np.max([2,x*0.2]) for x in np.sqrt(outputs.SquaredSNRMovingAverage[ind,:])])
 
            elif TypeDisturbance == 'CophasedThenForeground':
                CophasedInd = 50 ; ForegroundInd = 180
                CophasedRange = range(50,100)
                ForegroundRange = range(160,200)
                newThresholdGD = np.ones(NINmes)

                for ib in range(NINmes):
                    SNRcophased = np.mean(np.sqrt(outputs.SquaredSNRMovingAverage[CophasedRange,ib]))
                    SNRfg = np.mean(np.sqrt(outputs.SquaredSNRMovingAverage[ForegroundRange,ib]))
                    fgstd = np.std(np.sqrt(outputs.SquaredSNRMovingAverage[ForegroundRange,ib]))
                    cophasedstd = np.std(np.sqrt(outputs.SquaredSNRMovingAverage[CophasedRange,ib]))
                    
                    # Set threshold to a value between max and foreground with a lower limit defined by the std of foreground.
                    newThresholdGD[ib] = np.max([1.5,SNRfg + 5*fgstd,SNRfg+0.2*(SNRcophased-SNRfg)])
                    
                    if newThresholdGD[ib] ==0:
                        newThresholdGD[ib] = 10
                        
            newThresholdPD = 1e-3#np.min(newThresholdGD)/2
            
            config.FT['ThresholdGD'] = newThresholdGD
            config.FT['ThresholdPD'] = newThresholdPD
            config.FT['state'] = state

            sk.display('opd','snr','detector',wlOfTrack=1.6, pause=True,display=display)
            
        sk.update_config(DisturbanceFile=InitialDisturbanceFile, NT=InitNT,
                         verbose=verbose)
        updateFTparams(GainPD=gainPD, GainGD=gainGD, relock=relock,
                       ThresholdGD=newThresholdGD,ThresholdPD=newThresholdPD,
                       verbose=verbose)
        

    return newThresholdGD


def relockfunction(usaw):
    """
    Calculates a relock function for NA telescopes using the last relock command.

    Parameters
    ----------
    usaw : TYPE
        DESCRIPTION.

    Returns
    -------
    usaw : TYPE
        DESCRIPTION.

    """
    
    from . import outputs
    from .config import NA,dt
    from .outputs import it
    
    for ia in range(NA):
        it0 = config.FT['it0'][ia] ; it_last = config.FT['it_last'][ia]
        
        a = config.FT['sweep30s']/30
        sweep = config.FT['sweep0'] + a*(it-it0)*config.dt
        
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
        
    
    outputs.eps[it] = config.FT['eps']
    outputs.it_last[it] = config.FT['it_last']
    outputs.LastPosition[it] = config.FT['LastPosition']
    
    return usaw
  


def relockfunction_basical(usaw,it):
    
    a = config.FT['sweep30s']/30000
    sweep = config.FT['sweep0'] + a*(it-config.FT['it0'])*config.dt
    
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

def relockfunction3(usaw,it):
    
    a = config.FT['sweep30s']/30000
    sweep = config.FT['sweep0'] + a*(it-config.FT['it0'])*config.dt
    
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

def relockfunction_inc_basical(it):
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
    a = config.FT['sweep30s']/30000
    
    # Temps avant saut de frange
    sweep = config.FT['sweep0'] + a*(it-it0)*config.dt
    
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



def relockfunction_inc_sylvain(it, v):
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
        a = config.FT['sweep30s']/30000
        
        # Temps avant saut de frange
        sweep = config.FT['sweep0'] + a*(it-it0)*config.dt
        
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


def relockfunction_inc_sylvain_gestioncophased(it, v, covering, CophasedGroups):
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
        nbChanges = config.FT['nbChanges'][ia]
        
        # Coefficient directeur de la fonction d'augmentation du temps avant saut.
        a = config.FT['sweep30s']/30000
        
        # Temps avant saut de frange
        # sweep = config.FT['sweep0'] + a*(it-it0)*config.dt
        sweep = config.FT['sweep0']*config.FT['commonRatio']**nbChanges
        
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
            config.FT['nbChanges'][ia] += 1
            move[ia] = -config.FT['LastPosition'][ia]
            
            # Add some covering to prevent from missing fringes
            if np.abs(move[ia]) < 2*covering:
                coveringtemp = 0
            else:
                coveringtemp = covering
            if move[ia] > 0:
                move[ia] -= coveringtemp
            if move[ia] < 0:
                move[ia] += coveringtemp
                
            coveringtemp = covering
            
            # Cophased telescopes remain together when jumping
            tel=ia+1
            for CophasedGroup in CophasedGroups:
                l = [move[itel-1] for itel in CophasedGroup]
                generalmove = max(set(l), key = l.count)
                
                if tel in CophasedGroup:
                    move[ia] = generalmove            
            
            config.FT['LastPosition'][ia] = move[ia]
    
    return move


def relockfunction_incind(it):
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
        
        a = config.FT['sweep30s']/30000  # Coefficient directeur de la fonction d'augmentation du temps avant saut.
        sweep = config.FT['sweep0'] + a*(it-it0)*config.dt
        
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