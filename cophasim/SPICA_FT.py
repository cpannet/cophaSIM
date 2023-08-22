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
from scipy import signal

from .coh_tools import posk, poskfai

from . import config

def SPICAFT(*args, init=False, search=False, update=False, GainPD=0, GainGD=0, 
            Ngd=40, Nsnr=40, roundGD='round', Ncross=1,
            relock=True,SMdelay=1e3, sweep30s=10, maxVelocity=0.100, searchMinGD=500,
            sweepRelock=20,commonRatioRelock=1.2, coveringRelock=10,vfactorsRelock = [],
            sweepSearch=4000,commonRatioSearch=1, coveringSearch=40,vfactorsSearch=[],
            searchThreshGD=3,Nsearch=50,searchSNR='gd',searchSnrThreshold=2,
            CPref=True, BestTel=2, Ncp = 300, Nvar = 5, stdPD=0.07,stdGD=0.14,stdCP=0.07,
            cmdOPD=True, switch=1, continu=True,whichSNR='gd',
            ThresholdGD=2, ThresholdPD = 1.5, ThresholdPhot = 0.1,Nflux=100,ThresholdRELOCK=2,ratioThreshold=0.2,
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
        config.FT['Nsnr'] = Nsnr
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
        config.FT['ratioThreshold'] = ratioThreshold
        
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
        config.FT['sweep30s'] = sweep30s        # Sweep at 30s in seconds
        config.FT['maxVelocity'] = maxVelocity        # Sweep at 30s in seconds
        config.FT['sweepRelock'] = sweepRelock            # Starting sweep in seconds
        config.FT['commonRatioRelock'] = commonRatioRelock  # Common ratio of the geometrical sequence
        config.FT['coveringRelock'] = coveringRelock        # Covering of the sawtooth function in microns
        config.FT['sweepSearch'] = sweepSearch            # Starting sweep in seconds
        config.FT['commonRatioSearch'] = commonRatioSearch  # Common ratio of the geometrical sequence
        config.FT['coveringSearch'] = coveringSearch        # Covering of the sawtooth function in microns        config.FT['maxVelocity'] = maxVelocity  # Maximal slope given in µm/frame
        config.FT['searchThreshGD'] = searchThreshGD*config.wlOfTrack    # Maximal value of GD for considering fringe found.
        config.FT['Nsearch'] = Nsearch          
        config.FT['searchSNR'] = searchSNR      
        config.FT['searchMinGD'] = np.ones(NINmes)*searchMinGD  # Value of minimal reached value of GD during search
        config.FT['diffOffsets_best'] = np.zeros(NINmes)        # Vector that will contain the offsets of the fringes.
        config.FT['globalMaximumSnr'] = np.zeros(NINmes)        # Vector that will contain the maximal SNRs.
        config.FT['globalMaximumOffsets'] = np.zeros(NINmes)    # Vector that will contain the offsets of the maximal SNRs
        config.FT['secondMaximumSnr'] = np.zeros(NINmes)    # Vector that will contain the offsets of the maximal SNRs
        config.FT['expectedOffsetsExplored'] = np.ones(NINmes)*False    # Vector that will contain the offsets of the maximal SNRs      
        
        config.FT['searchSnrThreshold'] = searchSnrThreshold            # SNR threshold for SEARCH state
            
        # Version usaw vector
        config.FT['usaw'] = np.zeros([NT,NA])
        config.FT['moveSinceLastChange'] = np.zeros(NA)
        config.FT['it_last'] = np.zeros(NA)
        config.FT['it0'] = np.zeros(NA)
        config.FT['eps'] = np.ones(NA)
        config.FT['nbChanges'] = np.ones(NA)

        # Version usaw float
        # config.FT['usaw'] = np.zeros([NT])
        # config.FT['uRelock'] = np.zeros([NT,NA])
        # config.FT['moveSinceLastChange'] = np.zeros([NT+1,NA])
        # config.FT['it_last'] = 0
        # config.FT['it0'] = 0
        # config.FT['eps'] = 1
        
        config.FT['Nflux'] = Nflux                      # Length of flux average window
        config.FT['ThresholdPhot'] = ThresholdPhot      # Minimal photometry SNR for launching relock

        if (len(vfactorsRelock) == 0) and (len(vfactorsRelock) == NA):
            config.FT['vfactorsRelock'] = np.array(vfactorsRelock)
        else:
            if verbose:
                print("No or bad vfactorsRelock given. I create one.")

            if NA==2:
                config.FT['vfactorsRelock'] = np.array([-0.5,0.5])
                
            if NA==6:
                config.FT['vfactorsRelock'] = np.array([-8.25, -7.25, -4.25, 1.75, 3.75, 8.75])
                
            elif NA==7: # Fake values
                config.FT['vfactorsRelock'] = np.array([-8.25, -7.25, -4.25, 1.75, 3.75, 8.75, 10])
                
            elif NA==10:
                config.FT['vfactorsRelock'] = np.array([-24.9, -23.9, -18.9, -14.9,
                                                  -1.9,   1.1,   9.1,  16.1,
                                                  28.1, 30.1])
        
        if (len(vfactorsSearch) == 0) and (len(vfactorsSearch) == NA):
            config.FT['vfactorsSearch'] = np.array(vfactorsSearch)
        else:
            if verbose:
                print("No or bad vfactorsSearch given. I create one.")
                
            config.FT['vfactorsSearch'] = np.arange(NA)-NA//2+1
            
        if search==True:
            config.FT['Vfactors'] = config.FT['vfactorsSearch']
            config.FT['sweep0'] = config.FT['sweepSearch']
            config.FT['covering'] = config.FT['coveringSearch']
            config.FT['commonRatio'] = config.FT['commonRatioSearch']
        else:
            config.FT['Vfactors'] = config.FT['vfactorsRelock']
            config.FT['sweep0'] = config.FT['sweepRelock']
            config.FT['covering'] = config.FT['coveringRelock']
            config.FT['commonRatio'] = config.FT['commonRatioRelock']
            
        config.FT['vfactorsRelock'] = config.FT['vfactorsRelock']/np.ptp(config.FT['vfactorsRelock'])*maxVelocity     # The maximal OPD velocity is equal to slope/frame
        config.FT['vfactorsSearch'] = config.FT['vfactorsSearch']/np.ptp(config.FT['vfactorsSearch'])*maxVelocity     # The maximal OPD velocity is equal to slope/frame
        
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



def SearchState(coherentGroups=[]):
    
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
    # GradSNR = np.gradient(outputs.SearchSNR[timerange],axis=0)
    
    # Current Group-Delay
    NINmes = FS['NINmes'] ; R = FS['R'] ; Ncross = FT['Ncross']
    currGD = np.zeros(NINmes)
    for ib in range(NINmes):
        cfGDlmbdas = outputs.CfGD[it,:-Ncross,ib]*np.conjugate(outputs.CfGD[it,Ncross:,ib])
        cfGDmoy = np.sum(cfGDlmbdas)
        
        currGD[ib] = np.angle(cfGDmoy*np.exp(-1j*outputs.GDref[it,ib]))
        
    outputs.GDEstimated[it] = currGD
    GDmic = currGD *R*wlOfTrack/(2*np.pi)
    
    # MaxSNRCondition = (np.mean(GradSNR[:Nsearch//2],axis=0)*np.mean(GradSNR[Nsearch//2:],axis=0) < 0) # Sign change in first derivative

    #MaxSNRCondition = (np.mean(outputs.SearchSNR[it-Nsearch:it-Nsearch//2],axis=0)-outputs.SearchSNR[it-1]<0)
    # snrHigherThanThreshold = (outputs.SearchSNR[it] > config.FT['ThresholdGD'])
    SNR = np.sqrt(outputs.SquaredSNRMovingAverage[it])
    
    snrHigherThanThreshold = (SNR > FT['ThresholdGD'])
    lowEnoughGD = (np.abs(GDmic) < searchThreshGD)  # C'est pas ouf car latence dans les mesures
    # lowerGD = (np.abs(GDmic) < FT['searchMinGD'])
    higherSnr = (SNR > FT['searchSnrThreshold'])
    NoRecentChange = (config.FT['it_last'][0] < it-Ngd)
    
    
    # globalMaximumSnr = outputs.SearchSNR[it] * snrHigherThanThreshold * NoRecentChange * lowEnoughGD# * MaxSNRCondition)
    globalMaximumSnr = SNR * snrHigherThanThreshold * NoRecentChange * lowEnoughGD# * GDNullCondition# * GDNullCondition)# * NoRecentChange)# * MaxSNRCondition)
        
    for ib in range(NINmes):
        ia = int(FS['ich'][ib][0])-1
        iap = int(FS['ich'][ib][1])-1
        if globalMaximumSnr[ib] != 0:
            outputs.diffOffsets[it,ib] = outputs.EffectiveMoveODL[it,ia]-outputs.EffectiveMoveODL[it,iap]
    
            if higherSnr[ib]:
                FT['searchMinGD'][ib] = GDmic[ib]
                FT['diffOffsets_best'][ib] = outputs.diffOffsets[it,ib]
                FT['globalMaximumSnr'][ib] = globalMaximumSnr[ib]
        
    outputs.diffOffsets_best[it] = config.FT['diffOffsets_best']
    outputs.globalMaximumSnr[it] = config.FT['globalMaximumSnr']
                
    # Transpose the W matrix in the Piston-space
    Is = np.dot(config.FS['OPD2Piston_r'],np.dot(np.diag(FT['globalMaximumSnr']),config.FS['Piston2OPD_r']))
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
        
        Igd = np.dot(FS['Piston2OPD_r'],np.dot(VSdagUt,np.dot(FS['OPD2Piston_r'], np.diag(FT['globalMaximumSnr']))))
        
        uSearchOpd = np.dot(Igd,config.FT['diffOffsets_best'])
        
        uSearch = np.dot(FS['OPD2Piston'],uSearchOpd)
         
        outputs.CommandSearch[it+1] = uSearch
        # outputs.CommandRelock[it+1] = uSearch         # Patch to propagate the command to the fringe-tracker commands
        config.FT['Vfactors'] = config.FT['vfactorsRelock']    # Set Vfactors to the RELOCK values since it will never come back to SEARCH state.
        
        # Reinitialise parameters for sawtooth function
        config.FT['eps'] = np.ones(NA)
        config.FT['nbChanges'] = np.ones(NA)
        config.FT['sweep0'] = config.FT['sweepRelock']
        config.FT['covering'] = config.FT['coveringRelock']
        config.FT['commonRatio'] = config.FT['commonRatioRelock']
        
    else:
        config.FT['state'][it+1] = 2    # Remain in SEARCH state
        
        # newLostTelescopes = (outputs.LostTelescopes[it] - outputs.LostTelescopes[it-1] == 1)
        # TelescopesThatGotBackPhotometry = (outputs.noSignalOnTel[it-1] - outputs.noSignalOnTel[it] == 1)
        # # WeGotBackPhotometry = (sum(TelescopesThatGotBackPhotometry) > 0)
        
        # TelescopesThatNeedARestart = np.argwhere(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)
        # if len(TelescopesThatNeedARestart):
        #     # print(config.FT['moveSinceLastChange'])
        #     config.FT['moveSinceLastChange'][TelescopesThatNeedARestart] = 0
            
        if it>=1:
            if config.FT['state'][it-1] != 2:
                config.FT['eps'] = np.ones(NA)
                config.FT['it0'] = np.ones(NA)*it
                config.FT['it_last'] = np.ones(NA)*it
                    
        Increment = relockfunction_inc_sylvain_gestioncophased(it, config.FT['vfactorsSearch'], config.FT['covering'], coherentGroups)
    
        #You should send command only on telescope with flux
        #outputs.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(outputs.noSignalOnTel[it])
        #Increment = np.dot(outputs.NoPhotometryFiltration[it],Increment)
        outputs.CommandSearch[it+1] = outputs.CommandSearch[it] + Increment
        
        uSearch = outputs.CommandSearch[it+1]
        
    return uSearch


def SearchState2(coherentGroups=[]):
    # import scipy
    from . import outputs,config
    from . import skeleton as sk
    
    from .config import wlOfTrack, NA, FS, FT
    it=outputs.it 
    
    # searchThreshGD = FT['searchThreshGD']
    
    Nsearch = FT['Nsearch']
    
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
        
    if it>FT['Nsnr']/2:
        outputs.SearchSNR[it] = np.sqrt(np.nan_to_num(1/np.mean(varSignal,axis=0)))
    else:
        outputs.SearchSNR[it] = 0
    
    # Current Group-Delay
    NINmes = FS['NINmes'] ; R = FS['R'] ; Ncross = FT['Ncross']
    currGD = np.zeros(NINmes)
    for ib in range(NINmes):
        cfGDlmbdas = outputs.CfGD[it,:-Ncross,ib]*np.conjugate(outputs.CfGD[it,Ncross:,ib])
        cfGDmoy = np.sum(cfGDlmbdas)
        
        currGD[ib] = np.angle(cfGDmoy*np.exp(-1j*outputs.GDref[it,ib]))
        
    outputs.GDEstimated[it] = currGD
    coherenceLength = R*wlOfTrack
    # GDmic = currGD*coherenceLength/(2*np.pi)
    
    # scanVelocity = FT['vfactorsSearch']/np.ptp(FT['vfactorsSearch'])*FT['maxVelocity']
    # NTForCoherenceLength = coherenceLength/scanVelocity
    # waveletWindow = np.arange(NTForCoherenceLength)
    # snrEvolutionTemp = []
    # offsetsEvolutionTemp = []
        
    for ib in range(NINmes):
        ia = int(FS['ich'][ib][0])-1
        iap = int(FS['ich'][ib][1])-1
        offsetCurrent = outputs.EffectiveMoveODL[it,ia]-outputs.EffectiveMoveODL[it,iap]
                
        snrCurrent = outputs.SearchSNR[it,ib]

        snrEvolution = outputs.snrEvolution[ib].copy()
        offsetsEvolution = outputs.offsetsEvolution[ib].copy()
        
        # if global not already found, update the list of snr and offsets with the new snr and offset
        if not FT['globalMaximumSnr'][ib]:
            snrEvolution, offsetsEvolution = updateSnrEvolution(snrEvolution, offsetsEvolution, 
                                                                snrCurrent, offsetCurrent)
        
            outputs.snrEvolution[ib] = snrEvolution.copy()
            outputs.offsetsEvolution[ib] = offsetsEvolution.copy()
        
            offsetSteps = np.abs(FT['vfactorsSearch'][iap]-FT['vfactorsSearch'][ia])
            distance = 4/3*coherenceLength / offsetSteps
            
            # return 0,0 if no global max not found yet
            globalMaximumSnr, globalMaximumOffset, secondMaximumSnr = getGlobalMaximum(snrEvolution, offsetsEvolution, distance)

            if globalMaximumSnr:
                FT['globalMaximumSnr'][ib] = globalMaximumSnr
                FT['globalMaximumOffsets'][ib] = globalMaximumOffset
                FT['secondMaximumSnr'][ib] = secondMaximumSnr
        
    outputs.globalMaximumOffset[it] = FT['globalMaximumOffsets']
    outputs.globalMaximumSnr[it] = FT['globalMaximumSnr']    
    outputs.secondMaximumSnr[it] = FT['secondMaximumSnr']
    
    # Transpose the globalMaximumSnr matrix in the Piston-space for checking rank
    Is = np.dot(config.FS['OPD2Piston_r'],np.dot(np.diag(FT['globalMaximumSnr']),config.FS['Piston2OPD_r']))
    outputs.Is[it] = Is
    
    rankIs = np.linalg.matrix_rank(Is)
    outputs.rankIs[it] = rankIs
    
    allTelFound = (rankIs == NA-1)
    
    if allTelFound:
        
        # Singular-Value-Decomposition of the W matrix
        U, S, Vt = np.linalg.svd(Is)
        
        Ut = np.transpose(U)
        V = np.transpose(Vt)
        
        # Compute the least square matrix using the globalMaximumSnr
        reliablepistons = (S>1e-4)  #True at the positions of S verifying the condition
        Sdag = np.zeros([NA,NA])
        Sdag[reliablepistons,reliablepistons] = 1/S[reliablepistons]
        
        # Come back to the OPD-space
        VSdagUt = np.dot(V, np.dot(Sdag,Ut))
        
        Igd = np.dot(FS['Piston2OPD_r'],np.dot(VSdagUt,np.dot(FS['OPD2Piston_r'], np.diag(FT['globalMaximumSnr']))))
        
        # Compute expected offsets and check if the expected offsets of all baselines have already been explored
        FT['expectedOffsets'] = np.dot(Igd,FT['globalMaximumOffsets'])
        for ib in range(NINmes):
            FT['expectedOffsetsExplored'][ib] = ((FT['expectedOffsets'][ib] <= outputs.offsetsEvolution[ib]).any()\
                                                        and (FT['expectedOffsets'][ib] >= outputs.offsetsEvolution[ib]).any())
        
        """
        # Version that waits for all telescopes to be found before cophasing
        
        allExpectedOffsetsExplored = FT['expectedOffsetsExplored'].all()
        if allExpectedOffsetsExplored:  # We stop the scan since all offsets have been explored
            keepScanning = False
            
            # Update SNR thresholds
            newThresholdGD = np.ones(NINmes)*FT['ThresholdGD']
            for ib in range(NINmes):
                if FT['globalMaximumSnr'][ib]:
                    newThresholdGD[ib] = FT['secondMaximumSnr'][ib] + FT['ratioThreshold']*(FT['globalMaximumSnr'][ib]-FT['secondMaximumSnr'][ib])
            sk.updateFTparams(ThresholdGD=newThresholdGD, verbose=True)
        
        else:   # Keep scanning until exploring the expected offsets of all baselines
            keepScanning = True
            
        """
        
        keepScanning = False
        
        # Update SNR thresholds
        newThresholdGD = np.ones(NINmes)*FT['ThresholdGD']
        for ib in range(NINmes):
            if FT['globalMaximumSnr'][ib]:
                newThresholdGD[ib] = FT['secondMaximumSnr'][ib] + FT['ratioThreshold']*(FT['globalMaximumSnr'][ib]-FT['secondMaximumSnr'][ib])
        sk.updateFTparams(ThresholdGD=newThresholdGD, verbose=True)            
        
    else:   # Not enough independant baselines found, so keep scanning
        keepScanning = True
        
        
    if keepScanning:
        FT['state'][it+1] = 2    # Remains in SEARCH state
              
        if it>=1:
            if FT['state'][it-1] != 2:
                FT['eps'] = np.ones(NA)
                FT['it0'] = np.ones(NA)*it
                FT['it_last'] = np.ones(NA)*it
                    
        Increment = relockfunction_inc_sylvain_gestioncophased(it, FT['vfactorsSearch'], FT['covering'], coherentGroups)
    
        outputs.CommandSearch[it+1] = outputs.CommandSearch[it] + Increment
        
        uSearch = outputs.CommandSearch[it+1]
        
    else:
        print(FT['globalMaximumOffsets'])
        print(FT['globalMaximumSnr'])
        print(FT['secondMaximumSnr'])
        FT['state'][it+1] = 0    # Go to TRACK state
        
        # Set Vfactors to the RELOCK values since it will never come back to SEARCH state.
        FT['Vfactors'] = FT['vfactorsRelock']   
        
        # Reinitialise parameters for sawtooth function
        FT['eps'] = np.ones(NA)
        FT['nbChanges'] = np.ones(NA)
        
        # Compute the final OPD commands
        uSearchOpd = np.dot(Igd,FT['globalMaximumOffsets'])
        # Convert it into piston commands
        outputs.CommandSearch[it+1] = np.dot(FS['OPD2Piston'],uSearchOpd)
         
        uSearch = outputs.CommandSearch[it+1]
        
    return uSearch


def updateSnrEvolution(snrEvolution, offsetsEvolution, 
                       snrCurrent, offsetCurrent):
    """
    Update snrEvolution and offsetsEvolution according to the values of
    snrCurrent and offsetCurrent.
    
    Parameters
    ----------
    snrEvolution : LIST
        Evolution of the SNR.
    offsetsEvolution : LIST
        Evolution of the offsets.
    snrCurrent : FLOAT
        Current SNR.
    offsetCurrent : FLOAT
        Current offset.

    Returns
    -------
    snrEvolution : LIST
        Evolution of the SNR.
    offsetsEvolution : LIST
        Evolution of the offsets.

    """
    # snrEvolution = list(snrEvolution)
    # offsetsEvolution = list(offsetsEvolution)
    
    if len(offsetsEvolution)==0:
        snrEvolution.append(snrCurrent)
        offsetsEvolution.append(offsetCurrent)
    
    alreadyExploredOffset = (offsetCurrent <= np.array(offsetsEvolution)).any() \
        and (offsetCurrent >= np.array(offsetsEvolution)).any()
    
    if offsetCurrent >=0:
        # if offsetCurrent <= offsetsEvolution[-1]:
        if alreadyExploredOffset:
            sameOffsetIndex = np.argmin(np.abs(np.array(offsetsEvolution)-offsetCurrent))
            if snrCurrent > snrEvolution[sameOffsetIndex]:
                snrEvolution[sameOffsetIndex] = snrCurrent
                offsetsEvolution[sameOffsetIndex] = offsetCurrent

        else:
            snrEvolution.append(snrCurrent)
            offsetsEvolution.append(offsetCurrent)
    
    else:
        if alreadyExploredOffset:
            sameOffsetIndex = np.argmin(np.abs(np.array(offsetsEvolution)-offsetCurrent))
            if snrCurrent > snrEvolution[sameOffsetIndex]:
                snrEvolution[sameOffsetIndex] = snrCurrent
                offsetsEvolution[sameOffsetIndex] = offsetCurrent

        else:
            # Add current snr to the beginning of the list snrEvolution.
            Ltemp = snrEvolution[::-1]
            Ltemp.append(snrCurrent)
            snrEvolution = Ltemp[::-1]
            
            # Add current offsets to the beginning of the list offsetsEvolution.
            Ltemp = offsetsEvolution[::-1]
            Ltemp.append(offsetCurrent)
            offsetsEvolution = Ltemp[::-1]
    
    # offsetsEvolution = np.array(offsetsEvolution)
    # snrEvolution = np.array(snrEvolution)
    
    return snrEvolution, offsetsEvolution



def getGlobalMaximum(snrEvolution, offsetsEvolution, distance):
    """
    Compute the maximal value of the SNR and their asociated offsets.

    Parameters
    ----------
    snrEvolution : LIST or 1-D ARRAY
        Evolution of the SNR.
    offsetsEvolution : LIST or 1D-ARRAY
        Evolution of the SNR.
    distance : INTEGER
        Minimal distance between two local maxima in microns.

    Returns
    -------
    globalMaximumSnr : FLOAT
        Value of the maximal SNR.
    globalMaximumOffset : FLOAT
        Value of the offset corresponding to the maximal SNR.
    secondMaximum

    """

    globalMaximumSnr, globalMaximumOffsets, secondMaximumSnr = 0,0,0
    
    if len(snrEvolution) < 2:
        return globalMaximumSnr, globalMaximumOffsets, secondMaximumSnr
    
    elif len(snrEvolution) < 3*distance:
        return globalMaximumSnr, globalMaximumOffsets, secondMaximumSnr
    
    peaks = signal.find_peaks(snrEvolution, distance=distance)[0]
    
    localMaximaSnr = np.array(snrEvolution)[peaks]
    
    # Set the localMaxima which are under the detection limit (array tooLowSnr)
    # to the same value (the maximum of their values)
    # Necessary for removing noise in the first derivative test.
    tooLowSnrIndex = localMaximaSnr<config.FT['searchSnrThreshold']
    if tooLowSnrIndex.any():    # Check if there is a peak under detection limit.
        localMaximaSnr[tooLowSnrIndex] = np.max(localMaximaSnr[tooLowSnrIndex])
    
    localMaximaOffsets = np.array(offsetsEvolution)[peaks]
    
    # If a maximum is surrounded by two lower maxima, it's the global maximum.
    if (localMaximaSnr[-2] > localMaximaSnr[-3]) and (localMaximaSnr[-2] > localMaximaSnr[-1]):
        globalMaximumSnr = localMaximaSnr[-2]
        globalMaximumOffsets = localMaximaOffsets[-2]
        secondMaximumSnr = np.max([localMaximaSnr[-3],localMaximaSnr[-1]])

    """
    # Compute the first derivative of the localMaxima
    localMaximaSnrDeriv = np.gradient(localMaximaSnr)
    
    # If the derivative changes of sign, it means we reached the global maximum
    if not ((localMaximaSnrDeriv>=0).all() or (localMaximaSnrDeriv<0).all()):
        globalMaximumIndex = np.argmax(localMaximaSnr)
        if globalMaximumIndex == len(localMaximaSnr)-2:
            globalMaximumSnr = localMaximaSnr[globalMaximumIndex]
            globalMaximumOffsets = localMaximaOffsets[globalMaximumIndex]
            
            secondMaximumSnr = np.max([localMaximaSnr[globalMaximumIndex+1],localMaximaSnr[globalMaximumIndex-1]])
    """
    
    if globalMaximumSnr < config.FT['searchSnrThreshold']:
        globalMaximumSnr, globalMaximumOffsets, secondMaximumSnr = 0,0,0
        
    return globalMaximumSnr, globalMaximumOffsets, secondMaximumSnr




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
    
    from .config import NA
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
    
    Nflux = config.FT['Nflux']
    if it < Nflux:
        Nflux=it+1
    timerange = range(it-Nflux,it)
    for ia in range(NA):
        # Test if there were flux in the Nf last frames, before updating the average
        thereIsFlux = not outputs.noFlux[timerange,ia].any()
        if thereIsFlux:
            outputs.PhotometryAverage[it,ia] = np.mean(outputs.PhotometryEstimated[timerange,:,ia])
        else:
            outputs.PhotometryAverage[it,ia] = outputs.PhotometryAverage[it-1,ia]
            
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

    if it < Ngd:
        Ngd = it+1
    
    # Integrates GD with Ngd last frames (from it-Ngd to it)
    timerange = range(it+1-Ngd,it+1)
    for iot in timerange:
        # Subtract average (estimated) phase-delay to current coherent flux
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
            
    Nsnr = FT['Nsnr']
    if it < FT['Nsnr']:
        Nsnr = it+1
    
    Ncross = config.FT['Ncross']  # Distance between wavelengths channels for GD calculation
    R = config.FS['R']        
    
    """
    SEARCH State
    """
    
    if config.FT['state'][it] == 2:
        uSearch = SearchState2()
        CommandODL = uSearch
        # CommandODL = np.zeros(NA)
        return CommandODL
    else:
        outputs.CommandSearch[it+1] = outputs.CommandSearch[it]
        uSearch = outputs.CommandSearch[it+1]
    
    
        
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
        # outputs.CfGDlmbdas[it,:,ib] = cfGDlmbdas
        
        cfGDmoy = np.mean(cfGDlmbdas)*np.exp(-1j*outputs.GDref[it,ib])
        outputs.CfGDMeanOverLmbda[it,ib] = cfGDmoy
        
        currGD[ib] = np.angle(cfGDmoy)
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

    #     globalMaximumSnr = outputs.SearchSNR[it] * ((outputs.SearchSNR[it]-outputs.SearchSNR[it-1]<0)\
    #                                   and (np.abs(currGD) < searchThreshGD*wlOfTrack))
        
    #     zgd = [(0,0)]*NIN
    #     for ia in range(NA):
    #         for iap in range(ia+1,NA):
    #             ib = ct.posk(ia,iap,NA)
    #             if globalMaximumSnr[ib] != 0:
    #                 zgd[ib] = (outputs.EffectiveMoveODL[it,ia],outputs.EffectiveMoveODL[it,iap])
        
    #     Is = np.dot(np.transpose(config.FS['OPD2Piston']),np.dot(np.diag(globalMaximumSnr),config.FS['OPD2piston']))
        
    #     allTelFound = (np.linalg.matrix_rank(Is) < NA-1)
        
    #     if allTelFound:
    #         outputs.SearchState[it+1] = 0
            

    
    """
    Signal-to-noise ratio of the fringes ("Phase variance")
    The function getvar saves the inverse of the squared SNR ("Phase variance")
    in the global stack variable varPD [NT, MW, NIN]
    Eq. 12, 13 & 14
    """

    varcurrPD, varcurrGD = getvar()
    outputs.SquaredSnrPD[it] = np.nan_to_num(1/varcurrPD)
    outputs.SquaredSnrGD[it] = np.nan_to_num(1/varcurrGD)
    outputs.SquaredSnrGDUnbiased[it] = np.nan_to_num(1/outputs.varGDUnbiased[it])
    
    timerange = range(it+1-Nsnr, it+1)
    outputs.SquaredSNRMovingAveragePD[it] = np.nan_to_num(1/np.mean(outputs.varPD[timerange], axis=0))
    outputs.SquaredSNRMovingAverageGD[it] = np.nan_to_num(1/np.mean(outputs.varGD[timerange], axis=0))
    outputs.SquaredSNRMovingAverageGDUnbiased[it] = np.nan_to_num(1/np.mean(outputs.varGDUnbiased[timerange], axis=0))
    outputs.SquaredSNRMovingAverageGDnew[it] = np.nan_to_num(1/np.mean(outputs.varGDnew[timerange], axis=0))
    
    outputs.TemporalVariancePD[it] = np.var(outputs.PDEstimated[timerange], axis=0)
    timerange = range(it+1-Nsnr, it+1)
    outputs.TemporalVarianceGD[it] = np.var(outputs.GDEstimated[timerange], axis=0)
    
    if config.FT['whichSNR'] == 'pd':
        outputs.SquaredSNRMovingAverage[it] = outputs.SquaredSNRMovingAveragePD[it]
        outputs.SquaredSNR[it] = outputs.SquaredSnrPD[it]
    else:
        outputs.SquaredSNRMovingAverage[it] = outputs.SquaredSNRMovingAverageGD[it]
        outputs.SquaredSNR[it] = outputs.SquaredSnrGD[it]

    
    
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
        outputs.singularValuesSqrt[it] = np.sqrt(S[:-1])
        
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
        outputs.time_since_loss[it] = outputs.time_since_loss[it-1]+config.dt
        
        # FringeLost = (NotCophased and (IgdRank<np.linalg.matrix_rank(outputs.Igd[it-1]))
        # This situation could pose a problem but we don't manage it yet        
        if (outputs.time_since_loss[it] > config.FT['SMdelay']):
            
            Igdna = np.dot(config.FS['OPD2Piston_moy_r'],
                           np.dot(outputs.Igd[it],config.FS['Piston2OPD_r']))
            
            Kernel = np.identity(NA) - Igdna
            
            CophasedBaselines=np.where(np.diag(outputs.Igd[it])!=0)[0]
            coherentPairs=[]
            isolatedTels = list(np.arange(NA))
            for ib in CophasedBaselines:
                ia,iap = int(config.FS['ich'][ib][0])-1, int(config.FS['ich'][ib][1])-1
                coherentPairs.append([ia,iap])
                if ia in isolatedTels:
                    isolatedTels.remove(ia)
                if iap in isolatedTels:
                    isolatedTels.remove(iap)
                    
            coherentGroups = JoinOnCommonElements(coherentPairs)
            
            # Fringe loss
            outputs.LostBaselines[it] = (np.diag(outputs.Igd[it])==0)*1
            outputs.LostTelescopes[it] = (np.diag(Igdna) == 0)*1      # The positions of the lost telescopes get 1.
            # WeLostANewTelescope = (sum(newLostTelescopes) > 0)
            if (outputs.LostBaselines[it] != outputs.LostBaselines[it-1]).any() \
                and (outputs.LostTelescopes[it] != outputs.LostTelescopes[it-1]).any():
                    print("Time:",it*config.dt,"ms; Coherent:",coherentGroups,"; Isolated:", isolatedTels)
            
            # Photometry loss
            # We consider that flux is lost if the estimated flux is lower than a fraction of the average flux.
            # Parameter is the float config.FT['ThresholdPhot'].
            outputs.noFlux[it] = 1*(np.mean(outputs.PhotometryEstimated[it,:,:],axis=0) < config.FT['ThresholdPhot']*outputs.PhotometryAverage[it])
                
            outputs.noSignalOnTel[it] = outputs.noFlux[timerange].all(axis=0)
            
            # Evaluates if the two arrays are the same
            comparison = (outputs.noSignalOnTel == outputs.LostTelescopes[it])
            outputs.LossDueToInjection[it] = (comparison.all() and outputs.noSignalOnTel[it].any()) 
            
            if not outputs.LossDueToInjection[it]:     # The fringe loss is not due to an injection loss
                config.FT['state'][it] = 1
                # nbFrameBeforeRestart = int(config.FT['SMdelay']/config.dt)
                # newLostTelescopes = (outputs.LostTelescopes[it-nbFrameBeforeRestart] == 0 * (outputs.LostTelescopes[it-nbFrameBeforeRestart+1:] == 1).all(axis=0))
                newLostTelescopes = (outputs.LostTelescopes[it] - outputs.LostTelescopes[it-1] == 1)
                TelescopesThatGotBackPhotometry = (outputs.noSignalOnTel[it] - outputs.noSignalOnTel[it-1] == 1)
                # WeGotBackPhotometry = (sum(TelescopesThatGotBackPhotometry) > 0)
                
                TelescopesThatNeedARestart = np.argwhere(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)
                outputs.TelescopesThatNeedARestart[it] = np.ones(NA)*(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)
                
                # Null the value of the accumulative element that handle sawtooth jumps
                if len(TelescopesThatNeedARestart):
                    config.FT['moveSinceLastChange'][TelescopesThatNeedARestart] = 0
                    # config.FT['eps'][TelescopesThatNeedARestart] = 1
                    # config.FT['it0'][TelescopesThatNeedARestart] = it
                    # config.FT['it_last'][TelescopesThatNeedARestart] = it
                    # config.FT['nbChanges'][TelescopesThatNeedARestart] = 0
                    
                if config.FT['state'][it-1] != 1:
                    config.FT['eps'] = np.ones(NA)
                    config.FT['it0'] = np.ones(NA)*it
                    config.FT['it_last'] = np.ones(NA)*it
                
                Velocities = np.dot(Kernel,config.FT['vfactorsRelock'])
                
                # Increment = relockfunction_inc_sylvain_gestioncophased(it, Velocities, config.FT['covering'], coherentGroups)
            
                Increment = relockfunction_230515(it, Velocities, config.FT['covering'], coherentGroups, isolatedTels)
            
                #You should send command only on telescope with flux
                outputs.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(outputs.noSignalOnTel[it])
                
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
        outputs.GDCommandIntegrator[it+1] = outputs.GDCommandIntegrator[it] + FT['GainGD']*currGDerr*config.wlOfTrack*R/2/np.pi
        
        # From OPD to Pistons
        uGD = np.dot(FS['OPD2Piston_r'], outputs.GDCommandIntegrator[it+1])
        
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
    outputs.GDCommand[it+1] = np.dot(FS['Piston2OPD'],uGD)

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
    
    from .config import NA, MW
    
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
        
    varNum = np.zeros([MW,NINmes]) #; varNum2 = np.zeros([MW,NINmes])
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
    
    varNumGD = np.mean(varX+varY,axis=0)**2
    varNumGDMeanOverLmbda = np.mean(varNumGD)
    CfGDSquaredNorm = np.real(outputs.CfGDMeanOverLmbda[it]*np.conj(outputs.CfGDMeanOverLmbda[it]))  # Sum over lambdas of |CohFlux|² (modified eq.14)
    outputs.varGDnew[it] = varNumGDMeanOverLmbda/CfGDSquaredNorm
    
    # outputs.SNRPhotometry[it,:] = np.sum(outputs.PhotometryEstimated[it,:],axis=0)/np.sqrt(np.sum(varPhot,axis=0))
    
    varPD = outputs.varPD[it]
    varGD = outputs.varGD[it]
    
    return varPD, varGD


def SetThreshold(TypeDisturbance="CophasedThenForeground",nbSigma=0,minThreshold=0,
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
    Nsnr = config.FT['Nsnr']

    if nbSigma:
        
        NT=200 ;

        InitNT = config.NT
        
        foreground = 3*config.FS['R']*config.wlOfTrack*np.arange(config.NA)
        
#         underSampling = (np.ptp(foreground) >= config.nyquistCriterion)
#         if underSampling:
#             print(f"/!\  ATTENTION: one or more OPD value(s) doesn't respect Nyquist criterion \
# (OPD<{config.nyquistCriterion}µm).\n\
# The simulation might experience aliasing and the SNR values won't be correct. /!\ ")
        
        sk.update_config(foreground=foreground, NT = NT, verbose=verbose)
        
        # Initialize the fringe tracker with the gain
        gainPD,gainGD,relock,state=config.FT['GainPD'],config.FT['GainGD'],config.FT['relock'],config.FT['state']
        sk.updateFTparams(GainPD=0, GainGD=0, relock=False,search=False, verbose=verbose)
        
        config.FT['state'] = np.zeros(NT+1)
        
        # Launch the scan
        sk.loop(verbose=verbose)
        
        startComputing = 50
        newThresholdGD = np.ones(NINmes)

        timerange = range(startComputing,startComputing+Nsnr)
        for ib in range(NINmes):
            SNRfg = np.mean(np.sqrt(outputs.SquaredSNR[timerange,ib]))
            fgstd = np.std(np.sqrt(outputs.SquaredSNR[timerange,ib]))
            
            # Set threshold to mean(SNR) + nBsigma * rms(SNR)
            newThresholdGD[ib] = SNRfg + nbSigma*fgstd
            if minThreshold and (newThresholdGD[ib] < minThreshold):
                newThresholdGD[ib] = minThreshold
                
        
        newThresholdPD = 1e-3#np.min(newThresholdGD)/2
        
        config.FT['ThresholdGD'] = newThresholdGD
        config.FT['ThresholdPD'] = newThresholdPD
        config.FT['state'] = state

        sk.display('opd','snr','detector',wlOfTrack=1.6, pause=True,display=display)
        
        sk.update_config(NT=InitNT, foreground=[],verbose=verbose)
        sk.updateFTparams(GainPD=gainPD, GainGD=gainGD, relock=relock,
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
            
        InitialDisturbanceFile,InitNT = config.DisturbanceFile, config.NT
        
        sk.update_config(DisturbanceFile=DisturbanceFile, NT = NT,verbose=verbose)
        
        # Initialize the fringe tracker with the gain
        #SPICAFT(init=True, GainPD=0, GainGD=0,relock=False)
        gainPD,gainGD,relock=config.FT['GainPD'],config.FT['GainGD'],config.FT['relock']
        sk.updateFTparams(GainPD=0, GainGD=0, relock=False, verbose=verbose)
        
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
        sk.updateFTparams(GainPD=gainPD, GainGD=gainGD, relock=relock, 
                       ThresholdGD=newThresholdGD,
                       verbose=verbose)
    
    
    else:
        
        DisturbanceFile = TypeDisturbance
        
        NT=200
            
        InitialDisturbanceFile,InitNT = sk.config.DisturbanceFile, sk.config.NT
        
        sk.update_config(DisturbanceFile=DisturbanceFile, NT = NT, verbose=verbose)
        
        # Initialize the fringe tracker with the gain
        #SPICAFT(init=True, GainPD=0, GainGD=0,relock=False)
        gainPD,gainGD,relock,state=config.FT['GainPD'],config.FT['GainGD'],config.FT['relock'],config.FT['state']
        sk.updateFTparams(GainPD=0, GainGD=0, relock=False,search=False, verbose=verbose)
        
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
                CophasedInd = 50 ; ForegroundInd = 150
                CophasedRange = range(CophasedInd,CophasedInd+Nsnr)
                ForegroundRange = range(ForegroundInd,ForegroundInd+Nsnr)
                newThresholdGD = np.ones(NINmes)

                for ib in range(NINmes):
                    SNRcophased = np.mean(np.sqrt(outputs.SquaredSNRMovingAverage[CophasedRange,ib]))
                    SNRfg = np.mean(np.sqrt(outputs.SquaredSNRMovingAverage[ForegroundRange,ib]))
                    fgstd = np.std(np.sqrt(outputs.SquaredSNRMovingAverage[ForegroundRange,ib]))
                    # cophasedstd = np.std(np.sqrt(outputs.SquaredSNRMovingAverage[CophasedRange,ib]))
                    
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
        sk.updateFTparams(GainPD=gainPD, GainGD=gainGD, relock=relock,
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
    from .config import NA
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
            usaw[ia] = config.FT['moveSinceLastChange'][ia] + config.FT['eps'][ia]
            config.FT['moveSinceLastChange'][ia] = utemp
        
    
    outputs.eps[it] = config.FT['eps']
    outputs.it_last[it] = config.FT['it_last']
    outputs.moveSinceLastChange[it] = config.FT['moveSinceLastChange']
    
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
        usaw = config.FT['moveSinceLastChange'] + config.FT['eps']
        config.FT['moveSinceLastChange'] = utemp

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
        diff = config.FT['moveSinceLastChange'] - usaw
        config.FT['eps'] = -config.FT['eps']
        config.FT['it_last'] = it
        usaw = diff + config.FT['eps']
        config.FT['moveSinceLastChange'] = utemp

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
            config.FT['moveSinceLastChange'][ia]+=move[ia]
            
        else:   # Saut 
            change=True
            config.FT['eps'][ia] = -config.FT['eps'][ia]
            config.FT['it_last'][ia] = it
            move[ia] = -config.FT['moveSinceLastChange'][ia]
            config.FT['moveSinceLastChange'][ia] = move[ia]
    
    
    return move


def relockfunction_inc_sylvain_gestioncophased(it, v, covering, coherentGroups):
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
    
        it_last=config.FT['it_last'][ia]; eps=config.FT['eps'][ia]
        nbChanges = config.FT['nbChanges'][ia]
        
        # Temps avant saut de frange
        sweep = config.FT['sweep0']*config.FT['commonRatio']**nbChanges
        
        # Temps passé depuis dernier saut.
        time_since_last_change = (it-it_last)*config.dt
        
        change=False
        if time_since_last_change < sweep:  # Pas de saut
            move[ia] = eps*v[ia]
            
        else:   # Saut 
            change=True
            config.FT['eps'][ia] = -config.FT['eps'][ia]
            config.FT['it_last'][ia] = it
            config.FT['nbChanges'][ia] += 1
            move[ia] = -config.FT['moveSinceLastChange'][ia]   # Cancel all moves you did since last change on telescope ia
            
            # Cophased telescopes remain together when jumping
            for CophasedGroup in coherentGroups:
                l = [move[ia-1] for ia in CophasedGroup]
                generalmove = l[0]
                
                if ia in CophasedGroup:
                    move[ia] = generalmove
                    
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
            
            # Starts the new accumulation of commands until next change
            config.FT['moveSinceLastChange'][ia] = move[ia]
            
    # If there is no jump, the increment is normalised by the maximal authorized velocity
    if not change: 
        move = move / np.ptp(move) * config.FT['maxVelocity']
        config.FT['moveSinceLastChange'] += move  
    
    return move


def relockfunction_230515(it, v, covering, coherentGroups, isolatedTels):
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
    
    allGroups = coherentGroups + isolatedTels
    nbOfGroups = len(allGroups)
    v_groups = np.arange(nbOfGroups) - nbOfGroups/2
    
    for iGroup in range(nbOfGroups):
        group = allGroups[iGroup]
        
        if isinstance(group,list):
            ia0 = group[0]

            it_last=config.FT['it_last'][ia0]; eps=config.FT['eps'][ia0]
            nbChanges = config.FT['nbChanges'][ia0]
            
            # Temps avant saut de frange
            sweep = config.FT['sweep0']*config.FT['commonRatio']**nbChanges
            
            # Temps passé depuis dernier saut.
            time_since_last_change = (it-it_last)*config.dt
            
            change=False
            if time_since_last_change < sweep:  # Pas de saut
                move[ia0] = eps*v_groups[iGroup]
                
            else:   # Saut 
                change=True
                config.FT['eps'][ia0] = -config.FT['eps'][ia0]
                config.FT['it_last'][ia0] = it
                config.FT['nbChanges'][ia0] += 1
                move[ia0] = -config.FT['moveSinceLastChange'][ia0]   # Cancel all moves you did since last change on telescope ia
                
                # # Cophased telescopes remain together when jumping
                # if ia != group[0]:
                #     ia0 = group[0]
                #     # All telescopes of the group follow the first one.
                #     move[ia] = move[ia0]
                        
                # Add some covering to prevent from missing fringes
                if np.abs(move[ia0]) < 2*covering:
                    coveringtemp = 0
                else:
                    coveringtemp = covering
                    
                if move[ia0] > 0:
                    move[ia0] -= coveringtemp
                if move[ia0] < 0:
                    move[ia0] += coveringtemp
                    
                coveringtemp = covering
                
                # Starts the new accumulation of commands until next change
                config.FT['moveSinceLastChange'][ia0] = move[ia0]
            
            for ia in group[1:]:
                config.FT['eps'][ia] = config.FT['eps'][ia0]
                config.FT['it_last'][ia] = config.FT['it_last'][ia0]
                config.FT['nbChanges'][ia] = config.FT['nbChanges'][ia0]
                config.FT['moveSinceLastChange'][ia] = config.FT['moveSinceLastChange'][ia0]
                move[ia] = move[ia0]
                
                    
        else:
            ia = group
            
            it_last=config.FT['it_last'][ia]; eps=config.FT['eps'][ia]
            nbChanges = config.FT['nbChanges'][ia]
            
            # Temps avant saut de frange
            sweep = config.FT['sweep0']*config.FT['commonRatio']**nbChanges
            
            # Temps passé depuis dernier saut.
            time_since_last_change = (it-it_last)*config.dt
            
            change=False
            if time_since_last_change < sweep:  # Pas de saut
                move[ia] = eps*v_groups[iGroup]
                
            else:   # Saut 
                change=True
                config.FT['eps'][ia] = -config.FT['eps'][ia]
                config.FT['it_last'][ia] = it
                config.FT['nbChanges'][ia] += 1
                move[ia] = -config.FT['moveSinceLastChange'][ia]   # Cancel all moves you did since last change on telescope ia
                        
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
                
                # Starts the new accumulation of commands until next change
                config.FT['moveSinceLastChange'][ia] = move[ia]
            
    # If there is no jump, the increment is normalised by the maximal authorized velocity
    if not change:
        move = move / np.ptp(move) * config.FT['maxVelocity']
        config.FT['moveSinceLastChange'] += move  
    
    return move
    
    # for ia in range(NA):
    
    #     it_last=config.FT['it_last'][ia]; eps=config.FT['eps'][ia]
    #     nbChanges = config.FT['nbChanges'][ia]
        
    #     # Temps avant saut de frange
    #     sweep = config.FT['sweep0']*config.FT['commonRatio']**nbChanges
        
    #     # Temps passé depuis dernier saut.
    #     time_since_last_change = (it-it_last)*config.dt
        
    #     change=False
    #     if time_since_last_change < sweep:  # Pas de saut
    #         move[ia] = eps*v[ia]
            
    #         # Cophased telescopes remain together when jumping
    #         tel=ia+1
    #         for igroup in range(nbOfGroups):
    #             group = coherentGroups[igroup]
    #             if isinstance(group,list):
    #                 if tel in group:
    #                     move[ia] = eps*v_modified[igroup]
    #             elif isinstance(group,float):
    #                 if tel == group:
    #                     move[ia] = eps*v_modified[igroup]
            
    #     else:   # Saut 
    #         change=True
    #         config.FT['eps'][ia] = -config.FT['eps'][ia]
    #         config.FT['it_last'][ia] = it
    #         config.FT['nbChanges'][ia] += 1
    #         move[ia] = -config.FT['moveSinceLastChange'][ia]   # Cancel all moves you did since last change on telescope ia
            
    #         # Cophased telescopes remain together when jumping
    #         tel=ia+1
    #         for CophasedGroup in coherentGroups:
    #             l = [move[ia-1] for ia in CophasedGroup]
    #             generalmove = l[0]  #max(set(l), key = l.count)
                
    #             if tel in CophasedGroup:
    #                 move[ia] = generalmove
                    
    #         # Add some covering to prevent from missing fringes
    #         if np.abs(move[ia]) < 2*covering:
    #             coveringtemp = 0
    #         else:
    #             coveringtemp = covering
                
    #         if move[ia] > 0:
    #             move[ia] -= coveringtemp
    #         if move[ia] < 0:
    #             move[ia] += coveringtemp
                
    #         coveringtemp = covering
            
    #         # Starts the new accumulation of commands until next change
    #         config.FT['moveSinceLastChange'][ia] = move[ia]
            
    
    # if not change: # If there is no jump, the increment is normalised by the maximal authorized velocity
    #     move = move / np.ptp(move) * config.FT['maxVelocity']
    #     config.FT['moveSinceLastChange'] += move  
    
    
    # return move


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
            config.FT['moveSinceLastChange'][ia]+=eps
            
        else:
            change=True
            eps = -eps
            it_last = it
            
            # la fonction usaw prend la valeur qu'elle avait avant le précédent
            # saut.
            usaw[ia] = -config.FT['moveSinceLastChange'][ia]
            config.FT['it_last'][ia] = it_last
            config.FT['moveSinceLastChange'][ia]=-config.FT['moveSinceLastChange'][ia]
            
        config.FT['eps'][ia] = eps
    print(config.FT['moveSinceLastChange'])
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





