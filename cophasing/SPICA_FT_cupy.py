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
import cupy as cp

from .coh_tools import posk, poskfai,NB2NIN

from . import config

def SPICAFT(*args, init=False, GainPD=0, GainGD=0, Ngd=50, roundGD=True, Ncross=1,
            Sweep=100, Slope=100, Vfactors = [], CPref=True, Ncp = 300,
            ThresholdGD=2, ThresholdPD = 1.5, 
            Threshold=True, usePDref=True, useWmatrices=True,
            latencytime=1):
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
    Sweep : TYPE, optional
        DESCRIPTION. The default is 100.
    Slope : TYPE, optional
        DESCRIPTION. The default is 100.
    Vfactors : TYPE, optional
        DESCRIPTION. The default is [].
    Ncross : INT, optional
        Separation between two spectral channels for GD calculation. 
        The default is 1.
    CPref : BOOLEAN, optional
        If False, the Closure Phase is not subtracted for reference. 
        The default is True.
    Ncp : INT, optional
        Frame integration CP. The default is 1.
    roundGD : BOOLEAN, optional
        If True, the GD command is rounded to wavelength integers. 
        Advised to avoid a superposition of the PD and GD commands.
        The default is True.
    ThresholdGD : FLOAT, optional
        If the SNR of the estimated GD of a given baseline is lower than 
        ThresholdGD, then this baseline is weighted down.
        The default is 2.
    ThresholdPD : FLOAT, optional
        If the SNR of the estimated GD of a given baseline is lower than 
        ThresholdGD, then this baseline is weighted down.
        DESCRIPTION. The default is 1.5.
    Threshold : BOOLEAN, optional
        If False, the GD works also within a frange. Essentially for debugging.
    usePDref : BOOLEAN, optional
        If False, no reference vector
    useWmatrices: BOOLEAN, optional
        Wmatrices means Weighting matrices. If True, the weighting of the commands
        using the SNR of the measurements is used.

    Returns
    -------
    currCmd : ARRAY [NA]
        Piston Command to send to the ODL.

    """
    
    if init:

        # config.R = np.abs((config.MW-1)*config.PDspectra/(config.spectraM[-1] - config.spectraM[0]))
        config.FT['Name'] = 'SPICAfromGRAVITY'
        config.FT['func'] = SPICAFT
        config.FT['Ngd'] = Ngd
        config.FT['GainGD'] = GainGD
        config.FT['GainPD'] = GainPD
        config.FT['Sweep'] = Sweep
        config.FT['Slope'] = Slope*config.dt*1e-3
        config.FT['Ncross'] = Ncross
        config.FT['Ncp'] = Ncp
        config.FT['ThresholdGD'] = ThresholdGD
        config.FT['ThresholdPD'] = ThresholdPD
        config.FT['CPref'] = CPref
        config.FT['roundGD'] = roundGD
        config.FT['Threshold'] = Threshold
        config.FT['cmdOPD'] = True
        config.FT['usePDref'] = usePDref
        config.FT['useWmatrices'] = useWmatrices
        
        if len(Vfactors) != 0:
            config.FT['Vfactors'] = Vfactors
        else:
            config.FT['Vfactors'] = np.array([0, -10,-9,-6,2,7])/10
            print(f"Searching velocity factors are {config.FT['Vfactors']}")
        
        from .config import NA,NIN
        
        config.FT['Piston2OPD'] = np.zeros([NIN,NA])    # Piston to OPD matrix
        config.FT['OPD2Piston'] = np.zeros([NA,NIN])    # OPD to Pistons matrix
        
        # if not config.TELref:
        for ia in range(NA):
            for iap in range(ia+1,NA):
                ib = posk(ia,iap,NA)
                config.FT['Piston2OPD'][ib,ia] = 1
                config.FT['Piston2OPD'][ib,iap] = -1
                config.FT['OPD2Piston'][ia,ib] = 1
                config.FT['OPD2Piston'][iap,ib] = -1
            # config.FT['OPD2Piston'] = np.linalg.pinv(config.FT['Piston2OPD'])   # OPD to pistons matrix
        
        config.FT['OPD2Piston'] = config.FT['OPD2Piston']/NA
        
        if config.TELref:
            iTELref = config.TELref - 1
            L_ref = config.FT['OPD2Piston'][iTELref,:]
            config.FT['OPD2Piston'] = config.FT['OPD2Piston'] - L_ref
        
        return

    from . import simu

    it = simu.it
    
    currCfEstimated = args[0]

    currPD, currGD = ReadCf(currCfEstimated)
    
    simu.PDEstimated[it] = currPD
    simu.GDEstimated[it] = currGD*config.FS['R']
    
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
        - simu.ComplexVisibility                    [NT,MW,NIN]*1j
        - simu.CoherenceDegree                      [NT,MW,NIN]
    """
    
    from . import simu
    
    from .config import NA,NIN,NC
    from .config import MW
    
    it = simu.it            # Time
     
    """
    Photometries extraction
    [NT,MW,NA]
    """
    PhotEst = np.zeros([MW,NA])
    for ia in range(NA):
        PhotEst[:,ia] = np.real(currCfEstimated[:,ia*(NA+1)])
    
    # Extract NIN-sized coherence vector from NB-sized one. 
    # (eliminates photometric and conjugate terms)
    currCfEstimatedNIN = np.zeros([MW, NIN])*1j
    for imw in range(MW):    
        currCfEstimatedNIN[imw,:] = NB2NIN(currCfEstimated[imw,:])
        
    # Save coherent flux and photometries in stack
    simu.PhotometryEstimated[it] = PhotEst
    
    """
    Visibilities extraction
    [NT,MW,NIN]
    """
    
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib = posk(ia,iap,NA)
            Iaap = currCfEstimatedNIN[:,ib]                  # Interferometric intensity (a,a')
            Ia = PhotEst[:,ia]                     # Photometry pupil a
            Iap = PhotEst[:,iap]                   # Photometry pupil a'
            simu.ComplexVisibility[it,:,ib] = 2*Iaap/(Ia+Iap)          # Fringe ComplexVisibility of the base (a,a')
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
        # simu.CfPD[it,imw] = simu.CfPD[it,imw]*np.exp(-1j*simu.PDref)
        
    # Current Phase-Delay
    currPD = np.angle(np.sum(simu.CfPD[it,:,:], axis=0))
        
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
        # cfgd = cfgd*np.exp(-1j*simu.GDref)
        
        simu.CfGD[it,:,:] += cfgd


    currGD = np.zeros(NIN)
    for ib in range(NIN):
        cfGDlmbdas = simu.CfGD[it,:-Ncross,ib]*np.conjugate(simu.CfGD[it,Ncross:,ib])
        cfGDmoy = np.sum(cfGDlmbdas)

        currGD[ib] = np.angle(cfGDmoy)    # Group-delay on baseline (ib).
    
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
    
    timerange = range(it+1-Ncp,it+1)
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib = posk(ia,iap,NA)      # coherent flux (ia,iap)    
            cs1 = np.sum(simu.CfPD[timerange,:,ib], axis=1)     # Sum of coherent flux (ia,iap)
            cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
            cfGDmoy1 = np.sum(cfGDlmbdas,axis=1)     # Sum of coherent flux (ia,iap)  
            for iapp in range(iap+1,NA):
                ib = posk(iap,iapp,NA) # coherent flux (iap,iapp)    
                cs2 = np.sum(simu.CfPD[timerange,:,ib], axis=1) # Sum of coherent flux (iap,iapp)    
                cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
                cfGDmoy2 = np.sum(cfGDlmbdas,axis=1)
                
                ib = posk(ia,iapp,NA) # coherent flux (iapp,ia)    
                cs3 = np.sum(np.conjugate(simu.CfPD[timerange,:,ib]),axis=1) # Sum of 
                cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
                cfGDmoy3 = np.sum(cfGDlmbdas,axis=1)
                
                # The bispectrum of one time and one triangle adds up to
                # the Ncp last times
                ic = poskfai(ia,iap,iapp,NA)        # 0<=ic<NC=(NA-2)(NA-1) 
                
                bispectrumPD[ic]=np.sum(cs1*cs2*cs3)
                bispectrumGD[ic]=np.sum(cfGDmoy1*cfGDmoy2*np.conjugate(cfGDmoy3))
                    
    simu.ClosurePhasePD[it] = np.angle(bispectrumPD)
    simu.ClosurePhaseGD[it] = np.angle(bispectrumGD)
    
    if config.FT['CPref'] and (it == 0):                     # At time 0, we create the reference vectors
        for ia in range(1,NA-1):
            for iap in range(ia+1,NA):
                k = posk(ia,iap,NA)
                ic = poskfai(0,ia,iap,NA)   # Position of the triangle (0,ia,iap)
                
                if config.FT['usePDref']:
                    simu.PDref[k] = simu.ClosurePhasePD[0,ic]
                    simu.GDref[k] = simu.ClosurePhaseGD[0,ic]
                else:
                    simu.PDref[k] = 0
                    simu.GDref[k] = 0
                    
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
    
    from .config import NA,NB,NIN
    from .config import FT
    
    SNRgd = 2
    SNRpd = 1.5
    
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
        
        varcurrPD = getvar()
        
        # Raw Weighting matrix in the OPD-space
        W = np.zeros([NIN,NIN])
        
        for ib in range(NIN):
            if 1/np.mean(simu.varPD[it+1-Ngd:it+1,ib]) >= FT['ThresholdGD']**2:
                W[ib,ib] = 1/varcurrPD[ib]
        
        # Transpose the W matrix in the Piston-space
        MtWM = np.dot(FT['OPD2Piston'], np.dot(W,FT['Piston2OPD']))
        
        # Singular-Value-Decomposition of the W matrix
        U, S, Vt = np.linalg.svd(MtWM)
        
        Ut = np.transpose(U)
        V = np.transpose(Vt)
        
        """
        GD weighting matrix
        """
    
        Sdag = np.zeros([NA,NA])
        for ia in range(NA):
            if S[ia] > 0:
                Sdag[ia,ia] = 1/S[ia]
        
        # Come back to the OPD-space        
        VSdagUt = np.dot(V, np.dot(Sdag,Ut))
        
        # Calculates the weighting matrix
        currIgd = np.dot(FT['Piston2OPD'],np.dot(VSdagUt,np.dot(FT['OPD2Piston'], W)))
    
    
        """
        PD Weighting matrix
        """

        Sdag = np.zeros([NA,NA])
        for ia in range(NA):
            if S[ia] > SNRpd**2:
                Sdag[ia,ia] = 1/S[ia]
            else:
                Sdag[ia,ia] = S[ia]/FT['ThresholdPD']**4
        
        # Come back to the OPD-space        
        VSdagUt = np.dot(V, np.dot(Sdag,Ut))
        
        # Calculates the weighting matrix
        currIpd = np.dot(FT['Piston2OPD'],np.dot(VSdagUt,np.dot(FT['OPD2Piston'], W)))
            
    else:
        currIgd = np.identity(NIN)
        currIpd = np.identity(NIN)
    
    simu.Igd[it,:,:] = currIgd
    simu.Ipd[it,:,:] = currIpd
    
    """
    Group-Delay tracking
    """
    
    currGDerr = currGD - simu.GDref
    
    # Keep the GD between [-Pi, Pi]
    # Eq. 35
    for ib in range(NIN):
        if currGDerr[ib] > np.pi:
            currGDerr[ib] -= 2*np.pi
        elif currGDerr[ib] < -np.pi:
            currGDerr[ib] += 2*np.pi
    
    R = config.FS['R']
    # Store residual GD for display only [radians]
    simu.GDResidual[it] = currGDerr*R
    
    # Weights the GD (Eq.35)
        
    currGDerr = np.dot(currIgd,currGDerr)
     
    if FT['Threshold']:     # Threshold function (eq.36)
        for ib in range(NIN):
            if currGDerr[ib] > np.pi/R:
                currGDerr[ib] -= np.pi/R
            elif currGDerr[ib] < -np.pi/R:
                currGDerr[ib] += np.pi/R
            else:
                currGDerr[ib] = 0
    
    # Integrator (Eq.37)
    if FT['cmdOPD']:     # integrator on OPD
        # Integrator
        simu.GDCommand[it] = simu.GDCommand[it-1] + FT['GainGD']*currGDerr
        
        # From OPD to Pistons
        uGD = np.dot(FT['OPD2Piston'], simu.GDCommand[it])
        
    else:                       # integrator on Pistons
        # From OPD to Piston
        currPistonGD = np.dot(FT['OPD2Piston'], currGDerr)
        # Integrator
        uGD = simu.PistonGDCommand[it-1] + FT['GainGD']*currPistonGD
        
    # simu.GDCommand[it+1] = simu.GDCommand[it] + FT['GainGD']*currGDerr
    
    # From OPD to Piston
    # uGD = np.dot(FT['OPD2Piston'], simu.GDCommand[it+1])
    
    if config.FT['roundGD']:
        for ia in range(NA):
            jumps = round(uGD[ia]/config.PDspectra)
            uGD[ia] = jumps*config.PDspectra
            
    simu.PistonGDCommand[it] = uGD

    """
    Phase-Delay command
    """
    
    currPDerr = currPD - simu.PDref
 
    # Keep the PD between [-Pi, Pi]
    # Eq. 35
    for ib in range(NIN):
        if currPDerr[ib] > np.pi:
            currPDerr[ib] -= 2*np.pi
        elif currPDerr[ib] < -np.pi:
            currPDerr[ib] += 2*np.pi
    
    # Store residual PD and piston for display only
    simu.PDResidual[it] = currPDerr
    # simu.PistonResidual[it] = np.dot(OPD2Piston_, simu.PDResidual[it])/2/np.pi*config.PDspectra
    
    # Weights the PD (Eq.35)
    currPDerr = np.dot(currIpd,currPDerr)
    
    # Integrator (Eq.37)
            
    if FT['cmdOPD']:     # integrator on OPD
        # Integrator
        simu.PDCommand[it] = simu.PDCommand[it-1] + FT['GainPD']*currPDerr
        # From OPD to Pistons
        uPD = np.dot(FT['OPD2Piston'], simu.PDCommand[it])
        
    else:                       # integrator on Pistons
        # From OPD to Piston
        currPistonPD = np.dot(FT['OPD2Piston'], currPDerr)
        # Integrator
        uPD = simu.PistonPDCommand[it-1] + FT['GainPD']*currPistonPD
    
    simu.PistonPDCommand[it] = uPD
    
    """
    FRINGE SEARCHING command
    """
    
    usearch = np.zeros([NA])
    
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
    
    (NA,NB,NIN)=(config.NA,config.NB,config.NIN)
    (NW,MW)=(config.NW,config.MW)
    (MT,NT)=(config.MT,config.NT) 
    
    it = simu.it
    
    image = cp.array(simu.MacroImages[it,:,:])
    
    MW = np.shape(image)[0]
    M = config.M            # Amplification ratio camera
    
    sigsky = cp.array(config.FS['sigsky'])  # Background noise
    imsky = cp.array(config.FS['imsky'])    # Sky image before observation
    Demod = cp.array(config.FS['MacroP2VM'])    # Macro P2VM matrix used for demodulation
    
    # Flux variance calculation (eq. 12)
    varFlux = sigsky**2 + 1*(image - imsky)
    
    """
    Covariance calculation (eq. 13)
    """
    
    covar_ = cp.zeros([NT,MW,NB,NB])*1j
    for iw in range(MW):
        covar_[it,iw] = cp.dot(Demod[iw], cp.dot(cp.diag(varFlux[iw]),cp.transpose(Demod[iw])))
        
    # Phase variance calculation (eq. 14)
    # Put to a non-null very low value for the moment
    varPD = cp.ones([NIN])*1e-1

    Nvar = 5                # Integration time for phase variance
    if it < Nvar:
        Nvar = it+1
        
    vartemp = cp.zeros([MW,NIN])
    vartemp2 = cp.zeros([MW,NIN])*1j
    
    for iot in range(it+1-Nvar,it+1):
        for ia in range(NA):
            for iap in range(ia+1,NA):
                k = posk(ia,iap,NA)
                kp = ia*NA+iap
                vartemp[:,k] += (cp.abs(cp.real(covar_[iot,:,kp,kp]))+cp.abs(cp.imag(covar_[iot,:,kp,kp])))/Nvar
                vartemp2[:,k] += cp.array(simu.CfPD[iot,:,k])/Nvar
    
    varPD = cp.sum(vartemp,axis=0)/(2*cp.abs(cp.sum(vartemp2,axis=0))**2)
    
    varPDnumpy = cp.asnumpy(varPD)
    simu.varPD[it] = varPDnumpy
    
    return varPDnumpy


def searchfunction(NA,sweep,a,t):
    
    usaw = np.ones([NA])
    
    nbtooth = t//sweep
    tp = t%sweep + nbtooth//2*sweep
    if nbtooth%2 == 0:
        usaw *= a*tp
        
    elif nbtooth%2 == 1:
        usaw *= -a*tp
        
    return usaw

