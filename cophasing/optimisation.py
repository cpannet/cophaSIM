# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:17:41 2020

@author: cpannetier

Routines for optimising the fringe tracker

- OptimGain: Optimise on gain PD and GD
    
- OptimGainMulitprocess: Multiprocessing version of OptimGain.
    Doesn't work so far (16-11-2020)

"""

import numpy as np
import time

from . import config
import cophasing.skeleton as sk
import cophasing.coh_tools as ct
import glob

from importlib import reload  # Python 3.4+ only.

# def OptimGainMultiprocess(GainsPD=[],GainsGD=[],optim='FC',
#                 TimeBonds=100, WavelengthOfInterest=1.5, DIT=50,
#                 telescopes=0):
    
    
#     import multiprocessing
    
#     pool = multiprocessing.Pool()
    
#     from .config import NIN, NC
    
#     if len(GainsGD):
#         gainstr='GainGD'
#         Ngains = len(GainsGD)
#         Gains = GainsGD
#     elif len(GainsPD):
#         gainstr='GainPD'
#         Ngains = len(GainsPD)
#         Gains = GainsPD
#     else:
#         raise Exception('Need GainsPD or GainsGD.')
        
#     print(f"Start {gainstr} optimisation with sample gains={Gains}")
    
#     # SimpleIntegrator(init=True, GainPD=Gains[0], GainGD=0)
    
    
#     VarOPD = np.zeros([Ngains,NIN])     # Phase variances
#     VarCP = np.zeros([Ngains, NC])      # Cosure Phase variances
#     FCArray = np.zeros([Ngains, NIN])   # Contains the fringe contrasts
    
    
#     minValue = 10000
#     iOptim = -1
        
    
    # def GiveResults(G):
        
    #     config.FT['GainPD'] = G
        
    #     print(f'{gainstr}={G}')
        
    #     # Launch the simulator
    #     sk.loop()
    #     # Load the performance observables into simu module
    #     sk.ShowPerformance(TimeBonds, WavelengthOfInterest,DIT, display=False)
                
        
    #     return simu.VarOPD, simu.VarCPD, simu.VarCGD, simu.FringeContrast
        
       
        # elif (Value > 10*minValue) and (ig < Ngains-1):
        #     VarOPD[ig+1:,:] = 100*np.ones([Ngains-ig-1,NIN])
        #     VarCP[ig+1:,:] = 100*np.ones([Ngains-ig-1,NC])
        #     FCArray[ig+1:,:] = 100*np.ones([Ngains-ig-1,NIN])
        #     break
    
    
    # result = GiveResults(0.2)
    # myresult = pool.starmap(GiveResults, Gains.reshape([Ngains,1]))
    
    # print(myresult)
    
    # for ig in range(Ngains):
        
    #     # Initialise the comparison tables
    #     VarOPD[ig,:] = simu.VarOPD
    #     if gainstr=='GainPD':
    #         VarCP[ig,:] = simu.VarCPD
    #     elif gainstr=='GainGD':
    #         VarCP[ig,:] = simu.VarCGD
    #     FCArray[ig,:] = simu.FringeContrast
        
    #     if optim=='phase':      # Optimisation on Phase residues
    #         if not telescopes:
    #             Value = np.mean(VarOPD[ig,:])
    #         elif telescopes:
    #             itel1,itel2 = telescopes[0]-1, telescopes[1]-1
    #             ib = coh_tools.posk(itel1, itel2, config.NA)
    #             Value = VarOPD[ig,ib]
    #         if Value < minValue:    
    #             minValue = Value
    #             iOptim = ig
        
    #     elif optim == 'FC':     # Optimisation on final fringe contrast
    #         if not telescopes:
    #             Value = 1-np.mean(FCArray[ig,:])
    #         elif telescopes:
    #             itel1,itel2 = telescopes[0]-1, telescopes[1]-1
    #             ib = coh_tools.posk(itel1, itel2, config.NA)
    #             Value = 1-FCArray[ig,ib]
    #         print(Value, minValue)
    #         if Value < minValue:
    #             minValue = Value
    #             iOptim = ig
                
    #     elif optim=='CP':       # Optimisation on Closure Phase variance
    #         if not telescopes:
    #             Value = np.mean(VarCP[ig,:])
    #         elif telescopes:
    #             if len(telescopes) != 3:
    #                 raise Exception('For defining a closure phase, telescopes must be three.')
    #             itel1,itel2,itel3 = telescopes[0]-1, telescopes[1]-1, telescopes[2]-1
    #             ic = coh_tools.poskfai(itel1, itel2, itel3, config.NA)
    #             Value = VarCP[ig,ic]
    #         if Value < minValue:    
    #             minValue = Value
    #             iOptim = ig           
        
    #     elif (Value > 10*minValue) and (ig < Ngains-1):
    #         VarOPD[ig+1:,:] = 100*np.ones([Ngains-ig-1,NIN])
    #         VarCP[ig+1:,:] = 100*np.ones([Ngains-ig-1,NC])
    #         FCArray[ig+1:,:] = 100*np.ones([Ngains-ig-1,NIN])
    #         break
        
    # bestGain = Gains[iOptim]
    
#     print(f"Best gain: {Gains[iOptim]} \n\
# Average OPD Variance: {minValue} \n\
# Average CPD Variance: {minValue} \n\
# Average Fringe Contrast: {1-minValue}")
    


    # return #bestGain, iOptim, VarOPD, VarCP, FCArray


def OptimGain(GainsPD=[],GainsGD=[],optim='FC',filedir='',
                TimeBonds=100, WavelengthOfInterest=1.5, DIT=50,
                telescopes=0):
    
    from . import simu
    from . import config
    import pandas as pd
    
    from .config import NIN, NC
        
    if len(GainsGD):
        gainstr='GainGD'
        Ngains = len(GainsGD)
        Gains = GainsGD
    elif len(GainsPD):
        gainstr='GainPD'
        Ngains = len(GainsPD)
        Gains = GainsPD
    else:
        raise Exception('Need GainsPD or GainsGD.')
        
    print(f"Start {gainstr} optimisation with sample gains={Gains}")
    
    # SimpleIntegrator(init=True, GainPD=Gains[0], GainGD=0)
    
    
    VarOPD = np.zeros([Ngains,NIN])     # Phase variances
    VarCP = np.zeros([Ngains, NC])      # Cosure Phase variances
    FCArray = np.zeros([Ngains, NIN])   # Contains the fringe contrasts
    LockedRatio = np.zeros([Ngains, NIN])    #Locked ratio
    WLockedRatio = np.zeros([Ngains, NIN])   # Weigthed locked ratio
    
    minValue = 10000
    iOptim = -1
    EarlyStop=0
    NumberOfLoops = Ngains
    time0 = time.time() ; LoopNumber = 0
    for ig in range(Ngains):
        
        if (ig-iOptim>4) and (np.mean(FCArray[iOptim+1:ig,:]) > minValue):
            print("The higher gains won't do better. We stop the optimisation.")
            EarlyStop=ig
            break
        
        G = Gains[ig]
        
        config.FT[gainstr] = G
        
        print(f'{gainstr}={G}')
        
        if len(filedir):
            files = glob.glob(filedir+'*.fits')
            Nfiles = len(files) ; print(f"Found: {Nfiles} files")
        else:
            files = [config.DisturbanceFile]
            Nfiles = 1
            
        for ifile in range(Nfiles):
            DisturbanceFile = files[ifile]
            print(f'Reading file number {ifile+1} over {Nfiles}')
            
            sk.update_config(DisturbanceFile=DisturbanceFile, checkperiod=40)
        
            # Launch the simulator
            sk.loop()
            # Load the performance observables into simu module
            sk.ShowPerformance(TimeBonds, WavelengthOfInterest,DIT, display=False)
                
            
            # Initialise the comparison tables
            VarOPD[ig,:] += simu.VarOPD/Nfiles
            if gainstr=='GainPD':
                VarCP[ig,:] += simu.VarCPD/Nfiles
            elif gainstr=='GainGD':
                VarCP[ig,:] += simu.VarCGD/Nfiles
            FCArray[ig,:] += simu.FringeContrast[0]/Nfiles
            LockedRatio[ig,:] += simu.LockedRatio/Nfiles
            WLockedRatio[ig,:] += simu.WLockedRatio/Nfiles
        
        
        if optim=='opd':
            criteria = VarOPD
        elif optim=='FC':
            criteria = 1-FCArray
        elif optim == 'LockedRatio':
            criteria = 1-LockedRatio
        elif optim == 'WLockedRatio':
            criteria = 1-WLockedRatio
        elif optim == 'CP':
            if (not telescopes) and (len(telescopes) != 3):
                raise Exception('For defining a closure phase, telescopes must be three.')
            else:
                criteria = VarCP
                
                
        if not telescopes:
            Value = np.mean(criteria[ig,:])
        elif telescopes:
            itel1,itel2 = telescopes[0]-1, telescopes[1]-1
            ib = ct.posk(itel1, itel2, config.NA)
            Value = criteria[ig,ib]
        if Value < minValue:
            minValue = Value
            iOptim = ig
            
        
        # elif (Value > 10*minValue) and (ig < Ngains-1):
        #     VarOPD[ig+1:,:] = 100*np.ones([Ngains-ig-1,NIN])
        #     VarCP[ig+1:,:] = 100*np.ones([Ngains-ig-1,NC])
        #     FCArray[ig+1:,:] = 100*np.ones([Ngains-ig-1,NIN])
        #     break
    
        print(f'Current value={Value}, Minimal value={minValue} for Gain={Gains[iOptim]}')
        
        LoopNumber+=1
        Progress = LoopNumber/NumberOfLoops
        PassedTime = time.time() - time0
        RemainingTime = PassedTime/Progress - PassedTime
        
        
        print(f"\nProgression: {round(LoopNumber/NumberOfLoops*100)}% ({strtime(PassedTime)}) - \
Remains {strtime(RemainingTime)}")
        
    bestGain = Gains[iOptim]
    
    from tabulate import tabulate
    # ich = [12,13,14,15,16,23,24,25,26,34,35,36,45,46,56]
    ichint = [int(''.join([str(int(ic[0]+1)),str(int(ic[1]+1))])) for ic in config.ich] # Convert list of tuples into list of int
    
    criteriasBase = ["LR", "WLR", "FC", "VarOPD [µm]"]
    
    Ncb = len(criteriasBase)
    A=list(np.repeat(Gains, Ncb)) ; B = criteriasBase*Ngains
    
    base_3d = np.array([LockedRatio,WLockedRatio,FCArray,VarOPD])
    
    base_2d = base_3d.reshape([Ncb*Ngains,NIN], order='F').T  # The first index (criterias) changing fastest, then transpose for having baselines in rows
    
    # Need another dataframe only for Closure Phase (dimension 10)
    
    resultsBasedf = pd.DataFrame(data=base_2d, columns=[A,B], index=ichint)
    
    CPindexint = [int(''.join([str(int(cpindex[0]+1)),str(int(cpindex[1]+1)),str(int(cpindex[2]+1))])) for cpindex in config.CPindex]
    
    criteriasClosure = ["VarCP [µm]", "LR", "WLR", "SNR"]
    
    Ncc = len(criteriasClosure)
    A=list(np.repeat(Gains, Ncc)) ; B = criteriasClosure*Ngains
    
    # We only have VarCP so far so we populate the missing criteria with NaN values.
    CPLR = np.ones([Ngains,NC])*np.nan
    CPWLR = np.ones([Ngains,NC])*np.nan
    CPSNR = np.ones([Ngains,NC])*np.nan
    
    closure_3d = np.array([VarCP,CPLR,CPWLR,CPSNR])
    
    closure_2d = closure_3d.reshape([Ncc*Ngains,NC], order='F').T  # The first index (criterias) changing fastest, then transpose for having baselines in rows

    resultsClosuredf = pd.DataFrame(data=closure_2d, columns=[A,B],index=CPindexint)
    
    dico_results_base = {"Base": ichint, "LR":np.transpose(np.round(LockedRatio[iOptim]*100)),
                "WLR":np.transpose(np.round(WLockedRatio[iOptim]*100)),
                "FC":np.transpose(np.round(FCArray[iOptim]*100)),
                "Var [µm]":np.transpose(np.round(VarOPD[iOptim],2))}
    
    # dico_results_closure = {"Base": ichint, "LR":np.transpose(np.round(LockedRatio[iOptim]*100)),
    #             "WLR":np.transpose(np.round(WLockedRatio[iOptim]*100)),
    #             "FC":np.transpose(np.round(FCArray[iOptim]*100)),
    #             "Var [µm]":np.transpose(np.round(VarOPD[iOptim],2))}
    
    print(f"Best performances reached with gain={bestGain}")
    print(tabulate(dico_results_base, headers="keys"))
    
    print(f"Same for closure phases with gain={bestGain}")
    print(tabulate(resultsClosuredf[bestGain], headers="keys"))
    
    if len(GainsGD):
        config.FT['GainGD'] = bestGain
    elif len(GainsPD):
        config.FT['GainPD'] = bestGain
    
    
#     print("We launch again the simulation with the best gainsse gains on the last \
# disturbance file to show the results.")
    
#     sk.update_config(DisturbanceFile=DisturbanceFile,checkperiod=50)

#     # Launch the simulator
#     sk.loop()
#     sk.display('perftable',wl=WavelengthOfInterest)
    
#     print(f"Best gain: {Gains[iOptim]} \n\
# Average OPD Variance: {list(np.round(VarOPD[iOptim],2))} \n\
# Average CPD Variance: {VarCP[iOptim]} \n\
# Average Fringe Contrast: {1-FCArray[iOptim]} \n\
# Weighted Locked Ratio: {WLockedRatio[iOptim]} \n\
# Locked Ratio: {LockedRatio[iOptim]}")
    
    return bestGain, iOptim, resultsBasedf,  resultsClosuredf, EarlyStop



def OptimGain0(Ngd=40,GainsPD=np.arange(0,1,0.1), GainsGD=np.arange(0,1,0.1),display=False,base=0):
    
    from . import simu
    from . import config
    
    NIN = config.NIN
    
    
    # Create a disturbance pattern on the pupille 1
    # coher = sk.MakeAtmosphereCoherence(ampl=5, dist='randomatm')
    
    # from simu import CfObj, CfDisturbance, PistonDisturbance,OPDDisturbance
    # from simu import PistonDisturbance,PhotometryDisturbance
    from .simu import FreqSampling, DisturbancePSD
    

    """
    Optimization of the Group-Delay tracking
    """
    
    GainPD=0
    config.FT['GainPD'] = 0
    
    Ngains = len(GainsGD)
    rmsArrayGD = np.zeros([Ngains,NIN])
    
    print(f"Start GD gain optimisation with sample gains={GainsGD}")
    ig = 0
    minRMS = 100
    for ig in range(Ngains):
        
        reload(simu)
        simu.DisturbancePSD = DisturbancePSD
        simu.FreqSampling = FreqSampling
        
        GainGD = GainsGD[ig]
        sk.SimpleIntegrator(init=True, Ngd=Ngd, GainPD=GainPD, GainGD=GainGD)
        
        config.FT['GainGD'] = GainGD
        print(f'Gain GD={GainGD}')
        
        # Launch the simulator and save the data in the randomall.fits file
        sk.loop('randomall.fits')
        rmsOPD = calcRMS(50)
        rmsArrayGD[ig,:] = np.transpose(rmsOPD)
        
        print(f"RMS={np.mean(rmsOPD)}")
        
        if np.mean(rmsOPD) < minRMS:
            minRMS = np.mean(rmsOPD)
            print("Better correction found for GD={GainGD}")
        
        elif (np.mean(rmsOPD) > 10*minRMS) and (ig < Ngains-1):
            rmsArrayGD[ig+1:,:] = 100*np.ones([Ngains-ig-1,NIN])
            break
        
    ind = np.argmin(np.mean(rmsArrayGD,axis=1))
    bestGD = GainsGD[ind]
    
    print(f"The optimize gain is {GainsGD[ind]} with average OPD rms \
          {np.mean(rmsArrayGD[ind])}")
    
    GainGD = GainsGD[ind]
    
    print(f"Start PD gain optimisation with gain GD={GainGD} and sample \
gains={GainsGD}")
    ig = 0
    Ngains = len(GainsPD)
    rmsArrayPD = np.zeros([Ngains,NIN])
    minRMS = 100
    for ig in range(Ngains):
        
        reload(simu)
        simu.DisturbancePSD = DisturbancePSD
        simu.FreqSampling = FreqSampling
        
        GainPD = GainsPD[ig]
        config.FT['GainGD'] = GainGD
        config.FT['GainPD'] = GainPD
        sk.SimpleIntegrator(init=True, Ngd=Ngd, GainPD=GainPD, GainGD=GainGD)
        print(f'Gain PD={GainPD}')
        
        # Launch the simulator and save the data in the randomall.fits file
        sk.loop('randomall.fits')
        rmsOPD = calcRMS(50)
        rmsArrayPD[ig,:] = np.transpose(rmsOPD)
        
        print(f"RMS={np.mean(rmsOPD)}")
        if np.mean(rmsOPD) < minRMS:
            minRMS = np.mean(rmsOPD)
            print(f"Better correction found for PD={GainPD}")
        
        elif (np.mean(rmsOPD) > 5*minRMS) & (ig < Ngains-1):
            rmsArrayPD[ig+1:,:] = 100*np.ones([Ngains-ig-1,NIN])
            break
        
    ind = np.argmin(np.mean(rmsArrayPD,axis=1))
    bestPD = GainsPD[ind]
    
    print(f"The optimize GD gain is {bestGD}")
    print(f"The optimized gains are: GD={bestGD} and PD={bestPD} with average OPD rms \
          {np.mean(rmsArrayPD[ind])}")

        
    return rmsArrayGD, rmsArrayPD





def OptimGainsTogether_new(GainsPD=[],GainsGD=[],DITs=np.logspace(0,500,20), 
                           optimCriteria="FC",filedir='',Nsamples=5,
                           TimeBonds=100, WavelengthOfInterest=1.5,
                           telescopes=0, save_all='no',savepath='./',figsave='',
                           display=False,verbose=True,verbose2=False):
    """
    Estimates the best couple GD and PD gains after calculating the performance 
    (residual phase) of the servo loop on all the files contained in filedir.

    Parameters
    ----------
    GainsPD : TYPE, optional
        DESCRIPTION. The default is [].
    GainsGD : TYPE, optional
        DESCRIPTION. The default is [].
    optim : TYPE, optional
        DESCRIPTION. The default is 'opd'.
    filedir : TYPE, optional
        DESCRIPTION. The default is ''.
    TimeBonds : TYPE, optional
        DESCRIPTION. The default is 100.
    WavelengthOfInterest : TYPE, optional
        DESCRIPTION. The default is 1.5.
    DIT : TYPE, optional
        DESCRIPTION. The default is 50.
    telescopes : TYPE, optional
        DESCRIPTION. The default is 0.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    from . import simu
    from . import config
    import pandas as pd
    
    from .config import NIN, NC
        
    if not (len(GainsPD) and len(GainsPD)):
        raise Exception('Need GainsPD and GainsGD.')
    
    NgainsGD = len(GainsGD)
    NgainsPD = len(GainsPD)
    NDIT = len(DITs)
        
    if verbose2:
        print(f"Start optimisation with sample gains GD={GainsGD} and PD={GainsPD}")
    
    sk.update_config(checkperiod=110) # For not seeing the decount.
    
    VarOPD = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])         # Phase variances
    VarCP = np.zeros([NDIT,NgainsGD,NgainsPD,NC])           # Closure Phase variances
    SNR = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])            # Signal-to-noise ratio in the scientific instrument
    FCArray = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])        # Contains the fringe contrasts
    LockedRatio = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])    # Locked ratio
    WLockedRatio = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])   # Weigthed locked ratio
    LR2 = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])            # Locked ratio
    LR3 = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])            # Central fringe ratio
    
    minValue = 10000
    
    NumberOfLoops = NgainsGD*NgainsPD
    
    if len(filedir):
        files = [f.replace("\\","/") for f in glob.glob(filedir+'*.fits')]
        Nfiles = len(files)
        if verbose2:
            print(f"Found: {Nfiles} files")
        if Nsamples < Nfiles:
            Nfiles = Nsamples
        if verbose2:
            print(f"Take only the {Nsamples} first files")
    else:
        files = [config.DisturbanceFile]
        Nfiles = 1
    
    time0 = time.time() ; LoopNumber = 0
    iOptimGD=0; iOptimPD=0 ; IDs=[] ; ThresholdGDs=[]

    for ig in range(NgainsGD):
        Ggd = GainsGD[ig]    
        print(f"-----------Start optimising with gain GD={Ggd}------------")   
        
        for ip in range(NgainsPD):
            LoopNumber+=1
            igp = ig*NgainsPD + ip  # Position in the tables
            minindcurrentGD = ig*NgainsPD
            maxindcurrentGD = minindcurrentGD + NgainsPD
            
            Gpd = GainsPD[ip]
            
            config.FT['GainGD'] = Ggd
            config.FT['GainPD'] = Gpd
            
            if verbose:
                print("\n----------------------------------")
                print(f'## Gain GD={Ggd}; GainPD={Gpd} ##')
                
            for ifile in range(Nfiles):
                
                DisturbanceFile = files[ifile]
                if verbose2:
                    print(f'File {ifile+1}/{Nfiles}')
                
                sk.update_config(DisturbanceFile=DisturbanceFile)
            
                # Launch the simulator
                if save_all=='light':
                    sk.loop(savepath,LightSave=True,verbose=verbose)
                elif save_all=='yes':
                    sk.loop(savepath,LightSave=False,verbose=verbose)
                elif save_all=='no':
                    sk.loop(verbose=verbose)
                else:
                    raise Exception('save_all param must be "light", "yes" or "no".')
                    
                if len(figsave):
                    if isinstance(figsave,str):
                        sk.display(figsave,display=display,savedir=savepath,ext='pdf')
                    elif isinstance(figsave,list):
                        sk.display(*figsave,display=display,savedir=savepath,ext='pdf')
                        
                # Load the performance observables into simu module
                for idit in range(NDIT):
                    DIT=DITs[idit]
                    sk.ShowPerformance(TimeBonds, WavelengthOfInterest,DIT, display=False)
                    
                    # VarOPD, VarCP, SNRSI, FringeContrast, LockedRatio, WLockedRatio = perfs
                    
                    # Initialise the comparison tables
                    VarOPD[idit,ig,ip,:] += simu.VarOPD/Nfiles
                    VarCP[idit,ig,ip,:] += simu.VarCPD/Nfiles
                    # SNR[idit,ig,ip,:] += simu.SNRSI/Nfiles
                    FCArray[idit,ig,ip,:] += simu.FringeContrast[0]/Nfiles
                    LockedRatio[idit,ig,ip,:] += simu.LockedRatio/Nfiles
                    WLockedRatio[idit,ig,ip,:] += simu.WLockedRatio/Nfiles
                
                    LR2[idit,ig,ip,:] += simu.LR2/Nfiles # Doesn't depend on the integration time but need DIT dimension for dataframe
                    LR3[idit,ig,ip,:] += simu.LR3/Nfiles # Doesn't depend on the integration time but need DIT dimension for dataframe
            
            IDs.append(config.SimuTimeID)
            ThresholdGDs.append(np.min(config.FT['ThresholdGD']))
            
            if optimCriteria=="VarOPD":
                criteria = VarOPD
            elif optimCriteria=="FC":
                criteria = 1-FCArray
            elif optimCriteria == "LR":
                criteria = 1-LockedRatio
            elif optimCriteria == "LR2":
                criteria = 1-LR2
            elif optimCriteria == "WLR":
                criteria = 1-WLockedRatio
            elif optimCriteria == "VarCP":
                if (not telescopes) and (len(telescopes) != 3):
                    raise Exception('For defining a closure phase, telescopes must be three.')
                else:
                    criteria = VarCP
            else:
                raise Exception(f'The chosen criteria must be in this list: "LR", "LR2","WLR", "FC", "VarOPD", "VarCP"')
                    
                    
            if not telescopes:
                Value = np.max(np.mean(criteria[:,ig,ip,:], axis=1))   # Maximum (over DITs) of the averaged value (over baselines)
                bestDIT = DITs[np.argmax(np.mean(criteria[:,ig,ip,:], axis=1))]
            else:
                itel1,itel2 = telescopes[0]-1, telescopes[1]-1
                ib = ct.posk(itel1, itel2, config.NA)
                Value = np.max(criteria[:,ig,ip,ib])
                bestDIT = DITs[np.argmax(criteria[:,ig,ip,ib])]
            
            print("-------------------------------------------")
            print(f'\nComputed value={round(Value,5)}')
            if Value < minValue:    
                print(f"New value={round(Value,5)} lower than minValue={round(minValue,5)} obtained with (GD,PD)=({Ggd},{Gpd})")
                minValue = Value
                iOptim = igp
                iOptimGD = ig
                iOptimPD = ip
            else:
                print(f"We keep minimal value={round(minValue,5)} obtained with gains (GD,PD)=({GainsGD[iOptimGD]},{GainsPD[iOptimPD]})")

            Progress = LoopNumber/NumberOfLoops
            PassedTime = time.time() - time0
            RemainingTime = PassedTime/Progress - PassedTime
            
            
            print(f"Progression current optim: {round(LoopNumber/NumberOfLoops*100)}% ({strtime(PassedTime)}) - \
Remains {strtime(RemainingTime)}")
            print("-------------------------------------------\n")

    bestGains = GainsGD[iOptimGD], GainsPD[iOptimPD]
    
    
    from tabulate import tabulate
    # ich = [12,13,14,15,16,23,24,25,26,34,35,36,45,46,56]
    ichint = [int(''.join([str(int(ic[0]+1)),str(int(ic[1]+1))])) for ic in config.ich] # Convert list of tuples into list of int
    
    criteriasBase = ["LR", "LR2", "LR3", "WLR", "FC", "VarOPD [µm]"]
    
    Ncb = len(criteriasBase)
    # A=list(np.repeat(GainsGD, Ncb)) ; B = criteriasBase*Ngains
    
    A = list(np.repeat(DITs, NgainsGD*NgainsPD*Ncb))
    Btemp = list(np.repeat(GainsGD,Ncb*NgainsPD))
    B = Btemp * NDIT
    Ctemp = list(np.repeat(GainsPD,Ncb))
    C = Ctemp * NDIT * NgainsGD
    C2 = list(np.repeat(IDs,Ncb)) * NDIT            # ID of the last simu
    C3 = list(np.repeat(ThresholdGDs,Ncb)) * NDIT   # ThresholdGD
    D = criteriasBase * NDIT * NgainsGD * NgainsPD 
    
    base_5d = np.array([LockedRatio,LR2,LR3,WLockedRatio,FCArray,VarOPD])
    base_5d = np.transpose(base_5d, (0,3,2,1,4))      # Trick to get the levels DIT, GD and PD in this order
    
    base_2d = base_5d.reshape([NDIT*Ncb*NgainsGD*NgainsPD,NIN], order='F').T  # The first index (criterias) changing fastest, then transpose for having baselines in rows
    
    # Need another dataframe only for Closure Phase (dimension 10)
    
    resultsBasedf = pd.DataFrame(data=base_2d, columns=[A,B,C,C2,C3,D], index=ichint)
    # resultsBasedf['IDs'] = IDs
    # resultsBasedf['ThresholdGD_test'] = ThresholdGDs
    
    CPindexint = [int(''.join([str(int(cpindex[0]+1)),str(int(cpindex[1]+1)),str(int(cpindex[2]+1))])) for cpindex in config.CPindex]
    
    criteriasClosure = ["VarCP [µm]", "LR", "WLR", "SNR"]
    
    # Ncc = len(criteriasClosure)
    # A=list(np.repeat(GainsGD, NgainsPD*Ncc))
    # Btemp = list(np.repeat(GainsPD, Ncc))
    # B = Btemp * NgainsGD
    # C = criteriasClosure * NgainsGD * NgainsPD
    
    Ncc = len(criteriasClosure)
    A = list(np.repeat(DITs, NgainsGD*NgainsPD*Ncc))
    Btemp = list(np.repeat(GainsGD,Ncc*NgainsPD))
    B = Btemp * NDIT
    Ctemp = list(np.repeat(GainsPD,Ncc))
    C = Ctemp * NDIT * NgainsGD
    C2 = list(np.repeat(IDs,Ncc)) * NDIT
    C3 = list(np.repeat(ThresholdGDs,Ncc)) * NDIT
    D = criteriasClosure * NDIT * NgainsGD * NgainsPD 
    
    
    # We only have VarCP so far so we populate the missing criteria with NaN values.
    CPLR = np.ones([NDIT,NgainsGD, NgainsPD,NC])*np.nan
    CPWLR = np.ones([NDIT,NgainsGD,NgainsPD,NC])*np.nan
    CPSNR = np.ones([NDIT,NgainsGD,NgainsPD,NC])*np.nan
    
    closure_5d = np.array([VarCP,CPLR,CPWLR,CPSNR])
    closure_5d = np.transpose(closure_5d, (0,3,2,1,4))
    
    closure_2d = closure_5d.reshape([NDIT*Ncc*NgainsGD*NgainsPD,NC], order='F').T  # The first index (criterias) changing fastest, then transpose for having baselines in rows

    resultsClosuredf = pd.DataFrame(data=closure_2d, columns=[A,B,C,C2,C3,D],index=CPindexint)
    
    if not telescopes:
        Base_av = resultsBasedf.mean(axis=0).to_frame(name='Average')
        Closure_av = resultsClosuredf.mean(axis=0).to_frame(name='Average')
    else:
        itel1,itel2 = telescopes[0]-1, telescopes[1]-1
        ib = ct.posk(itel1, itel2, config.NA)
        Base_av = resultsBasedf.iloc[ib].to_frame(name=f"{telescopes[0]}{telescopes[1]}")
        Closure_av = resultsClosuredf.iloc[ib].to_frame(name=f"{telescopes[0]}{telescopes[1]}")
        
    if optimCriteria=="VarOPD":
        criteriaName = "VarOPD [µm]" 
    elif optimCriteria=="FC":
        criteriaName = "FC"
    elif optimCriteria == "LR":
        criteriaName = "LR"
    elif optimCriteria == "WLR":
        criteriaName = "WLR"
    elif optimCriteria == "VarCP":
        criteriaName = "VarCP [µm]"
    
    if criteriaName != "VarCP":
        bestCombi = Base_av.loc[(slice(None),slice(None),slice(None),
                                 slice(None),slice(None), 
                                 criteriaName)].idxmax(skipna=True)[0]
    else:
        bestCombi = Closure_av.loc[(slice(None),slice(None),slice(None)
                                    ,slice(None),slice(None), 
                                    criteriaName)].idxmax(skipna=True)[0]
        
    bestDIT, bestGainGD, bestGainPD = bestCombi[:3]

    print(f"Best performances reached with gains (GD,PD)={(bestGainGD, bestGainPD)} and DIT={bestDIT}ms")
    print(tabulate(resultsBasedf[bestCombi], headers="keys"))
    
    print(f"Same for closure phases with gain={bestGains}")
    print(tabulate(resultsClosuredf[bestCombi], headers="keys"))
   
    if display:
        print("We launch again the simulation with these gains on the last\
        disturbance file to show the results.")
        
        config.FT['GainGD'] = bestGainGD
        config.FT['GainPD'] = bestGainPD
        sk.update_config(DisturbanceFile=DisturbanceFile,checkperiod=40)
        
        # Launch the simulator
        sk.loop()
        sk.display(wl=WavelengthOfInterest)
    
    return bestCombi, resultsBasedf,  resultsClosuredf

a
def OptimGainsTogether_20220119(GainsPD=[],GainsGD=[],DITs=np.logspace(0,500,20), 
                           optimCriteria="FC",filedir='',Nsamples=5,
                           TimeBonds=100, WavelengthOfInterest=1.5,
                           telescopes=0, save_all='no',savepath='./',figsave='',
                           display=False,verbose=True,verbose2=False):
    """
    Estimates the best couple GD and PD gains after calculating the performance 
    (residual phase) of the servo loop on all the files contained in filedir.

    Parameters
    ----------
    GainsPD : TYPE, optional
        DESCRIPTION. The default is [].
    GainsGD : TYPE, optional
        DESCRIPTION. The default is [].
    optim : TYPE, optional
        DESCRIPTION. The default is 'opd'.
    filedir : TYPE, optional
        DESCRIPTION. The default is ''.
    TimeBonds : TYPE, optional
        DESCRIPTION. The default is 100.
    WavelengthOfInterest : TYPE, optional
        DESCRIPTION. The default is 1.5.
    DIT : TYPE, optional
        DESCRIPTION. The default is 50.
    telescopes : TYPE, optional
        DESCRIPTION. The default is 0.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    from . import simu
    from . import config
    import pandas as pd
    
    from .config import NIN, NC
        
    if not (len(GainsPD) and len(GainsPD)):
        raise Exception('Need GainsPD and GainsGD.')
    
    NgainsGD = len(GainsGD)
    NgainsPD = len(GainsPD)
    NDIT = len(DITs)
        
    if verbose2:
        print(f"Start optimisation with sample gains GD={GainsGD} and PD={GainsPD}")
    
    sk.update_config(checkperiod=110) # For not seeing the decount.
    
    VarOPD = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])         # Phase variances
    VarCP = np.zeros([NDIT,NgainsGD,NgainsPD,NC])           # Closure Phase variances
    SNR = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])            # Signal-to-noise ratio in the scientific instrument
    FCArray = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])        # Contains the fringe contrasts
    LockedRatio = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])    # Locked ratio
    WLockedRatio = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])   # Weigthed locked ratio
    LR2 = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])            # Locked ratio
    LR3 = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])            # Central fringe ratio
    
    minValue = 10000
    
    NumberOfLoops = NgainsGD*NgainsPD
    
    if len(filedir):
        files = [f.replace("\\","/") for f in glob.glob(filedir+'*.fits')]
        Nfiles = len(files)
        if verbose2:
            print(f"Found: {Nfiles} files")
        if Nsamples < Nfiles:
            Nfiles = Nsamples
        if verbose2:
            print(f"Take only the {Nsamples} first files")
    else:
        files = [config.DisturbanceFile]
        Nfiles = 1
    
    time0 = time.time() ; LoopNumber = 0
    iOptimGD=0; iOptimPD=0 ; IDs=[] ; ThresholdGDs=[]

    for ig in range(NgainsGD):
        Ggd = GainsGD[ig]    
        print(f"-----------Start optimising with gain GD={Ggd}------------")   
        
        for ip in range(NgainsPD):
            LoopNumber+=1
            igp = ig*NgainsPD + ip  # Position in the tables
            minindcurrentGD = ig*NgainsPD
            maxindcurrentGD = minindcurrentGD + NgainsPD
            
            Gpd = GainsPD[ip]
            
            config.FT['GainGD'] = Ggd
            config.FT['GainPD'] = Gpd
            
            if verbose:
                print("\n----------------------------------")
                print(f'## Gain GD={Ggd}; GainPD={Gpd} ##')
                
            for ifile in range(Nfiles):
                
                DisturbanceFile = files[ifile]
                if verbose2:
                    print(f'File {ifile+1}/{Nfiles}')
                
                sk.update_config(DisturbanceFile=DisturbanceFile)
            
                # Launch the simulator
                if save_all=='light':
                    sk.loop(savepath,LightSave=True,verbose=verbose)
                elif save_all=='yes':
                    sk.loop(savepath,LightSave=False,verbose=verbose)
                elif save_all=='no':
                    sk.loop(verbose=verbose)
                else:
                    raise Exception('save_all param must be "light", "yes" or "no".')
                    
                if len(figsave):
                    if isinstance(figsave,str):
                        sk.display(figsave,display=display,savedir=savepath,ext='pdf')
                    elif isinstance(figsave,list):
                        sk.display(*figsave,display=display,savedir=savepath,ext='pdf')
                        
                # Load the performance observables into simu module
                for idit in range(NDIT):
                    DIT=DITs[idit]
                    sk.ShowPerformance(TimeBonds, WavelengthOfInterest,DIT, display=False)
                    
                    # VarOPD, VarCP, SNRSI, FringeContrast, LockedRatio, WLockedRatio = perfs
                    
                    # Initialise the comparison tables
                    VarOPD[idit,ig,ip,:] += simu.VarOPD/Nfiles
                    VarCP[idit,ig,ip,:] += simu.VarCPD/Nfiles
                    # SNR[idit,ig,ip,:] += simu.SNRSI/Nfiles
                    FCArray[idit,ig,ip,:] += simu.FringeContrast[0]/Nfiles
                    LockedRatio[idit,ig,ip,:] += simu.LockedRatio/Nfiles
                    WLockedRatio[idit,ig,ip,:] += simu.WLockedRatio/Nfiles
                
                    LR2[idit,ig,ip,:] += simu.LR2/Nfiles # Doesn't depend on the integration time but need DIT dimension for dataframe
                    LR3[idit,ig,ip,:] += simu.LR3/Nfiles # Doesn't depend on the integration time but need DIT dimension for dataframe
            
            IDs.append(config.SimuTimeID)
            ThresholdGDs.append(np.min(config.FT['ThresholdGD']))
            
            if optimCriteria=="VarOPD":
                criteria = VarOPD
            elif optimCriteria=="FC":
                criteria = 1-FCArray
            elif optimCriteria == "LR":
                criteria = 1-LockedRatio
            elif optimCriteria == "LR2":
                criteria = 1-LR2
            elif optimCriteria == "WLR":
                criteria = 1-WLockedRatio
            elif optimCriteria == "VarCP":
                if (not telescopes) and (len(telescopes) != 3):
                    raise Exception('For defining a closure phase, telescopes must be three.')
                else:
                    criteria = VarCP
            else:
                raise Exception(f'The chosen criteria must be in this list: "LR", "LR2","WLR", "FC", "VarOPD", "VarCP"')
                    
                    
            if not telescopes:
                Value = np.max(np.mean(criteria[:,ig,ip,:], axis=1))   # Maximum (over DITs) of the averaged value (over baselines)
                bestDIT = DITs[np.argmax(np.mean(criteria[:,ig,ip,:], axis=1))]
            else:
                itel1,itel2 = telescopes[0]-1, telescopes[1]-1
                ib = ct.posk(itel1, itel2, config.NA)
                Value = np.max(criteria[:,ig,ip,ib])
                bestDIT = DITs[np.argmax(criteria[:,ig,ip,ib])]
            
            print("-------------------------------------------")
            print(f'\nComputed value={round(Value,5)}')
            if Value < minValue:    
                print(f"New value={round(Value,5)} lower than minValue={round(minValue,5)} obtained with (GD,PD)=({Ggd},{Gpd})")
                minValue = Value
                iOptim = igp
                iOptimGD = ig
                iOptimPD = ip
            else:
                print(f"We keep minimal value={round(minValue,5)} obtained with gains (GD,PD)=({GainsGD[iOptimGD]},{GainsPD[iOptimPD]})")

            Progress = LoopNumber/NumberOfLoops
            PassedTime = time.time() - time0
            RemainingTime = PassedTime/Progress - PassedTime
            
            
            print(f"Progression current optim: {round(LoopNumber/NumberOfLoops*100)}% ({strtime(PassedTime)}) - \
Remains {strtime(RemainingTime)}")
            print("-------------------------------------------\n")

    bestGains = GainsGD[iOptimGD], GainsPD[iOptimPD]
    
    
    from tabulate import tabulate
    # ich = [12,13,14,15,16,23,24,25,26,34,35,36,45,46,56]
    ichint = [int(''.join([str(int(ic[0]+1)),str(int(ic[1]+1))])) for ic in config.ich] # Convert list of tuples into list of int
    
    criteriasBase = ["LR", "LR2", "LR3", "WLR", "FC", "VarOPD [µm]"]
    
    Ncb = len(criteriasBase)
    # A=list(np.repeat(GainsGD, Ncb)) ; B = criteriasBase*Ngains
    
    A = list(np.repeat(DITs, NgainsGD*NgainsPD*Ncb))
    Btemp = list(np.repeat(GainsGD,Ncb*NgainsPD))
    B = Btemp * NDIT
    Ctemp = list(np.repeat(GainsPD,Ncb))
    C = Ctemp * NDIT * NgainsGD
    C2 = list(np.repeat(IDs,Ncb)) * NDIT            # ID of the last simu
    C3 = list(np.repeat(ThresholdGDs,Ncb)) * NDIT   # ThresholdGD
    D = criteriasBase * NDIT * NgainsGD * NgainsPD 
    
    base_5d = np.array([LockedRatio,LR2,LR3,WLockedRatio,FCArray,VarOPD])
    base_5d = np.transpose(base_5d, (0,3,2,1,4))      # Trick to get the levels DIT, GD and PD in this order
    
    base_2d = base_5d.reshape([NDIT*Ncb*NgainsGD*NgainsPD,NIN], order='F').T  # The first index (criterias) changing fastest, then transpose for having baselines in rows
    
    # Need another dataframe only for Closure Phase (dimension 10)
    
    resultsBasedf = pd.DataFrame(data=base_2d, columns=[A,B,C,C2,C3,D], index=ichint)
    # resultsBasedf['IDs'] = IDs
    # resultsBasedf['ThresholdGD_test'] = ThresholdGDs
    
    CPindexint = [int(''.join([str(int(cpindex[0]+1)),str(int(cpindex[1]+1)),str(int(cpindex[2]+1))])) for cpindex in config.CPindex]
    
    criteriasClosure = ["VarCP [µm]", "LR", "WLR", "SNR"]
    
    # Ncc = len(criteriasClosure)
    # A=list(np.repeat(GainsGD, NgainsPD*Ncc))
    # Btemp = list(np.repeat(GainsPD, Ncc))
    # B = Btemp * NgainsGD
    # C = criteriasClosure * NgainsGD * NgainsPD
    
    Ncc = len(criteriasClosure)
    A = list(np.repeat(DITs, NgainsGD*NgainsPD*Ncc))
    Btemp = list(np.repeat(GainsGD,Ncc*NgainsPD))
    B = Btemp * NDIT
    Ctemp = list(np.repeat(GainsPD,Ncc))
    C = Ctemp * NDIT * NgainsGD
    C2 = list(np.repeat(IDs,Ncc)) * NDIT
    C3 = list(np.repeat(ThresholdGDs,Ncc)) * NDIT
    D = criteriasClosure * NDIT * NgainsGD * NgainsPD 
    
    
    # We only have VarCP so far so we populate the missing criteria with NaN values.
    CPLR = np.ones([NDIT,NgainsGD, NgainsPD,NC])*np.nan
    CPWLR = np.ones([NDIT,NgainsGD,NgainsPD,NC])*np.nan
    CPSNR = np.ones([NDIT,NgainsGD,NgainsPD,NC])*np.nan
    
    closure_5d = np.array([VarCP,CPLR,CPWLR,CPSNR])
    closure_5d = np.transpose(closure_5d, (0,3,2,1,4))
    
    closure_2d = closure_5d.reshape([NDIT*Ncc*NgainsGD*NgainsPD,NC], order='F').T  # The first index (criterias) changing fastest, then transpose for having baselines in rows

    resultsClosuredf = pd.DataFrame(data=closure_2d, columns=[A,B,C,C2,C3,D],index=CPindexint)
    
    if not telescopes:
        Base_av = resultsBasedf.mean(axis=0).to_frame(name='Average')
        Closure_av = resultsClosuredf.mean(axis=0).to_frame(name='Average')
    else:
        itel1,itel2 = telescopes[0]-1, telescopes[1]-1
        ib = ct.posk(itel1, itel2, config.NA)
        Base_av = resultsBasedf.iloc[ib].to_frame(name=f"{telescopes[0]}{telescopes[1]}")
        Closure_av = resultsClosuredf.iloc[ib].to_frame(name=f"{telescopes[0]}{telescopes[1]}")
        
    if optimCriteria=="VarOPD":
        criteriaName = "VarOPD [µm]" 
    elif optimCriteria=="FC":
        criteriaName = "FC"
    elif optimCriteria == "LR":
        criteriaName = "LR"
    elif optimCriteria == "WLR":
        criteriaName = "WLR"
    elif optimCriteria == "VarCP":
        criteriaName = "VarCP [µm]"
    
    if criteriaName != "VarCP":
        bestCombi = Base_av.loc[(slice(None),slice(None),slice(None),
                                 slice(None),slice(None), 
                                 criteriaName)].idxmax(skipna=True)[0]
    else:
        bestCombi = Closure_av.loc[(slice(None),slice(None),slice(None)
                                    ,slice(None),slice(None), 
                                    criteriaName)].idxmax(skipna=True)[0]
        
    bestDIT, bestGainGD, bestGainPD = bestCombi[:3]

    print(f"Best performances reached with gains (GD,PD)={(bestGainGD, bestGainPD)} and DIT={bestDIT}ms")
    print(tabulate(resultsBasedf[bestCombi], headers="keys"))
    
    print(f"Same for closure phases with gain={bestGains}")
    print(tabulate(resultsClosuredf[bestCombi], headers="keys"))
   
    if display:
        print("We launch again the simulation with these gains on the last\
        disturbance file to show the results.")
        
        config.FT['GainGD'] = bestGainGD
        config.FT['GainPD'] = bestGainPD
        sk.update_config(DisturbanceFile=DisturbanceFile,checkperiod=40)
        
        # Launch the simulator
        sk.loop()
        sk.display(wl=WavelengthOfInterest)
    
    return bestCombi, resultsBasedf,  resultsClosuredf





def calcRMS(startframe):
    """
    Calculates the Standard Deviation of the OPD in micrometers.
        

    Parameters
    ----------
    startframe : integer
        Starting time in the loop. An integer, not in ms for the moment.

    Returns
    -------
    varOPD : float
        .
    varOGD : TYPE
        DESCRIPTION.

    """
    from .simu import OPDTrue
    
    rmsOPD = np.std(OPDTrue[startframe:,:],axis=0)
    
    return rmsOPD


def strtime(time_to_write):
    """
    Write a a time given in second on the format ##h##m##s

    Parameters
    ----------
    time_to_write : INT
        SECONDS.

    Returns
    -------
    str
        ##h##m##s.

    """
    
    return f"{int(time_to_write//3600)}h{int(time_to_write%3600/60)}m{int((time_to_write%3600)%60)}s"