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
#                 TimeBonds=100, WLOfScience=1.5, DIT=50,
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
    #     sk.ShowPerformance(TimeBonds, WLOfScience,DIT, display=False)
                
        
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

def OptimGainsTogether(GainsPD=[],GainsGD=[],DITs=np.logspace(0,500,20), 
                        optimCriteria="FC",filedir='',Nsamples=5,
                        TimeBonds=100, WLOfTrack=1.5,WLOfScience=0.75,
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
    WLOfScience : TYPE, optional
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
    
    from .config import NA, NIN, NC
        
    if not (len(GainsPD) and len(GainsPD)):
        raise Exception('Need GainsPD and GainsGD.')
    
    NgainsGD = len(GainsGD)
    NgainsPD = len(GainsPD)
    NDIT = len(DITs)
        
    if verbose2:
        print(f"Start optimisation with sample gains GD={GainsGD} and PD={GainsPD}")
    
    sk.update_config(checkperiod=110) # For not seeing the decount.
    
    VarOPD = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])         # Phase variances
    VarGDRes = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])       # GD Phase variances
    VarPDRes = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])       # PD Phase variances
    VarPiston = np.zeros([NDIT,NgainsGD,NgainsPD,NA])       # Piston variance
    VarPistonGD = np.zeros([NDIT,NgainsGD,NgainsPD,NA])     # Piston GD variance
    VarPistonPD = np.zeros([NDIT,NgainsGD,NgainsPD,NA])     # Piston PD variance
    VarCP = np.zeros([NDIT,NgainsGD,NgainsPD,NC])           # Closure Phase variances
    SNR = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])            # Signal-to-noise ratio in the scientific instrument
    FCArray = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])        # Contains the fringe contrasts
    LockedRatio = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])    # Locked ratio
    WLockedRatio = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])   # Weigthed locked ratio
    LR2 = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])            # Locked ratio
    LR3 = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])            # Central fringe ratio
    
    Vmod = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])
    Vangle = np.zeros([NDIT,NgainsGD,NgainsPD,NIN])
    
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
    iOptimGD=0; iOptimPD=0 ; IDs=[] ; ThresholdGDmins=[] ; ThresholdGDmaxs=[]

    indWLOfTrack = np.argmin(np.abs(config.spectra-WLOfTrack))
    
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
                    if figsave != 'onlyperf':
                        if isinstance(figsave,str):
                            sk.display(figsave,display=display,savedir=savepath,ext='pdf')
                        elif isinstance(figsave,list):
                            if 'perfarray' in figsave:
                                figsave.remove('perfarray')
                            sk.display(*figsave,display=display,savedir=savepath,ext='pdf')
                        
                # Load the performance observables into simu module
                for idit in range(NDIT):
                    DIT=DITs[idit]
                    sk.ShowPerformance(TimeBonds, WLOfScience,DIT, display=False)
                    
                    # VarOPD, VarCP, SNRSI, FringeContrast, LockedRatio, WLockedRatio = perfs
                    
                    # Initialise the comparison tables
                    VarOPD[idit,ig,ip,:] += simu.VarOPD/Nfiles
                    VarCP[idit,ig,ip,:] += simu.VarCPD/Nfiles
                    VarGDRes[idit,ig,ip,:] += simu.VarGDRes/Nfiles
                    VarPDRes[idit,ig,ip,:] += simu.VarPDRes/Nfiles
                    VarPiston[idit,ig,ip,:] += simu.VarPiston/Nfiles
                    VarPistonGD[idit,ig,ip,:] += simu.VarPistonGD/Nfiles
                    VarPistonPD[idit,ig,ip,:] += simu.VarPistonPD/Nfiles
                    # SNR[idit,ig,ip,:] += simu.SNRSI/Nfiles
                    FCArray[idit,ig,ip,:] += simu.FringeContrast[0]/Nfiles
                    LockedRatio[idit,ig,ip,:] += simu.LockedRatio/Nfiles
                    WLockedRatio[idit,ig,ip,:] += simu.WLockedRatio/Nfiles
                
                    LR2[idit,ig,ip,:] += simu.LR2/Nfiles # Doesn't depend on the integration time but need DIT dimension for dataframe
                    LR3[idit,ig,ip,:] += simu.LR3/Nfiles # Doesn't depend on the integration time but need DIT dimension for dataframe
            
                    Vmod[idit,ig,ip,:] = ct.NB2NIN(np.abs(simu.VisibilityObject[indWLOfTrack]))
                    Vangle[idit,ig,ip,:] = ct.NB2NIN(np.angle(simu.VisibilityObject[indWLOfTrack]))
            
            IDs.append(config.SimuTimeID)
            ThresholdGDmins.append(np.min(config.FT['ThresholdGD']))
            ThresholdGDmaxs.append(np.max(config.FT['ThresholdGD']))
            
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
    telint = np.arange(1,NA+1)
    criteriasBase = ["LR", "LR2", "LR3", "WLR", "FC", "VarOPD [µm]",
                     "VarGDRes","VarPDRes",
                     'Vmod','Vangle']
    
    Ncb = len(criteriasBase)
    # A=list(np.repeat(GainsGD, Ncb)) ; B = criteriasBase*Ngains
    
    A = list(np.repeat(DITs, NgainsGD*NgainsPD*Ncb))
    Btemp = list(np.repeat(GainsGD,Ncb*NgainsPD))
    B = Btemp * NDIT
    Ctemp = list(np.repeat(GainsPD,Ncb))
    C = Ctemp * NDIT * NgainsGD
    C2 = list(np.repeat(IDs,Ncb)) * NDIT            # ID of the last simu
    C3 = list(np.repeat(ThresholdGDmins,Ncb)) * NDIT   # ThresholdGD
    C4 = list(np.repeat(ThresholdGDmaxs,Ncb)) * NDIT   # ThresholdGD
    D = criteriasBase * NDIT * NgainsGD * NgainsPD 
    
    base_5d = np.array([LockedRatio,LR2,LR3,WLockedRatio,FCArray,VarOPD,
                        VarGDRes,VarPDRes,
                        Vmod,Vangle])
    base_5d = np.transpose(base_5d, (0,3,2,1,4))      # Trick to get the levels DIT, GD and PD in this order
    
    base_2d = base_5d.reshape([NDIT*Ncb*NgainsGD*NgainsPD,NIN], order='F').T  # The first index (criterias) changing fastest, then transpose for having baselines in rows
    
    # Need another dataframe only for Closure Phase (dimension 10)
    
    resultsBasedf = pd.DataFrame(data=base_2d, columns=[A,B,C,C2,C3,C4,D], index=ichint)
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
    C3 = list(np.repeat(ThresholdGDmins,Ncc)) * NDIT
    C4 = list(np.repeat(ThresholdGDmaxs,Ncc)) * NDIT
    D = criteriasClosure * NDIT * NgainsGD * NgainsPD 
    
    
    # We only have VarCP so far so we populate the missing criteria with NaN values.
    CPLR = np.ones([NDIT,NgainsGD, NgainsPD,NC])*np.nan
    CPWLR = np.ones([NDIT,NgainsGD,NgainsPD,NC])*np.nan
    CPSNR = np.ones([NDIT,NgainsGD,NgainsPD,NC])*np.nan
    
    closure_5d = np.array([VarCP,CPLR,CPWLR,CPSNR])
    closure_5d = np.transpose(closure_5d, (0,3,2,1,4))
    
    closure_2d = closure_5d.reshape([NDIT*Ncc*NgainsGD*NgainsPD,NC], order='F').T  # The first index (criterias) changing fastest, then transpose for having baselines in rows

    resultsClosuredf = pd.DataFrame(data=closure_2d, columns=[A,B,C,C2,C3,C4,D],index=CPindexint)
    
    """
    Residues on Telescopes
    """
    criteriasTel = ["VarPiston","VarPistonGD","VarPistonPD"]
    
    Nct = len(criteriasTel)
    # A=list(np.repeat(GainsGD, Ncb)) ; B = criteriasBase*Ngains
    
    A = list(np.repeat(DITs, NgainsGD*NgainsPD*Nct))
    Btemp = list(np.repeat(GainsGD,Nct*NgainsPD))
    B = Btemp * NDIT
    Ctemp = list(np.repeat(GainsPD,Nct))
    C = Ctemp * NDIT * NgainsGD
    C2 = list(np.repeat(IDs,Nct)) * NDIT            # ID of the last simu
    C3 = list(np.repeat(ThresholdGDmins,Nct)) * NDIT   # ThresholdGD
    C4 = list(np.repeat(ThresholdGDmaxs,Nct)) * NDIT   # ThresholdGD
    D = criteriasTel * NDIT * NgainsGD * NgainsPD 
    
    tel_5d = np.array([VarPiston,VarPistonGD,VarPistonPD])
    tel_5d = np.transpose(tel_5d, (0,3,2,1,4))      # Trick to get the levels DIT, GD and PD in this order
    
    tel_2d = tel_5d.reshape([NDIT*Nct*NgainsGD*NgainsPD,NA], order='F').T  # The first index (criterias) changing fastest, then transpose for having baselines in rows
    
    # Need another dataframe only for Closure Phase (dimension 10)
    
    resultsTeldf = pd.DataFrame(data=tel_2d, columns=[A,B,C,C2,C3,C4,D], index=telint)
    
    
    if not telescopes:
        Base_av = resultsBasedf.mean(axis=0).to_frame(name='Average')
        Closure_av = resultsClosuredf.mean(axis=0).to_frame(name='Average')
        Tel_av = resultsBasedf.mean(axis=0).to_frame(name='Average')
    else:
        itel1,itel2 = telescopes[0]-1, telescopes[1]-1
        ib = ct.posk(itel1, itel2, config.NA)
        ia = telescopes[0]-1  # histoire de mettre un truc, mais à modifier si besoin
        Base_av = resultsBasedf.iloc[ib].to_frame(name=f"{telescopes[0]}{telescopes[1]}")
        Closure_av = resultsClosuredf.iloc[ib].to_frame(name=f"{telescopes[0]}{telescopes[1]}")
        Tel_av = resultsTeldf.iloc[ia].to_frame(name=f"{telescopes[0]}")
        
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
    elif optimCriteria == "VarPiston":
        criteriaName = "VarPiston"
    
    if criteriaName not in ("VarCP","VarPiston"):
        bestCombi = Base_av.loc[(slice(None),slice(None),slice(None),
                                 slice(None),slice(None),slice(None), 
                                 criteriaName)].idxmax(skipna=True)[0]
    elif criteriaName == 'VarPiston':
        bestCombi = Tel_av.loc[(slice(None),slice(None),slice(None),
                                 slice(None),slice(None),slice(None), 
                                 criteriaName)].idxmax(skipna=True)[0]
    else:
        bestCombi = Closure_av.loc[(slice(None),slice(None),slice(None)
                                    ,slice(None),slice(None),slice(None),
                                    criteriaName)].idxmax(skipna=True)[0]
        
    bestDIT, bestGainGD, bestGainPD = bestCombi[:3]

    print(f"Best performances OPD reached with gains (GD,PD)={(bestGainGD, bestGainPD)} and DIT={bestDIT}ms")
    print(tabulate(resultsBasedf[bestCombi], headers="keys"))
    
    print(f"Same for closure phases with gain={bestGains}")
    print(tabulate(resultsClosuredf[bestCombi], headers="keys"))
    
    print(f"Same for telescopes with gain={bestGains}")
    print(tabulate(resultsTeldf[bestCombi], headers="keys"))
   
    if display or len(figsave):
        print("We launch again the simulation with these gains on the last\
        disturbance file to show the results.")
        
        config.FT['GainGD'] = bestGainGD
        config.FT['GainPD'] = bestGainPD
        sk.update_config(DisturbanceFile=DisturbanceFile,checkperiod=40)
        
        # Launch the simulator
        sk.loop()
        sk.display('perfarray',WLOfScience=WLOfScience,display=display,savedir=savepath,ext='pdf')
    
    return bestCombi, resultsBasedf,  resultsClosuredf, resultsTeldf




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