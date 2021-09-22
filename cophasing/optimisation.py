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

def OptimGainMultiprocess(GainsPD=[],GainsGD=[],optim='FC',
                TimeBonds=100, WavelengthOfInterest=1.5, DIT=50,
                telescopes=0):
    
    
    import multiprocessing
    
    pool = multiprocessing.Pool()
    
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
    
    
    minValue = 10000
    iOptim = -1
        
    
    def GiveResults(G):
        
        config.FT['GainPD'] = G
        
        print(f'{gainstr}={G}')
        
        # Launch the simulator
        sk.loop()
        # Load the performance observables into simu module
        sk.ShowPerformance(TimeBonds, WavelengthOfInterest,DIT, display=False)
                
        
        return simu.VarOPD, simu.VarCPD, simu.VarCGD, simu.FringeContrast
        
       
        # elif (Value > 10*minValue) and (ig < Ngains-1):
        #     VarOPD[ig+1:,:] = 100*np.ones([Ngains-ig-1,NIN])
        #     VarCP[ig+1:,:] = 100*np.ones([Ngains-ig-1,NC])
        #     FCArray[ig+1:,:] = 100*np.ones([Ngains-ig-1,NIN])
        #     break
    
    
    # result = GiveResults(0.2)
    myresult = pool.starmap(GiveResults, Gains.reshape([Ngains,1]))
    
    print(myresult)
    
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
    


    return #bestGain, iOptim, VarOPD, VarCP, FCArray


def OptimGain(GainsPD=[],GainsGD=[],optim='FC',filedir='',
                TimeBonds=100, WavelengthOfInterest=1.5, DIT=50,
                telescopes=0):
    
    from . import simu
    from . import config
    
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
            criteria = 1/FCArray
        elif optim == 'LockedRatio':
            criteria = 1/LockedRatio
        elif optim == 'WLockedRatio':
            criteria = 1/WLockedRatio
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
        
    bestGain = Gains[iOptim]
    
    from tabulate import tabulate
    ich = [12,13,14,15,16,23,24,25,26,34,35,36,45,46,56]
    
    dico_results = {"Base": ich, "LR [%]":np.transpose(np.round(LockedRatio[iOptim]*100)),
                "WLR [%]":np.transpose(np.round(WLockedRatio[iOptim]*100)),
                "FC [%]":np.transpose(np.round(FCArray[iOptim]*100)),
                "Var [µm]":np.transpose(np.round(VarOPD[iOptim],2))}
    print(f"Best performances reached with gain={bestGain}")
    print(tabulate(dico_results, headers="keys"))
    
    if len(GainsGD):
        config.FT['GainGD'] = bestGain
    elif len(GainsPD):
        config.FT['GainPD'] = bestGain
    
    
    print("We launch again the simulation with the best gainsse gains on the last \
disturbance file to show the results.")
    
    sk.update_config(DisturbanceFile=DisturbanceFile,checkperiod=50)

    # Launch the simulator
    sk.loop()
    sk.display('perftable',wl=WavelengthOfInterest)
    
#     print(f"Best gain: {Gains[iOptim]} \n\
# Average OPD Variance: {list(np.round(VarOPD[iOptim],2))} \n\
# Average CPD Variance: {VarCP[iOptim]} \n\
# Average Fringe Contrast: {1-FCArray[iOptim]} \n\
# Weighted Locked Ratio: {WLockedRatio[iOptim]} \n\
# Locked Ratio: {LockedRatio[iOptim]}")
    
    return bestGain, iOptim, WLockedRatio, LockedRatio, VarOPD, VarCP, FCArray, EarlyStop



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
    
    # GainPD_ = GainPD[ind]
    # for ig in range(Ngd):
    #     GainGD_ = GainGD[ig]
    #     print(f'Gain PD: {GainPD_}')
    #     print(f'Gain GD: {GainGD_}')
    #     # Launch the simulator and save the data in the randomall.fits file
    #     coh_turn_spica('randomall.fits')
    #     rms2, rms2bis = calcRMS(110)
    #     rmsOGD[ig] = np.transpose(rms2)
    #     # rmsOGDbis[ig] = np.transpose(rms2bis)
        
    # if display:
    #     plt.figure(config.newfig), plt.title('OPD rms function of PD gain\
    #                                          on most critical baseline.')
    #     plt.plot(GainPD, rmsOPD[:,base],'+')
    #     plt.ylabel('OPD rms [µm]')
    #     plt.xlabel('Gain')
    #     plt.show()
    #     plt.savefig('varOPD.png')
    #     config.newfig+=1
        
    #     plt.figure(config.newfig), plt.title('OPD rms function of GD gain.')
    #     plt.plot(GainGD, rmsOGD[:,base],'+')
    #     plt.ylabel('OPD rms [µm]')
    #     plt.xlabel('Gain')
    #     plt.show()
    #     plt.savefig('varOGD.png')
    #     config.newfig+=1
        
    return rmsArrayGD, rmsArrayPD





def OptimGainsTogether(GainsPD=[],GainsGD=[],optim='opd',filedir='',
                TimeBonds=100, WavelengthOfInterest=1.5, DIT=50,
                telescopes=0):
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
    
    from .config import NIN, NC
        
    if not (len(GainsPD) and len(GainsPD)):
        raise Exception('Need GainsPD and GainsGD.')
    
    NgainsGD = len(GainsGD)
    NgainsPD = len(GainsPD)
        
    print(f"Start optimisation with sample gains GD={GainsGD} and PD={GainsPD}")
    
    VarOPD = np.zeros([NgainsGD*NgainsPD,NIN])     # Phase variances
    VarCP = np.zeros([NgainsGD*NgainsPD, NC])      # Cosure Phase variances
    FCArray = np.zeros([NgainsGD*NgainsPD, NIN])   # Contains the fringe contrasts
    LockedRatio = np.zeros([NgainsGD*NgainsPD, NIN])    #Locked ratio
    WLockedRatio = np.zeros([NgainsGD*NgainsPD, NIN])   # Weigthed locked ratio
    
    minValue = 10000
    
    NumberOfLoops = NgainsGD*NgainsPD
    
    EarlyStop=0
    time1 = 0 ; time0 = time.time() ; LoopNumber = 0
    for ig in range(NgainsGD):
        lasttime=time1
        time1=time.time()
        print(f"--Time for last gain GD: {round(time1-lasttime,2)}s--")  
        # if (ig-iOptimGD>4) and (np.mean(FCArray[iOptimGD+1:ig,:]) > minValue):
        #         print("The higher gains won't do better. We stop the optimisation.")
        #         EarlyStop=ig
        #         break
        Ggd = GainsGD[ig]    
        print(f"-----------Start optimising with gain GD={Ggd}------------")   
        
        
        for ip in range(NgainsPD):
            LoopNumber+=1
            igp = ig*NgainsPD + ip  # Position in the tables
            minindcurrentGD = ig*NgainsPD
            maxindcurrentGD = minindcurrentGD + NgainsPD
            # if (ip-iOptimPD>4) and (np.mean(FCArray[iOptimPD+1:ig,:]) > minValue):
            #     print("The higher gains won't do better. We stop the optimisation.")
            #     EarlyStop=ig
            #     break
            
            Gpd = GainsPD[ip]
            
            config.FT['GainGD'] = Ggd
            config.FT['GainPD'] = Gpd
            
            print(f'Gain GD={Ggd}; GainPD={Gpd}')
            
            if len(filedir):
                files = glob.glob(filedir+'*.fits')
                Nfiles = len(files) ; print(f"Found: {Nfiles} files")
            else:
                files = [config.DisturbanceFile]
                Nfiles = 1
                
            for ifile in range(Nfiles):
                
                DisturbanceFile = files[ifile]
                print(f'Reading file number {ifile+1} over {Nfiles}')
                
                sk.update_config(DisturbanceFile=DisturbanceFile,checkperiod=40)
            
                # Launch the simulator
                sk.loop()
                # Load the performance observables into simu module
                sk.ShowPerformance(TimeBonds, WavelengthOfInterest,DIT, display=False)
                    
                # Initialise the comparison tables
                VarOPD[igp,:] += simu.VarOPD/Nfiles
                VarCP[igp,:] += simu.VarCPD/Nfiles
                VarCP[igp,:] += simu.VarCGD/Nfiles
                FCArray[igp,:] += simu.FringeContrast[0]/Nfiles
                LockedRatio[igp,:] += simu.LockedRatio/Nfiles
                WLockedRatio[igp,:] += simu.WLockedRatio/Nfiles
            
            
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
                Value = np.mean(criteria[igp,:])
            else:
                itel1,itel2 = telescopes[0]-1, telescopes[1]-1
                ib = ct.posk(itel1, itel2, config.NA)
                Value = criteria[igp,ib]
            
            if Value < minValue:    
                    minValue = Value
                    iOptim = igp
                    iOptimGD = ig
                    iOptimPD = ip
            
            # elif (Value > 10*minValue):
            #     if (ip < NgainsPD-1):
            #         # We put the NgainsPD-ip-1 next elements to 100.
            #         Nelements = NgainsPD-ip-1
            #         VarOPD[igp+1:igp+1+Nelements,:] = 100*np.ones([Nelements,NIN])
            #         VarCP[igp+1:igp+1+Nelements,:] = 100*np.ones([Nelements,NC])
            #         FCArray[igp+1:igp+1+Nelements,:] = 100*np.ones([Nelements,NIN])
            #         LockedRatio[igp+1:igp+1+Nelements,:] = 100*np.ones([Nelements,NIN])
            #         WLockedRatio[igp+1:igp+1+Nelements,:] = 100*np.ones([Nelements,NIN])
            #         break
                
            # if optim=='opd':      # Optimisation on OPD residues
            #     if not telescopes:
            #         Value = np.mean(VarOPD[igp,:])
            #     elif telescopes:
            #         itel1,itel2 = telescopes[0]-1, telescopes[1]-1
            #         ib = ct.posk(itel1, itel2, config.NA)
            #         Value = VarOPD[igp,ib]
            #     if Value < minValue:    
            #         minValue = Value
            #         iOptim = igp
            #         iOptimGD = ig
            #         iOptimPD = ip
            
            # elif optim == 'FC':     # Optimisation on final fringe contrast
            #     if not telescopes:
            #         Value = 1-np.mean(FCArray[igp,:])
            #     elif telescopes:
            #         itel1,itel2 = telescopes[0]-1, telescopes[1]-1
            #         ib = ct.posk(itel1, itel2, config.NA)
            #         Value = 1-FCArray[igp,ib]
            #     if Value < minValue:
            #         minValue = Value
            #         iOptim = igp
            #         iOptimGD = ig
            #         iOptimPD = ip
                    
            # elif optim=='CP':       # Optimisation on Closure Phase variance
            #     if not telescopes:
            #         Value = np.mean(VarCP[igp,:])
            #     elif telescopes:
            #         if len(telescopes) != 3:
            #             raise Exception('For defining a closure phase, telescopes must be three.')
            #         itel1,itel2,itel3 = telescopes[0]-1, telescopes[1]-1, telescopes[2]-1
            #         ic = ct.poskfai(itel1, itel2, itel3, config.NA)
            #         Value = VarCP[igp,ic]
            #     if Value < minValue:    
            #         minValue = Value
            #         iOptim = igp
            #         iOptimGD = ig
            #         iOptimPD = ip
                    
            # elif optim == 'LockedRatio':     # Optimisation on final fringe contrast
            #     if not telescopes:
            #         Value = np.mean(LockedRatio[igp,:])
            #     elif telescopes:
            #         itel1,itel2 = telescopes[0]-1, telescopes[1]-1
            #         ib = ct.posk(itel1, itel2, config.NA)
            #         Value = LockedRatio[igp,ib]
            #     if Value < minValue:
            #         minValue = Value
            #         iOptim = igp
            #         iOptimGD = ig
            #         iOptimPD = ip
                    
            # elif optim == 'WLockedRatio':     # Optimisation on final fringe contrast
            #     if not telescopes:
            #         Value = np.mean(WLockedRatio[igp,:])
            #     elif telescopes:
            #         itel1,itel2 = telescopes[0]-1, telescopes[1]-1
            #         ib = ct.posk(itel1, itel2, config.NA)
            #         Value = WLockedRatio[igp,ib]
            #     if Value < minValue:
            #         minValue = Value
            #         iOptim = igp
            #         iOptimGD = ig
            #         iOptimPD = ip
            
        #     elif (Value > 10*minValue) and (ip < NgainsPD-1):
        #         VarOPD[ip+1:,:] = 100*np.ones([NgainsPD-ip-1,NIN])
        #         VarCP[ip+1:,:] = 100*np.ones([NgainsPD-ip-1,NC])
        #         FCArray[ip+1:,:] = 100*np.ones([NgainsPD-ip-1,NIN])
        #         break
        # elif (Value > 10*minValue) and (ip < NgainsGD-1):
        #         VarOPD[ig+1:,:] = 100*np.ones([NgainsGD-ig-1,NIN])
        #         VarCP[ig+1:,:] = 100*np.ones([NgainsGD-ig-1,NC])
        #         FCArray[ig+1:,:] = 100*np.ones([NgainsGD-ig-1,NIN])
        #         break
            
            Progress = LoopNumber/NumberOfLoops
            PassedTime = time.time() - time0
            RemainingTime = PassedTime/Progress - PassedTime
            
            print(f'Gains (GD,PD)=({Ggd},{Gpd}) give value={round(Value,5)}')
            print(f"Minimal value={round(minValue,5)} with gains (GD,PD)=({GainsGD[iOptimGD]},{GainsPD[iOptimPD]})")
            print(f"Progression: {round(LoopNumber/NumberOfLoops*100)}% ({strtime(PassedTime)})")
            print(f"Remaining time: {strtime(RemainingTime)}")

        
    bestGains = GainsGD[iOptimGD], GainsPD[iOptimPD]
    
    
    from tabulate import tabulate
    ich = [12,13,14,15,16,23,24,25,26,34,35,36,45,46,56]
    
    dico_results = {"Base": ich, "LR [%]":np.transpose(np.round(LockedRatio[iOptim]*100)),
                "WLR [%]":np.transpose(np.round(WLockedRatio[iOptim]*100)),
                "FC [%]":np.transpose(np.round(FCArray[iOptim]*100)),
                "Var [µm]":np.transpose(np.round(VarOPD[iOptim],2))}
    print(f"Best performances reached with gains (GD,PD)={bestGains}")
    print(tabulate(dico_results, headers="keys"))
    
    
    config.FT['GainGD'] = bestGains[0]
    config.FT['GainPD'] = bestGains[1]
    
    print("We launch again the simulation with these gains on the last\
disturbance file to show the results.")
    
    sk.update_config(DisturbanceFile=DisturbanceFile,checkperiod=40)

    # Launch the simulator
    sk.loop()
    sk.display(wl=WavelengthOfInterest)
    
    return bestGains, iOptim, iOptimPD, iOptimGD, LockedRatio, WLockedRatio, VarOPD, VarCP, FCArray




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
    
    return f"{int(time_to_write//3600)}h{(int(time_to_write%3600)/60)}m{int((time_to_write%3600)%60)}s"