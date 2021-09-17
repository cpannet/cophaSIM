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
            for DisturbanceFile in files:
                print(f'Reading file number {DisturbanceFile[-6]}')
                sk.update_params(DisturbanceFile)
            
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
            
        else:
            # Launch the simulator
            sk.loop()
            # Load the performance observables into simu module
            sk.ShowPerformance(TimeBonds, WavelengthOfInterest,DIT, display=False)
                
            # Initialise the comparison tables
            VarOPD[ig,:] = simu.VarOPD
            if gainstr=='GainPD':
                VarCP[ig,:] = simu.VarCPD
            elif gainstr=='GainGD':
                VarCP[ig,:] = simu.VarCGD
            FCArray[ig,:] = simu.FringeContrast
        
        if optim=='phase':      # Optimisation on Phase residues
            if not telescopes:
                Value = np.mean(VarOPD[ig,:])
            elif telescopes:
                itel1,itel2 = telescopes[0]-1, telescopes[1]-1
                ib = ct.posk(itel1, itel2, config.NA)
                Value = VarOPD[ig,ib]
            if Value < minValue:    
                minValue = Value
                iOptim = ig
        
        elif optim == 'FC':     # Optimisation on final fringe contrast
            if not telescopes:
                Value = 1-np.mean(FCArray[ig,:])
            elif telescopes:
                itel1,itel2 = telescopes[0]-1, telescopes[1]-1
                ib = ct.posk(itel1, itel2, config.NA)
                Value = 1-FCArray[ig,ib]
            if Value < minValue:
                minValue = Value
                iOptim = ig
                
        elif optim=='CP':       # Optimisation on Closure Phase variance
            if not telescopes:
                Value = np.mean(VarCP[ig,:])
            elif telescopes:
                if len(telescopes) != 3:
                    raise Exception('For defining a closure phase, telescopes must be three.')
                itel1,itel2,itel3 = telescopes[0]-1, telescopes[1]-1, telescopes[2]-1
                ic = ct.poskfai(itel1, itel2, itel3, config.NA)
                Value = VarCP[ig,ic]
            if Value < minValue:    
                minValue = Value
                iOptim = ig           
        
        elif (Value > 10*minValue) and (ig < Ngains-1):
            VarOPD[ig+1:,:] = 100*np.ones([Ngains-ig-1,NIN])
            VarCP[ig+1:,:] = 100*np.ones([Ngains-ig-1,NC])
            FCArray[ig+1:,:] = 100*np.ones([Ngains-ig-1,NIN])
            break
        
        print(f'Current value={Value}, Minimal value={minValue} for Gain={Gains[iOptim]}')
        
    bestGain = Gains[iOptim]
    
    
    
    print(f"Best gain: {Gains[iOptim]} \n\
Average OPD Variance: {minValue} \n\
Average CPD Variance: {minValue} \n\
Average Fringe Contrast: {1-minValue}")
    
    return bestGain, iOptim, VarOPD, VarCP, FCArray, EarlyStop



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
    
    minValue = 10000
    
    EarlyStop=0
    time1 = 0
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
                for DisturbanceFile in files:
                    print(f'Reading file number {DisturbanceFile[-6]}')
                    sk.update_params(DisturbanceFile)
                
                    # Launch the simulator
                    sk.loop()
                    # Load the performance observables into simu module
                    sk.ShowPerformance(TimeBonds, WavelengthOfInterest,DIT, display=False)
                        
                    
                    # Initialise the comparison tables
                    VarOPD[igp,:] += simu.VarOPD/Nfiles
                    VarCP[igp,:] += simu.VarCPD/Nfiles
                    VarCP[igp,:] += simu.VarCGD/Nfiles
                    FCArray[igp,:] += simu.FringeContrast[0]/Nfiles
                
            else:
                # Launch the simulator
                sk.loop()
                # Load the performance observables into simu module
                sk.ShowPerformance(TimeBonds, WavelengthOfInterest,DIT, display=False)
                    
                # Initialise the comparison tables
                VarOPD[igp,:] = simu.VarOPD
                VarCP[igp,:] = simu.VarCPD
                VarCP[igp,:] = simu.VarCGD
                FCArray[igp,:] = simu.FringeContrast
            
            if optim=='opd':      # Optimisation on OPD residues
                if not telescopes:
                    Value = np.mean(VarOPD[igp,:])
                elif telescopes:
                    itel1,itel2 = telescopes[0]-1, telescopes[1]-1
                    ib = ct.posk(itel1, itel2, config.NA)
                    Value = VarOPD[igp,ib]
                if Value < minValue:    
                    minValue = Value
                    iOptim = igp
                    iOptimGD = ig
                    iOptimPD = ip
            
            elif optim == 'FC':     # Optimisation on final fringe contrast
                if not telescopes:
                    Value = 1-np.mean(FCArray[igp,:])
                elif telescopes:
                    itel1,itel2 = telescopes[0]-1, telescopes[1]-1
                    ib = ct.posk(itel1, itel2, config.NA)
                    Value = 1-FCArray[igp,ib]
                if Value < minValue:
                    minValue = Value
                    iOptim = igp
                    iOptimGD = ig
                    iOptimPD = ip
                    
            elif optim=='CP':       # Optimisation on Closure Phase variance
                if not telescopes:
                    Value = np.mean(VarCP[igp,:])
                elif telescopes:
                    if len(telescopes) != 3:
                        raise Exception('For defining a closure phase, telescopes must be three.')
                    itel1,itel2,itel3 = telescopes[0]-1, telescopes[1]-1, telescopes[2]-1
                    ic = ct.poskfai(itel1, itel2, itel3, config.NA)
                    Value = VarCP[igp,ic]
                if Value < minValue:    
                    minValue = Value
                    iOptim = igp
                    iOptimGD = ig
                    iOptimPD = ip
            
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
            
            print(f'Current value={Value}, Minimal value={minValue} for \n\
GainGD={GainsGD[iOptimGD]}\n\
GainsPD={GainsPD[iOptimPD]}')
        
    bestGains = GainsGD[iOptimGD], GainsPD[iOptimPD]
    
    
    
    print(f"Best gains (GD,PD): ({(GainsGD[iOptimGD],GainsPD[iOptimPD])} \n\
Average OPD Variance: {minValue} \n\
Average CPD Variance: {minValue} \n\
Average Fringe Contrast: {1-minValue}")
    
    return bestGains, iOptim, iOptimPD, iOptimGD, VarOPD, VarCP, FCArray




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