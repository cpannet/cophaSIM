# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:36:18 2023

@author: cpannetier

This file enables to read the telemetries saved by SPICA-FT instrument.
It formats the data to display it using display function.

"""

import struct
import numpy as np
import matplotlib.pyplot as plt
import mmap
import os
import datetime
from importlib import reload
from .tol_colors import tol_cset
from . import coh_tools as ct
from scipy.special import binom

import matplotlib.lines as mlines
from astropy.io import fits

from . import config,outputs

colors=['blue','red','green','brown','yellow','orange','pink','grey','cyan','black','magenta','lightblue','darkblue','darkgrey','lightgrey','indigo']

colors = tol_cset('muted')
telcolors = tol_cset('bright')

"""
Parameters of the script
"""

R=50 ; lmbda=1.6      # Important to have the correct values

NA=6 ; NIN = 15 ; NC = 10
nrows=int(np.sqrt(NA)) ; ncols=NA%nrows
len2 = NIN//2 ; len1 = NIN-len2

basecolors = colors[:len1]+colors[:len2]
basestyles = len1*['solid'] + len2*['dashed']
closurecolors = colors[:NC]

Piston2OPD = np.zeros([NIN,NA])
for ia in range(NA):
    for iap in range(ia+1,NA):
        ib = ct.posk(ia,iap,NA)
        Piston2OPD[ib,ia] = -1
        Piston2OPD[ib,iap] = 1

"""
RCPARAMS
"""

SS = 12     # Small size
MS = 14     # Medium size
BS = 16     # Big size
figsize = (10,8)

rcParamsForSlides = {"font.size":SS,
       "axes.titlesize":SS,
       "axes.labelsize":MS,
       "xtick.labelsize":SS,
       "ytick.labelsize":SS,
       "legend.fontsize":SS,
       "figure.titlesize":BS,
       "figure.constrained_layout.use": True,
       "figure.figsize":figsize}


figsize = (16,8)
rcParamsForBaselines = {"font.size":SS,
       "axes.titlesize":SS,
       "axes.labelsize":MS,
       "axes.grid":True,
       
       "xtick.labelsize":SS,
       "ytick.labelsize":SS,
       "legend.fontsize":SS,
       "figure.titlesize":BS,
       "figure.constrained_layout.use": False,
       #"figure.constrained_layout.h_pad": 0.08,
       "figure.figsize":figsize,
       'figure.subplot.hspace': 0.05,
       'figure.subplot.wspace': 0,
       'figure.subplot.left':0.1,
       'figure.subplot.right':0.95
       }


figsize = (16,8)
rcParamsForBaselines_SNR = {"font.size":SS,
       "axes.titlesize":SS,
       "axes.labelsize":MS,
       "axes.grid":True,
       
       "xtick.labelsize":SS,
       "ytick.labelsize":SS,
       "legend.fontsize":SS,
       "figure.titlesize":BS,
       "figure.constrained_layout.use": False,
       #"figure.constrained_layout.h_pad": 0.08,
       "figure.figsize":figsize,
       'figure.subplot.hspace': 0.1,
       'figure.subplot.wspace': 0,
       'figure.subplot.left':0.15,
       'figure.subplot.right':0.95
       }

figsize = (1.5*16,1.5*8)
SSFF = 16     # Small size
MSFF = 18     # Medium size
BSFF = 20     # Big size
rcParamsForFullScreen = {"font.size":SSFF,
       "axes.titlesize":SSFF,
       "axes.labelsize":MSFF,
       "axes.grid":True,
       
       "xtick.labelsize":SSFF,
       "ytick.labelsize":SSFF,
       "legend.fontsize":SSFF,
       "figure.titlesize":BSFF,
       "figure.constrained_layout.use": False,
       "figure.figsize":figsize,
       'figure.subplot.hspace': 0.05,
       'figure.subplot.wspace': 0,
       'figure.subplot.left':0.1,
       'figure.subplot.right':0.95
       }



#%%

def readDump(file, version='current'):
    
    fd=open(file,"r+b")
    
    buf=mmap.mmap(fd.fileno(),0)
    print(len(buf))
    
    # =============================================================================
    # VERSION OF THE DUMP AFTER 29/04/2021: see mail Sylvain 16:37
    # =============================================================================
    
    nbases=15
    ntel=6
    ntriplet=10
    
    
    if version == 'current':
        
        ###########################
        # Dump reading up-to-date #
        ###########################
        
        s = struct.Struct('@ Q 15d 6d 15d 15d 15d 15d 10d 10d 15d 15d 15d Q d d 6d 6d 15d 15d 15d 15d 6d 15d 6d 6d 6d 6d 6d 6d 2Q 2Q')
        slen = s.size
        
        nbframe = int(len(buf)/slen)
        
        print(f"Number of digits in the file: {len(buf)}")
        print(f"Number of frames: {nbframe}")
        print(f"Set number of digits per frame: {slen}")
        print(f"Number of digits browse by the loop (must be equal to {len(buf)}): {slen*nbframe}")
        
        """
        iFramePhi = 0; iGd = 1; iPhoto = 2; iPd = 3; iCurPdVar = 4; iAvPdVar = 5; iVisNorm = 6; 
        iGdClosure = 7; iPdClosure = 8; iAvVisSqNorm = 9;
        iFrameOpd = 10; iKgdKpd = 11; iCurGdPistonV = 12; iCurRefGdPistonV=13; iCurRefGd = 14; iCurRefPd=15;
        iCurErrPistonV=16; iGdWeights=17; iGdCorMic = 18;
        iFsCmd = 21;
        iTimeStamp = 22;
        """
        
        
        # Structure updated in 01/10/2021
        KgD=np.zeros((nbframe),dtype=np.double)
        KpD=np.zeros((nbframe),dtype=np.double)
        gD=np.zeros((nbframe,nbases),dtype=np.double)
        pD=np.zeros((nbframe,nbases),dtype=np.double)
        photometry=np.zeros((nbframe,ntel),dtype=np.double)
        curPdVar=np.zeros((nbframe,nbases),dtype=np.double)
        averagePdVar=np.zeros((nbframe,nbases),dtype=np.double) # average PD var over 40DIT generally
        normVisibility=np.zeros((nbframe,nbases),dtype=np.double)
        gdClosure=np.zeros((nbframe,ntriplet),dtype=np.double)
        pdClosure=np.zeros((nbframe,ntriplet),dtype=np.double)
        
        visLambSumMeanSqNorm=np.zeros((nbframe,nbases),dtype=np.double) # NDIT + Lambda coherent integration, then squared norm
        visVarDitMean=np.zeros((nbframe,nbases),dtype=np.double) # Average vis var (over NDITs for eq.14 Lacour numerator)
        avVisSpectralSqNorm=np.zeros((nbframe,nbases),dtype=np.double) # NDIT coherent integration (mean(vis)), then squared norm and incohernt integration over lambda
        #timestamps=np.zeros((nbframe,4),dtype=float)    # New structure - 20-04-2021
        
        curGdPistonMicrons=np.zeros((nbframe,ntel),dtype=np.double)
        curRefGDPistonMicrons=np.zeros((nbframe,ntel),dtype=np.double)
        curRefGD=np.zeros((nbframe,nbases),dtype=np.double)
        curRefPD=np.zeros((nbframe,nbases),dtype=np.double)
        curGdErrBaseMicrons=np.zeros((nbframe,nbases),dtype=np.double) # Output of eq.35
        curPdErrBaseMicrons=np.zeros((nbframe,nbases),dtype=np.double)
        curGdErrPistonMicrons=np.zeros((nbframe,ntel),dtype=np.double)
        gdWeightsPerBase=np.zeros((nbframe,nbases),dtype=np.double)
        gdDlCorMicrons=np.zeros((nbframe,ntel),dtype=np.double)
        gdDlCmdMicrons=np.zeros((nbframe,ntel),dtype=np.double)
        pdDlCmdMicrons=np.zeros((nbframe,ntel),dtype=np.double)
        
        dPdGdDlCmdMicrons=np.zeros((nbframe,ntel),dtype=np.double) # New structure - 06-11-2022
        FringeSearchCmdMicrons=np.zeros((nbframe,ntel),dtype=np.double) # New structure - 20-04-2021    
        dMetBoxOffsetsMicrons=np.zeros((nbframe,ntel),dtype=np.double) # New structure - 11-06-2022
        
        tBeforeProcessFrameCall=np.zeros((nbframe,2),dtype=np.double)
        tAfterProcessFrameCall=np.zeros((nbframe,2),dtype=np.double)
        
        icpt=0
        for iFrame in range(nbframe):
            i=0
            # Phase Sensor data
            fields=s.unpack(buf[icpt:icpt+slen])    
            frameNb=fields[i] ; i+=1
            gD[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            photometry[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            pD[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            curPdVar[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            averagePdVar[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            normVisibility[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            gdClosure[iFrame,:]=fields[i:i+ntriplet] ; i+=ntriplet
            pdClosure[iFrame,:]=fields[i:i+ntriplet] ; i+=ntriplet # 
            visLambSumMeanSqNorm[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            visVarDitMean[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            avVisSpectralSqNorm[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            
            # Fringe tracker data
        
            frameNb=fields[i] ; i+=1
            KgD[iFrame]=fields[i] ; i+=1
            KpD[iFrame]=fields[i] ; i+=1
            curGdPistonMicrons[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            curRefGDPistonMicrons[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            curRefGD[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            curRefPD[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            curGdErrBaseMicrons[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            curPdErrBaseMicrons[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            curGdErrPistonMicrons[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            gdWeightsPerBase[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            gdDlCorMicrons[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            gdDlCmdMicrons[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            pdDlCmdMicrons[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            dPdGdDlCmdMicrons[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            FringeSearchCmdMicrons[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            dMetBoxOffsetsMicrons[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            
            # OPD Monitoring data
            tBeforeProcessFrameCall[iFrame,:] = fields[i:i+2] ; i+=2
            tAfterProcessFrameCall[iFrame,:] = fields[i:i+2] ; i+=2
            
            # print(i*8)
            
            icpt+=slen
          
            
        data = {'fields':fields,
                'frameNb':frameNb,
                'gD':gD,
                'pD':pD,
                'photometry':photometry,
                'curPdVar':curPdVar,
                'averagePdVar':averagePdVar,
                'normVisibility':normVisibility,
                'gdClosure':gdClosure,
                'pdClosure':pdClosure,
                'visLambSumMeanSqNorm':visLambSumMeanSqNorm,
                'visVarDitMean':visVarDitMean,
                'avVisSpectralSqNorm':avVisSpectralSqNorm,
        
                'KgD':KgD,
                'KpD':KpD,
                'curGdPistonMicrons':curGdPistonMicrons,
                'curRefGDPistonMicrons':curRefGDPistonMicrons,
                'curRefGD':curRefGD,
                'curRefPD':curRefPD,
                'curGdErrBaseMicrons':curGdErrBaseMicrons,
                'curPdErrBaseMicrons':curPdErrBaseMicrons,
                'curGdErrPistonMicrons':curGdErrPistonMicrons,
                'gdWeightsPerBase':gdWeightsPerBase,
                'gdDlCorMicrons':gdDlCorMicrons,
                'gdDlCmdMicrons':gdDlCmdMicrons,
                'pdDlCmdMicrons':pdDlCmdMicrons,
                'dPdGdDlCmdMicrons':dPdGdDlCmdMicrons,
                'FringeSearchCmdMicrons':FringeSearchCmdMicrons,
                'dMetBoxOffsetsMicrons':dMetBoxOffsetsMicrons,
                
                'tBeforeProcessFrameCall':tBeforeProcessFrameCall,  # (seconds,nanoseconds)
                'tAfterProcessFrameCall':tAfterProcessFrameCall}
        
    else:
        
        s = struct.Struct('@ Q 15d 6d 15d 15d 15d 15d 10d 10d 15d 15d 15d Q d d 6d 6d 15d 15d 15d 15d 6d 15d 6d 6d 6d 6d 2Q 2Q')
        slen = s.size

        nbframe = int(len(buf)/slen)
        
        print(f"Number of digits in the file: {len(buf)}")
        print(f"Number of frames: {nbframe}")
        print(f"Set number of digits per frame: {slen}")
        print(f"Number of digits browse by the loop (must be equal to {len(buf)}): {slen*nbframe}")
        
        """
        iFramePhi = 0; iGd = 1; iPhoto = 2; iPd = 3; iCurPdVar = 4; iAvPdVar = 5; iVisNorm = 6; 
        iGdClosure = 7; iPdClosure = 8; iAvVisSqNorm = 9;
        iFrameOpd = 10; iKgdKpd = 11; iCurGdPistonV = 12; iCurRefGdPistonV=13; iCurRefGd = 14; iCurRefPd=15;
        iCurErrPistonV=16; iGdWeights=17; iGdCorMic = 18;
        iFsCmd = 21;
        iTimeStamp = 22;
        """
        
        
        # Structure updated in 01/10/2021
        KgD=np.zeros((nbframe),dtype=np.double)
        KpD=np.zeros((nbframe),dtype=np.double)
        gD=np.zeros((nbframe,nbases),dtype=np.double)
        pD=np.zeros((nbframe,nbases),dtype=np.double)
        photometry=np.zeros((nbframe,ntel),dtype=np.double)
        curPdVar=np.zeros((nbframe,nbases),dtype=np.double)
        averagePdVar=np.zeros((nbframe,nbases),dtype=np.double) # average PD var over 40DIT generally
        normVisibility=np.zeros((nbframe,nbases),dtype=np.double)
        gdClosure=np.zeros((nbframe,ntriplet),dtype=np.double)
        pdClosure=np.zeros((nbframe,ntriplet),dtype=np.double)

        visLambSumMeanSqNorm=np.zeros((nbframe,nbases),dtype=np.double) # NDIT + Lambda coherent integration, then squared norm
        visVarDitMean=np.zeros((nbframe,nbases),dtype=np.double) # Average vis var (over NDITs for eq.14 Lacour numerator)
        avVisSpectralSqNorm=np.zeros((nbframe,nbases),dtype=np.double) # NDIT coherent integration (mean(vis)), then squared norm and incohernt integration over lambda
        #timestamps=np.zeros((nbframe,4),dtype=float)    # New structure - 20-04-2021
        
        curGdPistonMicrons=np.zeros((nbframe,ntel),dtype=np.double)
        curRefGDPistonMicrons=np.zeros((nbframe,ntel),dtype=np.double)
        curRefGD=np.zeros((nbframe,nbases),dtype=np.double)
        curRefPD=np.zeros((nbframe,nbases),dtype=np.double)
        curGdErrBaseMicrons=np.zeros((nbframe,nbases),dtype=np.double) # Output of eq.35
        curPdErrBaseMicrons=np.zeros((nbframe,nbases),dtype=np.double)
        curGdErrPistonMicrons=np.zeros((nbframe,ntel),dtype=np.double)
        gdWeightsPerBase=np.zeros((nbframe,nbases),dtype=np.double)
        gdDlCorMicrons=np.zeros((nbframe,ntel),dtype=np.double)
        gdDlCmdMicrons=np.zeros((nbframe,ntel),dtype=np.double)
        pdDlCmdMicrons=np.zeros((nbframe,ntel),dtype=np.double)
        FringeSearchCmdMicrons=np.zeros((nbframe,ntel),dtype=np.double) # New structure - 20-04-2021    
        
        tBeforeProcessFrameCall=np.zeros((nbframe,2),dtype=np.double)
        tAfterProcessFrameCall=np.zeros((nbframe,2),dtype=np.double)
        
        icpt=0
        for iFrame in range(nbframe):
            i=0
            # Phase Sensor data
            fields=s.unpack(buf[icpt:icpt+slen])    
            frameNb=fields[i] ; i+=1
            gD[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            photometry[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            pD[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            curPdVar[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            averagePdVar[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            normVisibility[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            gdClosure[iFrame,:]=fields[i:i+ntriplet] ; i+=ntriplet
            pdClosure[iFrame,:]=fields[i:i+ntriplet] ; i+=ntriplet # 
            visLambSumMeanSqNorm[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            visVarDitMean[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            avVisSpectralSqNorm[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            
            # Fringe tracker data
        
            frameNb=fields[i] ; i+=1
            KgD[iFrame]=fields[i] ; i+=1
            KpD[iFrame]=fields[i] ; i+=1
            curGdPistonMicrons[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            curRefGDPistonMicrons[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            curRefGD[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            curRefPD[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            curGdErrBaseMicrons[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            curPdErrBaseMicrons[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            curGdErrPistonMicrons[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            gdWeightsPerBase[iFrame,:]=fields[i:i+nbases] ; i+=nbases
            gdDlCorMicrons[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            gdDlCmdMicrons[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            pdDlCmdMicrons[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            FringeSearchCmdMicrons[iFrame,:]=fields[i:i+ntel] ; i+=ntel
            
            # OPD Monitoring data
            tBeforeProcessFrameCall[iFrame,:] = fields[i:i+2] ; i+=2
            tAfterProcessFrameCall[iFrame,:] = fields[i:i+2] ; i+=2
            
            # print(i*8)
            
            icpt+=slen
          
            
        data = {'fields':fields,
                'frameNb':frameNb,
                'gD':gD,
                'pD':pD,
                'photometry':photometry,
                'curPdVar':curPdVar,
                'averagePdVar':averagePdVar,
                'normVisibility':normVisibility,
                'gdClosure':gdClosure,
                'pdClosure':pdClosure,
                'visLambSumMeanSqNorm':visLambSumMeanSqNorm,
                'visVarDitMean':visVarDitMean,
                'avVisSpectralSqNorm':avVisSpectralSqNorm,

                'KgD':KgD,
                'KpD':KpD,
                'curGdPistonMicrons':curGdPistonMicrons,
                'curRefGDPistonMicrons':curRefGDPistonMicrons,
                'curRefGD':curRefGD,
                'curRefPD':curRefPD,
                'curGdErrBaseMicrons':curGdErrBaseMicrons,
                'curPdErrBaseMicrons':curPdErrBaseMicrons,
                'curGdErrPistonMicrons':curGdErrPistonMicrons,
                'gdWeightsPerBase':gdWeightsPerBase,
                'gdDlCorMicrons':gdDlCorMicrons,
                'gdDlCmdMicrons':gdDlCmdMicrons,
                'pdDlCmdMicrons':pdDlCmdMicrons,
                'FringeSearchCmdMicrons':FringeSearchCmdMicrons,
                'tBeforeProcessFrameCall':tBeforeProcessFrameCall,  # (seconds,nanoseconds)
                'tAfterProcessFrameCall':tAfterProcessFrameCall}

    return data

def ReadFits(file,oldFits=False,computeCp=False,give_names=False,newDateFormat=True):
    global NT, NA, NINmes, TimeID, dt, timestamps,ich,whichSNR
    
    with fits.open(file) as hduL:

        NT, NA = hduL[1].data["pdDlCmdMicrons"].shape # microns
        _, NINmes = hduL[1].data["gD"].shape # microns
        
        config.NT = NT ; config.NA=NA; config.OW = 1
        config.NIN = int(NA*(NA-1)/2) ; config.NINmes=NINmes
        config.NB = int(NA**2) 
        config.NC = int(binom(NA,3))           # Number of closure phases
        config.ND = int((NA-1)*(NA-2)/2)       # Number of independant closure phases
        config.wlOfTrack = 1.6              # By default, it is spica-ft data from mircx.
        
        reload(outputs)
        
        outputs.outputsFile = file.split('\\')[-1].split('/')[-1]
        
        if not newDateFormat:
            timestr = outputs.outputsFile.split('.fits')[0][-20:]
        
            # Handle the fact that it can be 'Aug__8_07h08m18_2023' or 'Aug_13_07h08m18_2023'
            if len(timestr.split('__'))==2:
                timestr = timestr.split('__')[0] + "_0"+timestr.split('__')[1]
                
            recordTime = datetime.datetime.strptime(timestr, '%b_%d_%Hh%Mm%S_%Y')
        else:
            timestr = outputs.outputsFile.split('.fits')[0].split('.TELEMETRY.')[1]
            recordTime = datetime.datetime.strptime(timestr, '%Y-%m-%dT%H-%M-%S')
        
        outputs.TimeID = recordTime.strftime("%Y-%m-%dT%H-%M-%S")
        
        outputs.simulatedTelemetries = False
        
        config.Beam2Tel = hduL[0].header['Beam2Tel']
        config.Target = config.ScienceObject()
        InterfArray = ct.get_array('CHARA')
        
        config.FS['Piston2OPD'] = np.zeros([NIN,NA])
        for ia in range(NA):
            for iap in range(ia+1,NA):
                ib = ct.posk(ia,iap,NA)
                config.FS['Piston2OPD'][ib,ia] = 1
                config.FS['Piston2OPD'][ib,iap] = -1
                
        config.FS['OPD2Piston'] = np.linalg.pinv(config.FS['Piston2OPD'])   # OPD to pistons matrix
        config.FS['OPD2Piston'][np.abs(config.FS['OPD2Piston'])<1e-8]=0
        
        telNameLength=InterfArray.telNameLength
        beam2Tel = [config.Beam2Tel[i:i+telNameLength] for i in range(0, len(config.Beam2Tel), telNameLength)]
        
        basenamesInFile = []
        for ia in range(NA):
            for iap in range(ia+1,NA):
                basenamesInFile.append(f"{beam2Tel[ia]}{beam2Tel[iap]}")
        
        """
        Commands are sent to beams, which are associated to telescopes according
        to the Beam2Tel string. So they need to be sorted to correspond to
        the conventional order S1S2E1E2W1W2.
        """
        
        TelConventionalArrangement = InterfArray.TelNames# ['S1','S2','E1','E2','W1','W2']
        basenamesConventional = []
        for ia in range(NA):
            for iap in range(ia+1,NA):
                basenamesConventional.append(f"{TelConventionalArrangement[ia]}{TelConventionalArrangement[iap]}")
                
        sortBeams2Tels = np.zeros([NA,NA])
        
        for ia in range(NA):
            tel0 = TelConventionalArrangement[ia]
            associatedBeam = np.argwhere(np.array(beam2Tel)==tel0)[0][0]
            sortBeams2Tels[ia,associatedBeam]=1
            
        sortBeams2Tels = np.diag(np.ones(NA))
        sortBases2Conventional = np.zeros([NIN,NIN])
        # sortBases2Conventional = np.diag(np.ones(NIN))
        for ib in range(NIN):
            base0 = basenamesInFile[ib]
            if len(np.argwhere(np.array(basenamesConventional)==base0)):
                pos = np.argwhere(np.array(basenamesConventional)==base0)[0][0]
                
            else:
                base0 = base0[telNameLength:]+base0[0:telNameLength]
                pos = np.argwhere(np.array(basenamesConventional)==base0)[0][0]
            sortBases2Conventional[pos,ib]=1
            
        # sortPistons2Conventional = np.zeros([NA,NA])
        # for ia in range(NA):
        #     tel0 = beam2Tel[ia]
        #     pos = np.argwhere(np.array(TelConventionalArrangement)==tel0)[0][0]
        #     sortPistons2Conventional[pos,ia]=1
        # config.sortPistons2Conventional = sortPistons2Conventional.T
        

            
        # sortBases2Conventional = sortBases2Conventional.T
        config.sortBases2Conventional = sortBases2Conventional
        
        config.FT['whichSNR'] = 'pd'
        
        config.FS['R'] = 50
        config.FS['ich'] = np.array([[1,2], [1,3], [1,4], [1,5], [1,6], [2,3], [2,4], [2,5],[2,6],\
                         [3,4],[3,5],[3,6],[4,5],[4,6],[5,6]])
            
        config.InterfArray = config.Interferometer(name='chara')
        
        CommonOutputs = ["KgdKpd","PD","GD","curPdErrBaseMicrons",
                           "curGdErrBaseMicrons","PdClosure","GdClosure",
                           "curRefPD","curRefGD","Photometry","curPdVar",
                           "avPdVar","VisiNorm","pdDlCmdMicrons","gdDlCmdMicrons",
                           "curFsPosFromStartMicrons","MetBoxCurrentOffsetMicrons"]
        commonOutputsAssociatedNames = ["GainPD","GainGD","timestamps",
                                        "PD","GD","PDResidual","GDResidual",
                                        "ClosurePhasePD","ClosurePhaseGD",
                                        "PDref","GDref","PhotometryEstimated",
                                        "varPD","varGD","SquaredSnrGD","SquaredSnrPD",
                                        "SquaredSNRMovingAveragePD","SquaredSNRMovingAverageGD",
                                        "singularValuesSqrt","ThresholdPD","ThresholdGD",
                                        "VisibilityEstimated","PistonGDcorr",
                                        "PistonPDCommand","PistonGDCommand","CommandRelock",
                                        "CommandODL","PistonPDCommand","PistonGDCommand",
                                        "PistonGDcorr","PistonPDcorr",
                                        "GDPistonResidual","PDPistonResidual",
                                        "ClosurePhaseGDafter","ClosurePhasePDafter",
                                        "PDCommand","GDCommand","OPDCommand","OPDCommandRelock"]
        
        AdditionalOutputs = []
        for key in hduL[1].data.names:
            if key not in CommonOutputs:
                setattr(outputs,key,hduL[1].data[key])
                AdditionalOutputs.append(key)
                
        tBefore = hduL[1].data['tBeforeProcessFrameCall'][:,0] + hduL[1].data['tBeforeProcessFrameCall'][:,1]*1e-9  #Timestamps of the data (at frame reception)*
        tAfter = hduL[1].data['tAfterProcessFrameCall'][:,0] + hduL[1].data['tAfterProcessFrameCall'][:,1]*1e-9  #Timestamps of the data (at frame reception)
        outputs.timestamps = tBefore-tBefore[0]
        outputs.tAfter = tAfter - tAfter[0] 
        config.dt = np.mean(outputs.timestamps[1:]-outputs.timestamps[:-1])
        
        """Global variables analog to outputs module"""
        
        outputs.GainPD = hduL[1].data["KgdKpd"][:,1] # Gains PD [NT]
        outputs.GainGD = hduL[1].data["KgdKpd"][:,0] # Gains GD [NT]
        
        if (np.std(outputs.GainPD,axis=0) == 0).all():
            config.FT['GainPD'] = outputs.GainPD[0]
        else:   # add 1 for knowing that gain has changed, but add mean for knowing around what value it was
            config.FT['GainPD'] = 1+np.mean(outputs.GainPD)
        if (np.std(outputs.GainGD,axis=0) == 0).all():
            config.FT['GainGD'] = outputs.GainGD[0]
        else:   # add 1 for knowing that gain has changed, but add mean for knowing around what value it was
            config.FT['GainGD'] = 1+np.mean(outputs.GainGD)
            
        outputs.PDEstimated = np.matmul(sortBases2Conventional,hduL[1].data["PD"].T).T # Estimated baselines PD [NTxNINmes - rad]
        outputs.GDEstimated = np.matmul(sortBases2Conventional,hduL[1].data["GD"].T).T # Estimated baselines GD [NTxNINmes - rad]
        
        pdmic = outputs.PDEstimated*config.wlOfTrack/2/np.pi
        gdmic = outputs.GDEstimated*config.FS['R']*config.wlOfTrack/2/np.pi
        
        outputs.OPDTrue = np.unwrap(outputs.PDEstimated,axis=0)*config.wlOfTrack/2/np.pi
        outputs.reconstructedOPD = ct.reconstructOpenLoop(pdmic, gdmic, config.wlOfTrack)
        outputs.PDResidual = np.matmul(sortBases2Conventional,hduL[1].data["curPdErrBaseMicrons"].T).T/config.wlOfTrack*2*np.pi # Estimated residual PD = PD-PDref after Ipd (eq.35) [NTxNINmes - rad]
        outputs.GDResidual = np.matmul(sortBases2Conventional,hduL[1].data["curGdErrBaseMicrons"].T).T/R/config.wlOfTrack*2*np.pi # Estimated residual GD = GD-GDref after Ipd (eq.35) [NTxNINmes - rad]
        outputs.ClosurePhasePD = hduL[1].data["PdClosure"] # PD closure phase [NTxNC - rad]
        outputs.ClosurePhaseGD = hduL[1].data["GdClosure"] # GD closure phase [NTxNC - rad]
        outputs.PDref = np.matmul(sortBases2Conventional,hduL[1].data["curRefPD"].T).T/lmbda*2*np.pi # PD reference vector [NTxNINmes - rad]
        outputs.GDref = np.matmul(sortBases2Conventional,hduL[1].data["curRefGD"].T).T/R/lmbda*2*np.pi # GD reference vector [NTxNINmes - rad]
        
        outputs.PhotometryEstimated = hduL[1].data["Photometry"] # Estimated photometries [NTxNA - ADU]
        
        """ Before 2023-07-11 """
        
        # outputs.varPD = hduL[1].data["curPdVar"] # Estimated "PD variance" = 1/SNR² [NTxNINmes]
        # outputs.varGD = hduL[1].data["alternatePdVar"] # Estimated "GD variance" = 1/SNR² [NTxNINmes]
        # outputs.SquaredSNRMovingAveragePD = np.nan_to_num(1/hduL[1].data["avPdVar"],posinf=0) # Estimated SNR² averaged over N dit [NTxNINmes]
        
        """ After 2023-07-11 """
        whichCurPdVar = hduL[0].header['whichCurPdVar']
        if whichCurPdVar == "Sylvestre":
            config.FT['whichSNR'] = "pd"
        else:
            config.FT['whichSNR'] = "gd"
            
        outputs.varPD = np.matmul(sortBases2Conventional,hduL[1].data["pdVar"].T).T # Estimated "PD variance" = 1/SNR² [NTxNINmes]
        outputs.varGD = np.matmul(sortBases2Conventional,hduL[1].data["alternatePdVar"].T).T # Estimated "GD variance" = 1/SNR² [NTxNINmes]
        outputs.SquaredSnrGD = 1/outputs.varGD
        outputs.SquaredSnrPD = 1/outputs.varPD
        
        outputs.SquaredSNRMovingAveragePD = np.nan_to_num(np.matmul(sortBases2Conventional,1/hduL[1].data["averagePdVar"].T).T,posinf=0) # Estimated SNR² averaged over N dit [NTxNINmes]
        # outputs.whichVar = hduL[1].data['whichCurPdVar'][0]    # 0: varPd ; 1:varGd
        if config.FT['whichSNR']=="gd":
            outputs.SquaredSNRMovingAverageGD = np.copy(outputs.SquaredSNRMovingAveragePD)
            
        outputs.singularValuesSqrt = np.sqrt(hduL[1].data["sPdSingularValues"])
        
        config.FT['ThresholdPD'] = hduL[1].data['pdThreshold'][0]
        config.FT['ThresholdGD'] = hduL[1].data['gdThresholds'][0]
        outputs.ThresholdPD = hduL[1].data['pdThreshold']
        outputs.ThresholdGD = hduL[1].data['gdThresholds']
        
        outputs.VisibilityEstimated = np.nan_to_num(np.matmul(sortBases2Conventional,hduL[1].data["VisiNorm"].T).T,posinf=0) # Estimated fringe visibility [NTxNINmes]
        
        outputs.PistonGDcorr = hduL[1].data["gdDlCorMicrons"] # GD before round [NTxNA - microns]
        
        # outputs.PistonPDCommand = np.zeros([NT+1,NA])
        # outputs.PistonGDCommand = np.zeros([NT+1,NA])
        # outputs.SearchCommand = np.zeros([NT+1,NA])
        # outputs.CommandODL = np.zeros([NT+1,NA])
        
        outputs.PistonPDCommand[:-1] = hduL[1].data["pdDlCmdMicrons"] # PD command [NTxNA - microns]
        outputs.PistonGDCommand[:-1] = hduL[1].data["gdDlCmdMicrons"] # GD command [NTxNA - microns]
        outputs.CommandRelock[:-1] = hduL[1].data["curFsPosFromStartMicrons"] # Search command [NTxNA - microns]
        if not oldFits:
            outputs.CommandODL[:-1] = hduL[1].data["fullDlCmdMicrons"] # ODL command [NTxNA - microns] A VERIFIER
        else:
            outputs.CommandODL[:-1] = outputs.PistonPDCommand[:-1] + outputs.PistonGDCommand[:-1] # ODL command [NTxNA - microns]
    
    # #ALL DATA WHICH ARE COMMANDS-RELATED NEED TO BE SORTED IN SAME ORDER
    # THAN OPD, i.e. in the order of Beam2Tel

    for it in range(NT):
        outputs.PistonPDCommand[it] = np.dot(sortBeams2Tels,outputs.PistonPDCommand[it])
        outputs.PistonGDCommand[it] = np.dot(sortBeams2Tels,outputs.PistonGDCommand[it])
        outputs.CommandRelock[it] = np.dot(sortBeams2Tels,outputs.CommandRelock[it])
        outputs.CommandODL[it] = np.dot(sortBeams2Tels,outputs.CommandODL[it])
        outputs.PistonGDcorr[it] = np.dot(sortBeams2Tels,outputs.PistonGDcorr[it])
        outputs.GDPistonResidual[it] = np.dot(config.FS['OPD2Piston'],outputs.GDResidual[it])
        outputs.PDPistonResidual[it] = np.dot(config.FS['OPD2Piston'],outputs.PDResidual[it])
    
    if computeCp:
        
        outputs.ClosurePhaseGDafter=np.zeros([config.NT,config.NC])
        outputs.ClosurePhasePDafter=np.zeros([config.NT,config.NC])
        cpGdAfter=np.zeros([config.NT,config.NC])
        cpPdAfter=np.zeros([config.NT,config.NC])
        
        for it in range(NT):
            cpPdAfter[it]=ct.check_cp(outputs.PDEstimated[it])
            cpGdAfter[it]=ct.check_cp(outputs.GDEstimated[it])
            
        for it in range(NT):
            Ncp=300
            if it<Ncp:
                Ncp=it
            timerangeCp=range(it-Ncp,it+1)
            outputs.ClosurePhasePDafter[it]=np.mean(cpPdAfter[timerangeCp],axis=0)
            outputs.ClosurePhaseGDafter[it]=np.mean(cpGdAfter[timerangeCp],axis=0)
    # outputs.ClosurePhaseGDafter = ct.moving_average(outputs.ClosurePhaseGDafter, 300)
    # outputs.ClosurePhasePDafter = ct.moving_average(outputs.ClosurePhasePDafter, 300)
    
    # outputs.PDCommand = np.zeros([NT+1,NIN])
    # outputs.GDCommand = np.zeros([NT+1,NIN])
    # outputs.OPDCommand = np.zeros([NT+1,NIN])
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib = ct.posk(ia,iap,NA)
            outputs.PDCommand[:,ib] = outputs.PistonPDCommand[:,iap]-outputs.PistonPDCommand[:,ia]
            outputs.GDCommand[:,ib] = outputs.PistonGDCommand[:,iap]-outputs.PistonGDCommand[:,ia]
            outputs.OPDCommandRelock[:,ib] = outputs.CommandRelock[:,iap]-outputs.CommandRelock[:,ia]
            outputs.OPDCommand[:,ib] = outputs.CommandODL[:,iap]-outputs.CommandODL[:,ia]

    # Reload outputs parameters.
    # reload(outputs)
    
    if give_names:
        print("Load telemetries into outputs module:")
        for key in commonOutputsAssociatedNames+AdditionalOutputs:
            print(f"- {key}")
    
    return
        


def model(freq,delay,gain):
    z=np.exp(1J*2*np.pi*freq/(2*np.amax(freq)))
    ftr=1/(1+z**(-delay)*gain*z/(z-1))
    return np.abs(ftr)

def model_leak(freq,delay,leak,gain):
    z=np.exp(1J*2*np.pi*freq/(2*np.amax(freq)))
    ftr=1/(1+z**(-delay)*gain*z/(z-leak))
    return np.abs(ftr)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# def model_gd(freq,delay,leak,gain):
#     z=np.exp(1J*2*np.pi*freq/(2*np.amax(freq)))
#     n=leak
#     BO = gain/n*z**(-1) / (1-z**(-1)*((2*n-1)-gain)/n + z**(-2)*(n-1)/n)
#     ftr = 1/(1 + BO)
#     ftr=1/(1+z**(-delay)*gain*z/(z-leak))
#     return np.abs(ftr)

def BodeDiagrams(Output,Command,timestamps,Input=[],f1 = 0.3 , f2 = 5,
                 fbonds=[], mov_average=0, lmbda0=0.55, gain=0, details='', window='no',
                 display=True, figsave=False, figdir='',ext='pdf',only='ftbo',
                 displaytemporal=False):
    """
    Display the Bode Diagrams of FTBO, FTBF, FTrej and returns some characteristical values.

    Parameters
    ----------
    Output : TYPE
        DESCRIPTION.
    Command : TYPE
        DESCRIPTION.
    timestamps : TYPE
        DESCRIPTION.
    Input : TYPE, optional
        DESCRIPTION. The default is [].
    fbonds : TYPE, optional
        DESCRIPTION. The default is [].
    mov_average : TYPE, optional
        DESCRIPTION. The default is 0.
    lmbda0 : FLOAT, optional
        Wavelength of the measurement in microns. The default is 1.6.
        For computing the coherence time T0 (in ms).
    gain : TYPE, optional
        DESCRIPTION. The default is 0.
    details : TYPE, optional
        DESCRIPTION. The default is ''.
    window : TYPE, optional
        DESCRIPTION. The default is 'hanning'.
    display : TYPE, optional
        DESCRIPTION. The default is True.
    figsave : TYPE, optional
        DESCRIPTION. The default is False.
    figdir : TYPE, optional
        DESCRIPTION. The default is ''.
    ext : TYPE, optional
        DESCRIPTION. The default is 'pdf'.
    only : TYPE, optional
        DESCRIPTION. The default is 'ftbo'.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    nNT = len(timestamps) ; dt = np.mean(timestamps[1:]-timestamps[:-1])

    FrequencySampling1 = np.fft.fftfreq(nNT, dt)
    if len(fbonds):
        fmin, fmax = fbonds
    else:
        fmin=0
        fmax=np.max(FrequencySampling1)
    
    PresentFrequencies = (FrequencySampling1 > fmin) \
        & (FrequencySampling1 < fmax)
        
    FrequencySampling = FrequencySampling1[PresentFrequencies]
    
    if window =='hanning':
        windowsequence = np.hanning(nNT)
    else:
        windowsequence = np.ones(nNT)
        
    # pseudoOpenLoop = Command+Output
    
    Output2 = Output*windowsequence
    Command2 = Command*windowsequence
    
    FTResidues = np.fft.fft(Output2,norm="forward")[PresentFrequencies]
    FTCommands = np.fft.fft(Command2,norm="forward")[PresentFrequencies]
    
    # ftPseudoOpenLoop = np.fft.fft(pseudoOpenLoop,norm="forward")[PresentFrequencies]
    
    # FTrej = FTResidues/ftPseudoOpenLoop
    FTBO = FTCommands/FTResidues
    
    ModFTBO = np.abs(FTBO) ; AngleFTBO = np.angle(FTBO)
    ModFTCommands = np.abs(FTCommands)
    ModFTResidues = np.abs(FTResidues)
    
    if mov_average:
        ModFTBO = moving_average(ModFTBO,mov_average)
        AngleFTBO = moving_average(AngleFTBO,mov_average)
        ModFTResidues = moving_average(ModFTResidues, mov_average)
        ModFTCommands = moving_average(ModFTCommands, mov_average)
        FrequencySampling = moving_average(FrequencySampling,mov_average)
    
    # For model fitting later in the code
    NominalRegime = (FrequencySampling>f1)*(FrequencySampling<f2)
    logFrequencySampling = np.log10(FrequencySampling)
    
    if len(Input):
        Input2 = Input*windowsequence
        
    else:
        Input2 = Command*windowsequence
        
    FTTurb = np.fft.fft(Input2,norm="forward")[PresentFrequencies]    
    ModFTTurb = np.abs(FTTurb)
    
    if mov_average:
        ModFTTurb = moving_average(ModFTTurb,mov_average)
        
    coefs = np.polyfit(logFrequencySampling[NominalRegime], np.log10(ModFTTurb[NominalRegime]), 1)
    poly1d_fn = np.poly1d(coefs)
    ModFTTurbfit = 10**poly1d_fn(logFrequencySampling)
    coefDisturb = coefs[0]        
    
    ModFTrej = ModFTResidues/ModFTTurbfit
    ModFTBF = ModFTCommands/ModFTTurbfit

    coefs = np.polyfit(logFrequencySampling[NominalRegime], np.log10(np.abs(ModFTBO[NominalRegime])), 1)
    poly1d_fn = np.poly1d(coefs)
    ModFTBOfit = 10**poly1d_fn(logFrequencySampling)
    CutoffFrequency0dB = FrequencySampling[np.argmin(np.abs(20*np.log10(ModFTBOfit)))]
    
    coefs = np.polyfit(logFrequencySampling[NominalRegime], np.log10(np.abs(ModFTCommands[NominalRegime])), 1)
    poly1d_fn = np.poly1d(coefs)
    ModFTCommandsfit = 10**poly1d_fn(logFrequencySampling)
    
    # coefs = np.polyfit(logFrequencySampling[NominalRegime], np.log10(np.abs(ModFTTurb[NominalRegime])), 1)
    # poly1d_fn = np.poly1d(coefs)
    # ModFTTurbfit = 10**poly1d_fn(logFrequencySampling)
    
    index1Hz = np.argmin(np.abs(logFrequencySampling-1))
    PSDat1Hz = ModFTTurbfit[index1Hz]**2
    EstimatedT0 = (PSDat1Hz/2.84e-4/lmbda0**2)**(-3/5)
    
    results = {"CutoffFrequency0dB":CutoffFrequency0dB,
               "EstimatedT0":EstimatedT0,
               "FrequencySampling":FrequencySampling,
               "ModFTrej":ModFTrej,"ModFTBO":FTBO,"ModFTBF":ModFTBF}
    
    """ Something to implement: fit a model on the rejection function transfer
    # popt, *remain=curve_fit(model,freqs[f_plus],20*np.log10(np.abs(transfer_function[i,:][f_plus]))
    #                       ,p0=[2,0.1,0.99,0],bounds=[[0,0,0,-100],[5,1,0.999,100]])
    """
    
    if display:
        
        if not only:
            if not len(Input):
                raise ValueError('The "Input" parameter must be given. It is empty.')
                
            plt.rcParams.update(rcParamsForBaselines)
            title = f'{details} - Bode diagrams'
            plt.close(title)
            fig = plt.figure(title)
            fig.suptitle(title)
            ax1,ax2,ax3 = fig.subplots(nrows=3,sharex=True)
            
            ax1.plot(FrequencySampling, ModFTrej,color='k')
            
            if gain:
                gains = [gain] ; delays=np.arange(10,60,10)
                styles=['-','--',':']
                linestyles = []
                for ig in range(len(gains)):
                    gain=gains[ig]
                    linestyles.append(mlines.Line2D([],[],color='k',linestyle=styles[ig],label=f"Gain={gain}"))
                    for idel in range(len(delays)):
                        gain=gains[ig];delay=delays[idel]
                        ftr=model(FrequencySampling,delay,gain)
                        ax1.plot(FrequencySampling, ftr, color=colors[idel], linestyle=styles[ig])
                        if ig==len(gains)-1:
                            linestyles.append(mlines.Line2D([],[],color=colors[idel],linestyle='-',label=f'\u03C4={delay}'))
                ax1.legend(handles=linestyles)
                
    
            # plt.plot(FrequencySampling, FrequencySampling*10**(-2), linestyle='--')
            ax1.set_yscale('log') #; ax1.set_ylim(1e-3,5)
            ax1.set_ylabel('FTrej')
            
            ax2.plot(FrequencySampling, ModFTBO,color='k')
            ax2.set_yscale('log') #; ax2.set_ylim(1e-3,5)
            ax2.set_ylabel("FTBO")
            
            ax3.plot(FrequencySampling, ModFTBF,color='k')
        
            ax3.set_xlabel('Frequencies [Hz]')
            ax3.set_ylabel('FTBF')
            ax3.set_xscale('log')
            ax3.set_yscale('log')
            ax1.grid(True) ; ax2.grid(True) ; ax3.grid(True)

            if figsave:
                prefix = details.replace("=","").replace(";","").replace(" ","").replace(".","").replace('\n','_').replace('Phase-delay','PD').replace('Group-delay','GD')
                figname = "TransferFunctions"
                if isinstance(ext,list):
                    for extension in ext:
                        plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
                else:
                    plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
    
            generaltitle=f'{details} - Temporal sampling used'
            plt.close(generaltitle)
            fig = plt.figure(generaltitle)
            ax1,ax2,ax3 = fig.subplots(nrows=3,sharex=True)
            
            ax1.plot(timestamps, Input)
            # plt.plot(FrequencySampling, FrequencySampling*10**(-2), linestyle='--')
        
            ax1.set_ylabel('Open loop')
            
            ax2.plot(timestamps, Output)
            ax2.set_ylabel("Close loop")
            
            ax3.plot(timestamps,Command)
        
            ax3.set_xlabel('Timestamps [s]')
            ax3.set_ylabel('Command')

            fig.show()
            
            """
            Display FTrej and PSD of input (disturbance) and output (residues) signals
            """
            
            plt.rcParams.update(rcParamsForBaselines)
            if not details:
                title = 'FTrej'
            else:
                title = f'FTrej - {details}'
            plt.close(title)
            fig = plt.figure(title)
            fig.suptitle(title)
            ax1,ax2,ax3 = fig.subplots(nrows=3,sharex=True)
            
            ax1.plot(FrequencySampling, 40*np.log10(ModFTrej), "k")
            # ax1.plot(FrequencySampling, 40*np.log10(ModFTBOfit), 
            #          color=colors[0], linestyle='--')
            
            if gain:
                gains = [gain] ; delays=np.arange(10,60,10)
                styles=['-','--',':']
                linestyles = []
                for ig in range(len(gains)):
                    gain=gains[ig]
                    linestyles.append(mlines.Line2D([],[],color='k',linestyle=styles[ig],label=f"Gain={gain}"))
                    for idel in range(len(delays)):
                        gain=gains[ig];delay=delays[idel]
                        ftr=model(FrequencySampling,delay,gain)
                        ax1.plot(FrequencySampling, ftr, color=colors[idel], linestyle=styles[ig])
                        if ig==len(gains)-1:
                            linestyles.append(mlines.Line2D([],[],color=colors[idel],linestyle='-',label=f'\u03C4={delay}'))
                ax1.legend(handles=linestyles)
    
            ax1.set_ylabel('Gain [dB]')
            ct.addtext(ax1,"|FTrej|²",loc='upper center',fontsize='x-large')
            
            ax2.plot(FrequencySampling, ModFTResidues**2, "k")
            
            ax2.set_ylabel("PSD [µm²/Hz]")
            ax2.set_yscale("log")
            ct.addtext(ax2,"Residue",loc='upper center',fontsize='x-large')
            
            ax3.plot(FrequencySampling, ModFTTurb**2, "k")
            ax3.plot(FrequencySampling, ModFTTurbfit**2, color=colors[0])
            ax3.set_ylabel('PSD [µm²/Hz]')
            
            annX = 2
            annYindex = np.argmin(np.abs(FrequencySampling-annX))
            annY = ModFTTurbfit[annYindex]**2*2
            ax3.annotate(f"{round(2*coefDisturb*3,2)}/3",(annX,annY),color=colors[0])
            
            ct.addtext(ax3,"Disturbance",loc='upper center',fontsize='x-large')
            
            ax3.set_xlabel('Frequencies [Hz]')
            ax3.set_xscale('log')
            ax3.set_yscale("log")
            
            ax1.grid(True) ; ax2.grid(True) ; ax3.grid(True)
            ax3.sharey(ax2)
            
            ct.setaxelim(ax3, ydata=ModFTTurbfit**2)
            
            fig.show()
        
        if only == 'ftbo':
            
            
            """ Display gain and phase of FTBO """
            
            plt.rcParams.update(rcParamsForBaselines)
            if not details:
                title = 'Bode Diagrams FTBO'
            else:
                title = f'Bode Diagrams FTBO - {details}'
            
            plt.close(title)
            fig = plt.figure(title)
            fig.suptitle(title)
            ax1,ax2 = fig.subplots(nrows=2,sharex=True)
            
            ax1.plot(FrequencySampling, 40*np.log10(ModFTBO), "k")
            ax1.plot(FrequencySampling, 40*np.log10(ModFTBOfit), 
                     color=colors[0], linestyle='--')
            
            ax2.plot(FrequencySampling, AngleFTBO*180/np.pi, "k")
            
            ax1.set_ylabel('Gain [dB]')
            ax2.set_ylabel("Phase [°]")
            
            ax1.grid(True) ; ax2.grid(True)
            
            ax2.set_xscale('log')
            
            fig.show()
            
            if figsave:
                prefix = details.replace("=","").replace(";","").replace(" ","").replace(".","").replace('\n','_').replace('Phase-delay','PD').replace('Group-delay','GD')
                figname = f"BodeDiagrams_{only}"
                if isinstance(ext,list):
                    for extension in ext:
                        plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
                else:
                    plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
            
            """
            Display FTBO and PSD of command and residues
            """
            
            plt.rcParams.update(rcParamsForBaselines)
            if not details:
                title = 'FTBO'
            else:
                title = f'FTBO - {details}'
            plt.close(title)
            fig = plt.figure(title)
            fig.suptitle(title)
            ax1,ax2,ax3 = fig.subplots(nrows=3,sharex=True)
            
            ax1.plot(FrequencySampling, 40*np.log10(ModFTBO), "k")
            ax1.plot(FrequencySampling, 40*np.log10(ModFTBOfit), 
                     color=colors[0], linestyle='--')
            
            if gain:
                gains = [gain] ; delays=np.arange(10,60,10)
                styles=['-','--',':']
                linestyles = []
                for ig in range(len(gains)):
                    gain=gains[ig]
                    linestyles.append(mlines.Line2D([],[],color='k',linestyle=styles[ig],label=f"Gain={gain}"))
                    for idel in range(len(delays)):
                        gain=gains[ig];delay=delays[idel]
                        ftr=model(FrequencySampling,delay,gain)
                        ax1.plot(FrequencySampling, ftr, color=colors[idel], linestyle=styles[ig])
                        if ig==len(gains)-1:
                            linestyles.append(mlines.Line2D([],[],color=colors[idel],linestyle='-',label=f'\u03C4={delay}'))
                ax1.legend(handles=linestyles)
    
            ax1.set_ylabel('Gain [dB]')
            ct.addtext(ax1,"|FTBO|²",loc='upper center',fontsize='x-large')
            
            ax2.plot(FrequencySampling, ModFTResidues**2, "k")
            
            ax2.set_ylabel("PSD [µm²/Hz]")
            ax2.set_yscale("log")
            ct.addtext(ax2,"Residues",loc='upper center',fontsize='x-large')
            
            
            ax3.plot(FrequencySampling, ModFTCommands**2, "k")
            ax3.plot(FrequencySampling, ModFTCommandsfit**2, color=colors[0])
            ax3.set_ylabel('PSD [µm²/Hz]')
            ct.addtext(ax3,"Commands",loc='upper center',fontsize='x-large')
            
            
            ax3.set_xlabel('Frequencies [Hz]')
            ax3.set_xscale('log')
            ax3.set_yscale("log")
            
            ax1.grid(True) ; ax2.grid(True) ; ax3.grid(True)
            ax3.sharey(ax2)
            
            ct.setaxelim(ax3, ydata=[ModFTCommands**2,ModFTResidues**2])
            
            fig.show()
            
            if figsave:
                prefix = details.replace("=","").replace(";","").replace(" ","").replace(".","").replace('\n','_').replace('Phase-delay','PD').replace('Group-delay','GD')
                figname = f"Amplitude_{only}"
                if isinstance(ext,list):
                    for extension in ext:
                        plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
                else:
                    plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
            
            
            if displaytemporal:
                """
                Display temporal sequences of command and residues
                """
                 
                plt.rcParams.update(rcParamsForBaselines)
                if not details:
                    title = 'FTBO and temporal sequences'
                else:
                    title = f'FTBO and temporal sequences - {details}'
                plt.close(title)
                fig = plt.figure(title)
                fig.suptitle(title)
                ax1,axGhost,ax2,ax3 = fig.subplots(nrows=4, gridspec_kw={"height_ratios":[2,0.5,2,2]})
                axGhost.remove()
                
                ax1.plot(FrequencySampling, 20*np.log10(ModFTBO**2), "k")
                
                if gain:
                    gains = [gain] ; delays=np.arange(10,60,10)
                    styles=['-','--',':']
                    linestyles = []
                    for ig in range(len(gains)):
                        gain=gains[ig]
                        linestyles.append(mlines.Line2D([],[],color='k',linestyle=styles[ig],label=f"Gain={gain}"))
                        for idel in range(len(delays)):
                            gain=gains[ig];delay=delays[idel]
                            ftr=model(FrequencySampling,delay,gain)
                            ax1.plot(FrequencySampling, ftr, color=colors[idel], linestyle=styles[ig])
                            if ig==len(gains)-1:
                                linestyles.append(mlines.Line2D([],[],color=colors[idel],linestyle='-',label=f'\u03C4={delay}'))
                    ax1.legend(handles=linestyles)
                    
                #ax1.set_yscale('log')
                ax1.set_xscale('log')
                ax1.set_ylabel('G(|FTBO|²)')
                ax1.set_xlabel('Frequencies [Hz]')
                
                ax2.plot(timestamps, Output, "k")
                ax2.set_ylabel("Residues\n[µm]")
                ax2.set_ylim(-1.5*lmbda/2,1.5*lmbda/2)
                
                ax3.plot(timestamps, Command, "k")
                
                ax3.set_ylabel('Command\n[µm]')
                ax3.set_xlabel('Time [s]')
                ax2.sharex(ax3)
                
                ax1.grid(True) ; ax2.grid(True) ; ax3.grid(True)
                fig.show()
        
    return results


def BodeDiagrams_pseudoloop(Output,Command,timestamps,Input=[],f1 = 0.3 , f2 = 5,
                 fbonds=[], mov_average=0, lmbda0=0.55, gain=0, details='', window='no',
                 display=True, figsave=False, figdir='',ext='pdf',only='ftbo',
                 displaytemporal=False):
    """
    Display the Bode Diagrams of FTBO, FTBF, FTrej and returns some characteristical values.

    Parameters
    ----------
    Output : TYPE
        DESCRIPTION.
    Command : TYPE
        DESCRIPTION.
    timestamps : TYPE
        DESCRIPTION.
    Input : TYPE, optional
        DESCRIPTION. The default is [].
    fbonds : TYPE, optional
        DESCRIPTION. The default is [].
    mov_average : TYPE, optional
        DESCRIPTION. The default is 0.
    lmbda0 : FLOAT, optional
        Wavelength of the measurement in microns. The default is 1.6.
        For computing the coherence time T0 (in ms).
    gain : TYPE, optional
        DESCRIPTION. The default is 0.
    details : TYPE, optional
        DESCRIPTION. The default is ''.
    window : TYPE, optional
        DESCRIPTION. The default is 'hanning'.
    display : TYPE, optional
        DESCRIPTION. The default is True.
    figsave : TYPE, optional
        DESCRIPTION. The default is False.
    figdir : TYPE, optional
        DESCRIPTION. The default is ''.
    ext : TYPE, optional
        DESCRIPTION. The default is 'pdf'.
    only : TYPE, optional
        DESCRIPTION. The default is 'ftbo'.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    nNT = len(timestamps) ; dt = np.mean(timestamps[1:]-timestamps[:-1])

    FrequencySampling1 = np.fft.fftfreq(nNT, dt)
    if len(fbonds):
        fmin, fmax = fbonds
    else:
        fmin=0
        fmax=np.max(FrequencySampling1)
    
    PresentFrequencies = (FrequencySampling1 > fmin) \
        & (FrequencySampling1 < fmax)
        
    FrequencySampling = FrequencySampling1[PresentFrequencies]
    
    if window =='hanning':
        windowsequence = np.hanning(nNT)
    else:
        windowsequence = np.ones(nNT)
        
    # pseudoOpenLoop = Command+Output
    
    Output2 = Output*windowsequence
    Command2 = Command*windowsequence
    
    FTResidues = np.fft.fft(Output2,norm="forward")[PresentFrequencies]
    FTCommands = np.fft.fft(Command2,norm="forward")[PresentFrequencies]
    
    # ftPseudoOpenLoop = np.fft.fft(pseudoOpenLoop,norm="forward")[PresentFrequencies]
    
    # FTrej = FTResidues/ftPseudoOpenLoop
    FTBO = FTCommands/FTResidues
    
    ModFTBO = np.abs(FTBO) ; AngleFTBO = np.angle(FTBO)
    ModFTCommands = np.abs(FTCommands)
    ModFTResidues = np.abs(FTResidues)
    
    if mov_average:
        ModFTBO = moving_average(ModFTBO,mov_average)
        AngleFTBO = moving_average(AngleFTBO,mov_average)
        ModFTResidues = moving_average(ModFTResidues, mov_average)
        ModFTCommands = moving_average(ModFTCommands, mov_average)
        FrequencySampling = moving_average(FrequencySampling,mov_average)
    
    if not only:
        Input2 = Input*windowsequence
        FTTurb = np.fft.fft(Input2,norm="forward")[PresentFrequencies]    
        
        FTrej = FTResidues/FTTurb
        FTBF = FTCommands/FTTurb
        
        ModFTrej = np.abs(FTrej) ; AngleFTrej = np.angle(FTrej)
        ModFTBF = np.abs(FTBF) ; AngleFTBF = np.angle(FTBF)
        ModFTTurb = np.abs(FTTurb)
        
        if mov_average:
            ModFTrej = moving_average(ModFTrej,mov_average)
            AngleFTrej = moving_average(AngleFTrej,mov_average)
            ModFTBF = moving_average(ModFTBF,mov_average)
            AngleFTBF = moving_average(AngleFTBF,mov_average)
            ModFTTurb = moving_average(ModFTTurb,mov_average)
 
    f1 = 0.3 ; f2 = 10
    NominalRegime = (FrequencySampling>f1)*(FrequencySampling<f2)
    logFrequencySampling = np.log10(FrequencySampling)
    coefs = np.polyfit(logFrequencySampling[NominalRegime], np.log10(np.abs(ModFTBO[NominalRegime])), 1)
    poly1d_fn = np.poly1d(coefs)
    ModFTBOfit = 10**poly1d_fn(logFrequencySampling)
    CutoffFrequency0dB = FrequencySampling[np.argmin(np.abs(20*np.log10(ModFTBOfit)))]
    
    coefs = np.polyfit(logFrequencySampling[NominalRegime], np.log10(np.abs(ModFTCommands[NominalRegime])), 1)
    poly1d_fn = np.poly1d(coefs)
    ModFTCommandsfit = 10**poly1d_fn(logFrequencySampling)
    
    coefs = np.polyfit(logFrequencySampling[NominalRegime], np.log10(np.abs(ModFTTurb[NominalRegime])), 1)
    poly1d_fn = np.poly1d(coefs)
    ModFTTurbfit = 10**poly1d_fn(logFrequencySampling)
    
    index1Hz = np.argmin(np.abs(logFrequencySampling-1))
    PSDat1Hz = ModFTCommandsfit[index1Hz]
    EstimatedT0 = (PSDat1Hz/2.84e-4/lmbda0**2)**(-3/5)
    
    results = {"CutoffFrequency0dB":CutoffFrequency0dB,
               "EstimatedT0":EstimatedT0}
    
    """ Something to implement: fit a model on the rejection function transfer
    # popt, *remain=curve_fit(model,freqs[f_plus],20*np.log10(np.abs(transfer_function[i,:][f_plus]))
    #                       ,p0=[2,0.1,0.99,0],bounds=[[0,0,0,-100],[5,1,0.999,100]])
    """
    
    if display:
        
        if not only:
            if not len(Input):
                raise ValueError('The "Input" parameter must be given. It is empty.')
                
            plt.rcParams.update(rcParamsForBaselines)
            title = f'{details} - Bode diagrams'
            plt.close(title)
            fig = plt.figure(title)
            fig.suptitle(title)
            ax1,ax2,ax3 = fig.subplots(nrows=3,sharex=True)
            
            ax1.plot(FrequencySampling, ModFTrej)
            
            if gain:
                gains = [gain] ; delays=np.arange(10,60,10)
                styles=['-','--',':']
                linestyles = []
                for ig in range(len(gains)):
                    gain=gains[ig]
                    linestyles.append(mlines.Line2D([],[],color='k',linestyle=styles[ig],label=f"Gain={gain}"))
                    for idel in range(len(delays)):
                        gain=gains[ig];delay=delays[idel]
                        ftr=model(FrequencySampling,delay,gain)
                        ax1.plot(FrequencySampling, ftr, color=colors[idel], linestyle=styles[ig])
                        if ig==len(gains)-1:
                            linestyles.append(mlines.Line2D([],[],color=colors[idel],linestyle='-',label=f'\u03C4={delay}'))
                ax1.legend(handles=linestyles)
                
    
            # plt.plot(FrequencySampling, FrequencySampling*10**(-2), linestyle='--')
            ax1.set_yscale('log') #; ax1.set_ylim(1e-3,5)
            ax1.set_ylabel('FTrej')
            
            ax2.plot(FrequencySampling, ModFTBO)
            ax2.set_yscale('log') #; ax2.set_ylim(1e-3,5)
            ax2.set_ylabel("FTBO")
            
            ax3.plot(FrequencySampling, ModFTBF)
        
            ax3.set_xlabel('Frequencies [Hz]')
            ax3.set_ylabel('FTBF')
            ax3.set_xscale('log')
            ax3.set_yscale('log')
            ax1.grid(True) ; ax2.grid(True) ; ax3.grid(True)

            if figsave:
                prefix = details.replace("=","").replace(";","").replace(" ","").replace(".","").replace('\n','_').replace('Phase-delay','PD').replace('Group-delay','GD')
                figname = "TransferFunctions"
                if isinstance(ext,list):
                    for extension in ext:
                        plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
                else:
                    plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
    
            generaltitle=f'{details} - Temporal sampling used'
            plt.close(generaltitle)
            fig = plt.figure(generaltitle)
            ax1,ax2,ax3 = fig.subplots(nrows=3,sharex=True)
            
            ax1.plot(timestamps, Input)
            # plt.plot(FrequencySampling, FrequencySampling*10**(-2), linestyle='--')
        
            ax1.set_ylabel('Open loop')
            
            ax2.plot(timestamps, Output)
            ax2.set_ylabel("Close loop")
            
            ax3.plot(timestamps,Command)
        
            ax3.set_xlabel('Timestamps [s]')
            ax3.set_ylabel('Command')

            fig.show()
            
            """
            Display FTrej and PSD of input (disturbance) and output (residues) signals
            """
            
            plt.rcParams.update(rcParamsForBaselines)
            if not details:
                title = 'FTrej'
            else:
                title = f'FTrej - {details}'
            plt.close(title)
            fig = plt.figure(title)
            fig.suptitle(title)
            ax1,ax2,ax3 = fig.subplots(nrows=3,sharex=True)
            
            ax1.plot(FrequencySampling, 40*np.log10(ModFTrej), "k")
            # ax1.plot(FrequencySampling, 40*np.log10(ModFTBOfit), 
            #          color=colors[0], linestyle='--')
            
            if gain:
                gains = [gain] ; delays=np.arange(10,60,10)
                styles=['-','--',':']
                linestyles = []
                for ig in range(len(gains)):
                    gain=gains[ig]
                    linestyles.append(mlines.Line2D([],[],color='k',linestyle=styles[ig],label=f"Gain={gain}"))
                    for idel in range(len(delays)):
                        gain=gains[ig];delay=delays[idel]
                        ftr=model(FrequencySampling,delay,gain)
                        ax1.plot(FrequencySampling, ftr, color=colors[idel], linestyle=styles[ig])
                        if ig==len(gains)-1:
                            linestyles.append(mlines.Line2D([],[],color=colors[idel],linestyle='-',label=f'\u03C4={delay}'))
                ax1.legend(handles=linestyles)
    
            ax1.set_ylabel('Gain [dB]')
            ct.addtext(ax1,"|FTrej|²",loc='upper center',fontsize='x-large')
            
            ax2.plot(FrequencySampling, ModFTResidues**2, "k")
            
            ax2.set_ylabel("PSD [µm²/Hz]")
            ax2.set_yscale("log")
            ct.addtext(ax2,"Residues",loc='upper center',fontsize='x-large')
            
            
            ax3.plot(FrequencySampling, ModFTTurb**2, "k")
            ax3.plot(FrequencySampling, ModFTTurbfit**2, color=colors[0])
            ax3.set_ylabel('PSD [µm²/Hz]')
            ct.addtext(ax3,"Disturbances",loc='upper center',fontsize='x-large')
            
            
            ax3.set_xlabel('Frequencies [Hz]')
            ax3.set_xscale('log')
            ax3.set_yscale("log")
            
            ax1.grid(True) ; ax2.grid(True) ; ax3.grid(True)
            ax3.sharey(ax2)
            
            ct.setaxelim(ax3, ydata=[ModFTCommands**2,ModFTResidues**2])
            
            fig.show()
        
        if only == 'ftbo':
            
            
            """ Display gain and phase of FTBO """
            
            plt.rcParams.update(rcParamsForBaselines)
            if not details:
                title = 'Bode Diagrams FTBO'
            else:
                title = f'Bode Diagrams FTBO - {details}'
            
            plt.close(title)
            fig = plt.figure(title)
            fig.suptitle(title)
            ax1,ax2 = fig.subplots(nrows=2,sharex=True)
            
            ax1.plot(FrequencySampling, 40*np.log10(ModFTBO), "k")
            ax1.plot(FrequencySampling, 40*np.log10(ModFTBOfit), 
                     color=colors[0], linestyle='--')
            
            ax2.plot(FrequencySampling, AngleFTBO*180/np.pi, "k")
            
            ax1.set_ylabel('Gain [dB]')
            ax2.set_ylabel("Phase [°]")
            
            ax1.grid(True) ; ax2.grid(True)
            
            ax2.set_xscale('log')
            
            fig.show()
            
            if figsave:
                prefix = details.replace("=","").replace(";","").replace(" ","").replace(".","").replace('\n','_').replace('Phase-delay','PD').replace('Group-delay','GD')
                figname = f"BodeDiagrams_{only}"
                if isinstance(ext,list):
                    for extension in ext:
                        plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
                else:
                    plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
            
            """
            Display FTBO and PSD of command and residues
            """
            
            plt.rcParams.update(rcParamsForBaselines)
            if not details:
                title = 'FTBO'
            else:
                title = f'FTBO - {details}'
            plt.close(title)
            fig = plt.figure(title)
            fig.suptitle(title)
            ax1,ax2,ax3 = fig.subplots(nrows=3,sharex=True)
            
            ax1.plot(FrequencySampling, 40*np.log10(ModFTBO), "k")
            ax1.plot(FrequencySampling, 40*np.log10(ModFTBOfit), 
                     color=colors[0], linestyle='--')
            
            if gain:
                gains = [gain] ; delays=np.arange(10,60,10)
                styles=['-','--',':']
                linestyles = []
                for ig in range(len(gains)):
                    gain=gains[ig]
                    linestyles.append(mlines.Line2D([],[],color='k',linestyle=styles[ig],label=f"Gain={gain}"))
                    for idel in range(len(delays)):
                        gain=gains[ig];delay=delays[idel]
                        ftr=model(FrequencySampling,delay,gain)
                        ax1.plot(FrequencySampling, ftr, color=colors[idel], linestyle=styles[ig])
                        if ig==len(gains)-1:
                            linestyles.append(mlines.Line2D([],[],color=colors[idel],linestyle='-',label=f'\u03C4={delay}'))
                ax1.legend(handles=linestyles)
    
            ax1.set_ylabel('Gain [dB]')
            ct.addtext(ax1,"|FTBO|²",loc='upper center',fontsize='x-large')
            
            ax2.plot(FrequencySampling, ModFTResidues**2, "k")
            
            ax2.set_ylabel("PSD [µm²/Hz]")
            ax2.set_yscale("log")
            ct.addtext(ax2,"Residues",loc='upper center',fontsize='x-large')
            
            
            ax3.plot(FrequencySampling, ModFTCommands**2, "k")
            ax3.plot(FrequencySampling, ModFTCommandsfit**2, color=colors[0])
            ax3.set_ylabel('PSD [µm²/Hz]')
            ct.addtext(ax3,"Commands",loc='upper center',fontsize='x-large')
            
            
            ax3.set_xlabel('Frequencies [Hz]')
            ax3.set_xscale('log')
            ax3.set_yscale("log")
            
            ax1.grid(True) ; ax2.grid(True) ; ax3.grid(True)
            ax3.sharey(ax2)
            
            ct.setaxelim(ax3, ydata=[ModFTCommands**2,ModFTResidues**2])
            
            fig.show()
            
            if figsave:
                prefix = details.replace("=","").replace(";","").replace(" ","").replace(".","").replace('\n','_').replace('Phase-delay','PD').replace('Group-delay','GD')
                figname = f"Amplitude_{only}"
                if isinstance(ext,list):
                    for extension in ext:
                        plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
                else:
                    plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
            
            
            if displaytemporal:
                """
                Display temporal sequences of command and residues
                """
                 
                plt.rcParams.update(rcParamsForBaselines)
                if not details:
                    title = 'FTBO and temporal sequences'
                else:
                    title = f'FTBO and temporal sequences - {details}'
                plt.close(title)
                fig = plt.figure(title)
                fig.suptitle(title)
                ax1,axGhost,ax2,ax3 = fig.subplots(nrows=4, gridspec_kw={"height_ratios":[2,0.5,2,2]})
                axGhost.remove()
                
                ax1.plot(FrequencySampling, 20*np.log10(ModFTBO**2), "k")
                
                if gain:
                    gains = [gain] ; delays=np.arange(10,60,10)
                    styles=['-','--',':']
                    linestyles = []
                    for ig in range(len(gains)):
                        gain=gains[ig]
                        linestyles.append(mlines.Line2D([],[],color='k',linestyle=styles[ig],label=f"Gain={gain}"))
                        for idel in range(len(delays)):
                            gain=gains[ig];delay=delays[idel]
                            ftr=model(FrequencySampling,delay,gain)
                            ax1.plot(FrequencySampling, ftr, color=colors[idel], linestyle=styles[ig])
                            if ig==len(gains)-1:
                                linestyles.append(mlines.Line2D([],[],color=colors[idel],linestyle='-',label=f'\u03C4={delay}'))
                    ax1.legend(handles=linestyles)
                    
                #ax1.set_yscale('log')
                ax1.set_xscale('log')
                ax1.set_ylabel('G(|FTBO|²)')
                ax1.set_xlabel('Frequencies [Hz]')
                
                ax2.plot(timestamps, Output, "k")
                ax2.set_ylabel("Residues\n[µm]")
                ax2.set_ylim(-1.5*lmbda/2,1.5*lmbda/2)
                
                ax3.plot(timestamps, Command, "k")
                
                ax3.set_ylabel('Command\n[µm]')
                ax3.set_xlabel('Time [s]')
                ax2.sharex(ax3)
                
                ax1.grid(True) ; ax2.grid(True) ; ax3.grid(True)
                fig.show()
        
    return results


def PowerSpectralDensity(signal, timestamps, *AdditionalSignals, SignalName='Signal [µm]',
                         f1 = 0.3 , f2 = 5,
                         fbonds=[], details='', window='no',mov_average=0,model=True,
                         cumStd=False,
                         display=True, figsave=False, figdir='',ext='pdf'):
    
    nNT = len(timestamps) ; dt = np.mean(timestamps[1:]-timestamps[:-1])
    
    FrequencySampling1 = np.fft.fftfreq(nNT, dt)
    if len(fbonds):
        fmin, fmax = fbonds
    else:
        fmin=0
        fmax=np.max(FrequencySampling1)
    
    PresentFrequencies = (FrequencySampling1 > fmin) \
        & (FrequencySampling1 < fmax)
        
    FrequencySampling = FrequencySampling1[PresentFrequencies]
    
    if window =='hanning':
        windowsequence = np.hanning(nNT)
    else:
        windowsequence = np.ones(nNT)
        
    signal_filtered = signal*windowsequence
    
    PSD = 2*np.abs(np.fft.fft(signal_filtered,norm="forward")[PresentFrequencies])**2

    if cumStd:
        cumulativeStd = np.sqrt(np.cumsum(PSD))
        cumFrequencySampling = np.copy(FrequencySampling)

    if mov_average:
        PSD = moving_average(PSD, mov_average)
        FrequencySampling = moving_average(FrequencySampling,mov_average)

    # print(np.var(signal)) ; print(np.var(signal_filtered))
    # print(np.sum(np.abs(signal_filtered)**2*dt))
    # print(np.sum(np.abs(PSD)))
    
    # f1 = 0.3 ; f2 = 10
    logFrequencySampling = np.log10(FrequencySampling)
    NominalRegime = (FrequencySampling>f1)*(FrequencySampling<f2)
    coefs = np.polyfit(logFrequencySampling[NominalRegime], np.log10(PSD[NominalRegime]), 1)
    poly1d_fn = np.poly1d(coefs)
    psdFit = 10**poly1d_fn(logFrequencySampling)   # model sampled in direct space
    powPsd=coefs[0]
    # val0 = 10**coefs[0]#FTSignalfit[np.abs(np.argmin(logFrequencySampling-1))]
    
    if len(AdditionalSignals):
        addSig=[] ; addPSD=[] ; addCumStd=[]
    
    for sig in AdditionalSignals:
        addSig.append(sig)
        tempPSD = 2*np.abs(np.fft.fft(sig*windowsequence, norm="forward")[PresentFrequencies])**2
        
        if mov_average:
            addPSD.append(moving_average(tempPSD, mov_average))
        else:
            addPSD.append(tempPSD)
    
        addCumStd.append(np.cumsum(tempPSD))
    
    if display:
                      
        plt.rcParams.update(rcParamsForBaselines)
        # plt.rcParams.update({"figure.subplot.hspace":0.2})
        title = f'{details} - PSD'
        plt.close(title)
        fig = plt.figure(title)
        fig.suptitle(title)
        
        if not cumStd:
            ax1,ax2 = fig.subplots(nrows=2)
            
            ax1.plot(FrequencySampling, PSD,'k')
            ax2.plot(timestamps, signal, 'k')
            
            for i in range(len(AdditionalSignals)):
                ax1.plot(FrequencySampling, addPSD[i],color=colors[i])
                ax2.plot(timestamps, addSig[i], colors[i])
            
            if model:
                ax1.plot(FrequencySampling, psdFit)#10*FrequencySampling**(-8/3))
                annX = 2
                annYindex = np.argmin(np.abs(FrequencySampling-annX))
                annY = psdFit[annYindex]*2
                ax1.annotate(f"{round(powPsd*3,2)}/3",(annX,annY),color=colors[0])
            # ax3.plot(timestamps, windowsequence,'k--', label='Window')
            
            # ct.setaxelim(ax1, 
            ax1.set_yscale('log') ; ax1.set_xscale('log')
            ax1.set_ylabel('PSD [µm²/Hz]')
            ax2.set_ylabel(SignalName)
            #ax2.legend()#handles=linestyles)
            
            ax1.set_xlabel('Frequencies [Hz]')
            ax2.set_xlabel('Time (s)')
            #ax3.set_ylabel("Window amplitude")
            ax1.grid(True); ax2.grid(True);# ax3.grid(False)

        else:
            ax1,ax2,axGhost,ax3 = fig.subplots(nrows=4,gridspec_kw={"height_ratios":[5,5,1.5,5]})
            axGhost.axis("off")
            ax1.plot(FrequencySampling, PSD,'k')
            ax2.plot(cumFrequencySampling, cumulativeStd, 'k')
            ax3.plot(timestamps, signal, 'k')
            
            for i in range(len(AdditionalSignals)):
                ax1.plot(FrequencySampling, addPSD[i],color=colors[i])
                ax2.plot(cumFrequencySampling, addCumStd[i],color=colors[i])
                ax3.plot(timestamps, addSig[i], colors[i])
            
            if model:
                ax1.plot(FrequencySampling, psdFit)#10*FrequencySampling**(-8/3))
                annX = 2
                annYindex = np.argmin(np.abs(FrequencySampling-annX))
                annY = psdFit[annYindex]*2
                ax1.annotate(f"{round(powPsd*3,2)}/3",(annX,annY),color=colors[0])
            # ax3.plot(timestamps, windowsequence,'k--', label='Window')
            
            ax1.set_yscale('log') ; ax1.set_xscale('log')
            ax2.set_xscale('log')
            ax2.sharex(ax1) ;# ax1.tick_params.labelbottom(False)
            ax1.set_ylabel('PSD [µm²/Hz]')
            ax2.set_ylabel("Cumulative STD [µm]")
            ax3.set_ylabel(SignalName)
            
            ax2.set_xlabel('Frequencies [Hz]')
            ax3.set_xlabel('Time (s)')
            ax1.grid(True); ax2.grid(True); ax3.grid(True)

        if figsave:
            prefix = details.replace("=","").replace(";","").replace(" ","").replace(".","").replace('\n','_').replace('Phase-delay','PD').replace('Group-delay','GD')
            figname = "PSD"
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        
        fig.show()
    
    if cumStd:
        return FrequencySampling, PSD, cumStd, psdFit

    else:
        return FrequencySampling, PSD, psdFit


def FitAtmospherePsd(signal, timestamps, *AdditionalSignals, SignalName='Signal [µm]',
                     fbonds=[], details='', window='no',mov_average=0,model=True,
                     cumStd=False,
                     atmParams={"V":10,"L0":30,"d":1,"pows":(-2/3,-8/3,-17/3)},
                     display=True, figsave=False, figdir='',ext='pdf'):    
    
    nNT = len(timestamps) ; dt = np.mean(timestamps[1:]-timestamps[:-1])
    # T = timestamps[-1]
    
    FrequencySampling1 = np.fft.fftfreq(nNT, dt)
    if len(fbonds):
        fmin, fmax = fbonds
    else:
        fmin=0
        fmax=np.max(FrequencySampling1)
    
    PresentFrequencies = (FrequencySampling1 > fmin) \
        & (FrequencySampling1 < fmax)
        
    FrequencySampling = FrequencySampling1[PresentFrequencies]
    
    if window =='hanning':
        windowsequence = np.hanning(nNT)
    else:
        windowsequence = np.ones(nNT)
        
    signal_filtered = signal*windowsequence
    
    PSD = 2*np.abs(np.fft.fft(signal_filtered,norm="forward")[PresentFrequencies])**2

    if cumStd:
        cumulativeStd = np.sqrt(np.cumsum(PSD))
        cumFrequencySampling = np.copy(FrequencySampling)

    if mov_average:
        PSD = moving_average(PSD, mov_average)
        FrequencySampling = moving_average(FrequencySampling,mov_average)
        
    df = FrequencySampling[1]-FrequencySampling[0]
    # print(np.var(signal)) ; print(np.var(signal_filtered))
    # print(np.sum(np.abs(signal_filtered)**2*dt))
    # print(np.sum(np.abs(PSD)))
    
    V = atmParams["V"] ; L0 = atmParams["L0"] ; d=atmParams["d"]
    nu1 = V/L0                          # Low cut-off frequency
    nu2 = 0.3*V/d                       # High cut-off frequency
        
    if not "pows" in atmParams.keys():
        atmParams["pows"] = (-2/3, -8/3, -17/3) # Conan et al
    
    pows = atmParams["pows"]
    filtre = modelAtmosphere(FrequencySampling,nu1,nu2, pows)
    psdFit = filtre
    # f1 = 0.3 ; f2 = 10
    # logFrequencySampling = np.log10(FrequencySampling)
    # NominalRegime = (FrequencySampling>f1)*(FrequencySampling<f2)
    # coefs = np.polyfit(logFrequencySampling[NominalRegime], np.log10(PSD[NominalRegime]), 1)
    # poly1d_fn = np.poly1d(coefs)
    # psdFit = 10**poly1d_fn(logFrequencySampling)   # model sampled in direct space
    # val0 = 10**coefs[0]#FTSignalfit[np.abs(np.argmin(logFrequencySampling-1))]
    
    if len(AdditionalSignals):
        addSig=[] ; addPSD=[] ; addCumStd=[]
    
    for sig in AdditionalSignals:
        addSig.append(sig)
        tempPSD = 2*np.abs(np.fft.fft(sig*windowsequence, norm="forward")[PresentFrequencies])**2
        
        if mov_average:
            addPSD.append(moving_average(tempPSD, mov_average))
        else:
            addPSD.append(tempPSD)
    
        addCumStd.append(np.cumsum(tempPSD))
    
    if display:
        
        linestyles = [mlines.Line2D([],[],color='k',label="Signal"),
                      mlines.Line2D([],[],color='k',linestyle='--', label="Window")]
                      
        plt.rcParams.update(rcParamsForBaselines)
        # plt.rcParams.update({"figure.subplot.hspace":0.2})
        title = f'{details} - PSD'
        plt.close(title)
        fig = plt.figure(title)
        fig.suptitle(title)
        
        if not cumStd:
            ax1,ax2 = fig.subplots(nrows=2)
            
            ax1.plot(FrequencySampling, PSD,'k')
            ax2.plot(timestamps, signal, 'k')
            
            for i in range(len(AdditionalSignals)):
                ax1.plot(FrequencySampling, addPSD[i],color=colors[i])
                ax2.plot(timestamps, addSig[i], colors[i])
            
            if model:
                ax1.plot(FrequencySampling, filtre)#10*FrequencySampling**(-8/3))
            # ax3.plot(timestamps, windowsequence,'k--', label='Window')
            
            # ct.setaxelim(ax1, 
            ax1.set_yscale('log') ; ax1.set_xscale('log')
            ax1.set_ylabel('PSD [µm²/Hz]')
            ax2.set_ylabel(SignalName)
            #ax2.legend()#handles=linestyles)
            
            ax1.set_xlabel('Frequencies [Hz]')
            ax2.set_xlabel('Time (s)')
            #ax3.set_ylabel("Window amplitude")
            ax1.grid(True); ax2.grid(True);# ax3.grid(False)

        else:
            ax1,ax2,axGhost,ax3 = fig.subplots(nrows=4,gridspec_kw={"height_ratios":[5,5,1.5,5]})
            axGhost.axis("off")
            ax1.plot(FrequencySampling, PSD,'k')
            ax2.plot(cumFrequencySampling, cumulativeStd, 'k')
            ax3.plot(timestamps, signal, 'k')
            
            for i in range(len(AdditionalSignals)):
                ax1.plot(FrequencySampling, addPSD[i],color=colors[i])
                ax2.plot(cumFrequencySampling, addCumStd[i],color=colors[i])
                ax3.plot(timestamps, addSig[i], colors[i])
            
            if model:
                ax1.plot(FrequencySampling, psdFit)#10*FrequencySampling**(-8/3))
            # ax3.plot(timestamps, windowsequence,'k--', label='Window')
            
            ax1.set_yscale('log') ; ax1.set_xscale('log')
            ax2.set_xscale('log')
            ax2.sharex(ax1) ;# ax1.tick_params.labelbottom(False)
            ax1.set_ylabel('PSD [µm²/Hz]')
            ax2.set_ylabel("Cumulative STD [µm]")
            ax3.set_ylabel(SignalName)
            
            ax2.set_xlabel('Frequencies [Hz]')
            ax3.set_xlabel('Time (s)')
            ax1.grid(True); ax2.grid(True); ax3.grid(True)

        if figsave:
            prefix = details.replace("=","").replace(";","").replace(" ","").replace(".","").replace('\n','_').replace('Phase-delay','PD').replace('Group-delay','GD')
            figname = "PSD"
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        
        fig.show()
    
    if cumStd:
        return FrequencySampling, PSD, cumStd, psdFit

    else:
        return FrequencySampling, PSD, psdFit


def modelAtmosphere(FrequencySampling, nu1,nu2,pows):
    
    """"""""""""""""""""""""
    """  modelAtmosphere """
    """"""""""""""""""""""""
    
    # V = 0.31*r0/t0*1e3              # Average wind velocity in its direction [m/s]
    pow1, pow2, pow3 = pows
        
    b0 = nu1**(pow1-pow2)           # offset for continuity
    b1 = b0*nu2**(pow2-pow3)        # offset for continuity

    # if verbose:
    #     print(f'Atmospheric cutoff frequencies: {nu1:.2}Hz and {nu2:.2}Hz')
    
    nNT = len(FrequencySampling)
    filtre = np.zeros(nNT)
    
    # Define the three frequency regimes
    lowregim = (np.abs(FrequencySampling)>0) * (np.abs(FrequencySampling)<nu1)
    medregim = (np.abs(FrequencySampling)>=nu1) * (np.abs(FrequencySampling)<nu2)
    highregim = np.abs(FrequencySampling)>=nu2
    
    filtre[lowregim] = np.abs(FrequencySampling[lowregim])**pow1
    filtre[medregim] = np.abs(FrequencySampling[medregim])**pow2*b0
    filtre[highregim] = np.abs(FrequencySampling[highregim])**pow3*b1
    
    return filtre



def SpectralAnalysis(BOTelemetries, BFTelemetries, infos, details='',
                     bonds=[],window='no',
                     display=True, 
                     figsave=False, figdir='', ext='pdf'):
    
    from config import NA
    
    TelConventionalArrangement = config.InterfArray.TelNames
    telNameLength=len(TelConventionalArrangement[0])
    if 'TelescopeArrangement' in infos.keys():
        TelArrangement = infos['TelescopeArrangement']
    elif 'Beam2Tel' in vars(config):
        telNameLength2 = int(len(config.Beam2Tel)/NA)
        TelArrangement = [config.Beam2Tel[i:i+telNameLength2] for i in range(0, NA, telNameLength2)]
    else:
        TelArrangement = TelConventionalArrangement
        
    base = infos['Base']
    
    if 'gains' in infos.keys():
        gainGD,gainPD = infos['gains']
    else:
        gainGD,gainPD = 0,0
        
    tel1, tel2 = base[:telNameLength],base[telNameLength:]
        
    iTel1mes = np.argwhere(tel1==np.array(TelArrangement))[0][0]
    iTel2mes = np.argwhere(tel2==np.array(TelArrangement))[0][0]
    
    iTel1cmd = np.argwhere(tel1==np.array(TelConventionalArrangement))[0][0]
    iTel2cmd = np.argwhere(tel2==np.array(TelConventionalArrangement))[0][0]
    
    iBaseCmd = ct.posk(iTel1cmd,iTel2cmd,NA)
    iBaseMes = ct.posk(iTel1mes,iTel2mes,NA)
    
    timestamps = BOTelemetries["timestamps"]
    
    if len(bonds):
        timerange = range(bonds[0],bonds[1])
    else:
        timerange = range(0,len(timestamps))
    
    t = timestamps[timerange]
    
    BOgd=BOTelemetries["gD"][timerange,iBaseMes]*R*lmbda/2/np.pi #microns
    BOpd=BOTelemetries["pD"][timerange,iBaseMes]*lmbda/2/np.pi #microns
    BFgd=BFTelemetries["gD"][timerange,iBaseMes]*R*lmbda/2/np.pi #microns
    BFpd=BFTelemetries["pD"][timerange,iBaseMes]*lmbda/2/np.pi #microns

    Commandgd=BFTelemetries["OPDCmdGD"][:,iBaseCmd] #microns
    Commandpd=BFTelemetries["OPDCmdPD"][:,iBaseCmd] #microns

    fbonds = (0,800)
    
    details = f"{infos['details']}\nGroup-delay control"
    resultsgd = BodeDiagrams(BFgd, Commandgd, t, Input=BOgd, gain=gainGD,
                             fbonds=fbonds, window=window,
                             details=details,
                             display=display,only=False,
                             figsave=figsave, figdir=figdir, ext=ext)
    
    FrequencySampling, FTrejgd, FTBOgd, FTBFgd = resultsgd
    
    details = f"{infos['details']}\nPhase-delay control"
    resultspd = BodeDiagrams(BFpd, Commandpd, t, Input=BOgd, gain=gainPD,
                             fbonds=fbonds, window=window,
                             details=details,
                             display=display,only=False,
                             figsave=figsave, figdir=figdir, ext=ext)
    
    FrequencySampling, FTrejpd, FTBOpd, FTBFpd = resultspd
    
    
    # details = f"{infos['details']}\nPhase-delay control 2"
    # resultspd2 = BodeDiagrams(BFpd, Commandpd, t, Input=BOpd, gain=gainPD,
    #                          fbonds=fbonds, window='no',
    #                          details=details,
    #                          display=display,
    #                          figsave=figsave, figdir=figdir, ext=ext)
    
    # FrequencySampling, FTrejpd, FTBOpd, FTBFpd = resultspd2
    
    details = f"{infos['details']}\nOpen loop GD"
    PSDbo = PowerSpectralDensity(BOgd, t, SignalName='Group-delay [µm]',
                                 fbonds=fbonds, window=window, 
                                 details=details,
                                 display=display,
                                 figsave=figsave, figdir=figdir, ext=ext)
    
    details = f"{infos['details']}\nCommand PD"
    PSDpd = PowerSpectralDensity(Commandpd, t, SignalName='Command PD [µm]',
                                 fbonds=fbonds, window=window, 
                                 details=details,
                                 display=display,
                                 figsave=figsave, figdir=figdir, ext=ext)
    
    details = f"{infos['details']}\nResidues PD"
    PSDpdres = PowerSpectralDensity(BFpd, t, SignalName='Residues PD [µm]',
                                 fbonds=fbonds, window=window, 
                                 details=details,
                                 display=display,
                                 figsave=figsave, figdir=figdir, ext=ext)
    
    results = {"FrequencySampling":FrequencySampling,
               "FTrejgd":FTrejgd, "FTBOgd":FTBOgd, "FTBFgd":FTBFgd,
               "FTrejpd":FTrejpd, "FTBOpd":FTBOpd, "FTBFpd":FTBFpd,
               "PSDbo":PSDbo, "PSDpd":PSDpd}
    
    return results


def setaxelim(ax, xdata=[],ydata=[],xmargin=0.1,ymargin=0.1, absmargin=False,**kwargs):
    
    if len(xdata):
        
        if isinstance(xdata,list):
            xvalues = xdata[0].ravel()
            for arr in xdata:
                xvalues = np.concatenate((xvalues, arr.ravel()))
        else:
            xvalues = xdata
        
        xmin = (1+xmargin)*np.min(xvalues) ; xmax = (1+xmargin)*np.max(xvalues)
        ax.set_xlim([xmin,xmax])
        
    if ydata != []:
        if isinstance(ydata,float):
            ymax=(1+ymargin)*np.abs(ydata)
            
            if 'ymin' in kwargs.keys():
                ymin=kwargs['ymin']
            else:
                ymin = -ymax

        else:
            if isinstance(ydata,list):
                yvalues = ydata[0].ravel()
                for arr in ydata:
                    yvalues = np.concatenate((yvalues, arr.ravel()))
            else:
                yvalues=ydata
            
            if absmargin:
                ymax = (1+ymargin)*np.max(np.abs(yvalues))
                ymin = -ymax
            else:
                ymax = (1+ymargin)*np.max(yvalues)
                if not 'ymin' in kwargs.keys():
                    ymin = (1+ymargin)*np.min(yvalues)
                else:
                    ymin=kwargs['ymin']
            
        ax.set_ylim([ymin,ymax])


def LoadData(file, STDbonds, DIT=0, version='current'):

    STDbonds=list(STDbonds)
    data = readDump(file,version=version)

    nbframe = np.shape(data['gD'])[0]-1
    timerange = range(nbframe)

    """LOAD TEMPORAL SAMPLING INFORMATION"""
    tBefore = data['tBeforeProcessFrameCall'][:nbframe,0] + data['tBeforeProcessFrameCall'][:nbframe,1]*1e-9  #Timestamps of the data (at frame reception)
    tAfter = data['tAfterProcessFrameCall'][:nbframe,0] + data['tAfterProcessFrameCall'][:nbframe,1]*1e-9  #Timestamps of the data (when the command is sent)
    
    timestamps = tBefore-tBefore[0]        # Times in seconds
    
    t=timestamps
    dt=np.mean(timestamps[1:]-timestamps[:-1]) # Temporal sampling in seconds



    """LOAD TELEMETRIES"""
    KpD = data['KpD'] ; KgD = data['KgD']
    gD = data['gD'][:nbframe] ; pD=data['pD'][:nbframe] ; gdClosure=data['gdClosure'][:nbframe] ; pdClosure=data['pdClosure'][:nbframe]
    photometry = data['photometry'][:nbframe]
    
    gdDlCmdMicrons = data['gdDlCmdMicrons'][:nbframe] ; pdDlCmdMicrons = data['pdDlCmdMicrons'][:nbframe]
    normVisibility = data['normVisibility'][:nbframe]

    OPDCmdGD = np.zeros([nbframe,NIN]) ; OPDCmdPD = np.zeros([nbframe,NIN])
    for iFrame in range(nbframe):
        OPDCmdGD[iFrame] = np.dot(Piston2OPD,gdDlCmdMicrons[iFrame])
        OPDCmdPD[iFrame] = np.dot(Piston2OPD,pdDlCmdMicrons[iFrame])

    averagePdVar = data['averagePdVar'][:nbframe] ; curRefPD=data['curRefPD'][:nbframe] ; curRefGD=data['curRefGD'][:nbframe]
    curGdErrBaseMicrons = data['curGdErrBaseMicrons'][:nbframe]
    curPdErrBaseMicrons = data['curPdErrBaseMicrons'][:nbframe]

    curGdPistonMicrons = data['curGdPistonMicrons'][:nbframe]
    curRefGDPistonMicrons = data['curRefGDPistonMicrons'][:nbframe]

    visLambSumMeanSqNorm = data['visLambSumMeanSqNorm'][:nbframe]
    visVarDitMean = data['visVarDitMean'][:nbframe]
    avVisSpectralSqNorm = data['avVisSpectralSqNorm'][:nbframe]

    """COMPUTE STANDARD-DEVIATIONS"""
    if STDbonds[1] > nbframe:
        STDbonds[1]=nbframe
    
    if DIT:
        FrameLength = int(DIT//dt)
        Nframes = int(nbframe // FrameLength)
        
    else:
        FrameLength = STDbonds[1]-STDbonds[0]
        Nframes = 1
        
    print("Nframes:",Nframes)
    print("FrameLength",FrameLength)

    RMSgd = 0 ; RMSpd = 0 ; RMSgdc = 0 ; Meangdc = 0 ; RMSpdc = 0
    Meanpdc = 0 ; RMSgdref = 0 ; Meangdref = 0 ; RMSpdref = 0
    Meanpdref = 0 ; RMSgderr = 0 ; RMSpderr = 0 ; RMSphot = 0
    MeanPhot = 0 ; MeanSquaredSNR = 0 ; RMSSquaredSNR = 0

    FrameIn = STDbonds[0]
    for iframe in range(Nframes):
        FrameOut = FrameIn + FrameLength
        
        rms_interval_calculation = range(FrameIn,FrameOut)
        
        RMSgd += np.std(gD[rms_interval_calculation,:],axis=0)/Nframes
        RMSpd += np.std(pD[rms_interval_calculation,:],axis=0)/Nframes
        RMSgdc += np.std(gdClosure[rms_interval_calculation,:],axis=0)/Nframes
        Meangdc += np.mean(gdClosure[rms_interval_calculation,:],axis=0)/Nframes
        RMSpdc += np.std(pdClosure[rms_interval_calculation,:],axis=0)/Nframes
        Meanpdc += np.mean(pdClosure[rms_interval_calculation,:],axis=0)/Nframes
        RMSgdref += np.std(curRefGD[rms_interval_calculation,:],axis=0)/Nframes
        Meangdref += np.mean(curRefGD[rms_interval_calculation,:],axis=0)/Nframes
        RMSpdref += np.std(curRefPD[rms_interval_calculation,:],axis=0)/Nframes
        Meanpdref += np.mean(curRefPD[rms_interval_calculation,:],axis=0)/Nframes
        RMSgderr += np.std(curGdErrBaseMicrons[rms_interval_calculation,:],axis=0)/Nframes
        RMSpderr += np.std(curPdErrBaseMicrons[rms_interval_calculation,:],axis=0)/Nframes
        RMSphot += np.std(photometry[rms_interval_calculation,:],axis=0)/Nframes
        MeanPhot += np.mean(photometry[rms_interval_calculation,:],axis=0)/Nframes
        MeanSquaredSNR += np.mean(1/averagePdVar[rms_interval_calculation,:],axis=0)/Nframes
        RMSSquaredSNR += np.std(1/averagePdVar[rms_interval_calculation,:],axis=0)/Nframes
    
        FrameIn += FrameLength    
        
        
    Telemet = {
        "nbframe":nbframe,
        "timerange":timerange,
        "tBefore":tBefore,
        "tAfter":tAfter,
        "timestamps":timestamps,
        "dt":dt,
        "gD":gD,
        "pD":pD,
        "photometry":photometry,
        "gdClosure":gdClosure,
        "pdClosure":pdClosure,
        "gdDlCmdMicrons":gdDlCmdMicrons,
        "pdDlCmdMicrons":pdDlCmdMicrons,
        "normVisibility":normVisibility,
        "OPDCmdGD":OPDCmdGD,
        "OPDCmdPD":OPDCmdPD,
        "averagePdVar":averagePdVar,
        "visLambSumMeanSqNorm":visLambSumMeanSqNorm,
        "visVarDitMean":visVarDitMean,
        "avVisSpectralSqNorm":avVisSpectralSqNorm,
        "KpD":KpD,
        "KgD":KgD,
        "curRefPD":curRefPD,
        "curRefGD":curRefGD,
        "curGdErrBaseMicrons":curGdErrBaseMicrons,
        "curPdErrBaseMicrons":curPdErrBaseMicrons,
        "curGdPistonMicrons":curGdPistonMicrons,
        "curRefGDPistonMicrons":curRefGDPistonMicrons,
        "RMSgd":RMSgd,
        "RMSpd":RMSpd,
        "RMSgdc":RMSgdc,
        "Meangdc":Meangdc,
        "RMSpdc":RMSpdc,
        "Meanpdc":Meanpdc,
        "RMSgdref":RMSgdref,
        "Meangdref":Meangdref,
        "RMSpdref":RMSpdref,
        "Meanpdref":Meanpdref,
        "RMSgderr":RMSgderr,
        "RMSpderr":RMSpderr,
        "RMSphot":RMSphot,
        "MeanPhot":MeanPhot,
        "MeanSquaredSNR":MeanSquaredSNR,
        "RMSSquaredSNR":RMSSquaredSNR
    }
    
    return Telemet



def DisplayAll(Telemetries, infos, *args, figsave=False,figdir='',ext='pdf',**kwargs):
    
    """ Set the directory of the saved figures """
    if not len(figdir):
        figdir=os.getcwd()+'figures/'
        
    details = infos['details']
    
    prefix = details.replace("=","").replace(";","").replace(" ","").replace(".","")
    displayall = False
    if not len(args):
        displayall=True
    
    
    """
    Telescope Arrangements
    """
    
    TelConventionalArrangement = config.InterfArray.TelNames
    if 'TelescopeArrangement' in infos.keys():
        tels = infos['TelescopeArrangement']
    else:
        tels = TelConventionalArrangement
        
    Tel2Beam = np.zeros([6,6])
    for ia in range(NA):
        tel = tels[ia] ; tel0 = TelConventionalArrangement[ia] 
        pos = np.argwhere(np.array(tels)==tel0)[0][0]
        Tel2Beam[pos,ia]=1

    
    baselines = [] ; baselinesForPDref=[0]*(NA-1)
    itel=0
    for tel1 in tels:
        for tel2 in tels[itel+1:]:
            baselines.append(f'{tel1}{tel2}')
        itel+=1
    
    # For distribute baselines between two subplots
    NIN = len(baselines)
    len2 = NIN//2 ; len1 = NIN-len2
    
    closures = []  ; baselinesForPDref=baselines.copy()
    tel1=tels[0] ; itel1=0 ; itel2=itel1+1 
    for tel2 in tels[itel2:]:
        itel3=itel2+1
        for tel3 in tels[itel2+1:]:
            closures.append(f'{tel1}{tel2}{tel3}')
            ib = ct.posk(itel2, itel3, NA)
            baselinesForPDref[ib] = baselinesForPDref[ib] + f"\n({tel1}{tel2}{tel3})"
            itel3+=1
        itel2+=1
    
    
    PlotTel = [False]*NA ; PlotTelOrigin=[False]*NA
    PlotBaseline = [False]*NIN
    PlotClosure = [False]*NC
    TelNameLength = 2
    
    if 'TelsToDisplay' in infos.keys():
        TelsToDisplay = infos['TelsToDisplay']
        for ia in range(NA):
            tel = tels[ia] ; tel2 = TelConventionalArrangement[ia]
            if tel in TelsToDisplay:
                PlotTel[ia]=True
            if tel2 in TelsToDisplay:  
                PlotTelOrigin[ia]=True
                
        if not 'BaselinesToDisplay' in infos.keys():
            for ib in range(NIN):
                baseline = baselines[ib]
                tel1,tel2=baseline[:TelNameLength],baseline[TelNameLength:]
                if (tel1 in TelsToDisplay) \
                    and (tel2 in TelsToDisplay):
                        PlotBaseline[ib] = True
                    
        if not 'ClosuresToDisplay' in infos.keys():
            for ic in range(NC):
                closure = closures[ic]
                tel1,tel2,tel3=closure[:TelNameLength],closure[TelNameLength:2*TelNameLength],closure[2*TelNameLength:]
                if (tel1 in TelsToDisplay) \
                    and (tel2 in TelsToDisplay) \
                        and (tel3 in TelsToDisplay):
                            PlotClosure[ic] = True
                
    if 'BaselinesToDisplay' in infos.keys():
        BaselinesToDisplay = infos['BaselinesToDisplay']
        for ia in range(NA):
            tel = tels[ia] ; tel2 = TelConventionalArrangement[ia]
            if tel in "".join(BaselinesToDisplay):
                PlotTel[ia]=True
            if tel2 in "".join(BaselinesToDisplay):  
                PlotTelOrigin[ia]=True
                    
        for ib in range(NIN):
            baseline = baselines[ib]
            if baseline in BaselinesToDisplay:
                PlotBaseline[ib] = True
        
        if not 'ClosuresToDisplay' in infos.keys():
            for ic in range(NC):
                closure = closures[ic]
                base1, base2,base3=closure[:2*TelNameLength],closure[TelNameLength:],"".join([closure[:TelNameLength],closure[2*TelNameLength:]])
                if (base1 in BaselinesToDisplay) \
                    and (base2 in BaselinesToDisplay) \
                        and (base3 in BaselinesToDisplay):
                            PlotClosure[ic] = True
                            
    if 'ClosuresToDisplay' in infos.keys():
        ClosuresToDisplay = infos['ClosuresToDisplay']
        for ia in range(NA):
            tel = tels[ia] ; tel2 = TelConventionalArrangement[ia]
            if tel in "".join(ClosuresToDisplay):
                PlotTel[ia]=True
            if tel2 in "".join(ClosuresToDisplay):
                PlotTelOrigin[ia]=True
        
        for ib in range(NIN):
            baseline = baselines[ib]
            for closure in ClosuresToDisplay:
                if baseline in closure:
                    PlotBaseline[ib] = True
        
        for ic in range(NC):
            closure = closures[ic]
            if closure in ClosuresToDisplay:
                PlotClosure[ic] = True
                
    if not (('TelsToDisplay' in infos.keys()) \
            or ('BaselinesToDisplay' in infos.keys()) \
                or ('ClosuresToDisplay' in infos.keys())):
        PlotTel = [True]*NA ; PlotTelOrigin = [True]*NA
        PlotBaseline = [True]*NIN
        PlotClosure = [True]*NC
        
    PlotBaselineIndex = np.argwhere(PlotBaseline).ravel()
    
    """All values in Telemetries become local variables"""
    for key,val in Telemetries.items():
        globals()[key] = val
    t = timestamps
            
    OPDCmdGD_rearranged = np.zeros([nbframe,NIN])
    OPDCmdPD_rearranged = np.zeros([nbframe,NIN])
    pdDlCmdMicrons_rearranged = np.zeros([nbframe,NA])
    gdDlCmdMicrons_rearranged = np.zeros([nbframe,NA])
    
    for iFrame in range(nbframe):
        pdDlCmdMicrons_rearranged[iFrame] = np.dot(Tel2Beam,pdDlCmdMicrons[iFrame])
        gdDlCmdMicrons_rearranged[iFrame] = np.dot(Tel2Beam,gdDlCmdMicrons[iFrame])
        OPDCmdPD_rearranged[iFrame] = np.dot(Piston2OPD,pdDlCmdMicrons_rearranged[iFrame])
        OPDCmdGD_rearranged[iFrame] = np.dot(Piston2OPD,gdDlCmdMicrons_rearranged[iFrame])   
    
    ylimSNR = [0,1.1*np.max(1/averagePdVar[timerange[200:],:])]
    maxabs = 1.1*np.max(np.abs(gD[timerange[200:],:]*R*lmbda/2/np.pi))
    ylimGD = [-maxabs,+maxabs]
    ylimPD = [-1.3*lmbda/2,1.3*lmbda/2]

    xlabelpad = -3

    
    if 'which' in args:
        print("Possible entry parameters:\n\
                - 'phot': Photometries and standard-deviation.\n\
                - 'gains': Gains PD and GD and SNR PD and GD thresholds. \n\
                - 'GDonly': Only the GD and SNR values.\n\
                - 'PDonly': Only the PD and SNR values.\n\
                - 'GDPD': GD and PD estimators (eq.11 & 16), SNR and their standard-deviations.\n\
                - 'GDPDwithoutSNR': GD and PD estimators (eq.11 & 16) and their standard-deviations.\n\
                - 'GDPDerr': GD and PD errors (eq.35), SNR and their standard-deviations.\n\
                - 'GDPDdiffRef': GD and PD errors (estimator - reference), SNR and standard-deviations.\n\
                - 'GDPDref': GD and PD reference vectors, SNR and their standard-deviations.\n\
                - 'PisCommands': Commands in Piston space.\n\
                - 'OpdCommands': Commands in OPD space (computed from Piston commands).\n\
                - 'vis': Norm of the visibilities.\n\
                - 'cp': Closure Phases and their standard-deviations.")
        return
    

    """ Photometries """
    if displayall or ('phot' in args):
        plt.rcParams.update(rcParamsForBaselines)
        title=f'Photometries - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        ax1,ax2 = fig.subplots(nrows=2, gridspec_kw={"height_ratios":[4,1.5]})

        for iTel in range(NA):
            if PlotTel[iTel]:
                ax1.plot(t[timerange],photometry[timerange,iTel],color=telcolors[iTel], label=f"Tel{iTel+1}")

            #ax2.errorbar(tels[iTel], MeanPhot[iTel], yerr=RMSphot[iTel],fmt='x')
        ax2.bar(tels,MeanPhot, color=telcolors, yerr=RMSphot)
        
        ax1.set_ylabel('Photometries [ADU]')
        ax2.set_ylabel("<F>")
        ax1.set_xlabel('Time [s]') ; ax2.set_xlabel('Telescope')
        ax2.set_box_aspect(1/15)
        
        figname = '_'.join(title.split(' ')[:3])
        figname = 'Photometries'
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()
    
    if displayall or ("gains" in args):
        ylimGainsPD = (0,1) ; ylimGainsGD=(0,0.1)
        SNRpd = np.ones_like(gD)*1.5 ; SNRgd = np.ones_like(gD)*2
        plt.rcParams.update(rcParamsForBaselines)
        title=f'Gains - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        ax1,ax2 = fig.subplots(nrows=2,ncols=1, sharex=True,gridspec_kw={"height_ratios":[1,1]})
        ax3 = ax1.twinx()
        
        linestyles = [mlines.Line2D([],[],color=telcolors[0],label='$G_{pd}$'),
                      mlines.Line2D([],[],color=telcolors[1],label='$G_{gd}$'),
                      mlines.Line2D([],[],color=telcolors[2],label='$SNR_{pd}$'),
                      mlines.Line2D([],[],color=telcolors[3],label='$SNR_{gd}$')]
        
        ax1.plot(t[timerange],KpD[timerange],color=telcolors[0])
        ax3.plot(t[timerange],KgD[timerange],color=telcolors[1])
        ax2.plot(t[timerange],SNRpd[timerange],color=telcolors[2])
        ax2.plot(t[timerange],SNRgd[timerange],color=telcolors[3])

        ax1.set_ylim(ylimGainsPD) ; ax3.set_ylim(ylimGainsGD) ; 
        ct.setaxelim(ax2,ydata=np.concatenate([SNRpd,SNRgd]),ymin=0)
        
        ax1.set_ylabel('Gains PD')
        ax2.set_ylabel('SNR thresholds')
        ax3.set_ylabel('Gains GD')
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")
        # ax5.set_anchor('S') ; ax6.set_anchor('S')
        # ax5.set_box_aspect(1/20) ; ax6.set_box_aspect(1/20)
        ax2.set_xlabel("Time (s)", labelpad=xlabelpad) 
    
        ax2.legend(handles=linestyles)
    
        figname = '_'.join(title.split(' ')[:3])
        figname = f"Gains&Threholds"
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()

    
    if displayall or ("GDonly" in args):
        plt.rcParams.update(rcParamsForBaselines)
        title=f'Only GD - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        (ax1,ax3),(ax2,ax4), (ax5,ax6) = fig.subplots(nrows=3,ncols=2, gridspec_kw={"height_ratios":[2,2,1]})
        ax1.set_title("First serie of baselines, from 12 to 25")
        ax3.set_title("Second serie of baselines, from 26 to 56")


        for iBase in range(len1):   # First serie
            if PlotBaseline[iBase]:
                ax1.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
                ax2.plot(t[timerange],gD[timerange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])

        for iBase in range(len1,NIN):   # Second serie
            if PlotBaseline[iBase]:
                ax3.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
                ax4.plot(t[timerange],gD[timerange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])


        p1=ax5.bar(baselines[:len1],RMSgd[:len1]*R*lmbda/2/np.pi, color=basecolors[:len1])
        p2=ax6.bar(baselines[len1:],RMSgd[len1:]*R*lmbda/2/np.pi, color=basecolors[len1:])
        

        ax1.sharex(ax2) ; ax3.sharex(ax4)
        ax1.set_ylim(ylimSNR) ; 
        ax3.sharey(ax1) ; ax3.tick_params(labelleft=False)
        ax4.sharey(ax2) ; ax4.tick_params(labelleft=False)
        ax6.sharey(ax5) ; ax6.tick_params(labelleft=False)

        ax4.set_ylim(ylimGD)
        ct.setaxelim(ax5, ydata=RMSgd*R*lmbda/2/np.pi, ymargin=0.4,ymin=0)

        ax1.set_ylabel('SNR²')
        ax2.set_ylabel('Group-Delays [µm]')
        ax5.set_ylabel('GD rms\n[µm]') ;
        ax5.bar_label(p1,label_type='edge',fmt='%.2f')
        ax6.bar_label(p2,label_type='edge',fmt='%.2f')
        ax5.set_anchor('S') ; ax6.set_anchor('S')
        ax5.set_box_aspect(1/8) ; ax6.set_box_aspect(1/8)
        ax2.set_xlabel("Time (s)", labelpad=xlabelpad) ; ax4.set_xlabel("Time (s)", labelpad=xlabelpad)

        figname = '_'.join(title.split(' ')[:3])
        figname = f"GDonly"
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()
    
    if displayall or ("GDhist2" in args):
        plt.rcParams.update(rcParamsForBaselines)
        title=f'Histogram GD - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        (ax1,ax3),(ax2,ax4), (ax5,ax6) = fig.subplots(nrows=3,ncols=2, gridspec_kw={"height_ratios":[1,2,2]})
        ax1.set_title("First serie of baselines, from 12 to 25")
        ax3.set_title("Second serie of baselines, from 26 to 56")


        for iBase in range(len1):   # First serie
            if PlotBaseline[iBase]:
                ax1.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
                ax2.plot(t[timerange],gD[timerange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                ax5.hist(gD[timerange,iBase]*R*lmbda/2/np.pi, bins=1000,range=(-5,5),color=basecolors[iBase])
                
        for iBase in range(len1,NIN):   # Second serie
            if PlotBaseline[iBase]:
                ax3.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
                ax4.plot(t[timerange],gD[timerange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                ax6.hist(gD[timerange,iBase]*R*lmbda/2/np.pi, bins=1000, range=(-5,5),color=basecolors[iBase])
        

        ax1.sharex(ax2) ; ax3.sharex(ax4)
        ax1.set_ylim(ylimSNR) ; 
        ax3.sharey(ax1) ; ax3.tick_params(labelleft=False)
        ax4.sharey(ax2) ; ax4.tick_params(labelleft=False)
        ax6.sharey(ax5) ; ax6.tick_params(labelleft=False)

        ax4.set_ylim(ylimGD)
        # setaxelim(ax5, ydata=RMSgd*R*lmbda/2/np.pi, ymargin=0.4,ymin=0)

        ax1.set_ylabel('SNR²')
        ax2.set_ylabel('Group-Delays [µm]')
        ax5.set_xlabel('Group-Delays [µm]') 
        # ax5.set_anchor('S') ; ax6.set_anchor('S')
        # ax5.set_box_aspect(1/8) ; ax6.set_box_aspect(1/8)
        ax2.set_xlabel("Time (s)", labelpad=xlabelpad) ; ax4.set_xlabel("Time (s)", labelpad=xlabelpad)

        figname = '_'.join(title.split(' ')[:3])
        figname = f"GDonly"
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()
    

    if displayall or ("GDhist" in args):
        plt.rcParams.update(rcParamsForBaselines)
        title=f'Histogram GD - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        axes = fig.subplots(nrows=4,ncols=4, sharex=True, sharey=True)
        axes[-1,-1].remove()
        axes = axes.ravel()
        
        pmax=0
        for iBase in range(NIN):   # First serie
            p=axes[iBase].hist(gD[timerange,iBase]*R*lmbda/2/np.pi, bins=1000, range=(-5,5),color='k')
            if np.max(p[0]) > pmax:
                pmax = np.max(p[0])
                
            if iBase%4 == 0:
                axes[iBase].set_ylabel("Occurences")
                
            if iBase > 4*3-1:
                axes[iBase].set_xlabel("Group-delays [µm]")
            
            
        for iBase in range(NIN):
            axes[iBase].annotate(baselines[iBase],(-4,0.9*pmax))
        
        # ax1.sharex(ax2) ; ax3.sharex(ax4)
        axes[0].set_ylim(0,1.1*pmax) ; 
        axes[0].set_xlim(-5,5)
        # ax3.sharey(ax1) ; ax3.tick_params(labelleft=False)
        # ax4.sharey(ax2) ; ax4.tick_params(labelleft=False)
        # ax6.sharey(ax5) ; ax6.tick_params(labelleft=False)

        # setaxelim(ax5, ydata=RMSgd*R*lmbda/2/np.pi, ymargin=0.4,ymin=0)

        figname = '_'.join(title.split(' ')[:3])
        figname = f"Histogram GD"
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()

    
    
    if displayall or ("SNRonly" in args):
        plt.rcParams.update(rcParamsForBaselines)
        title=f'Squared SNR - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        if len(details):
            fig.suptitle(title)
        OneAxe = False ; NumberOfBaselinesToPlot = np.sum(PlotBaseline)
        if NumberOfBaselinesToPlot < len1:
            OneAxe=True
            ax1, ax2 = fig.subplots(nrows=2,ncols=1, gridspec_kw={"height_ratios":[3,1]})
        else:    
            (ax1,ax3),(ax2,ax4) = fig.subplots(nrows=2,ncols=2, gridspec_kw={"height_ratios":[3,1]})
            ax1.set_title("First serie of baselines")
            ax3.set_title("Second serie of baselines")
        
        ct.setaxelim(ax1, ydata=1/averagePdVar[:,PlotBaseline], ymargin=0.2, ymin=0)
        ct.setaxelim(ax2, ydata=MeanSquaredSNR[PlotBaseline]+RMSSquaredSNR[PlotBaseline], ymargin=0.2,ymin=0)
    
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel('SNR²')
        ax2.set_ylabel('<SNR²>')
        
        if OneAxe:
            baselinestemp = [baselines[iBase] for iBase in PlotBaselineIndex]
            basecolorstemp = basecolors[:NumberOfBaselinesToPlot]
            barbasecolors = ['grey']*NIN
            
            ax2.set_anchor('S')
            ax2.set_box_aspect(1/15)
            
            k=0
            for iBase in PlotBaselineIndex:   # First serie
                ax1.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolorstemp[k])
                barbasecolors[iBase] = basecolorstemp[k]
                k+=1
                
            p1=ax2.bar(baselines,MeanSquaredSNR, color=barbasecolors, yerr=RMSSquaredSNR)
        
        else:    
            for iBase in range(len1):   # First serie
                if PlotBaseline[iBase]:
                    ax1.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])

            for iBase in range(len1,NIN):   # Second serie
                if PlotBaseline[iBase]:
                    ax3.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
            
            p1=ax2.bar(baselines[:len1],MeanSquaredSNR[:len1], color=basecolors[:len1], yerr=RMSSquaredSNR[:len1])
            p2=ax4.bar(baselines[len1:],MeanSquaredSNR[len1:], color=basecolors[len1:], yerr=RMSSquaredSNR[len1])
    
            ax3.sharex(ax1)
            ax3.sharey(ax1) ; ax3.tick_params(labelleft=False)
            ax4.sharey(ax2) ; ax4.tick_params(labelleft=False)
            ax4.bar_label(p2,label_type='edge',fmt='%.2f')
            ax3.set_xlabel("Time (s)")#, labelpad=xlabelpad)
            ax2.set_anchor('S') ; ax4.set_anchor('S')
            ax2.set_box_aspect(1/8) ; ax4.set_box_aspect(1/8)
    
        ax2.bar_label(p1,label_type='edge',fmt='%.2f')
    
    
        figname = '_'.join(title.split(' ')[:3])
        figname = f"SNRonly"
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()
    
    
    if displayall or ("PDonly" in args):
        plt.rcParams.update(rcParamsForBaselines)
        title=f'Only PD - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        (ax1,ax3),(ax2,ax4), (ax5,ax6) = fig.subplots(nrows=3,ncols=2, gridspec_kw={"height_ratios":[2,2,1]})
        ax1.set_title("First serie of baselines, from 12 to 25")
        ax3.set_title("Second serie of baselines, from 26 to 56")


        for iBase in range(len1):   # First serie
            if PlotBaseline[iBase]:
                ax1.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
                ax2.plot(t[timerange],pD[timerange,iBase]*lmbda/2/np.pi,color=basecolors[iBase])

        for iBase in range(len1,NIN):   # Second serie
            if PlotBaseline[iBase]:
                ax3.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
                ax4.plot(t[timerange],pD[timerange,iBase]*lmbda/2/np.pi,color=basecolors[iBase])


        p1=ax5.bar(baselines[:len1],RMSpd[:len1]*lmbda/2/np.pi*1e3, color=basecolors[:len1])
        p2=ax6.bar(baselines[len1:],RMSpd[len1:]*lmbda/2/np.pi*1e3, color=basecolors[len1:])

        ax1.sharex(ax2) ; ax3.sharex(ax4)
        ax1.set_ylim(ylimSNR) ; 
        ax3.sharey(ax1) ; ax3.tick_params(labelleft=False)
        ax4.sharey(ax2) ; ax4.tick_params(labelleft=False)
        ax6.sharey(ax5) ; ax6.tick_params(labelleft=False)
        ct.setaxelim(ax5, ydata=RMSpd*lmbda/2/np.pi*1e3, ymargin=0.4,ymin=0)
        #setaxelim(ax6, ydata=RMSpd, ymargin=0.2,ymin=0)

        ax4.set_ylim(ylimPD)

        ax1.set_ylabel('SNR²')
        ax2.set_ylabel('Phase-Delays [µm]')
        ax5.set_ylabel('PD rms\n[nm]')
        ax5.bar_label(p1,label_type='edge',fmt='%.0f')
        ax6.bar_label(p2,label_type='edge',fmt='%.0f')
        ax5.set_anchor('S') ; ax6.set_anchor('S')
        ax5.set_box_aspect(1/8) ; ax6.set_box_aspect(1/8)
        ax2.set_xlabel("Time (s)", labelpad=xlabelpad) ; ax4.set_xlabel("Time (s)", labelpad=xlabelpad)

        figname = '_'.join(title.split(' ')[:3])
        figname = f"PDonly"
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()
    
    
    """ GD and PD estimators"""
    if displayall or ("GDPDwithoutSNR" in args):
        plt.rcParams.update(rcParamsForBaselines_SNR)
        title=f'GD and PD - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        if len(details):
            fig.suptitle(title)
        
        OneAxe = False ; NumberOfBaselinesToPlot = np.sum(PlotBaseline)
        if NumberOfBaselinesToPlot < len1:
            OneAxe=True
            ax1, ax2, axghost1,ax3, ax4 = fig.subplots(nrows=5,ncols=1, gridspec_kw={"height_ratios":[3,3,0.5,1,1]})
        else:    
            (ax1,ax5),(ax2,ax6), (axghost1,axghost2),(ax3,ax7), (ax4,ax8) = fig.subplots(nrows=5,ncols=2, gridspec_kw={"height_ratios":[4,4,0.5,1,1]})
            ax1.set_title("First serie of baselines")
            ax5.set_title("Second serie of baselines")

        ax1.set_ylabel('Group-Delays [µm]')
        ax2.set_ylabel('Phase-Delays [µm]')
        ax3.set_ylabel('GD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
        ax4.set_ylabel('PD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')

        ax2.set_xlabel("Time (s)")#, labelpad=xlabelpad)
        ax4.set_xlabel('Baselines')
        
        ax1.sharex(ax2) ; ax3.sharex(ax4)
        ax1.tick_params(labelbottom=False)
        ax3.tick_params(labelbottom=False)
        
        axghost1.set_visible(False)
        
        barbasecolors = ['grey']*NIN

        if OneAxe:
            baselinestemp = [baselines[iBase] for iBase in PlotBaselineIndex]
            basecolorstemp = basecolors[:NumberOfBaselinesToPlot]
            baselinestyles=['-']*len1 + ['--']*len2
            baselinehatches=['']*len1 + ['/']*len2
            

            k=0
            for iBase in PlotBaselineIndex:
                ax1.plot(t[timerange],gD[timerange,iBase]*R*lmbda/2/np.pi,color=basecolors[k])
                ax2.plot(t[timerange],pD[timerange,iBase]*lmbda/2/np.pi,color=basecolors[k])
                barbasecolors[iBase] = basecolors[k]
                k+=1
            
            # p1=ax3.bar(baselinestemp,RMSgd[PlotBaseline]*R*lmbda/2/np.pi, color=basecolorstemp)
            # p2=ax4.bar(baselinestemp,RMSpd[PlotBaseline]*lmbda/2/np.pi, color=basecolorstemp)
            p1=ax3.bar(baselines,RMSgd*R*lmbda/2/np.pi, color=barbasecolors)
            p2=ax4.bar(baselines,RMSpd*lmbda/2/np.pi, color=barbasecolors)


        else:
            boxratio = 1/14
            ax3.set_anchor('S'); ax4.set_anchor('S')
            ax3.set_box_aspect(boxratio); ax4.set_box_aspect(boxratio)
            ax7.set_anchor('S'); ax8.set_anchor('S')
            ax7.set_box_aspect(boxratio); ax8.set_box_aspect(boxratio)
            
            for iBase in range(len1):   # First serie
                if PlotBaseline[iBase]:
                    ax1.plot(t[timerange],gD[timerange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                    ax2.plot(t[timerange],pD[timerange,iBase]*lmbda/2/np.pi,color=basecolors[iBase])
                    barbasecolors[iBase] = basecolors[iBase]
                    
            for iBase in range(len1,NIN):   # Second serie
                if PlotBaseline[iBase]:
                    ax5.plot(t[timerange],gD[timerange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                    ax6.plot(t[timerange],pD[timerange,iBase]*lmbda/2/np.pi,color=basecolors[iBase])
                    barbasecolors[iBase] = basecolors[iBase]
                    
            p1=ax3.bar(baselines[:len1],RMSgd[:len1]*R*lmbda/2/np.pi, color=barbasecolors[:len1])
            p2=ax4.bar(baselines[:len1],RMSpd[:len1]*lmbda/2/np.pi, color=barbasecolors[:len1])
            p3=ax7.bar(baselines[len1:],RMSgd[len1:]*R*lmbda/2/np.pi, color=barbasecolors[len1:])
            p4=ax8.bar(baselines[len1:],RMSpd[len1:]*lmbda/2/np.pi, color=barbasecolors[len1:])
    
            #ax7.sharex(ax8) ; ax8.sharex(ax4)
            ax5.sharey(ax1) ; ax5.tick_params(labelleft=False, labelbottom=False) ; ct.setaxelim(ax1,ydata=gD[timerange]*R*lmbda/2/np.pi)
            ax6.sharey(ax2) ; ax6.tick_params(labelleft=False) ; ct.setaxelim(ax2,ydata=pD[timerange]*lmbda/2/np.pi)
            ax7.sharey(ax3) ; ax7.tick_params(labelleft=False, labelbottom=False)
            ax8.sharey(ax4) ; ax8.tick_params(labelleft=False)

            ax6.set_xlabel("Time [s]")#, labelpad=xlabelpad)
            ax8.set_xlabel('Baselines')
            
            ax7.bar_label(p3,label_type='center',fmt='%.2f')
            ax8.bar_label(p4,label_type='center',fmt='%.2f')
            
            axghost2.set_visible(False)

        ct.setaxelim(ax3,ydata=RMSgd*R*lmbda/2/np.pi,ymin=0, ymargin=0.5)
        ct.setaxelim(ax4,ydata=RMSpd*lmbda/2/np.pi,ymin=0, ymargin=0.5)
        ax3.bar_label(p1,label_type='center',fmt='%.2f')
        ax4.bar_label(p2,label_type='center',fmt='%.2f')
            
        figname = '_'.join(title.split(' ')[:3])
        figname = 'GD&PDestimators'
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()
        
        
        if 'zoom' in kwargs.keys():
            bonds = kwargs['zoom']  # tuple of first and last frame in seconds
            
            iStart = int((bonds[0]-t[0])/dt)
            iStop = int((bonds[1]-t[0])/dt)
            print("DT = ",dt)
            zoomrange = range(iStart,iStop)
            
            gdCommand_zoom = OPDCmdGD_rearranged[zoomrange] - np.median(OPDCmdGD_rearranged[zoomrange],axis=0)
            
            plt.rcParams.update(rcParamsForBaselines_SNR)
            title=f'GD and PD zoom - {details}'
            plt.close(title)
            fig=plt.figure(title, clear=True)
            fig.suptitle(title)
            (ax1,ax6),(ax2,ax7), (ax3,ax8),(ax4,ax9),(ax5,ax10) = fig.subplots(nrows=5,ncols=2, sharex='col',gridspec_kw={"height_ratios":[1,4,4,1,1]})
            ax1.set_title("First serie of baselines, from 12 to 25")
            ax6.set_title("Second serie of baselines, from 26 to 56")
    
            for iBase in range(len1):   # First serie
                if PlotBaseline[iBase]:
                    ax1.plot(t[zoomrange],1/averagePdVar[zoomrange,iBase],color=basecolors[iBase])
                    ax2.plot(t[zoomrange],gD[zoomrange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                    ax2.plot(t[zoomrange],gdCommand_zoom[:,iBase],color=basecolors[iBase],linestyle='-.')
                    ax3.plot(t[zoomrange],pD[zoomrange,iBase]*lmbda/2/np.pi,color=basecolors[iBase], label=baselines[iBase])
                    ax4.plot(t[zoomrange],curRefGD[zoomrange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                    ax5.plot(t[zoomrange],curRefPD[zoomrange,iBase]*lmbda/2/np.pi,color=basecolors[iBase])
                    
            for iBase in range(len1,NIN):   # Second serie
                if PlotBaseline[iBase]:
                    ax6.plot(t[zoomrange],1/averagePdVar[zoomrange,iBase],color=basecolors[iBase])
                    ax7.plot(t[zoomrange],gD[zoomrange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                    ax7.plot(t[zoomrange],gdCommand_zoom[:,iBase],color=basecolors[iBase],linestyle='-.')
                    ax8.plot(t[zoomrange],pD[zoomrange,iBase]*lmbda/2/np.pi,color=basecolors[iBase], label=baselines[iBase])
                    ax9.plot(t[zoomrange],curRefGD[zoomrange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                    ax10.plot(t[zoomrange],curRefPD[zoomrange,iBase]*lmbda/2/np.pi,color=basecolors[iBase])
    
            ax6.sharey(ax1) ; ax6.tick_params(labelleft=False) ; ct.setaxelim(ax1,ydata=[1/averagePdVar[zoomrange,iBase] for iBase in PlotBaselineIndex],ymin=0)
            ax7.sharey(ax2) ; ax7.tick_params(labelleft=False) ; ct.setaxelim(ax2,ydata=[gD[zoomrange,iBase]*R*lmbda/2/np.pi for iBase in PlotBaselineIndex]+
                                                                                      [gdCommand_zoom[:,PlotBaseline]])
            ax8.sharey(ax3) ; ax8.tick_params(labelleft=False) ; ct.setaxelim(ax3,ydata=[pD[zoomrange,iBase]*lmbda/2/np.pi for iBase in PlotBaselineIndex])
            ax9.sharey(ax4) ; ax9.tick_params(labelleft=False) ; ct.setaxelim(ax4,ydata=[curRefGD[zoomrange,iBase]*R*lmbda/2/np.pi for iBase in PlotBaselineIndex],ymargin=0.3,absmargin=True)
            ax10.sharey(ax5) ; ax10.tick_params(labelleft=False) ; ct.setaxelim(ax5,ydata=lmbda/2,ymargin=0,absmargin=True)
    
            ax1.set_ylabel('SNR²')
            ax2.set_ylabel('Group-Delays [µm]')
            ax3.set_ylabel('Phase-Delays [µm]')
            ax4.set_ylabel('$GD_{ref}$\n[µm]',rotation=1,labelpad=60,loc='bottom')
            ax5.set_ylabel('$PD_{ref}$\n[µm]',rotation=1,labelpad=60,loc='bottom')

            ax5.set_xlabel('Time (s)') ; ax10.set_xlabel('Time (s)')
            
            ax3.legend(loc='lower left'); ax8.legend(loc='lower left')
            
            figname = '_'.join(title.split(' ')[:3])
            figname = f'GD&PDestimators_zoom{int(bonds[0])}-{int(bonds[1])}'
            if figsave:
                if isinstance(ext,list):
                    for extension in ext:
                        plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
                else:
                    plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
            fig.show()
        
        
    """ GD and PD estimators with SNR info"""
    if displayall or ("GDPD" in args):
        plt.rcParams.update(rcParamsForBaselines_SNR)
        title=f'GD and PD - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        (ax1,ax6),(ax2,ax7), (ax3,ax8),(ax11,ax12),(ax4,ax9),(ax5,ax10) = fig.subplots(nrows=6,ncols=2, gridspec_kw={"height_ratios":[1,4,4,0.5,1,1]})
        ax1.set_title("First serie of baselines, from 12 to 25")
        ax6.set_title("Second serie of baselines, from 26 to 56")

        for iBase in range(len1):   # First serie
            if PlotBaseline[iBase]:
                ax1.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
                ax2.plot(t[timerange],gD[timerange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                ax3.plot(t[timerange],pD[timerange,iBase]*lmbda/2/np.pi,color=basecolors[iBase])

        for iBase in range(len1,NIN):   # Second serie
            if PlotBaseline[iBase]:
                ax6.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
                ax7.plot(t[timerange],gD[timerange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                ax8.plot(t[timerange],pD[timerange,iBase]*lmbda/2/np.pi,color=basecolors[iBase])


        ax4.bar(baselines[:len1],RMSgd[:len1]*R*lmbda/2/np.pi, color=basecolors[:len1])
        ax5.bar(baselines[:len1],RMSpd[:len1]*lmbda/2/np.pi, color=basecolors[:len1])

        ax9.bar(baselines[len1:],RMSgd[len1:]*R*lmbda/2/np.pi, color=basecolors[len1:])
        ax10.bar(baselines[len1:],RMSpd[len1:]*lmbda/2/np.pi, color=basecolors[len1:])

        ax1.sharex(ax3) ; ax2.sharex(ax3); ax6.sharex(ax8) ; ax7.sharex(ax8)
        ax6.sharey(ax1) ; ax6.tick_params(labelleft=False) ; ct.setaxelim(ax1,ydata=1/averagePdVar[timerange],ymin=0)
        ax7.sharey(ax2) ; ax7.tick_params(labelleft=False) ; ct.setaxelim(ax2,ydata=gD[timerange]*R*lmbda/2/np.pi)
        ax8.sharey(ax3) ; ax8.tick_params(labelleft=False) ; ct.setaxelim(ax3,ydata=pD[timerange]*lmbda/2/np.pi)
        ax9.sharey(ax4) ; ax9.tick_params(labelleft=False) ; ct.setaxelim(ax4,ydata=RMSgd*R*lmbda/2/np.pi,ymin=0)
        ax10.sharey(ax5) ; ax10.tick_params(labelleft=False) ; ct.setaxelim(ax5,ydata=RMSpd*lmbda/2/np.pi,ymin=0)

        ax4.sharex(ax5) ; ax4.tick_params(labelbottom=False)
        ax9.sharex(ax10) ; ax9.tick_params(labelbottom=False)

        ax1.set_ylabel('SNR²')
        ax2.set_ylabel('Group-Delays [µm]')
        ax3.set_ylabel('Phase-Delays [µm]')
        ax4.set_ylabel('GD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
        ax5.set_ylabel('PD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')

        ax11.remove() ; ax12.remove()       # These axes are here to let space for ax3 and ax8 labels

        ax3.set_xlabel("Time (s)", labelpad=xlabelpad) ; ax8.set_xlabel("Time (s)", labelpad=xlabelpad)
        ax5.set_xlabel('Baselines') ; ax10.set_xlabel('Baselines')
        figname = '_'.join(title.split(' ')[:3])
        figname = 'GD&PDestimators'
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()
        
        
        if 'zoom' in kwargs.keys():
            bonds = kwargs['zoom']  # tuple of first and last frame in seconds
            
            iStart = int((bonds[0]-t[0])/dt)
            iStop = int((bonds[1]-t[0])/dt)
            print("DT = ",dt)
            zoomrange = range(iStart,iStop)
            
            gdCommand_zoom = OPDCmdGD_rearranged[zoomrange] - np.median(OPDCmdGD_rearranged[zoomrange],axis=0)
            
            plt.rcParams.update(rcParamsForBaselines_SNR)
            title=f'GD and PD zoom - {details}'
            plt.close(title)
            fig=plt.figure(title, clear=True)
            fig.suptitle(title)
            (ax1,ax6),(ax2,ax7), (ax3,ax8),(ax4,ax9),(ax5,ax10) = fig.subplots(nrows=5,ncols=2, sharex='col',gridspec_kw={"height_ratios":[1,4,4,1,1]})
            ax1.set_title("First serie of baselines, from 12 to 25")
            ax6.set_title("Second serie of baselines, from 26 to 56")
    
            for iBase in range(len1):   # First serie
                if PlotBaseline[iBase]:
                    ax1.plot(t[zoomrange],1/averagePdVar[zoomrange,iBase],color=basecolors[iBase])
                    ax2.plot(t[zoomrange],gD[zoomrange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                    ax2.plot(t[zoomrange],gdCommand_zoom[:,iBase],color=basecolors[iBase],linestyle='-.')
                    ax3.plot(t[zoomrange],pD[zoomrange,iBase]*lmbda/2/np.pi,color=basecolors[iBase], label=baselines[iBase])
                    ax4.plot(t[zoomrange],curRefGD[zoomrange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                    ax5.plot(t[zoomrange],curRefPD[zoomrange,iBase]*lmbda/2/np.pi,color=basecolors[iBase])
                    
            for iBase in range(len1,NIN):   # Second serie
                if PlotBaseline[iBase]:
                    ax6.plot(t[zoomrange],1/averagePdVar[zoomrange,iBase],color=basecolors[iBase])
                    ax7.plot(t[zoomrange],gD[zoomrange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                    ax7.plot(t[zoomrange],gdCommand_zoom[:,iBase],color=basecolors[iBase],linestyle='-.')
                    ax8.plot(t[zoomrange],pD[zoomrange,iBase]*lmbda/2/np.pi,color=basecolors[iBase], label=baselines[iBase])
                    ax9.plot(t[zoomrange],curRefGD[zoomrange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                    ax10.plot(t[zoomrange],curRefPD[zoomrange,iBase]*lmbda/2/np.pi,color=basecolors[iBase])
    
            ax6.sharey(ax1) ; ax6.tick_params(labelleft=False) ; ct.setaxelim(ax1,ydata=[1/averagePdVar[zoomrange,iBase] for iBase in PlotBaselineIndex],ymin=0)
            ax7.sharey(ax2) ; ax7.tick_params(labelleft=False) ; ct.setaxelim(ax2,ydata=[gD[zoomrange,iBase]*R*lmbda/2/np.pi for iBase in PlotBaselineIndex]+
                                                                                      [gdCommand_zoom[:,PlotBaseline]])
            ax8.sharey(ax3) ; ax8.tick_params(labelleft=False) ; ct.setaxelim(ax3,ydata=[pD[zoomrange,iBase]*lmbda/2/np.pi for iBase in PlotBaselineIndex])
            ax9.sharey(ax4) ; ax9.tick_params(labelleft=False) ; ct.setaxelim(ax4,ydata=[curRefGD[zoomrange,iBase]*R*lmbda/2/np.pi for iBase in PlotBaselineIndex],ymargin=0.3,absmargin=True)
            ax10.sharey(ax5) ; ax10.tick_params(labelleft=False) ; ct.setaxelim(ax5,ydata=[curRefPD[zoomrange,iBase]*lmbda/2/np.pi for iBase in PlotBaselineIndex],ymargin=0.3,absmargin=True)
    
            ax1.set_ylabel('SNR²')
            ax2.set_ylabel('Group-Delays [µm]')
            ax3.set_ylabel('Phase-Delays [µm]')
            ax4.set_ylabel('$GD_{ref}$\n[µm]',rotation=1,labelpad=60,loc='bottom')
            ax5.set_ylabel('$PD_{ref}$\n[µm]',rotation=1,labelpad=60,loc='bottom')

            ax5.set_xlabel('Time (s)') ; ax10.set_xlabel('Time (s)')
            
            ax3.legend(loc='lower left'); ax8.legend(loc='lower left')
            
            figname = '_'.join(title.split(' ')[:3])
            figname = f'GD&PDestimators_zoom{int(bonds[0])}-{int(bonds[1])}'
            if figsave:
                if isinstance(ext,list):
                    for extension in ext:
                        plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
                else:
                    plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
            fig.show()


        
        
        
        
    """ GD - ref ; PD - ref  = errors"""
    if displayall or ("GDPDdiffRef" in args):
        plt.rcParams.update(rcParamsForBaselines_SNR)
        title=f'GD-GDref and PD-PDref - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        (ax1,ax6),(ax2,ax7), (ax3,ax8),(ax11,ax12),(ax4,ax9),(ax5,ax10) = fig.subplots(nrows=6,ncols=2, gridspec_kw={"height_ratios":[1,4,4,0.5,1,1]})
        ax1.set_title("First serie of baselines, from 12 to 25")
        ax6.set_title("Second serie of baselines, from 26 to 56")

        for iBase in range(len1):   # First serie
            if PlotBaseline[iBase]:
                ax1.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
                ax2.plot(t[timerange],(gD-curRefGD)[timerange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                ax3.plot(t[timerange],(pD-curRefPD)[timerange,iBase]*lmbda/2/np.pi,color=basecolors[iBase])

        for iBase in range(len1,NIN):   # Second serie
            if PlotBaseline[iBase]:
                ax6.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
                ax7.plot(t[timerange],(gD-curRefGD)[timerange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                ax8.plot(t[timerange],(pD-curRefPD)[timerange,iBase]*lmbda/2/np.pi,color=basecolors[iBase])


        ax4.bar(baselines[:len1],RMSgd[:len1]*R*lmbda/2/np.pi, color=basecolors[:len1])
        ax5.bar(baselines[:len1],RMSpd[:len1]*lmbda/2/np.pi, color=basecolors[:len1])

        ax9.bar(baselines[len1:],RMSgd[len1:]*R*lmbda/2/np.pi, color=basecolors[len1:])
        ax10.bar(baselines[len1:],RMSpd[len1:]*lmbda/2/np.pi, color=basecolors[len1:])

        ax1.sharex(ax3) ; ax2.sharex(ax3); ax6.sharex(ax8) ; ax7.sharex(ax8)
        ax6.sharey(ax1) ; ax6.tick_params(labelleft=False) ; ct.setaxelim(ax1,ydata=1/averagePdVar[timerange[200:1000]],ymin=0)
        ax7.sharey(ax2) ; ax7.tick_params(labelleft=False) ; ct.setaxelim(ax2,ydata=(gD-curRefGD)[timerange[200:]]*R*lmbda/2/np.pi)
        ax8.sharey(ax3) ; ax8.tick_params(labelleft=False) ; ct.setaxelim(ax3,ydata=(pD-curRefPD)[timerange[200:]]*lmbda/2/np.pi)
        ax9.sharey(ax4) ; ax9.tick_params(labelleft=False) ; ct.setaxelim(ax4,ydata=RMSgd*R*lmbda/2/np.pi,ymin=0)
        ax10.sharey(ax5) ; ax10.tick_params(labelleft=False) ; ct.setaxelim(ax5,ydata=RMSpd*lmbda/2/np.pi,ymin=0)

        ax4.sharex(ax5) ; ax4.tick_params(labelbottom=False)
        ax9.sharex(ax10) ; ax9.tick_params(labelbottom=False)

        ax1.set_ylabel('SNR²')
        ax2.set_ylabel('Group-Delays [µm]')
        ax3.set_ylabel('Phase-Delays [µm]')
        ax4.set_ylabel('GD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
        ax5.set_ylabel('PD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')

        ax11.remove() ; ax12.remove()       # These axes are here to let space for ax3 and ax8 labels

        ax3.set_xlabel("Time (s)", labelpad=xlabelpad) ; ax8.set_xlabel("Time (s)", labelpad=xlabelpad)
        ax5.set_xlabel('Baselines') ; ax10.set_xlabel('Baselines')
        figname = '_'.join(title.split(' ')[:3])
        figname = 'GD&PDdiffRef'
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()

    if displayall or ("GDPDerr" in args):
        plt.rcParams.update(rcParamsForBaselines_SNR)
        title=f'GDErr & PDErr - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        (ax1,ax6),(ax2,ax7), (ax3,ax8),(ax11,ax12),(ax4,ax9),(ax5,ax10) = fig.subplots(nrows=6,ncols=2, gridspec_kw={"height_ratios":[1,4,4,0.5,1,1]})
        ax1.set_title("First serie of baselines, from 12 to 25")
        ax6.set_title("Second serie of baselines, from 26 to 56")

        for iBase in range(len1):   # First serie
            if PlotBaseline[iBase]:
                ax1.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
                ax2.plot(t[timerange],curGdErrBaseMicrons[timerange,iBase],color=basecolors[iBase])
                ax3.plot(t[timerange],curPdErrBaseMicrons[timerange,iBase],color=basecolors[iBase])

        for iBase in range(len1,NIN):   # Second serie
            if PlotBaseline[iBase]:
                ax6.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
                ax7.plot(t[timerange],curGdErrBaseMicrons[timerange,iBase],color=basecolors[iBase])
                ax8.plot(t[timerange],curPdErrBaseMicrons[timerange,iBase],color=basecolors[iBase])


        ax4.bar(baselines[:len1],RMSgderr[:len1]*R*lmbda/2/np.pi, color=basecolors[:len1])
        ax5.bar(baselines[:len1],RMSpderr[:len1]*lmbda/2/np.pi, color=basecolors[:len1])

        ax9.bar(baselines[len1:],RMSgderr[len1:]*R*lmbda/2/np.pi, color=basecolors[len1:])
        ax10.bar(baselines[len1:],RMSpderr[len1:]*lmbda/2/np.pi, color=basecolors[len1:])

        ax1.sharex(ax3) ; ax2.sharex(ax3); ax6.sharex(ax8) ; ax7.sharex(ax8)
        ax6.sharey(ax1) ; ax6.tick_params(labelleft=False) ; ct.setaxelim(ax1,ydata=1/averagePdVar[timerange[200:1000]],ymin=0)
        ax7.sharey(ax2) ; ax7.tick_params(labelleft=False) ; ct.setaxelim(ax2,ydata=curGdErrBaseMicrons[timerange[200:]])
        ax8.sharey(ax3) ; ax8.tick_params(labelleft=False) ; ct.setaxelim(ax3,ydata=(pD-curRefPD)[timerange[200:]]*lmbda/2/np.pi)
        ax9.sharey(ax4) ; ax9.tick_params(labelleft=False) ; ct.setaxelim(ax4,ydata=RMSgd*R*lmbda/2/np.pi,ymin=0)
        ax10.sharey(ax5) ; ax10.tick_params(labelleft=False) ; ct.setaxelim(ax5,ydata=RMSpd*lmbda/2/np.pi,ymin=0)

        ax4.sharex(ax5) ; ax4.tick_params(labelbottom=False)
        ax9.sharex(ax10) ; ax9.tick_params(labelbottom=False)

        ax1.set_ylabel('SNR²')
        ax2.set_ylabel('Group-Delays [µm]')
        ax3.set_ylabel('Phase-Delays [µm]')
        ax4.set_ylabel('GD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
        ax5.set_ylabel('PD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')

        ax11.remove() ; ax12.remove()       # These axes are here to let space for ax3 and ax8 labels

        ax3.set_xlabel("Time (s)", labelpad=xlabelpad) ; ax8.set_xlabel("Time (s)", labelpad=xlabelpad)
        ax5.set_xlabel('Baselines') ; ax10.set_xlabel('Baselines')
        figname = '_'.join(title.split(' ')[:3])
        figname = 'GD&PDerrors'
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()
        

    """ GD and PD reference vectors"""
    if displayall or ("GDPDref" in args):
        plt.rcParams.update(rcParamsForBaselines)
        title=f'RefGD and RefPD - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        (ax1,ax6),(ax2,ax7), (ax3,ax8),(ax11,ax12),(ax4,ax9),(ax5,ax10) = fig.subplots(nrows=6,ncols=2, gridspec_kw={"height_ratios":[1,4,4,1,1,1]})
        ax1.set_title("First serie of baselines, from 12 to 25")
        ax6.set_title("Second serie of baselines, from 26 to 56")

        for iBase in range(len1):   # First serie
            if PlotBaseline[iBase]:
                ax1.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
                ax2.plot(t[timerange],curRefGD[timerange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                ax3.plot(t[timerange],curRefPD[timerange,iBase]*lmbda/2/np.pi,color=basecolors[iBase])

        for iBase in range(len1,NIN):   # Second serie
            if PlotBaseline[iBase]:
                ax6.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
                ax7.plot(t[timerange],curRefGD[timerange,iBase]*R*lmbda/2/np.pi,color=basecolors[iBase])
                ax8.plot(t[timerange],curRefPD[timerange,iBase]*lmbda/2/np.pi,color=basecolors[iBase])

        ax4.bar(baselines[:len1],Meangdref[:len1]*R*lmbda/2/np.pi, yerr=RMSgdref[:len1]*R*lmbda/2/np.pi,color=basecolors[:len1])
        ax5.bar(baselines[:len1],Meanpdref[:len1]*lmbda/2/np.pi, yerr=RMSpdref[:len1]*lmbda/2/np.pi,color=basecolors[:len1])

        ax9.bar(baselines[len1:],Meangdref[len1:]*R*lmbda/2/np.pi, yerr=RMSgdref[len1:]*R*lmbda/2/np.pi,color=basecolors[len1:])
        ax10.bar(baselines[len1:],Meanpdref[len1:]*lmbda/2/np.pi, yerr=RMSpdref[len1:]*lmbda/2/np.pi,color=basecolors[len1:])

        ax1.sharex(ax3) ; ax2.sharex(ax3); ax6.sharex(ax8) ; ax7.sharex(ax8)
        ax6.sharey(ax1) ; ax6.tick_params(labelleft=False) ; ct.setaxelim(ax1,ydata=1/averagePdVar[timerange[200:1000]], ymin=0)
        ax7.sharey(ax2) ; ax7.tick_params(labelleft=False) ; ct.setaxelim(ax2,ydata=curRefGD[timerange[200:]]*R*lmbda/2/np.pi)
        ax3.sharey(ax8) ; ax8.tick_params(labelleft=False) ; ct.setaxelim(ax3,ydata=curRefPD[timerange[200:]]*lmbda/2/np.pi)
        ax9.sharey(ax4) ; ax9.tick_params(labelleft=False) ;# setaxelim(ax4,ydata=Meangdref*R*lmbda/2/np.pi,absmargin=True)
        ax10.sharey(ax5) ; ax10.tick_params(labelleft=False) ;# setaxelim(ax5,ydata=Meanpdref*lmbda/2/np.pi,absmargin=True)

        ax4.sharex(ax5) ; ax4.tick_params(labelbottom=False)
        ax9.sharex(ax10) ; ax9.tick_params(labelbottom=False)

        ax1.set_ylabel('SNR²')
        ax2.set_ylabel('$GD_{ref}$ [µm]')
        ax3.set_ylabel('$PD_{ref}$ [µm]')
        ax4.set_ylabel('$<GD_{ref}>$\n[µm]',rotation=1,labelpad=60,loc='bottom')
        ax5.set_ylabel('$<PD_{ref}>$\n[µm]',rotation=1,labelpad=60,loc='bottom')

        ax11.remove() ; ax12.remove()       # These axes are here to let space for ax3 and ax8 labels

        ax3.set_xlabel("Time (s)") ; ax8.set_xlabel("Time (s)")
        ax5.set_xlabel('Baselines') ; ax10.set_xlabel('Baselines')
        figname = '_'.join(title.split(' ')[:3])
        figname = 'GD&PDref'
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()

    """ GD and PD commands"""
    if displayall or ("PisCommands" in args):
        plt.rcParams.update(rcParamsForBaselines)
        title=f'GD and PD Piston Commands - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        ax1,ax2 = fig.subplots(nrows=2, sharex=True)

        for iTel in range(NA):
            if PlotTel[iTel]:
                ax1.plot(t[timerange],gdDlCmdMicrons_rearranged[timerange,iTel],color=telcolors[iTel], label=f"{tels[iTel]}")
                ax2.plot(t[timerange],pdDlCmdMicrons_rearranged[timerange,iTel],color=telcolors[iTel])


        ax1.set_ylabel('Command GD [µm]')
        ax2.set_ylabel('Command PD [µm]')
        ax2.set_xlabel('Time [s]')
        ax1.legend()
        figname = '_'.join(title.split(' ')[:3])
        figname = 'CommandsPistonSpace'
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()
        
    if displayall or ("PisCommands2" in args):
        plt.rcParams.update(rcParamsForBaselines)
        title=f'GD and PD Piston Commands - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        ax1,ax2,ax3 = fig.subplots(nrows=3, sharex=True)

        for iTel in range(NA):
            if PlotTel[iTel]:
                ax1.plot(t[timerange],gdDlCmdMicrons_rearranged[timerange,iTel],color=telcolors[iTel], label=f"{tels[iTel]}")
                ax2.plot(t[timerange],pdDlCmdMicrons_rearranged[timerange,iTel],color=telcolors[iTel])
                ax3.plot(t[timerange],np.gradient(t[timerange],pdDlCmdMicrons_rearranged[timerange,iTel]),color=telcolors[iTel])


        ax1.set_ylabel('Command GD [µm]')
        ax2.set_ylabel('Command PD [µm]')
        ax3.set_ylabel('Gradient Command PD')
        ax3.set_xlabel('Time [s]')
        ax1.legend()
        figname = '_'.join(title.split(' ')[:3])
        figname = 'CommandsPistonSpace'
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()
        
    
    """GD and PD commands in OpdSpace"""
    if displayall or ("OpdCommands" in args):
        plt.rcParams.update(rcParamsForBaselines)
        title=f'GD and PD Opd Commands - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        (ax1,ax3),(ax2,ax4) = fig.subplots(nrows=2,ncols=2, gridspec_kw={"height_ratios":[1,1]})
        ax1.set_title("First serie of baselines, from 12 to 25")
        ax3.set_title("Second serie of baselines, from 26 to 56")

        for iBase in range(len1):   # First serie
            if PlotBaseline[iBase]:
                # ax1.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
                ax1.plot(t[timerange],OPDCmdGD_rearranged[timerange,iBase],color=basecolors[iBase])
                ax2.plot(t[timerange],OPDCmdPD_rearranged[timerange,iBase],color=basecolors[iBase],label=baselines[iBase])
                # ax2.plot(t[timerange],OPDCmdPD[timerange,iBase]*lmbda/2/np.pi,color=basecolors[iBase],label=baselines[iBase])

        for iBase in range(len1,NIN):   # Second serie
            if PlotBaseline[iBase]:
                ax3.plot(t[timerange],OPDCmdGD_rearranged[timerange,iBase],color=basecolors[iBase])
                ax4.plot(t[timerange],OPDCmdPD_rearranged[timerange,iBase],color=basecolors[iBase],label=baselines[iBase])
                # ax4.plot(t[timerange],OPDCmdPD[timerange,iBase]*lmbda/2/np.pi,color=basecolors[iBase],label=baselines[iBase])
                # ax8.plot(t[timerange],OPDCmdPD[timerange,iBase],color=basecolors[iBase])

        ax1.sharex(ax2) ; ax3.sharex(ax4)
        ax1.tick_params(labelbottom=False) ; ax3.tick_params(labelbottom=False)

        ax3.sharey(ax1) ; ax3.tick_params(labelleft=False) ; ct.setaxelim(ax1,ydata=OPDCmdGD_rearranged[timerange])
        ax4.sharey(ax2) ; ax4.tick_params(labelleft=False) ; ct.setaxelim(ax2,ydata=OPDCmdPD_rearranged[timerange])

        ax1.set_ylabel('GD Commands [µm]')
        ax2.set_ylabel('PD Commands [µm]')

        ax2.set_xlabel("Time (s)") ; ax4.set_xlabel("Time (s)")

        ax2.legend(loc='upper left') ; ax4.legend(loc='upper left')

        figname = '_'.join(title.split(' ')[:3])
        figname = 'CommandsOpdSpace'
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()

        
    """ Visibility norms"""
    if displayall or ("vis" in args):
        plt.rcParams.update(rcParamsForBaselines_SNR)
        title=f'Norm Visibility - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        ax1,ax2 = fig.subplots(nrows=1,ncols=2)
        ax1.set_title("First serie of baselines, from 12 to 25")
        ax2.set_title("Second serie of baselines, from 26 to 56")

        for iBase in range(len1):   # First serie
            if PlotBaseline[iBase]:
                ax1.plot(t[timerange],normVisibility[timerange,iBase],color=basecolors[iBase], label=baselines[iBase])

        for iBase in range(len1,NIN):   # Second serie
            if PlotBaseline[iBase]:
                ax2.plot(t[timerange],normVisibility[timerange,iBase],color=basecolors[iBase], label=baselines[iBase])

        ax1.sharex(ax2)
        ct.setaxelim(ax1,ydata=normVisibility[timerange[200:]],ymin=0)

        ax1.set_ylabel('Norm Visibility')
        ax1.legend() ; ax2.legend()
        ax1.set_xlabel("Time (s)", labelpad=xlabelpad)

        figname = '_'.join(title.split(' ')[:3])
        figname = 'NormVisibility'
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()


    """ GD and PD Closure Phases"""
    if displayall or ('cp' in args):
        plt.rcParams.update(rcParamsForBaselines)
        title=f'Closure phases - {details}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        (ax1,ax5),(ax2,ax6),(ax11,ax12), (ax3,ax7) = fig.subplots(nrows=4,ncols=2, gridspec_kw={"height_ratios":[5,5,1,2]})
        ax1.set_title("First serie of closure phases")
        ax5.set_title("Second serie of closure phases")

        len2cp = NC//2 ; len1cp=NC-len2cp
        for iClosure in range(len1cp):
            if PlotClosure[iClosure]:
                # ax1.plot(t[timerange],1/averagePdVar[timerange,iClosure],color=basecolors[iClosure])
                ax1.plot(t[timerange],gdClosure[timerange,iClosure]*R*180/np.pi,color=basecolors[iClosure])
                ax2.plot(t[timerange],pdClosure[timerange,iClosure]*180/np.pi,color=basecolors[iClosure])

            ax3.errorbar(iClosure,Meangdc[iClosure]*R*180/np.pi,yerr=RMSgdc[iClosure]*R*180/np.pi, marker='o',color=basecolors[iClosure],ecolor='k')
            ax3.errorbar(iClosure,Meanpdc[iClosure]*180/np.pi,yerr=RMSpdc[iClosure]*180/np.pi, marker='x',color=basecolors[iClosure],ecolor='k')

        for iClosure in range(len1cp,NC):
            if PlotClosure[iClosure]:
                # ax6.plot(t[timerange],1/averagePdVar[timerange,iClosure],color=basecolors[iClosure])
                ax5.plot(t[timerange],gdClosure[timerange,iClosure]*R*180/np.pi,color=basecolors[iClosure])
                ax6.plot(t[timerange],pdClosure[timerange,iClosure]*180/np.pi,color=basecolors[iClosure])

            ax7.errorbar(iClosure,Meangdc[iClosure]*R*180/np.pi,yerr=RMSgdc[iClosure]*R*180/np.pi, marker='o',color=basecolors[iClosure],ecolor='k')
            ax7.errorbar(iClosure,Meanpdc[iClosure]*180/np.pi,yerr=RMSpdc[iClosure]*180/np.pi, marker='x',color=basecolors[iClosure],ecolor='k')
    
        ax1.sharex(ax2) ; ax5.sharex(ax6)
        ax5.sharey(ax1) ; ax5.tick_params(labelleft=False) ; ct.setaxelim(ax1,ydata=gdClosure[timerange[200:]]*R*180/np.pi)
        ax6.sharey(ax2) ; ax6.tick_params(labelleft=False) ; ct.setaxelim(ax2,ydata=pdClosure[timerange[200:]]*180/np.pi)
        ax7.sharey(ax3) ; ax7.tick_params(labelleft=False) ; ct.setaxelim(ax3,ydata=Meangdc*R*180/np.pi,absmargin=True)
    
        ax3.set_xticks(np.arange(len1cp))
        ax7.set_xticks(np.arange(len1cp,NC))
        ax3.set_xticklabels(closures[:len1cp],rotation=45)
        ax7.set_xticklabels(closures[len1cp:],rotation=45)

        ax3.sharey(ax7); ax3.set_ylim(-270,270)
        ax3.set_yticks([-180,-90,0,90,180])
        ax3.set_yticklabels([-180,'',0,'',180])
        #ax7.set_yticks([-180,-90,0,90,180])
        
        ax1.set_ylabel('GD closure [°]')
        ax2.set_ylabel('PD closure [°]')
        ax3.set_ylabel('Mean \nvalues\n[°]',rotation=90,labelpad=10,loc='bottom')
        ax1.legend() ; ax5.legend()
        ax11.remove() ; ax12.remove() # These axes are here to let space for ax3 and ax8 labels

        ax2.set_xlabel("Time (s)") ; ax6.set_xlabel("Time (s)")

        figname = '_'.join(title.split(' ')[:3])
        figname = 'GD&PDcp'
        if figsave:
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        fig.show()

    

