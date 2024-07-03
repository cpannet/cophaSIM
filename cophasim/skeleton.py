# -*- coding: utf-8 -*-

import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from astropy.io import fits
import datetime

from importlib import reload

from . import coh_tools as ct
from . import config

from .decorators import timer

import pkg_resources #Enable to load data included in this package
from matplotlib.backends.backend_pdf import PdfPages

from cophasim.tol_colors import tol_cset
colors = tol_cset('muted')

SS = 12     # Small size
MS = 14     # Medium size
BS = 16     # Big size
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
                        "figure.figsize":figsize,
                        'figure.subplot.hspace': 0.05,
                        'figure.subplot.wspace': 0,
                        'figure.subplot.left':0.1,
                        'figure.subplot.right':0.95
                        }

""" If wanted, change the display font """
# plt.rc('font', **{'family' : 'serif', 'serif' : ['Computer Modern Roman']})
# plt.rc('text', usetex = True)

def initialize(Interferometer, ObsFile, DisturbanceFile, NT=512, OT=1, MW = 5, 
               ND=1,
               spectra = [], spectraM=[],wlOfTrack=0, spectrum = [],
               mode = 'search',
               fs='default', TELref=0, FSfitsfile='', R = 0.5, dt=1e-3,sigmap=[],imsky=[],
               ft = 'integrator', state = 0,
               noise=True, phnoise = 0, noiseParams = {},
               seedPh=100, seedRon=100, seedDist=100,seedDark=100,
               starttracking=50e-3, latencytime=0,
               piston_average=0, foreground=[], display=False,
               checktime=True, checkperiod=10):
    """
    

    Parameters
    ----------
    Interferometer : TYPE
        DESCRIPTION.
    ObsFile : TYPE
        DESCRIPTION.
    DisturbanceFile : TYPE
        DESCRIPTION.
    NT : TYPE, optional
        DESCRIPTION. The default is 512.
    OT : TYPE, optional
        DESCRIPTION. The default is 1.
    MW : TYPE, optional
        DESCRIPTION. The default is 5.
    ND : TYPE, optional
        DESCRIPTION. The default is 1.
    spectra : TYPE, optional
        DESCRIPTION. The default is [].
    spectraM : TYPE, optional
        DESCRIPTION. The default is [].
    wlOfTrack : TYPE, optional
        DESCRIPTION. The default is 0.
    spectrum : TYPE, optional
        DESCRIPTION. The default is [].
    mode : TYPE, optional
        DESCRIPTION. The default is 'search'.
    fs : TYPE, optional
        DESCRIPTION. The default is 'default'.
    TELref : TYPE, optional
        DESCRIPTION. The default is 0.
    FSfitsfile : TYPE, optional
        DESCRIPTION. The default is ''.
    R : TYPE, optional
        DESCRIPTION. The default is 0.5.
    dt : TYPE, optional
        DESCRIPTION. The default is 1.
    sigmap : TYPE, optional
        DESCRIPTION. The default is [].
    imsky : TYPE, optional
        DESCRIPTION. The default is [].
    ft : TYPE, optional
        DESCRIPTION. The default is 'integrator'.
    state : TYPE, optional
        DESCRIPTION. The default is 0.
    noise : TYPE, optional
        DESCRIPTION. The default is True.
    ron : TYPE, optional
        DESCRIPTION. The default is 0.
    qe : TYPE, optional
        DESCRIPTION. The default is 0.5.
    phnoise : TYPE, optional
        DESCRIPTION. The default is 0.
    G : TYPE, optional
        DESCRIPTION. The default is 1.
    enf : TYPE, optional
        DESCRIPTION. The default is 1.5.
    M : TYPE, optional
        DESCRIPTION. The default is 1.
    seedPh : TYPE, optional
        DESCRIPTION. The default is 100.
    seedRon : TYPE, optional
        DESCRIPTION. The default is 100.
    seedDist : TYPE, optional
        DESCRIPTION. The default is 100.
    starttracking : TYPE, optional
        DESCRIPTION. The default is 50.
    latencytime : TYPE, optional
        DESCRIPTION. The default is 0.
    piston_average : TYPE, optional
        DESCRIPTION. The default is 0.
    display : TYPE, optional
        DESCRIPTION. The default is False.
    checktime : TYPE, optional
        DESCRIPTION. The default is True.
    checkperiod : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    None.

    """
    
    """
    NAME: initialize - Initializes a structure to simulate an interferometer by \
        COPHASIM
    
    PURPOSE:
        This procedure creates the coh structure and intializes it with the \
        FS-given information.
        Other user-information are filled with coh_user.
        
    OUTPUT:
        coh: dictionnary which defines the interferometric setup.
        
    INPUTS:
        MANDATORY:
        NA : (integer) Number of input beams
        NW  : (integer) Number of waves in the simulation
        MW  : (integer) Number of macro-waves (detector spectral channels)
        spectra   : (float array) Set of reference wavelengths
        
    OPTIONAL:
        fs : (string) the name of the Fringe Sensor procedure coh_fs_name
        FSfitsfile : (string) the name of the experimental V2PM fitsfile
        mode : the algorithm (pd, gd,...)
        lmbda  : (float array) Value of wavelengths
        spectrum   : ([NB,NW] float array) XX TODO How to weight waveNUMBERS 
        on the corresponding baseline
        filename : (string) Filename for automatic saves
        step_resp: (FLTARR(xT)) ODL response to a Heaviside step
        /openloop : to simulate an openloop with coh_ft_openloop
        [DEFAULT]=closed loop with coh_ft_integrator
        random : random start for the pseudo-random sequence
        version : (optional input) prints version number before execution. 
        help    : (optional input) prints the documentation and exits.
        F : Amplification factor of an EM-CCD camera, by default = 1 
        piston_average : INT
            - 0 : The pistons are as it comes in the file
            - 1 : The first value of pistons is subtracted to each telescope
            - 2 : The average of the first pistons is subtracted to each telescope
            - 3 : The temporal average of each piston is subtracted: each piston has a null average
    
EXAMPLE:
    See unitary test at the end of file 
    
RESTRICTIONS:
    This code is copyright (c) ONERA, 2009. 
    
SOURCE:
    This code is born from Choquet and Cassaing's IDL code developped at ONERA\
    (2009)
    
"""
        

    """
    LOAD INTERFEROMETER INFOS
    """

    InterfArray = ct.get_array(name=Interferometer)

    Obs, Target = ct.get_ObsInformation(ObsFile)

    # Number of telescopes
    NA=InterfArray.NA
    
    # Number of unknown phase and photometries
    NB = NA**2
    
    # Number of baselines
    NIN = int(NA*(NA-1)/2)
    
    # Number of Closure Phases
    from scipy.special import binom
    NC = int(binom(NA,3))
    
    # Number of independant closure phases
    ND = int((NA-1)*(NA-2)/2)
    
    # TEMPORAL PARAMETERS
    
    MT=int(NT/OT)                           # Total number of temporal samples
    timestamps = np.arange(NT)*dt           # Time sampling in [s]
    
    # SPECTRAL PARAMETERS
    
    if len(spectra) == 0:
        raise ValueError('Lambda array required')      # Array which contains our signal's 
    NW = len(spectra)
    
    if len(spectraM) == 0:
        raise ValueError('Macro lambda array required')
    MW = len(spectraM)
    
    if wlOfTrack==0:
        wlOfTrack = np.median(spectraM) 

    # Disturbance Pattern
    
    OW = int(NW/MW)
    if OW != NW/MW:
        raise ValueError('Oversampling might be integer.')
    
    nyquistCriterion = config.FS['R']*wlOfTrack*OW/2

    # CONFIG PARAMETERS
    
    dyn=0.                                  # to be updtated later by FS (for ex via coh_algo)
    
    # Observation parameters
    config.ObservationFile = ObsFile
    config.Obs = Obs
    config.Target = Target
    config.InterferometerFile = Interferometer
    config.InterfArray = InterfArray
    config.DisturbanceFile = DisturbanceFile
    config.piston_average = piston_average
    config.NA=NA
    config.NB=NB
    config.NC=NC
    config.ND=ND
    config.NIN=NIN
    config.MT=MT
    config.NT=NT
    config.OT=OT
    config.timestamps = timestamps
    config.foreground = foreground
 
    # Fringe Sensor parameters
    config.NW=NW
    config.MW=MW
    config.OW=OW
    config.nyquistCriterion=nyquistCriterion
    config.NX=0
    config.NY=0
    config.ND=ND
    config.dt=dt   # s

    # Noises
    config.noise=noise
    if len(noiseParams):
        print("Update noise parameters of the fringe-sensor:")
        for key, value in zip(list(noiseParams.keys()),list(noiseParams.values())):
            oldval=config.FS[key]
            if key == 'qe':
                if isinstance(value, float):
                    value=value*np.ones([MW,1])
            config.FS[key] = value
            print(f' - Parameter "{key}" changed from {oldval} to {value}')
    
    if imsky:
        config.FS['imsky'] = imsky
    if sigmap:
        config.FS['sigmap'] = sigmap
    
    if noise:
        np.random.seed(seedRon+60)
        config.FS['sigmap'] = np.random.randn(MW,config.FS['NP'])*config.FS['ron']
    
    config.phnoise=phnoise

    if latencytime == 0:
        config.latency = config.dt
    else:
        config.latency = round(latencytime/config.dt)
    
    
    # Random Statemachine seeds
    config.seedPh=seedPh
    config.seedRon=seedRon
    config.seedDist=seedDist
    
    # Fringe tracker
    config.starttracking = starttracking
    
    
    # Source description
    config.spectra=spectra
    config.spectraM=spectraM
    config.wlOfTrack=wlOfTrack
    config.dyn=dyn
    
    # Simulation parameters
    # config.SimuFilename=SimuFilename        # Where to save the data
    config.TELref=TELref             # For display only: reference delay line
    
    if config.TELref:
        iTELref = config.TELref - 1
        L_ref = config.FS['OPD2Piston'][iTELref,:]
        config.FS['OPD2Piston'] = config.FS['OPD2Piston'] - L_ref
    
    config.checkperiod = checkperiod
    config.checktime = checktime
    config.ich = np.zeros([NIN,2])
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib = ct.posk(ia,iap,NA)
            config.ich[ib] = [ia,iap]
    
    config.CPindex = np.zeros([NC,3])
    for ia in range(NA):
        for iap in range(ia+1,NA):
            for iapp in range(iap+1,NA):
                config.CPindex[ct.poskfai(ia,iap,iapp,NA)] = [ia,iap,iapp]#int(''.join([ia+1,iap+1,iapp+1]))

    return 


def update_config(verbose=False,**kwargs):
    """
    Update any parameter the same way than first done in initialise function.

    Returns
    -------
    None.

    """
    
    if verbose:
        print("Update config parameters:")
    for key,value in zip(list(kwargs.keys()),list(kwargs.values())):
        if not hasattr(config, key):
            raise ValueError(f"{key} is not an attribute of config. Maybe it is in FS or FT ?")
        else:    
            oldval = getattr(config,key)
            setattr(config,key,value)
            if verbose:
                if isinstance(value,str):
                    if "/" in value:
                        oldval = oldval.split("/")[-1]
                        value = value.split("/")[-1]
                if verbose:
                    print(f' - Parameter "{key}" changed from {oldval} to {value}')
            
    if len(config.timestamps) != config.NT:
        config.timestamps = np.arange(config.NT)*config.dt
        from . import outputs
        outputs.timestamps = np.copy(config.timestamps)

        if verbose:
            print(' - New NT not equal to len(timestamps) so we change timestamps.')
            
    if "noiseParams" in kwargs.keys():
        ron = config.FS['ron']
        config.noise=True
        config.FS['sigmap'] = np.random.randn(config.MW,config.FS['NP'])*ron
        if verbose:
            print(f' - Updated sigmap for ron={ron}.')
            
    return

def updateFSparams(verbose=False,**kwargs):
    
    if verbose:
        print("Update fringe-sensor parameters:")
    for key, value in zip(list(kwargs.keys()),list(kwargs.values())):
        oldval=config.FS[key]
        config.FS[key] = value
        
        if verbose:
            if isinstance(value,str):
                if "/" in value:
                    oldval = oldval.split("/")[-1]
                    value = value.split("/")[-1]
            print(f' - Parameter "{key}" changed from {oldval} to {value}')

def updateFTparams(verbose=False,**kwargs):
    
    if verbose:
        print("Update fringe-tracker parameters:")
    for key, value in zip(list(kwargs.keys()),list(kwargs.values())):
        oldval=config.FT[key]
        if (key=='ThresholdGD') and (isinstance(value,(float,int))):
            config.FT['ThresholdGD'] = np.ones(config.FS['NINmes'])*value
        else:
            config.FT[key] = value
            if (key == 'search'):
                if value==True:
                    config.FT['state'][0] = 2
                else:
                    config.FT['state'] = np.zeros(config.NT)
                
        if verbose:
            if isinstance(value,str):
                if "/" in value:
                    oldval = oldval.split("/")[-1]
                    value = value.split("/")[-1]
            print(f' - Parameter "{key}" changed from {oldval} to {value}')

def save_config():
    
    a_dict={}

    var_names = [name for name in dir(config) 
                 if ("_" not in name) 
                 and ("V2PM" not in name) 
                 and ("P2VM" not in name)
                 and ("timestamp" not in name)]
    
    for var_name in var_names:
        a_dict[var_name] = eval(var_name)
    # Tests with classes
    # PC = FringeTracker.PistonCalculator()
    # FS = FringeTracker.FringeSensor()
    
    with open('myfile.txt', 'w') as f:
        print(a_dict, file=f)

def MakeAtmosphereCoherence(filepath, InterferometerFile, overwrite=False,
                            spectra=[], RefLambda=0, NT=1000,NTend=0,dt=1e-3,
                            ampl=0, seed=100, dist='step', startframe = 10, 
                            f_fin=300, value_start=0, value_end=0,
                            r0=0.15,t0=10, L0=25, direction=0, d=1,
                            Levents=[],
                            TransDisturb={}, PhaseJumps={},
                            debug=False, tel=0, highCF=True,pows=[],
                            verbose=True,
                            **kwargs):
    '''
    CREATES or LOAD a disturbance scheme INTO or FROM a FITSFILE.
    If filepath is empty, raise error.
    If filepath already exists and overwrite is False: 
        the disturbance pattern is loaded and its Coherent Flux is returned.
    If filepath doesn't exist or overwrite is True:
        a new disturbance pattern is created and saved to filepath according to
        the parameters.
    
    INPUTS:
        - coh: dictionnary which defines the interferometer: contains NA = number of apertures
        - ampl: amplitude of the piston disturbance
        - dist: disturbance model chosen
            - 'scan': a simple case disturbance to begin
            - 'random': a randomly generated set of pistons with modelisation:
                - if highCF=True: high cutoff frequency 
                - if highCF=False: no high cutoff frequency
        - startframe: disturbance starting frame
    
    FITS STRUCTURE:
        -PRIMARY:
            - HEADER:
                hdr['TYPE'] =dist
                hdr['AMPL'] = ampl
                hdr['TEL'] = tel
        - HDU1: [ImageHDU] RealCf
        - HDU2: [ImageHDU] ImagCf
        - HDU3: [ImageHDU] Piston
        - HDU4: [ImageHDU] Transmission
        - if random:
            -HDU5: [ImageHDU] DisturbancePSD
            -HDU6: [ImageHDU] FreqSampling
        
    OUTPUT:
        - CoherentFlux: [NTxNWxNB] the coherence matrix resulting from all pairs of apertures
        - OPTIONAL PistonDisturbance: [NTxNWxNB] pistons
        - OPTIONAL TransmissionDisturbance: [NTxNWxNB] amplitudes
    '''
    
    import os
    from astropy.io import fits

    if len(filepath) == 0:
        raise Exception('No disturbance filepath given.')
        
    elif os.path.exists(filepath):
        if verbose:
            print(f'Disturbance file {filepath} exists.')
        if overwrite:
            os.remove(filepath)
            if verbose:
                print(f'Parameter OVERWRITE is {overwrite}.')
        else:
            if verbose:
                print(f'Parameter OVERWRITE is {overwrite}. Loading the disturbance scheme.')
            
            with fits.open(filepath) as hdu:
                CoherentFlux = hdu['RealCf'].data + hdu['ImagCf'].data*1j
                timestamps = hdu['TimeSampling'].data['timestamps']
                spectra = hdu['SpectralSampling'].data['spectra']
                
            return CoherentFlux, timestamps, spectra
       
    else:
        if verbose:
            print(f'Creating the disturbance pattern and saving it in {filepath}')
        

    if not os.path.exists(InterferometerFile):
        try:
            InterferometerFile = pkg_resources.resource_stream(__name__, InterferometerFile)
        except:
            raise Exception(f"{InterferometerFile} doesn't exist.")
    
    with fits.open(InterferometerFile) as hdu:
        ArrayParams = hdu[0].header
        NA = ArrayParams['NA']
    
    obstime = NT*dt                         # Observation time [s]
    timestamps = np.arange(NT)*dt           # Time sampling [s]
    
    # lmbdamin = 1/np.max(spectra)
    
    
# =============================================================================
#       TRANSMISSION DISTURBANCE
# =============================================================================
    NW=len(spectra)
    TransmissionDisturbance = np.ones([NT,NW,NA])
    
    if TransDisturb:        # TransDisturb not empty
    
        typeinfo = TransDisturb['type'] # can be "sample" or "manual"
        if typeinfo == "sample":
            TransmissionDisturbance = TransDisturb['values']
            
        elif typeinfo == "manual":  # Format: TransDisturb['TELi']=[[time, duration, amplitude],...]
            
            NW = len(spectra)
            TransmissionDisturbance = np.ones([NT,NW,NA])
            
            for itel in range(1,NA+1):
                if f'TEL{itel}' not in TransDisturb.keys():
                    pass
                else:
                    tab = TransDisturb[f'TEL{itel}']
                    Nevents = np.shape(tab)[0]
                    for ievent in range(Nevents):
                        tstart, duration, amplitude = tab[ievent]
                        istart = tstart//dt ; idur = duration//dt
                        TransmissionDisturbance[istart:istart+idur+1,:,itel-1] = amplitude
                    
        elif typeinfo == "fileMIRCx":
            
            file = TransDisturb["file"]
            hdu=fits.open(file) 
            p=hdu['PHOTOMETRY'].data

            spectra = hdu['WAVELENGTH'].data
            NW = len(spectra)
            TransmissionDisturbance = np.ones([NT,NW,NA])
            
            NT1,NT2,NW,NAfile = p.shape     # MIRCx data have a particular shape due to the camera reading mode
            inj = np.reshape(p[:,:,:,:],[NT1*NT2,NW,NAfile], order='C')
            inj = inj - np.min(inj)         # Because there are negative values
            inj = inj/np.max(inj)
            if verbose:
                print(f"Max value: {np.max(inj)}, Moy: {np.mean(inj)}")
            NTfile = NT1*NT2
            
            for ia in range(NA):
                injtemp = inj[np.random.randint(NTfile//2):,:,3]
                if np.shape(injtemp)[0] < NT:
                    TransmissionDisturbance[:,:,ia] = repeat_sequence(injtemp, NT)
                else:
                    TransmissionDisturbance[:,:,ia] = injtemp/np.mean(injtemp)
                
                TransmissionDisturbance[:,:,ia] = TransmissionDisturbance[:,:,ia]/np.mean(TransmissionDisturbance[:,:,ia]) # Average to 1
                TransmissionDisturbance[:,:,ia] = TransmissionDisturbance[:,:,ia] - np.std(TransmissionDisturbance[:,:,ia]) # Average to 1
                
            if verbose:
                print(f"Longueur sequence: {np.shape(TransmissionDisturbance)[0]} \n\
Longueur timestamps: {len(timestamps)}")
            
            # # inj = p[:, :, 0, 0].ravel()
            # NTfile = len(inj)
            # timestampsfile = TransDisturb["timestamps"]
            # sequence = inj
            
            # if timestampsfile[-1] <= timestamps[-1]:
            #     newtimestamps = timestampsfile
            #     dtfile = timestampsfile[1] - timestampsfile[0]
                
            #     if timestampsfile[0] == 0:
            #         while newtimestamps[-1] <= timestamps[-1]:
            #             sequence = np.concatenate([sequence,inj])
            #             newtimestamps = np.concatenate([newtimestamps, newtimestamps+newtimestamps[-1]+dtfile])
            #     else:
            #         while newtimestamps[-1] <= timestamps[-1]:
            #             sequence = np.concatenate([sequence,inj])
            #             newtimestamps = np.concatenate([newtimestamps, newtimestamps+newtimestamps[-1]])
            #     timestampsfile = newtimestamps     
                        
            # for ia in range(NA):
            #     istart = np.random.randint(NTfile)      # Take a random position among all available
            #     sequence = np.concatenate([sequence[istart:],sequence[:istart]])
            #     sequence = np.reshape(np.repeat(sequence[:,np.newaxis],NW,1),[NT,NW])
            #     TransmissionDisturbance[:,:,ia] = sequence
    
        elif typeinfo == "both":
            
            file = TransDisturb["file"]
            hdu=fits.open(file) 
            p=hdu['PHOTOMETRY'].data
            
#             print("A file for the photometries has been given. It defines the spectral sampling of \
# the DisturbanceFile.")
            spectra = hdu['WAVELENGTH'].data
            NW = len(spectra)
            Lc = np.abs(1/(spectra[0]-spectra[1]))      # Coherence length
    
            TransmissionDisturbance = np.ones([NT,NW,NA])
            
            NT1,NT2,NW,NAfile = p.shape     # MIRCx data have a particular shape due to the camera reading mode
            inj = np.reshape(p[:,:,:,:],[NT1*NT2,NW,NAfile], order='C')
            inj = inj - np.min(inj)         # Because there are negative values
            inj = inj/np.max(inj)
            if verbose:
                print(f"Max value: {np.max(inj)}, Moy: {np.mean(inj)}")
            NTfile = NT1*NT2
            
            if NTfile < NT:
                TransmissionDisturbance = repeat_sequence(inj, NT)
            else:
                TransmissionDisturbance = inj
            
            if verbose:
                print(f"Longueur sequence: {np.shape(TransmissionDisturbance)[0]} \n\
Longueur timestamps: {len(timestamps)}")
            
            
            for itel in range(1,NA+1):
                if f'TEL{itel}' not in TransDisturb.keys():
                    pass
                else:
                    tab = TransDisturb[f'TEL{itel}']
                    Nevents = np.shape(tab)[0]
                    for ievent in range(Nevents):
                        tstart, duration, amplitude = tab[ievent]
                        istart = tstart//dt ; idur = duration//dt
                        TransmissionDisturbance[istart:istart+idur+1,:,itel-1] *= amplitude

    
# =============================================================================
#     PISTON DISTURBANCE
# =============================================================================
    
    PistonDisturbance = np.zeros([NT,NA])

    if dist == 'coherent':
        if verbose:
            print('No piston')
        
    elif dist == 'step':
        if tel <=0:
            itel=0
        else:
            itel = tel-1
        PistonDisturbance[startframe:,itel] = ampl*np.ones(NT-startframe)
        
    elif dist == 'pair':
        PistonDisturbance[startframe:,:2] = ampl*np.ones([NT-startframe,2])
        
    elif dist == 'kstep':
        T = int(0.8*NT/NA)          # Time between two consecutive steps
        t = int(0.5*T)              # Duration of a step
        for ia in range(NA):
            PistonDisturbance[startframe+T*ia:startframe+T*ia+t,ia] = ampl*np.ones(t)
        
    elif dist == 'manualsteps':
        """
        Choose the moment of each step on each telescope thanks to the list:
            Levents=[(t_i, t0, duration, amplitude)]_i - times in ms and ampl in µm
        """
        if Levents == []:
            raise ValueError("The input parameter 'Levents' is empty")
        for event in Levents:
            ia, t0, duration, ampl = event
            startframe = t0//dt ; t = duration//dt
            PistonDisturbance[startframe:startframe+t,ia] = ampl*np.ones(t)
        pass
        
    # elif dist == 'scan':
    #     for ia in range(NA):
    #         PistonDisturbance[:,ia] = ampl*(-1)**(ia+1)*((0.5*ia+np.arange(NT))/NT-0.5)  # pistons in µm
    
    elif dist == 'slope':
        if value_end:
            if ampl:
                print("ATTENTION: You gave 'value_end' and 'ampl', only 'value_end' is used.")
            if NTend:
                ampl = (value_end-value_start)/NTend
            else:
                ampl = (value_end-value_start)/NT
                
        itel = (tel-1 if tel else 0)
        if NTend:
            PistonDisturbance[:NTend,itel] = value_start + np.arange(NTend) * ampl
        else:
            PistonDisturbance[:,itel] = value_start + np.arange(NT) * ampl
        
    # The first telil sees a range of piston from -Lc to Lc
    elif dist == 'browse':
        PistonDisturbance[:,1] = (np.arange(NT)/NT - 1/2) * 2*Lc
    
    elif dist == 'random':
        
        if 'old' in kwargs.keys():
            rmsOPD = ampl
            rmsPiston = rmsOPD/np.sqrt(2)
            freq = np.fft.fftshift(np.fft.fftfreq(NT,d=dt))
            freqfft=freq

            filtre = np.zeros(NT)
    
            # Atmospheric disturbance from Conan et al 1995
            for i in range(NT):
                if freq[i] < 0.02:
                    filtre[i] = 0
                elif freq[i] >= 0.02 or freq[i] < 3:    # Low frequencies regime 
                    filtre[i] = freq[i]**(-4/3)
                else:                                   # High frequencies regime
                    filtre[i] = freq[i]**(-8.5/3)
            
            filtre = filtre/np.max(filtre)
    
            if tel:                     # Disturbances on only one pupil
                itel = tel - 1
                np.random.seed(seed)    # Set the seed state for getting always the same disturbance
                dsp = np.fft.fftshift(np.fft.fft(np.random.randn(NT)))
                
                newdsp = dsp*filtre
                motif = np.real(np.fft.ifft(np.fft.ifftshift(newdsp)))
    
                PistonDisturbance[:,itel]= rmsPiston*motif/np.std(motif)/3
            
            else:
                for ia in range(NA):
                    np.random.seed(seed+ia)
                    dsp = np.fft.fftshift(np.fft.fft(np.random.randn(NT)))
                    newdsp = dsp*filtre
                    motif = np.real(np.fft.ifft(np.fft.ifftshift(newdsp)))
        
                    PistonDisturbance[:,ia]= rmsPiston*motif/np.std(motif)/3

            # outputs.DisturbancePSD = np.abs(newdsp[freqfft>=0])**2    # Save the PSD of the last disturbance for example
            # outputs.FreqSampling = freqfft[freqfft>=0]

        elif 'new' in kwargs.keys():        # Correct: The DSP are true DSP
            
            if 'baselines' in kwargs.keys():
                baselines = kwargs['baselines']
            else:
                InterfArray = ct.get_array(config.Name)
                baselines = InterfArray.BaseNorms
            
            V = 0.31*r0/t0*1e3              # Average wind velocity in its direction [m/s]
            L0 = L0                         # Outer scale [m]
            direction = direction           # Orientation from the North [deg] (positive toward East)
            d = d                           # Telescopes diameter [m]
                
            if ampl==0:
                wl_r0 = 0.55                # Wavelength at which r0 is defined
                # rmsOPD = np.sqrt(6.88*(L0/r0)**(5/3))*wl_r0/(2*np.pi)    # microns
                rmsOPD = 30*np.sqrt((0.12/r0)**(5/3)) # on fixe rmsOPD = 15µm pour r0=12cm (value of 30 gives 15 at the end)
                if verbose:
                    print(f'RMS OPD={rmsOPD}')
            
            else:
                rmsOPD = ampl
                
            rmsPiston = rmsOPD/np.sqrt(2)
            for ia in range(NA):
                if tel:                 # Disturbances on only one pupil
                    itel = tel - 1
                    if ia != itel:
                        continue
    
                if verbose:
                    print(f'Piston on pupil {ia}')
    
                dfreq = np.min([0.008,1/(2.2*NT*dt)]) # Minimal sampling wished
                freqmax = 1/(2*dt)                  # Maximal frequency derived from given temporal sampling
                
                Npix = int(freqmax/dfreq)*2         # Array length (taking into account aliasing)
                
                freqfft = (np.arange(Npix)-Npix//2)*dfreq
                timefft = (np.arange(Npix)-Npix//2)*dt  #s
            
                #nu0 = 0.2*V/B                      # Very low cut-off frequency
                nu1 = V/L0                          # Low cut-off frequency
                nu2 = 0.3*V/d                       # High cut-off frequency
            
                if not pows:
                    pow1, pow2, pow3 = (-2/3, -8/3, -17/3)  # Conan et al
                else:
                    pow1, pow2, pow3 = pows
                    
                b0 = nu1**(pow1-pow2)           # offset for continuity
                b1 = b0*nu2**(pow2-pow3)        # offset for continuity
                
                if verbose:
                    print(f'Atmospheric cutoff frequencies: {nu1:.2}Hz and {nu2:.2}Hz')
                
                if highCF:
                    filtre = np.zeros(Npix)
                    
                    # Define the three frequency regimes
                    lowregim = (np.abs(freqfft)>0) * (np.abs(freqfft)<nu1)
                    medregim = (np.abs(freqfft)>=nu1) * (np.abs(freqfft)<nu2)
                    highregim = np.abs(freqfft)>=nu2
                    
                    filtre[lowregim] = np.abs(freqfft[lowregim])**pow1
                    filtre[medregim] = np.abs(freqfft[medregim])**pow2*b0
                    filtre[highregim] = np.abs(freqfft[highregim])**pow3*b1

    
                """
                MOST REALISTIC DISTURBANCE SO FAR
                No high frequency cut
                """
                if not highCF:
                    lowregim = np.abs(freqfft)<nu1
                    highregim = np.abs(freqfft)>=nu1
                    
                    filtre = np.zeros(Npix)
                    
                    filtre[lowregim] = np.abs(freqfft[lowregim])**pow1
                    filtre[highregim] = np.abs(freqfft[highregim])**pow2*b0
                
                area_filtre = np.sum(filtre)/Npix
                filtrePSD = filtre/area_filtre*rmsPiston**2   # Scale the variance to rms²
                
                # Add a gaussian noise
                np.random.seed(seed+ia)
                whitenoise = np.random.randn(Npix)  # Gaussian white noise with sig²=1
                whitePSD = np.abs(np.fft.fftshift(np.fft.fft(whitenoise,norm="ortho")))**2
                
                signalPSD = whitePSD*filtrePSD      # DSP of a filtered gaussian signal with sig² = sig²(filtre)
                
                signalTF = np.sqrt(signalPSD)       # Transform of the signal
                
                motif0 = np.real(np.fft.ifft(np.fft.ifftshift(signalTF), norm="ortho"))
                keeptime = (timefft>=0)*(timefft<obstime)
                stdtime = (timefft>=0)#*(timefft<30)              # We'll compute the standard deviation on a sample of 10s
                            
                motif = motif0[keeptime]
    
                # calibmotif = motif/np.std(motif0[stdtime])
                
                PistonDisturbance[:,ia] = motif#rmsPiston*calibmotif
                
                # PistonDisturbance[:,ia] = PistonDisturbance[:,ia] - PistonDisturbance[startframe,ia]
                
                # Just for compatibility with ancient version
                newdsp = signalTF
                filtre = np.sqrt(filtrePSD)

        else:           # Inspired from Conan but more smooth
            if 'baselines' in kwargs.keys():
                baselines = kwargs['baselines']
            else:
                InterfArray = ct.get_array(name=config.Name)
            
            V = 0.31*r0/t0*1e3              # Average wind velocity in its direction [m/s]
            L0 = L0                         # Outer scale [m]
            direction = direction           # Orientation from the North [deg] (positive toward East)
            d = d                           # Telescopes diameter [m]
                
            if ampl==0:
                wl_r0 = 0.55                # Wavelength at which r0 is defined
                rmsOPD = np.sqrt(6.88*(L0/r0)**(5/3))*wl_r0/(2*np.pi)    # microns
                if verbose:
                    print(f'RMS OPD={rmsOPD}')
                
            else:
                rmsOPD = ampl
                
            rmsPiston = rmsOPD/np.sqrt(2)

            for ia in range(NA):
                if tel:                 # Disturbances on only one pupil
                    itel = tel - 1
                    if ia != itel:
                        continue
    
                tstart=time.perf_counter()
                if verbose:
                    print(f'Piston on pupil {ia}')
    
                dfreq = np.min([0.008,1/(2*NT*dt)]) # Minimal sampling wished
                freqmax = 1/(2*dt)                  # Maximal frequency derived from given temporal sampling
                
                Npix = int(freqmax/dfreq)*2         # Array length (taking into account aliasing)
                
                np.random.seed(seed+ia)
                dsp = np.fft.fftshift(np.fft.fft(np.random.randn(Npix)))
                freqfft = (np.arange(Npix)-Npix//2)*dfreq
                timefft = (np.arange(Npix)-Npix//2)*dt  #ms
                
    
                # nu0 = V/L0                    # Low cut-off frequency: Von Karman
                nu1 = 0.2*V/L0                      # Low cut-off frequency: Von Karman outer scale filtration
                nu2 = 0.3*V/d                   # High cut-off frequency
    
                lowregim = np.abs(freqfft)<nu1
                medregim = (np.abs(freqfft)>=nu1) * (np.abs(freqfft)<nu2)
                highregim = np.abs(freqfft)>=nu2
            
                if not pows:
                    pow1, pow2, pow3 = (-2/3, -8/3, -17/3)
                else:
                    pow1, pow2, pow3 = pows
                    
                b0 = nu1**(pow1-pow2)           # offset for continuity
                b1 = b0*nu2**(pow2-pow3)        # offset for continuity
                
                if verbose:
                    print(f'Atmospheric cutoff frequencies: {nu1:.2}Hz and {nu2:.2}Hz')
                
                if highCF:
                    filtre = np.zeros(Npix)
                
                    for i in range(Npix):
                        checkpoint = int(Npix/10)
                        if i%checkpoint == 0:
                            if verbose:
                                print(f'Filtering....{i/Npix*100}%')
                        if freqfft[i] == 0:
                            filtre[i] = 0
                        elif np.abs(freqfft[i]) < nu1:
                            filtre[i] = np.abs(freqfft[i])**pow1
                        elif (np.abs(freqfft[i]) >= nu1) and (np.abs(freqfft[i]) < nu2):
                            filtre[i] = np.abs(freqfft[i])**pow2*b0
                        else:
                            filtre[i] = np.abs(freqfft[i])**pow3*b1

                    

    
                """
                MOST REALISTIC DISTURBANCE SO FAR
                No high frequency cut
                """
                if not highCF:
                    filtre = np.zeros(Npix)
                    # print('Number of pixels:',Npix)
                    for i in range(Npix):
                        checkpoint = int(Npix/2)
                        if i%checkpoint == 0:
                            print(f'Filtering....{i/Npix*100}%')
                        if freqfft[i] == 0:
                            filtre[i] = 0
                        elif np.abs(freqfft[i]) < nu1:
                            filtre[i] = np.abs(freqfft[i])
                        elif np.abs(freqfft[i]) >= nu1: #and np.abs(freqfft[i]) < nu2:
                            b0 = nu1**(7/3)
                            filtre[i] = np.abs(freqfft[i])**(-4/3)*b0
                

                filtre = filtre/np.max(filtre)
                
                newdsp = dsp*filtre
                
                # var_newdsp = np.sum(newdsp)/Npix
                
                motif0 = np.real(np.fft.ifft(np.fft.ifftshift(newdsp), norm="ortho"))
                keeptime = (timefft>=0)*(timefft<obstime)
                stdtime = (timefft>=0)#*(timefft<30)              # We'll compute the standard deviation on a sample of 10s
                            
                motif = motif0[keeptime]
    
                calibmotif = motif/np.std(motif0[stdtime])
    
                PistonDisturbance[:,ia] = rmsPiston*calibmotif
                
                # PistonDisturbance[:,ia] = PistonDisturbance[:,ia] - PistonDisturbance[startframe,ia]
                ElapsedTime = time.perf_counter() - tstart
                if verbose:
                    print(f'Done. Ellapsed time: {ElapsedTime}s')
        
    elif dist == 'chirp':
        
        omega_fin = 2*np.pi*f_fin
        t_fin = timestamps[-1]
        a = omega_fin/(2*t_fin)
        chirp = lambda phi0,t : np.sin(phi0 + a*t**2)
        if tel:
            if verbose:
                print(f'CHIRP on telescope {tel}')
            itel = tel-1
            PistonDisturbance[:,itel] = ampl*chirp(0,timestamps)
            newdsp = np.fft.fftshift(np.fft.fft(PistonDisturbance[:,itel], norm='ortho'))
            freqfft = np.fft.fftshift(np.fft.fftfreq(NT, dt))
        else:
            for ia in range(NA):
                PistonDisturbance[:,ia] = chirp(ia*2*np.pi/NA,timestamps)
            newdsp = np.fft.fftshift(np.fft.fft(PistonDisturbance[:,-1], norm='ortho'))
            freqfft = np.fft.fftshift(np.fft.fftfreq(NT, dt))
        
        dfreq = freqfft[1]-freqfft[0]
        # Put first time at null-piston for reference closure phase measurement
        # PistonDisturbance[:startframe] = np.zeros([startframe,NA])


    if PhaseJumps:
        
        for itel in range(1,NA+1):
            if f'TEL{itel}' not in PhaseJumps.keys():
                pass
            else:
                tab = PhaseJumps[f'TEL{itel}']
                Nevents = np.shape(tab)[0]
                for ievent in range(Nevents):
                    tstart, duration, piston_jump = tab[ievent]
                    istart = tstart//dt; idur = duration//dt
                    
                    if idur:
                        PistonDisturbance[istart:istart+idur+1,itel-1] += piston_jump
                    else:
                        PistonDisturbance[istart:,itel-1] += piston_jump
                    
    if debug:
        return CoherentFlux, PistonDisturbance, TransmissionDisturbance
    

    hdr = fits.Header()
    hdr['TYPE'] =dist
    hdr['AMPL'] = ampl
    hdr['TEL'] = tel
    hdr['NT'] = NT
    hdr['dt'] = dt
    
    if dist == 'random' or dist == 'chirp':
        hdr['df'] = dfreq
        
    if dist == 'random':
        hdr['r0'] = round(r0*1e2)       # Fried parameter in centimeter
        hdr['t0'] = t0                  # Coherence time in ms
        hdr['Windspeed'] = V            # m/s
        hdr['direction'] = direction    
        hdr['d'] = d            # m
        hdr['L0'] = L0          # m
        hdr['nu1'] = nu1        # Hz
        hdr['nu2'] = nu2        # Hz
        hdr['rmsOPD'] = round(rmsOPD,2)  # OPD rms (µm)
        hdr['highCF'] = highCF  # Boolean
        
    primary = fits.PrimaryHDU(header=hdr)
    
    col1 = fits.Column(name='lambdas', format='1D', array=spectra)
    hdu1 = fits.BinTableHDU.from_columns([col1], name='LambdaSampling' )

    im1 = fits.ImageHDU(PistonDisturbance, name='Piston')
    im2 = fits.ImageHDU(TransmissionDisturbance, name='Transmission')
    
    hdu = fits.HDUList([primary,hdu1,im1,im2])
    
    if dist == 'random':

        DisturbancePSD = np.abs(newdsp[freqfft>=0])**2      # Save the PSD of the last disturbance for example
        filtrePSD = np.abs(filtre)**2                       # Convert the filter to PSD space
        im3 = fits.ImageHDU(DisturbancePSD, name='LAST TEL PSD')
        im4 = fits.ImageHDU(filtrePSD[freqfft>=0], name='Disturbance Filter')
        hdu.append(im3)
        hdu.append(im4)

    if verbose:
        print(f'Saving file into {filepath}')
    filedir = '/'.join(filepath.split('/')[:-1])    # remove the filename to get the file directory only
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    hdu.writeto(filepath)
    hdu.close()
    if verbose:
        print('Saved.')
    
    return

      
# @timer
def loop(*args, LightSave=True, overwrite=False, verbose=False,verbose2=True):
    """
    Core of the simulator. This routine calls the different modules according
    to the simulation timeline. 
    At the end, the data must be stored in a file. It has not yet been 
    developped.
    
    Parameters
    ----------
    DisturbanceFitsfile : STRING
        Fits file containing the disturbance data.

    Returns
    -------
    None.

    """
    
    from . import outputs

    from .config import NT, NA,timestamps, spectra, OW, MW, checktime, checkperiod, foreground
    
    outputs.TimeID=time.strftime("%Y-%m-%dT%H-%M-%S")
    outputs.simulatedTelemetries = True
    # Reload outputs module for initialising the observables with their shape
    if verbose:
        print('Reloading outputs for reinitialising the observables.')
    reload(outputs)
    
    # Importation of the object
    if NA>=3:
        CfObj, VisObj, CPObj = ct.get_CfObj(config.ObservationFile,spectra)
    else:
        CfObj, VisObj = ct.get_CfObj(config.ObservationFile,spectra)
        
    #scaling it to the spectral sampling  and integration time dt
    delta_wav = np.abs(spectra[1]-spectra[2])
    
    CfObj = CfObj * delta_wav           # Photons/spectralchannel/second at the entrance of the FS
    CfObj = CfObj * config.dt      # Photons/spectralchannel/DIT at the entrance of the FS
    
    if NA>=3:
        outputs.ClosurePhaseObject = CPObj
        
        BestTel=config.FT['BestTel'] ; itelbest=BestTel-1
        if config.FT['CPref']:                     # At time 0, we create the reference vectors
            for ia in range(NA-1):
                for iap in range(ia+1,NA):
                    if not (ia==itelbest or iap==itelbest):
                        ib = ct.posk(ia,iap,NA)
                        if itelbest>iap:
                            ic = ct.poskfai(ia,iap,itelbest,NA)   # Position of the triangle (0,ia,iap)
                        elif itelbest>ia:
                            ic = ct.poskfai(ia,itelbest,iap,NA)   # Position of the triangle (0,ia,iap)
                        else:
                            ic = ct.poskfai(itelbest,ia,iap,NA)
                    
                        outputs.OPDrefObject[ib] = np.median(outputs.ClosurePhaseObject[:,ic])/(2*np.pi)*config.wlOfTrack
        
    outputs.CoherentFluxObject = CfObj
    outputs.VisibilityObject = VisObj
    
    # Importation of the disturbance
    CfDist, PistonDist, TransmissionDist = ct.get_CfDisturbance(config.DisturbanceFile,spectra, timestamps,
                                                                foreground=foreground,verbose=verbose)          
    
    outputs.CfDisturbance = CfDist
    outputs.PistonDisturbance = PistonDist
    outputs.TransmissionDisturbance = TransmissionDist    
    # outputs.PhotometryDisturbance = np.zeros([config.NT,config.NW,config.NA])
    
    for ia in range(config.NA):
        PhotometryObject = np.abs(CfObj[:,ia*(config.NA+1)])
        outputs.PhotometryDisturbance[:,:,ia] = outputs.TransmissionDisturbance[:,:,ia]*PhotometryObject

    outputs.FTmode[:config.starttracking] = np.zeros(config.starttracking)

    if verbose2:
        print("Processing simulation ...")
    
    outputs.it = 0
    time0 = time.time()
    for it in range(NT):                        # We browse all the (macro)times
        outputs.it = it
        
        # Coherence of the ODL
        CfODL = coh__pis2coh(-outputs.EffectiveMoveODL[it,:],1/config.spectra)
        
        currCfTrue = CfObj * outputs.CfDisturbance[it,:,:] * CfODL
        outputs.CfTrue[it,:,:] = currCfTrue   #NB
        
        """
        Fringe Sensor: From oversampled true coherences to macrosampled 
        measured coherences
        """
        fringesensor = config.FS['func']
        currCfEstimated = fringesensor(currCfTrue)
        outputs.CfEstimated[it,:,:] = currCfEstimated   # NBmes


        """
        FRINGE TRACKER: From measured coherences to ODL commands
        """
        GainGD = config.FT['GainGD']
        GainPD = config.FT['GainPD']
        
        if outputs.FTmode[it] == 0:
            config.FT['GainGD'] = 0
            config.FT['GainPD'] = 0
            
        fringetracker = config.FT['func']
        CmdODL = fringetracker(currCfEstimated)
        
        config.FT['GainGD'] = GainGD
        config.FT['GainPD'] = GainPD
        
        outputs.CommandODL[it+1,:] = CmdODL
        
        # Very simple step response of the delay lines: if latency==1, the odl response is instantaneous.
        outputs.EffectiveMoveODL[it+config.latency] = CmdODL
        
        checkpoint = int(checkperiod/100 * NT)
        if (it>checkpoint) and (it%checkpoint == 0) and (it!=0) and checktime:
            processedfraction = it/NT
            # LeftProcessingTime = (time.time()-time0)*(1-processedfraction)/processedfraction
            if verbose:
                print(f'Processed: {processedfraction*100}%, Elapsed time: {round(time.time()-time0)}s')


    print(f"Done. (Total: {round(time.time()-time0)}s)")
    
    # Process observables for visualisation
    outputs.PistonTrue = outputs.PistonDisturbance - outputs.EffectiveMoveODL[:-config.latency]

    # Save true OPDs in an observable
    for ia in range(config.NA):
        for iap in range(ia+1,config.NA):
            ib = ct.posk(ia,iap,config.NA)
            outputs.OPDTrue[:,ib] = outputs.PistonTrue[:,ia] - outputs.PistonTrue[:,iap]
            outputs.OPDDisturbance[:,ib] = outputs.PistonDisturbance[:,ia] - outputs.PistonDisturbance[:,iap]
            outputs.OPDCommand[:,ib] = outputs.CommandODL[:,ia] - outputs.CommandODL[:,iap]    
            outputs.EffectiveOPDMove[:,ib] = outputs.EffectiveMoveODL[:,ia] - outputs.EffectiveMoveODL[:,iap]    
            
            # if 'search' in config.FT.keys():
            #     outputs.OPDSearchCommand[:,ib] = outputs.SearchCommand[:,ia] - outputs.SearchCommand[:,iap]
            
            for iow in range(MW):
                GammaObject = outputs.CoherentFluxObject[iow*OW,ia*NA+iap]/np.sqrt(outputs.CoherentFluxObject[iow*OW,ia*(NA+1)]*outputs.CoherentFluxObject[iow*OW,iap*(NA+1)])
                
                Ia = np.abs(outputs.CfTrue[:,iow*OW,ia*(NA+1)])    # Photometry pupil a
                Iap = np.abs(outputs.CfTrue[:,iow*OW,iap*(NA+1)])  # Photometry pupil a'
                Iaap = np.abs(outputs.CfTrue[:,iow*OW,ia*NA+iap])  # Mutual intensity aa'
                
                Lc = config.FS['R']*spectra[iow*OW]
                outputs.VisibilityTrue[:,iow,ib] = Iaap/np.sqrt(Ia*Iap)*np.abs(GammaObject)*np.sinc(outputs.OPDTrue[:,ib]/Lc)*np.exp(1j*2*np.pi*outputs.OPDTrue[:,ib]/spectra[iow*OW])
    
    # Check if one of all OPD values is higher than R*lmbda*OW/2 (Nyquist criterion)
    underSampling = (np.abs(outputs.OPDTrue) >= config.nyquistCriterion).any()
    if underSampling:
        print(f"\n /!\  ATTENTION : one or more OPD value(s) doesn't respect Nyquist criterion \
(OPD<{config.nyquistCriterion:.0f}µm).\n\
The simulation might experience aliasing. /!\\n")
    
    if len(args):
        filepath = args[0]+f"results_{outputs.TimeID}.fits"
        save_data(outputs, config, filepath, LightSave=LightSave, overwrite=overwrite, verbose=verbose)
    
    return


def display(*args, outputsData=[],timebonds=(0,-1),DIT=10,wlOfScience=0.75,
            topAxe='snr',infos={'details':''},smoothObs=0,
            pause=False, display=True,verbose=False,mergedPdf=True,
            savedir='',ext='pdf'):
    """
    
    Purpose
    ---------
        This procedure plots different quantities available in the outputs module.
        Special functions have been developped to display main quantities (see below), yet
        it is possible to plot every quantities present in outputs module by using
        'outputsData' optional parameter.
        Using skeleton.ShowPerformance function, it also computes and save in outputs interesting quantities like:
            - variance of many quantities (pd, gd, cp, etc...)
            - fringeJumpsPeriod
            - LockRatio
            
        By default, quantities related to all telescopes/baselines/closures are plotted.
        Yet, it is possible to plot only a set of tels/bases/closures using the optional parameter infos.
            
    Parameters
    ----------
    *args : LIST OF STRING, optional argument
        Quantities to display among available ones (see below).
        If args is empty, the function displays four main plots:
            - pertable
            - perfarray
            - gdHist
            - fluxHist
            
    outputsData : LIST OF STRINGS, optional. The default is [].
        List of variable names present in outputs to plot it.
    timebonds : TUPLE, optional. The default is (0,-1).
        Enables to plot only a part of the sequence.
        Quantities must be given is seconds.
        If (0,-1), plots all sequence.
    DIT : FLOAT, optional. The default is 100.
        Integration time for the variances, averages, etc...
    wlOfScience : FLOAT, optional. The default is 0.75.
        Wavelength at which the science instrument is working, for SNR computation.
    topAxe : STRING. The default is 'snr'.
        If 'snr': adds an axe at the top of some figures and plots SNR.
        If 'dispersion': adds an axe at the top of some figures and plots dispersion (GD-PD).
        Else: don't add axe.
    infos : DICTIONARY, optional. The default is {'title':''}.
        Enables to:
            - personnalise a title for the plot thanks to 'title' keyword.
            - plot only a subset of the telescopes/baselines/triangles.
        It empty, it plots all telescopes and put standard title.
        Expected keywords for plotting subset of telescopes:
            'telsToDisplay': telescopes to display
            'basesToDisplay': baselines to display
            'trianglesToDisplay': closure phases to display
        Example: to plot baselines concerning only S1S2E1, you can either write:
            - 'telsToDisplay':['S1','S2','E1'] or
            - 'basesToDisplay':['S1S2','S1E1','S2E1']
    pause : BOOLEAN, optional. The default is False.
        Enables to show plots during a simulation.
    display : BOOLEAN, optional. The default is True.
        If True, displays plots.
        If False, generate plots but don't display. 
    verbose : BOOLEAN, optional. The default is False.
        If True, writes information in the terminal.
    savedir : STRING, optional. The default is ''.
        Directory path for saving the files. If empty, don't save plots.
    ext : STRING or LIST OF STRINGS, optional. The default is 'pdf'.
        Extension of the file.
        If a list of file is given, it saves in all extensions included in the list.
        
    Returns
    -------
    None.

    Available observables to display
    --------------------------------
        
    ##############################################
    # Variables recorded in SPICA-FT telemetries #
    ##############################################
    
    ### SINGLE VARIABLE PLOT ###
    
    # OPD-space quantities #
    
    - 'gd': gdEst - Estimated GD - eq. 33
    - 'gdLsq': gdLsq - Estimated GD after filtered least square - eq. 35
    - 'pd': pdEst - Estimated PD - eq. 33 (adapted to PD)
    - 'pdLsq': pdLsq - Estimated PD after filtered least square - eq. 35 (adapted to PD)
    - 'snr': Estimated smoothed SNR
    - 'snrPd': Estimated smoothed SNR with coherent spectral addition
    - 'snrGd': Estimated smoothed SNR with uncoherent spectral addition
    - 'gdCmd': GD commands
    - 'pdCmd': PD commands
    - 'cmdOpd': Commands sent to the delay lines (converted in opd-space)
    - 'gdHist': Histogram of the estimated GD
    
    # Piston-space quantities #
    
    - 'estFlux': Estimated photometries
    - 'cmdPis': Commands sent to the delay lines
    - 'gdCmdPis': Piston GD commands
    - 'pdCmdPis': Piston PD commands
    - 'fluxHist': Histogram of the estimated photometries
    
    # Closure-space quantities #
    
    - 'cpd': PD closure phase
    - 'cgd': GD closure phase
    
    ### MULTIPLE VARIABLE PLOT ###
    
    - 'perftable': snr/dispersion*, PDest, GDest, fringe jumps and rms(pd)
    - 'gdPdEst': snr/dispersion*, gdEst, pdEst, rms(gdEst), rms(pdEst)
    - 'gdPdLsq': snr/dispersion*, gdRes, pdLsq, rms(gdLsq), rms(pdLsq)
    - 'gdPdCmd': snr/dispersion*, gdCmd, pdCmd, rms(gdCmd), rms(pdCmd)
    - 'gdPdCmdDiff': snr/dispersion*, gdCmdDiff, pdCmdDiff, fringe jumps, rms(pdCmdDiff)
    - 'cgdCpd': cgd, cpd, rms(cgd), rms(cpd)
    - 'cgdCpd_all': cgd, cpd, rms(cgd), rms(cpd) computed for all triangles using gdEst and pdEst
    *plotted depending on 'topAxe' input parameter. It plots SNR by default.
    
    ### GLOBAL PERFORMANCE IN ARRAY VISUALISATION ###
    
    - 'perfarray': the CHARA array with colorful lines representing performance
    
    ########################################
    # Quantities only recorded by simulator #
    ########################################
    
    ### SINGLE QUANTITY PLOT ###
    
    # OPD-space quantities #
    
    - 'gd2': gdEst - Estimated GD with patch
    - 'gdErr': gdErr - Estimated GD after subtraction of reference - eq.34
    - 'pd2': pdEst - Estimated PD - eq. 33 (adapted to PD)
    - 'pdErr': pdErr - Estimated PD after subtraction of reference - eq.34 (adapted to PD)
    - 'distOpd': disturbance opd
    - 'opd': true opd
    - 'estVis': Estimated squared visibilities
    - 'trueVis': True squared visibilities
    
    # Piston-space quantities #
    
    - 'trueFlux': True flux
    - 'distPis': Piston disturbances
    
    ### MULTIPLE QUANTITIES PLOT ###
    
    - 'gdPdEst2': gdEst, pdEst, rms(gdEst), rms(pdEst)
    - 'gdPdErr': gdErr, pdErr, rms(gdErr), rms(pdErr)
    - 'detector': evolution of the images with time.

    """
    
    from . import display_module
    from . import outputs
    
    from .config import NA,NIN,NC,ND,NT,wlOfTrack
    NINmes = config.FS['NINmes']
    
    if (len(savedir)) and (not os.path.exists(savedir)):
        os.makedirs(savedir)
    
    ms=1e3
    ich = config.FS['ich']
    R = config.FS['R']
    
    dt=config.dt
    whichSNR = config.FT['whichSNR']
    
    from .outputs import timestamps, TimeID
    
    if outputs.simulatedTelemetries:
        wlIndex = np.argmin(np.abs(config.spectraM-wlOfTrack))
        filenamePrefix = f"Simu{TimeID}"
        
    else:
        filenamePrefix = outputs.outputsFile.split('.fits')[0]
        #filenamePrefix = f"TT{TimeID}"
    
    InterfArray = config.InterfArray

    if len(timebonds)==2:
        if timebonds[1]==-1:
            timerange = range(np.argmin(np.abs(timestamps-timebonds[0])),NT-1)
        else:
            timerange = range(np.argmin(np.abs(timestamps-timebonds[0])),np.argmin(np.abs(timestamps-timebonds[1])))
    else:
        stationaryregim_start = config.starttracking+(config.NT-config.starttracking)*1//3
        if stationaryregim_start >= NT: stationaryregim_start=config.NT*1//3
        timerange = np.arange(stationaryregim_start,NT-1)
        
    display_module.timerange = timerange
    
    effDIT = min(DIT, config.NT - config.starttracking -1)

    if not ('opdcontrol' in args):
        timeBonds = [timestamps[timerange][0],timestamps[timerange][-1]]
        ShowPerformance(timeBonds, wlOfScience, effDIT, display=False,verbose=verbose)
    else:
        if verbose:
            print("don't compute performances")
            
    timestamps = timestamps[timerange]
    t = timestamps*ms # time in ms
    
    if verbose:
        print('Displaying observables...')
        print(f'First fig is Figure {config.newfig}')
    
    if (len(args)==0) and (len(outputsData)==0):
        args = ['perftable','gdEstMatricial','pdEstMatricial','gdCmdMatricial',
                'estFlux','fluxHist','gdHist','pdPsd','pdCumStd','cmdPsd']
        
    if 'focusRelock' in args:
        args = ['gdPdEst','snr','gdPdCmd','gd']
        outputsData=['OPDCommandRelock','CommandRelock']
    
    NAdisp = 10
    NumberOfTelFigures = 1+NA//NAdisp - 1*(NA % NAdisp==0)
    telcolors = colors[:NAdisp]*NumberOfTelFigures
    
    """
    HANDLE THE POSSILIBITY TO SHOW ONLY A PART OF THE TELESCOPES/BASELINES/CLOSURES
    """
    
    TelConventionalArrangement = InterfArray.TelNames    
    telNameLength=len(TelConventionalArrangement[0])
    if 'TelescopeArrangement' in infos.keys():
        telescopes = infos['TelescopeArrangement']
    elif 'Beam2Tel' in vars(config):
        telNameLength=2
        telescopes = [config.Beam2Tel[i:i+telNameLength] for i in range(0, len(config.Beam2Tel), telNameLength)]
    else:
        telescopes = TelConventionalArrangement
        
    # to be validated (it unvalidates the 10 lines above) - must be coherent with TrueTelemetries module
    telescopes = TelConventionalArrangement
    #
    
    display_module.telescopes = telescopes
        
    beam_patches = []
    for ia in range(NA):
        beam_patches.append(mpatches.Patch(color=telcolors[ia],label=telescopes[ia]))
        
    display_module.beam_patches = beam_patches
    
    baselinesNIN = []
    itel=0
    for tel1 in telescopes:
        for tel2 in telescopes[itel+1:]:
            baselinesNIN.append(f'{tel1}{tel2}')
        itel+=1
    baselinesNIN = np.array(baselinesNIN) 
    display_module.baselinesNIN = baselinesNIN
    
    baselines = []
    itel=0
    for ib in range(NINmes):
        ia, iap = int(ich[ib][0])-1,int(ich[ib][1])-1
        tel1,tel2 = telescopes[ia],telescopes[iap]
        baselines.append(f'{tel1}{tel2}')
        
    baselines = np.array(baselines)
    if isinstance(config.baselineArrangement,(list,np.ndarray)):
        baselines = np.array(config.baselineArrangement)
    display_module.baselines = baselines
    
    if NC>=1:
        closures = []
        tel1=telescopes[0] ; itel1=0 
        for tel1 in telescopes:
            itel2 = itel1+1
            for tel2 in telescopes[itel1+1:]:
                itel3=itel2+1
                for tel3 in telescopes[itel2+1:]:
                    closures.append(f'{tel1}{tel2}{tel3}')
                    ib = ct.poskfai(itel1,itel2, itel3, NA)
                    itel3+=1
                itel2+=1
            itel1+=1
        
        closures = np.array(closures)
        display_module.closures = closures
    
    
    plotTel = [False]*NA ; plotTelOrigin=[False]*NA
    plotBaselineNIN = [False]*NIN
    plotBaseline = [False]*NINmes
    plotClosure = [False]*NC
    plotClosureND = [False]*ND
    telNameLength = len(InterfArray.TelNames[0])
    
    if 'telsToDisplay' in infos.keys():
        telsToDisplay = infos['telsToDisplay']
        for ia in range(NA):
            tel = telescopes[ia] ; tel2 = TelConventionalArrangement[ia]
            if tel in telsToDisplay:
                plotTel[ia]=True
            if tel2 in telsToDisplay:
                plotTelOrigin[ia]=True
                
        if not 'basesToDisplay' in infos.keys():
            for ib in range(NIN):
                baseline = baselinesNIN[ib]
                tel1,tel2=baseline[:telNameLength],baseline[telNameLength:]
                if (tel1 in telsToDisplay) \
                    and (tel2 in telsToDisplay):
                        plotBaselineNIN[ib] = True
                        
            for ib in range(NINmes):
                baseline = baselines[ib]
                tel1,tel2=baseline[:telNameLength],baseline[telNameLength:]
                if (tel1 in telsToDisplay) \
                    and (tel2 in telsToDisplay):
                        plotBaseline[ib] = True
                    
        if (not 'trianglesToDisplay' in infos.keys()) and (NC>=1):
            for ic in range(NC):
                closure = closures[ic]
                tel1,tel2,tel3=closure[:telNameLength],closure[telNameLength:2*telNameLength],closure[2*telNameLength:]
                if (tel1 in telsToDisplay) \
                    and (tel2 in telsToDisplay) \
                        and (tel3 in telsToDisplay):
                            plotClosure[ic] = True
                            
            for ic in range(config.ND):
                closure = closures[ic]
                tel1,tel2,tel3=closure[:telNameLength],closure[telNameLength:2*telNameLength],closure[2*telNameLength:]
                if (tel1 in telsToDisplay) \
                    and (tel2 in telsToDisplay) \
                        and (tel3 in telsToDisplay):
                            plotClosureND[ic] = True
                
    if 'basesToDisplay' in infos.keys():
        basesToDisplay = infos['basesToDisplay']
        for ia in range(NA):
            tel = telescopes[ia] ; tel2 = TelConventionalArrangement[ia]
            if tel in "".join(basesToDisplay):
                plotTel[ia]=True
            if tel2 in "".join(basesToDisplay):  
                plotTelOrigin[ia]=True
                    
        for ib in range(NIN):
            baseline = baselinesNIN[ib]
            if (baseline in basesToDisplay) or (baseline[telNameLength:]+baseline[:telNameLength] in basesToDisplay):
                plotBaselineNIN[ib] = True
        
        for ib in range(NINmes):
            baseline = baselines[ib]
            if (baseline in basesToDisplay) or (baseline[telNameLength:]+baseline[:telNameLength] in basesToDisplay):
                plotBaseline[ib] = True
        
        if (not 'trianglesToDisplay' in infos.keys()) and (NC>=1):
            for ic in range(NC):
                closure = closures[ic]
                base1, base2,base3=closure[:2*telNameLength],closure[telNameLength:],"".join([closure[:telNameLength],closure[2*telNameLength:]])
                if (base1 in basesToDisplay) \
                    and (base2 in basesToDisplay) \
                        and (base3 in basesToDisplay):
                            plotClosure[ic] = True
                            
            for ic in range(ND):
                closure = closures[ic]
                base1, base2,base3=closure[:2*telNameLength],closure[telNameLength:],"".join([closure[:telNameLength],closure[2*telNameLength:]])
                if (base1 in basesToDisplay) \
                    and (base2 in basesToDisplay) \
                        and (base3 in basesToDisplay):
                            plotClosureND[ic] = True
                            
    if ('trianglesToDisplay' in infos.keys()) and (NC>=1):
        trianglesToDisplay = infos['trianglesToDisplay']
        for ia in range(NA):
            tel = telescopes[ia] ; tel2 = TelConventionalArrangement[ia]
            if tel in "".join(trianglesToDisplay):
                plotTel[ia]=True
            if tel2 in "".join(trianglesToDisplay):
                plotTelOrigin[ia]=True
        
        for ib in range(NIN):
            baseline = baselinesNIN[ib]
            for closure in trianglesToDisplay:
                if baseline in closure:
                    plotBaselineNIN[ib] = True
        
        for ib in range(NINmes):
            baseline = baselines[ib]
            for closure in trianglesToDisplay:
                if baseline in closure:
                    plotBaseline[ib] = True
        
        for ic in range(NC):
            closure = closures[ic]
            if closure in trianglesToDisplay:
                plotClosure[ic] = True
                
        for ic in range(ND):
            closure = closures[ic]
            if closure in trianglesToDisplay:
                plotClosureND[ic] = True
                
    if not (('telsToDisplay' in infos.keys()) \
            or ('basesToDisplay' in infos.keys()) \
                or ('trianglesToDisplay' in infos.keys())):
        plotTel = [True]*NA ; plotTelOrigin = [True]*NA
        plotBaselineNIN = [True]*NIN
        plotBaseline = [True]*NINmes
        plotClosure = [True]*NC
        plotClosureND = [True]*ND
        
    # plotBaselineNINIndex = np.argwhere(plotBaselineNIN).ravel()
    # plotBaselineIndex = np.argwhere(plotBaseline).ravel()
    
    display_module.plotTel = plotTel
    display_module.plotTelOrigin = plotTelOrigin
    display_module.plotBaselineNIN = plotBaselineNIN
    display_module.plotBaseline = plotBaseline
    display_module.plotClosure = plotClosure
    display_module.plotClosureND = plotClosureND
    display_module.telNameLength = telNameLength
    # display_module.plotBaselineIndex = plotBaselineIndex
    # display_module.plotBaselineNINIndex = plotBaselineNINIndex

    """
    COMPUTATION RMS
    """
    
    # Estimated, before patch (eq. 33)
    GD = outputs.GDEstimated ; PD=outputs.PDEstimated 
    
    # Estimated, after patch
    GD2 = outputs.GDEstimated2 ; PD2 = outputs.PDEstimated2   
    
    # Residual, after subtraction of reference vectors (eq. 34)
    GDerr2 = outputs.GDResidual2 ; PDerr2 = outputs.PDResidual2
    
    # Residual, after Igd and Ipd (eq. 36)
    GDerr = outputs.GDResidual ; PDerr = outputs.PDResidual
    
    # Reference vectors
    GDrefmic = outputs.GDref*R*wlOfTrack/2/np.pi ; PDrefmic = outputs.PDref*wlOfTrack/2/np.pi
    
    GDmic = GD[timerange]*R*wlOfTrack/2/np.pi ; PDmic = PD[timerange]*wlOfTrack/2/np.pi
    GDerrmic2 = GDerr2[timerange]*R*wlOfTrack/2/np.pi ; PDerrmic2 = PDerr2[timerange]*wlOfTrack/2/np.pi
    
    GDrefmic = outputs.GDref[timerange]*R*wlOfTrack/2/np.pi ; PDrefmic = outputs.PDref[timerange]*wlOfTrack/2/np.pi

    
    if outputs.simulatedTelemetries:
        GDmic2 = GD2[timerange]*R*wlOfTrack/2/np.pi ; PDmic2 = PD2[timerange]*wlOfTrack/2/np.pi
        GDerrmic = GDerr[timerange]*R*wlOfTrack/2/np.pi ; PDerrmic = PDerr[timerange]*wlOfTrack/2/np.pi

    """
    SIGNAL TO NOISE RATIOS
    """
    
    SNR_pd = np.sqrt(outputs.SquaredSNRMovingAveragePD[timerange])
    SNR_gd = np.sqrt(outputs.SquaredSNRMovingAverageGD[timerange])
    if outputs.simulatedTelemetries:
        SNRGD = np.sqrt(outputs.SquaredSNRMovingAverageGDUnbiased[timerange])
    
    if whichSNR == 'pd':
        SNR = SNR_pd
    else:
        SNR = SNR_gd
    
    
    """
    POWER SPECTRAL DISTRIBUTION AND CUMULATIVE STANDARD DEVIATION
    """
    
    if 'psd' in ''.join(args).casefold():
        if 'cumstd' in ''.join(args).casefold():
            if 'pdcumstd' in [x.casefold() for x in args]:
                frequencySampling,pdPsd,frequencySamplingSmoothed,pdPsdSmoothed,pdCumStd = ct.getPsd(PDmic,timestamps,cumStd=True,mov_average=20)

            if 'gdcumstd' in [x.casefold() for x in args]:
                frequencySampling,gdPsd,frequencySamplingSmoothed,gdPsdSmoothed,gdCumStd = ct.getPsd(GDmic,timestamps,cumStd=True,mov_average=20)
            
        else:   # only computes psd because cumStd computation takes time
            if 'pdpsd' in [x.casefold() for x in args]:
                frequencySampling,pdPsd,frequencySamplingSmoothed,pdPsdSmoothed = ct.getPsd(PDmic,timestamps,mov_average=20)

            if 'gd' in [x.casefold() for x in args]:
                frequencySampling,gdPsd,frequencySamplingSmoothed,gdPsdSmoothed = ct.getPsd(GDmic,timestamps,mov_average=20)
    
        if 'cmdpsd' in [x.casefold() for x in args]:
            frequencySampling,cmdPsd,frequencySamplingSmoothed,cmdPsdSmoothed = ct.getPsd(outputs.OPDCommand,timestamps,mov_average=20)
        
        outputs.frequencySampling = frequencySampling
        
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""" PREPARE PDF FILE FOR SAVING FIGURES """"""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""
    
    saveIndividualFigs = False
    if mergedPdf:
        pdf = PdfPages(savedir+outputs.outputsFile.split('.fits')[0]+"_figures.pdf")
        d = pdf.infodict()
        d['Title'] = 'Performance Plots'
        d['AttachedFile'] = outputs.outputsFile
        d['CreationDate'] = datetime.datetime.today()
    elif len(savedir):
        saveIndividualFigs = True
    
    """""""""""""""""""""""""""""""""""""""
    """"""""  NOW YOU CAN DISPLAY  """"""""
    """""""""""""""""""""""""""""""""""""""

    if len(outputsData):
        
        for obsName in outputsData:
            if not (obsName in vars(outputs)):     # case-insensitive test
                print(f"{obsName} not in outputs module, I can't plot it")
                continue

            obs = getattr(outputs, obsName)[timerange]
            obsBar = np.std(obs,axis=0)
            
            generalTitle = obsName
            obsType = obsName
            if saveIndividualFigs:
                filename= savedir+f"{filenamePrefix}_{obsType}"
            else:
                filename=''
            
            if "pis" in obsName.casefold():
                fig = display_module.simpleplot_tels(timestamps, obs,obsBar,generalTitle,plotTel,
                                          obsName=obsName,
                                          display=display,filename=filename,ext=ext,infos=infos,
                                          verbose=verbose)
                
            elif ("opd" in obsName.casefold())\
                or ("snr" in obsName.casefold()):
                    
                fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaselineNIN,
                                          obsName=obsName,
                                          display=display,filename=filename,ext=ext,infos=infos,
                                          verbose=verbose)
                
            else:
                if obs.shape[-1] < 10:
                    print(f"{obsName} plotted with piston-oriented display.")
                    fig = display_module.simpleplot_tels(timestamps, obs,obsBar,generalTitle,plotTel,
                                              obsName=obsName,
                                              display=display,filename=filename,ext=ext,infos=infos,
                                              verbose=verbose)
                else:
                    print(f"{obsName} plotted with OPD-oriented display.")
                    fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaselineNIN,
                                              obsName=obsName,
                                              display=display,filename=filename,ext=ext,infos=infos,
                                              verbose=verbose)
            if mergedPdf:
                pdf.savefig(fig)



    """ PLOT OF GD AND PD TYPES OBSERVABLES """

    if 'perftable' in args:
        generalTitle = "GD and PD estimated"
        obsType = "perftable"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''
        
        GDobs = GDmic
        PDobs = PDmic
        dispersion = GDobs-PDobs
        
        gdBar = outputs.fringeJumpsPeriod
        pdBar = np.sqrt(outputs.VarPDEst)*config.wlOfTrack/2/np.pi

        if topAxe.casefold()=='snr':
            fig = display_module.perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,gdBar,pdBar,
                                     plotBaseline,generalTitle,SNR=SNR,obsType=obsType, 
                                     display=display,filename=filename,ext=ext,infos=infos)
        elif topAxe.casefold()=='dispersion':
            fig = display_module.perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,gdBar,pdBar,
                                     plotBaseline,generalTitle,dispersion=dispersion, obsType=obsType, 
                                     display=display,filename=filename,ext=ext,infos=infos)
        else:
            fig = display_module.perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,gdBar,pdBar,
                                     plotBaseline,generalTitle,obsType=obsType, 
                                     display=display,filename=filename,ext=ext,infos=infos)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'gdPdEst' in args:
        generalTitle = "GD and PD estimated"
        obsType = "GDPDest"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''
        
        GDobs = GDmic
        PDobs = PDmic
        dispersion = GDobs-PDobs
        
        gdBar = np.sqrt(outputs.VarGDEst)*config.wlOfTrack*config.FS['R']/2/np.pi
        pdBar = np.sqrt(outputs.VarPDEst)*config.wlOfTrack/2/np.pi
        
        if topAxe.casefold()=='snr':
            fig = display_module.perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,gdBar,pdBar,
                                     plotBaseline,generalTitle,SNR=SNR,obsType=obsType, 
                                     display=display,filename=filename,ext=ext,infos=infos)
        elif topAxe.casefold()=='dispersion':
            fig = display_module.perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,gdBar,pdBar,
                                     plotBaseline,generalTitle,dispersion=dispersion, obsType=obsType, 
                                     display=display,filename=filename,ext=ext,infos=infos)
            
        else:
            fig = display_module.perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,gdBar,pdBar,
                                     plotBaseline,generalTitle,display=display,
                                     filename=filename,ext=ext,infos=infos)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'gdPdEst2' in args:
        generalTitle = "GD and PD estimated, after patch"
        obsType = "GDPDest2"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''
            
        GDobs = GDmic2
        PDobs = PDmic2
        
        gdBar = np.std(GDobs,axis=0)
        pdBar = np.std(PDobs,axis=0)
        
        if topAxe.casefold()=='snr':
            fig = display_module.perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,gdBar,pdBar,
                                     plotBaseline,generalTitle,SNR=SNR,display=display,
                                     filename=filename,ext=ext,infos=infos)
        else:
            fig = display_module.perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,gdBar,pdBar,
                                     plotBaseline,generalTitle,display=display,
                                     filename=filename,ext=ext,infos=infos)
        if mergedPdf:
            pdf.savefig(fig)
        
    if 'gdPdLsq' in args:
        generalTitle = "GD and PD estimated, after filtered least square"
        obsType = "gdPdLsq"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''
            
        GDobs = GDerrmic
        PDobs = PDerrmic
        
        gdBar = np.sqrt(outputs.VarGDRes)*config.wlOfTrack*config.FS['R']/2/np.pi
        pdBar = np.sqrt(outputs.VarPDRes)*config.wlOfTrack/2/np.pi
        
        if topAxe.casefold()=='snr':
            fig = display_module.perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,gdBar,pdBar,
                                     plotBaseline,generalTitle,SNR=SNR,display=display,
                                     filename=filename,ext=ext,infos=infos)
        else:
            fig = display_module.perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,gdBar,pdBar,
                                     plotBaseline,generalTitle,display=display,
                                     filename=filename,ext=ext,infos=infos)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'gdPdErr' in args:
        generalTitle = "GD and PD errors"
        obsType = "gdPdErr"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''
        
        GDobs = GDerrmic2
        PDobs = PDerrmic2
        
        gdBar = np.std(GDobs,axis=0)
        pdBar = np.std(PDobs,axis=0)
        
        if topAxe.casefold()=='snr':
            fig = display_module.perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,gdBar,pdBar,
                                     plotBaseline,generalTitle,SNR=SNR,display=display,
                                     filename=filename,ext=ext,infos=infos)
        else:
            fig = display_module.perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,gdBar,pdBar,
                                     plotBaseline,generalTitle,display=display,
                                     filename=filename,ext=ext,infos=infos)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'gdPdCmd' in args:
        generalTitle = "GD and PD commands in OPD-space"
        obsType = "gdPdCmd"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''
            
        GDobs = outputs.GDCommand[timerange]-outputs.GDCommand[timerange[0]]
        PDobs = outputs.PDCommand[timerange]-outputs.PDCommand[timerange[0]]
        
        gdBar = np.std(GDobs,axis=0)
        pdBar = np.std(PDobs,axis=0)
        
        if topAxe.casefold()=='snr':
            fig = display_module.perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,gdBar,pdBar,
                                     plotBaseline,generalTitle,SNR=SNR,display=display,
                                     filename=filename,ext=ext,infos=infos)
        else:
            fig = display_module.perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,gdBar,pdBar,
                                     plotBaseline,generalTitle,display=display,
                                     filename=filename,ext=ext,infos=infos)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'gdPdCmdDiff' in args:
        generalTitle = "GD and PD commands diff in OPD-space"
        obsType = "gdPdCmdDiff"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''
            
        GDobs = outputs.GDCommand[timerange][1:] - outputs.GDCommand[timerange][:-1]
        PDobs = outputs.PDCommand[timerange][1:] - outputs.PDCommand[timerange][:-1]
        GDobs[-1] = 0 ; PDobs[-1] = 0
        
        fringeJumpsPeriod = outputs.fringeJumpsPeriod#durationSeconds/np.sum(np.abs(GDobs),axis=0)
        # fringeJumpsPeriod[fringeJumpsPeriod==np.inf] = durationSeconds
        pdBar = np.std(PDobs,axis=0)
        
        timestmp = timestamps[1:]
        PDrefmic_short = PDrefmic[1:]
        GDrefmic_short = GDrefmic[1:]
        SNR_short = SNR[1:]
        
        if topAxe.casefold()=='snr':
            fig = display_module.perftable(timestmp, PDobs,GDobs,GDrefmic_short,PDrefmic_short,fringeJumpsPeriod,pdBar,
                                     plotBaseline,generalTitle,SNR=SNR_short,obsType=obsType,display=display,
                                     filename=filename,ext=ext,infos=infos)
        else:
            fig = display_module.perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,fringeJumpsPeriod,pdBar,
                                     plotBaseline,generalTitle,obsType=obsType,display=display,
                                     filename=filename,ext=ext,infos=infos)
        if mergedPdf:
            pdf.savefig(fig)

    if 'cgdCpd' in args:
        generalTitle = "GD and PD closure phases"
        obsType = "closure"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''
        
        GDobs = outputs.ClosurePhaseGD[timerange]*180/np.pi
        PDobs = outputs.ClosurePhasePD[timerange]*180/np.pi
        
        GDObsInfo = np.sqrt(outputs.VarCGD)*180/np.pi
        PDObsInfo = np.sqrt(outputs.VarCPD)*180/np.pi

        fig = display_module.perftable_cp(timestamps, PDobs,GDobs,GDObsInfo,PDObsInfo,
                                    plotClosureND,generalTitle,obsType=obsType,
                                    display=display,
                                    filename=filename,ext=ext,infos=infos)
        if mergedPdf:
            pdf.savefig(fig)
        
    if 'cgdCpd_all' in args:
        generalTitle = "GD and PD closure phases"
        obsType = "closureAfter"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''
        
        GDobs = outputs.ClosurePhaseGDafter[timerange]*180/np.pi
        PDobs = outputs.ClosurePhasePDafter[timerange]*180/np.pi
        
        GDObsInfo = np.mean(GDobs,axis=0)
        PDObsInfo = np.mean(PDobs,axis=0)

        fig = display_module.perftable_cp(timestamps, PDobs,GDobs,GDObsInfo,PDObsInfo,
                                    plotClosure,generalTitle,obsType=obsType,
                                    display=display,
                                    filename=filename,ext=ext,infos=infos)
        if mergedPdf:
            pdf.savefig(fig)

    """ PLOT OF A SINGLE OBSERVABLES (flux, PD,GD,OPD,SNR,etc...) IN OPD-SPACE """

    if 'distOpd' in args:
        generalTitle = 'OPD Disturbances'
        obsType = "distOpd"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''
            
        obs = outputs.OPDDisturbance[timerange]
        obsBar = np.std(obs,axis=0)
        
        fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaselineNIN,
                                  obsName='OPD [µm]',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'opd' in args:
        generalTitle = 'True OPDs'
        obsType = "OPDtrue"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''
            
        obs = outputs.OPDTrue[timerange]
        obsBar = np.std(obs,axis=0)
        
        fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaselineNIN,
                                  obsName='OPD [µm]',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'pd' in args:
        generalTitle = 'Phase-delays'
        obsType='PD'
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''

        obs = PDmic
        obsBar = np.sqrt(outputs.VarPDEst)*config.wlOfTrack/2/np.pi
        
        fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaseline,
                                  obsName='PD [µm]',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'pd2' in args:
        generalTitle = 'Phase-delays 2'
        obsType = "PD2"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''
        
        obs = PDmic2    
        obsBar = np.std(obs,axis=0)
        
        fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaseline,
                                  obsName='PD [µm]',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'gd' in args:

        generalTitle = 'Group-delays'
        obsType = "GD"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''        
            
        obs = GDmic
        obsBar = np.sqrt(outputs.VarGDEst)*config.FS['R']*config.wlOfTrack/2/np.pi
        
        fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaseline,
                                  obsName='GD [µm]',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'gd2' in args:
        generalTitle = 'Group-delays 2'
        obsType = "GD2"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''        
            
        obs = GDmic2
        obsBar = np.std(obs,axis=0)
        fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaseline,
                                  obsName='GD2 [µm]',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'gdLsq' in args:
        generalTitle = 'Group-delays error'
        obsType = "GDerr"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''        
            
        obs = GDerrmic
        obsBar = np.sqrt(outputs.VarGDRes)*config.FS['R']*config.wlOfTrack/2/np.pi
        
        fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaseline,
                                  obsName='GDerr [µm]',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)    
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'gdErr' in args:
        generalTitle = 'Group-delays filtered'
        obsType = "GDerr2"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''        
            
        obs = GDerrmic2
        obsBar = np.std(obs,axis=0)
        fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaseline,
                                  obsName='GDerr2 [µm]',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'trueVis' in args:
        generalTitle = 'True visibilities'
        obsType = "SquaredVisTrue"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''        
            
        obs = np.abs(outputs.VisibilityTrue[timerange,wlIndex,:])**2
        obsBar = np.std(np.abs(outputs.VisibilityTrue[timerange,wlIndex,:])**2,axis=0)
        fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaseline,
                                  obsName='Square Vis True',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'estVis' in args:
        generalTitle = 'Estimated visibilities'
        obsType = "SquaredVisEst"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''        
            
        obs = np.abs(outputs.VisibilityEstimated[timerange,wlIndex,:])**2
        obsBar = np.std(np.abs(outputs.VisibilityEstimated[timerange,wlIndex,:])**2,axis=0)
        fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaseline,
                                  obsName='Square Vis Estimated',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
        
    if 'snr' in args:
        generalTitle = 'SNR'
        obsType = "SNR"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''        
            
        obs = SNR
        obsBar = np.std(obs,axis=0)
        fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaseline,
                                  obsName='SNR',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)        
        
    if 'snrPd' in args:

        generalTitle = 'SNR PD'
        obsType="SNRPD"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''     
            
        obs = SNR_pd
        obsBar = np.std(obs,axis=0)
        
        fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaseline,
                                  obsName='SNR',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'snrGd' in args:
        generalTitle = 'SNR GD'
        obsType="SNRGD"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''     
            
        obs = SNR_gd
        obsBar = np.std(obs,axis=0)
        fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaseline,
                                  obsName='SNR',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'gdCmd' in args:
        generalTitle = 'GD Commands'
        obsType="GDCmd"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''     
            
        obs = outputs.GDCommand[timerange]
        obsBar = np.std(obs,axis=0)
        fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaseline,
                                  obsName='Commands [µm]',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
        
    if 'pdCmd' in args:
        generalTitle = 'PD Commands'
        obsType="PDCmd"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''     
            
        obs = outputs.PDCommand[timerange]
        obsBar = np.std(obs,axis=0)
        fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaseline,
                                  obsName='Commands [µm]',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
        
    if 'cmdOpd' in args:
        generalTitle = 'OPD Commands'
        obsType="OPDCmd"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''     
            
        obs = outputs.OPDCommand[timerange]
        obsBar = np.std(obs,axis=0)
        fig = display_module.simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotBaseline,
                                  obsName='Commands [µm]',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
        
    """ PISTONS-SPACE (flux, pistons) """    
    
    if 'estFlux' in args:
        generalTitle = 'Estimated Flux'
        obsType = "estFlux"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''
            
        if outputs.simulatedTelemetries:
            obs = outputs.PhotometryEstimated[timerange,wlIndex]
            obsBar = np.mean(obs,axis=0)
        else:
            obs = outputs.PhotometryEstimated[timerange]
            obsBar = np.mean(obs,axis=0)
        
        if smoothObs:
            obs = ct.moving_average(obs, smoothObs)
        
        fig = display_module.simpleplot_tels(timestamps, obs,obsBar,generalTitle,plotTel,
                                  obsName='Flux [ph]',barName='Average',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
    
    
    if 'trueFlux' in args:
        generalTitle = 'True Flux'
        obsType = "trueFlux"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''
            
        if outputs.simulatedTelemetries:
            obs = outputs.PhotometryDisturbance[timerange,wlIndex]
            obsBar = np.mean(obs,axis=0)
        else:
            obs = outputs.PhotometryDisturbance[timerange]
            obsBar = np.mean(obs,axis=0)
        
        fig = display_module.simpleplot_tels(timestamps, obs,obsBar,generalTitle,plotTel,
                                  obsName='Flux [ph]',barName='Average',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
    
    if 'distPis' in args:
        generalTitle = 'Piston Disturbances'
        obsType="pisDist"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''     
            
        obs = outputs.PistonDisturbance[timerange]
        obsBar = np.std(obs,axis=0)
        fig = display_module.simpleplot_tels(timestamps, obs,obsBar,generalTitle,plotTel,
                                  obsName='Disturbances [µm]',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'cmdPis' in args:
        generalTitle = 'Piston Commands'
        obsType="pisCmd"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''     
            
        obs = outputs.CommandODL[timerange]
        obsBar = np.std(obs,axis=0)
        fig = display_module.simpleplot_tels(timestamps, obs,obsBar,generalTitle,plotTel,
                                  obsName='Commands [µm]',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'gdCmdPis' in args:
        generalTitle = 'Piston GD Commands'
        obsType="gdPisCmd"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''     
            
        obs = outputs.PistonGDCommand[timerange]
        obsBar = np.std(obs,axis=0)
        fig = display_module.simpleplot_tels(timestamps, obs,obsBar,generalTitle,plotTel,
                                  obsName='Commands [µm]',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'pdCmdPis' in args:
        generalTitle = 'Piston PD Commands'
        obsType="pdCmdPis"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''     
            
        obs = outputs.PistonPDCommand[timerange]
        obsBar = np.std(obs,axis=0)
        fig = display_module.simpleplot_tels(timestamps, obs,obsBar,generalTitle,plotTel,
                                  obsName='Commands [µm]',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
        
    if 'truePis' in args:
        generalTitle = 'Piston True'
        obsType="pisTrue"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''     
            
        obs = outputs.PistonTrue[timerange]
        obsBar = np.std(obs,axis=0)
        fig = display_module.simpleplot_tels(timestamps, obs,obsBar,generalTitle,plotTel,
                                  obsName='Piston [µm]',display=display,
                                  filename=filename,ext=ext,infos=infos,
                                  verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
        
    if 'gdEstMatricial' in args:
        plt.rcParams.update(rcParamsForBaselines)
        generalTitle='Group-Delays'
        obsType="gdEstMatrix"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename='' 
            
        obs = GDmic
        obsBar = np.sqrt(outputs.VarPDEst)*config.FS['R']*config.wlOfTrack/2/np.pi
        
        fig = display_module.plotMatricial(obs,generalTitle,plotBaseline,xQuantity=timestamps, 
                                     obsName='GD [µm]',obsBar=obsBar,
                                     whichFormat='standard', infos=infos,
                                     display=display,filename=filename,ext=ext,
                                     verbose=verbose)        
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'gdCmdMatricial' in args:
        generalTitle = 'GD Commands'
        obsType="GDCmdMatrix"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''     
            
        obs = outputs.GDCommand[timerange] - outputs.GDCommand[0]
        
        fig = display_module.plotMatricial(obs,generalTitle,plotBaseline,
                                           xQuantity=timestamps, obsName='GD Command\n[µm]',
                                         whichFormat='standard', infos=infos,
                                         display=display,filename=filename,ext=ext,
                                         verbose=verbose)    
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'pdEstMatricial' in args:
        plt.rcParams.update(rcParamsForBaselines)
        generalTitle='Phase-Delays'
        obsType="pdEstMatrix"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename=''
            
        obs = PDmic
        obsBar = np.sqrt(outputs.VarPDEst)*config.wlOfTrack/2/np.pi
        
        fig = display_module.plotMatricial(obs,generalTitle,plotBaseline,xQuantity=timestamps, 
                                     obsName='PD [µm]',obsBar=obsBar,
                                     whichFormat='standard', infos=infos,
                                     display=display,filename=filename,ext=ext,
                                     verbose=verbose)
        
        if mergedPdf:
            pdf.savefig(fig)
        
    if "gdHist" in args:
        plt.rcParams.update(rcParamsForBaselines)
        generalTitle='Histogram GD'
        obsType="gdHist"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename='' 
            
        obs = GDmic
        
        fig = display_module.plotMatricial(obs,generalTitle,plotBaseline,obsName='GD [µm]',
                                     whichFormat='hist',infos=infos,
                                     display=display,filename=filename,ext=ext,
                                     verbose=verbose)
        
        if mergedPdf:
            pdf.savefig(fig)
        
    if "fluxHist" in args:
        plt.rcParams.update(rcParamsForBaselines)
        generalTitle='Histogram Flux'
        obsType="fluxHist"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename='' 
            
        obs = outputs.PhotometryEstimated
        if obs.ndim==3:
            obs = np.mean(obs,axis=1)
        
        fig = display_module.plotMatricial(obs,generalTitle,plotTel,obsName='Flux [ADU]',
                                     whichFormat='hist',infos=infos,
                                     display=display,filename=filename,ext=ext,
                                     verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
             
    if 'pdPsd' in args:
        generalTitle="Phase-Delay PSD"
        obsType="pdPsd"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename='' 

        obs = pdPsd ; obs2 = pdPsdSmoothed
        xQuantity = frequencySampling ; xQuantity2 = frequencySamplingSmoothed
        
        fig = display_module.plotMatricial(obs,generalTitle,plotBaseline,xQuantity=xQuantity,
                                     obs2=obs2,xQuantity2=xQuantity2,
                                     obsName='PSD [µm²/Hz]',
                                     whichFormat='standard loglog',infos=infos,
                                     display=display,filename=filename,ext=ext,
                                     verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'gdPsd' in args:
        generalTitle="Group-Delay PSD"
        obsType="gdPsd"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename='' 

        obs = gdPsd ; obs2 = gdPsdSmoothed
        xQuantity = frequencySampling ; xQuantity2 = frequencySamplingSmoothed
        
        fig = display_module.plotMatricial(obs,generalTitle,plotBaseline,xQuantity=xQuantity,
                                     obs2=obs2,xQuantity2=xQuantity2,
                                     obsName='PSD [µm²/Hz]',
                                     whichFormat='standard loglog',infos=infos,
                                     display=display,filename=filename,ext=ext,
                                     verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'cmdPsd' in args:
        generalTitle="FT commands PSD"
        obsType="cmdPsd"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename='' 

        obs = cmdPsd ; obs2 = cmdPsdSmoothed
        xQuantity = frequencySampling ; xQuantity2 = frequencySamplingSmoothed
        
        fig = display_module.plotMatricial(obs,generalTitle,plotBaseline,xQuantity=xQuantity,
                                     obs2=obs2,xQuantity2=xQuantity2,
                                     obsName='PSD [µm²/Hz]',
                                     whichFormat='standard loglog',infos=infos,
                                     display=display,filename=filename,ext=ext,
                                     verbose=verbose)    
        if mergedPdf:
            pdf.savefig(fig)
        
    if 'pdCumStd' in args:
        generalTitle='PD Cumulative STD'
        obsType="pdCumStd"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename='' 
            
        obs = pdCumStd
        xQuantity = frequencySampling
        
        fig = display_module.plotMatricial(obs,generalTitle,plotBaseline,xQuantity=xQuantity,
                                     obsName='PD Cumulative STD [µm]',
                                     whichFormat='standard xlog',infos=infos,
                                     display=display,filename=filename,ext=ext,
                                     verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'gdCumStd' in args:
        generalTitle='GD Cumulative STD'
        obsType="gdCumStd"
        if saveIndividualFigs:
            filename= savedir+f"{filenamePrefix}_{obsType}"
        else:
            filename='' 
            
        obs = gdCumStd
        xQuantity = frequencySampling
        
        fig = display_module.plotMatricial(obs,generalTitle,plotBaseline,xQuantity=xQuantity,
                                     obsName='GD Cumulative STD [µm]',
                                     whichFormat='standard xlog',infos=infos,
                                     display=display,filename=filename,ext=ext,
                                     verbose=verbose)
        if mergedPdf:
            pdf.savefig(fig)
            
    """ GRAPH OF PERFORMANCE OF THE COPHASING (need to be updated) """
        
    if 'perfarray' in args:
        plt.rcParams.update(rcParamsForBaselines)
        
        from .tol_colors import tol_cmap as tc
        import matplotlib as mpl
        
        #visibilities, _,_,_=ct.VanCittert(wlOfScience,config.Obs,config.Target)
        #outputs.VisibilityAtPerfWL = visibilities
        visibilities = ct.NB2NIN(outputs.VisibilityObject[wlIndex])
        vismod = np.abs(visibilities) ; visangle = np.angle(visibilities)
        PhotometricBalance = config.FS['PhotometricBalance']
        
        cm = tc('rainbow_PuRd').reversed() ; Nnuances = 256
        
        # plt.rcParams['figure.figsize']=(16,12)
        # font = {'family' : 'DejaVu Sans',
        #         'weight' : 'normal',
        #         'size'   : 22}
        
        # plt.rc('font', **font)
        title="perfarray"
        fig=plt.figure(title, clear=True)
        (ax1,ax2)=fig.subplots(ncols=2, sharex=True, sharey=True)
        ax1.set_title(f"Target visibility and photometric balance ({wlOfTrack:.3}µm)")
        ax2.set_title(f"Fringe contrast and Time on central fringe ({wlOfScience:.3}µm)")
        
        for ia in range(NA):
            name1,(x1,y1) = InterfArray.TelNames[ia],InterfArray.TelCoordinates[ia,:2]
            ax1.scatter(x1,y1,color='k',linewidth=10)
            ax1.annotate(name1, (x1+6,y1+1),color="k")
            ax1.annotate(f"({ia+1})", (x1+21,y1+1),color=colors[0])
            for iap in range(ia+1,NA):
                ib=ct.posk(ia,iap,NA)
                x2,y2 = InterfArray.TelCoordinates[iap,:2]
                if PhotometricBalance[ib]>0:
                    ls = (0,(10*PhotometricBalance[ib]/np.max(PhotometricBalance),10*(1-PhotometricBalance[ib]/np.max(PhotometricBalance))))
                    im=ax1.plot([x1,x2],[y1,y2],linestyle=ls,
                            linewidth=3,
                            color=cm(int(vismod[ib]*Nnuances)))
                else:
                    im=ax1.plot([x1,x2],[y1,y2],linestyle='solid',
                            linewidth=1,
                            color=cm(int(vismod[ib]*Nnuances)))
        ax1.set_xlabel("X [m]")
        ax1.tick_params(labelleft=False)
        
        for ia in range(NA):
            name1,(x1,y1) = InterfArray.TelNames[ia],InterfArray.TelCoordinates[ia,:2]
            ax2.scatter(x1,y1,color='k',linewidth=10)
            ax2.annotate(name1, (x1+6,y1+1),color="k")
            ax2.annotate(f"({ia+1})", (x1+21,y1+1),color=colors[0])
            for iap in range(ia+1,NA):
                ib=ct.posk(ia,iap,NA)
                x2,y2 = InterfArray.TelCoordinates[iap,:2]
                ls = (0,(10*outputs.LR4[ib],np.max([0,10*(1-outputs.LR4[ib])])))
                if PhotometricBalance[ib]>0:
                    im=ax2.plot([x1,x2],[y1,y2],linestyle=ls,
                            linewidth=3,
                            color=cm(int(outputs.FringeContrast[ib]*Nnuances)))
                else:
                    im=ax2.plot([x1,x2],[y1,y2],linestyle=ls,
                            linewidth=1,
                            color=cm(int(outputs.FringeContrast[ib]*Nnuances)))
        ax2.set_xlabel("X [m]")
        ax2.set_ylabel("Y [m]")
        ax2.set_xlim([-210,160]) ; ax2.set_ylim([-50,350])
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # mpl.colorbar.ColorbarBase(cbar_ax, cmap=cm,
        #                           orientation='vertical',
        #                           label=f"Fringe Contrast at {wlOfScience:.3}µm")


        
        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.1, 0.05, 0.85, 0.05])
        mpl.colorbar.ColorbarBase(cbar_ax, cmap=cm,
                                  orientation='horizontal')

        if saveIndividualFigs:
            if verbose:
                print("Saving perfarray figure.")
            plt.savefig(savedir+f"{filenamePrefix}_perfarray.{ext}")
        
        if mergedPdf:
            pdf.savefig(fig)


    plt.rcParams.update(plt.rcParamsDefault)

        
    if 'cp' in args:
        """
        CLOSURE PHASES
        """
        
        
        linestyles=[] ; linestyles1=[] ; linestyles2=[]
        linestyles.append(mlines.Line2D([], [],linestyle='solid',
                                        label='Estimated PD'))    
        linestyles.append(mlines.Line2D([], [],linestyle=':',
                                        label='Estimated GD')) 
        linestyles.append(mlines.Line2D([], [],linestyle='--',
                                        label='Object'))
        linestyles.append(mlines.Line2D([],[], color='black',
                                        linestyle=':', label='Start tracking'))
        
        
        ymax = np.pi*1.1
        ylim = [-ymax, ymax]
        fig = plt.figure('Closure Phases')
        fig.suptitle('Closure Phases')
        (ax1,ax3), (ax2,ax4) = fig.subplots(nrows=2, ncols=2, gridspec_kw={"width_ratios":[5,1]})
        
        # Plot on ax1 the (NA-1)(NA-2)/2 independant Closure Phases
        for ia in range(1,NA):
            for iap in range(ia+1,NA):
                ic = ct.poskfai(0,ia,iap,NA)
                
                ax1.plot(timestamps, outputs.ClosurePhasePD[:,ic],
                         color=colors[ic])
                ax1.plot(timestamps, outputs.ClosurePhaseGD[:,ic],
                         color=colors[ic],linestyle=':')
                ax1.hlines(outputs.ClosurePhaseObject[wlIndex,ic], 0, t[-1], 
                           color=colors[ic], linestyle='--')
                linestyles1.append(mlines.Line2D([],[], color=colors[ic],
                                                linestyle='-', label=f'{1}{ia+1}{iap+1}'))
                
        # Plot on ax2 the (NA-1)(NA-2)/2 (independant?) other Closure Phases
        for ia in range(1,NA):
            for iap in range(ia+1,NA):
                for iapp in range(iap+1,NA):
                    ic = ct.poskfai(ia,iap,iapp,NA)
                    colorindex = int(ic - config.NC//2)
                    ax2.plot(timestamps, outputs.ClosurePhasePD[:,ic],
                             color=colors[colorindex])
                    ax2.plot(timestamps, outputs.ClosurePhaseGD[:,ic],
                             color=colors[colorindex],linestyle=':')
                    ax2.hlines(outputs.ClosurePhaseObject[wlIndex,ic], 0, t[-1],
                               color=colors[colorindex], linestyle='--')
                    linestyles2.append(mlines.Line2D([],[], color=colors[colorindex],
                                                    linestyle='-', label=f'{ia+1}{iap+1}{iapp+1}'))
        
        ax1.vlines(config.starttracking*dt,ylim[0],ylim[1],
                   color='k', linestyle=':')
        # ax2.vlines(config.starttracking*dt,ylim[0],ylim[1],
        #            color='k', linestyle=':')
        plt.xlabel('Time [s]')
        plt.ylabel('Closure Phase [rad]')
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)
        ax1.legend(handles=linestyles)
        ax3.legend(handles=linestyles1,loc='upper left') ; ax3.axis('off')
        ax4.legend(handles=linestyles2,loc='upper left') ; ax4.axis('off')
        
        if display:
            plt.show()
        config.newfig+=1
        
        if saveIndividualFigs:
            fig.savefig(savedir+f"{filenamePrefix}_cp.{ext}")
        if mergedPdf:
            pdf.savefig(fig)
            
    if 'detector' in args:
        """
        DETECTOR VIEW
        """
        title="Detector&SNR"
        fig = plt.figure(title,clear=True)
        axes = fig.subplots()
        plt.suptitle(f'Sequence of intensities on pixels corresponding to {wlOfTrack:.2f}µm')

        if config.FS['name'] == 'MIRCxFS':
            ABCDchip = False
        else:
            ABCDchip=True

        if ABCDchip:
            NIN = config.NIN
            NP = config.FS['NP']
            NMod = config.FS['NMod']
                        
            for ip in range(NP):
                
                ax = plt.subplot(NIN,NMod,ip+1)
                if ip < NMod:
                    ax.set_title(config.FS['Modulation'][ip])
                im = plt.imshow(np.transpose(np.dot(np.reshape(outputs.MacroImages[:,wlIndex,ip],[NT,1]), \
                                                    np.ones([1,100]))), vmin=np.min(outputs.MacroImages), vmax=np.max(outputs.MacroImages))    
                    
                plt.tick_params(axis='y',left='off')
                if ip//NMod == ip/NMod:
                    plt.ylabel(ich[ip//NMod])
                    
                if ip>=NP-NMod:
                    plt.xticks([0,NT],[0,NT*dt])
                    plt.xlabel('Time (ms)') 
                else:
                    plt.xticks([],[])
                plt.yticks([],[])
                    
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.3, 0.05, 0.6])
            fig.colorbar(im, cax=cbar_ax)
            plt.grid(False)
            
            
        else:
            # H,L = config.FS['Dsize']
            # posp = config.FS['posp']
            # posi_center = config.FS['posi_center']
            # PSDwindow = config.FS['PSDwindow']
            # p = config.FS['p']
            
            # realimage = np.zeros([NT,MW,L])
            # for ia in range(NA):
            #     realimage[:,posp[ia]] = outputs.MacroImages[:,:,ia]
            
            # realimage[:,:,posi_center//p-(NP-NA)//2:posi_center//p+(NP-NA)//2] = outputs.MacroImages[:,:,NA:]
            ax = plt.subplot()
            im = plt.imshow(outputs.MacroImages[:,wlIndex,:], vmin=np.min(outputs.MacroImages), vmax=np.max(outputs.MacroImages))
            
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.3, 0.05, 0.6])
            fig.colorbar(im, cax=cbar_ax)
            
            ax.set_ylabel('Time')
            ax.set_xlabel("Pixels")
            cbar_ax.set_ylabel("Pixel intensity")
            cbar_ax.yaxis.set_label_position("left")
            
            ax.grid(False)

        # linestyles=[mlines.Line2D([],[], color='black',
        #                                 linestyle='solid', label='Maximal SNR²'),
        #             mlines.Line2D([],[], color='black',
        #                                 linestyle=':', label='Start tracking')]
        # if 'ThresholdGD' in config.FT.keys():
        #     linestyles.append(mlines.Line2D([],[], color='black',
        #                                 linestyle='--', label='Squared Threshold GD'))
                    
        fig.subplots_adjust(bottom=0.3)
        snr_ax = fig.add_axes([0.18, 0.05, 0.72, 0.1])
        snr_ax.set_title("SNR & Thresholds")

        SNRnonan = np.nan_to_num(np.mean(SNRGD,axis=0))
        snr_ax.bar(baselines,SNRnonan, color=colors[6])
        snr_ax.bar(baselines,config.FT['ThresholdGD'], fill=False,edgecolor='k')

        snr_ax.hlines(config.FT['ThresholdPD'],-0.5,NIN-0.5,color='r',linestyle='-.')
        
        snr_ax.set_yscale('log') ; snr_ax.grid(True)
        #snr_ax.set_ylabel('SNR &\n Thresholds')

        snr_ax.set_xlabel('Baseline')
        ct.setaxelim(snr_ax,ydata=SNRnonan,ymin=0.5)

        if display:
            if pause:
                plt.pause(0.1)
            else:
                plt.show()
        if saveIndividualFigs:
            fig.savefig(savedir+f"{filenamePrefix}_detector.{ext}")
        if mergedPdf:
            pdf.savefig(fig)
    
    if ('state' in args):
        # ylim=[-0.1,2*config.FT['ThresholdGD']**2]
        ylim=[1e-1,np.max(SNR_pd)]
        # State-Machine and SNR
        title="State"
        fig = plt.figure(title,clear=True)
        fig.suptitle("SNR² and State-Machine")
        ax,ax2 = fig.subplots(nrows=2,ncols=1, sharex=True)
        for ib in range(NIN):
            ax.plot(timestamps, SNR_pd[:,ib],
                    label=f"{ich[ib]}")
            
        ax.hlines(config.FT['ThresholdGD']**2,0,t[-1],
                  linestyle='--',label='Threshold GD')
        
        ax2.plot(timestamps, config.FT['state'][:-1], 
                color='blue', label='State-machine')
        ax2.plot(timestamps, outputs.IgdRank, 
                color='k', label='Rank')
        
        ax.set_ylim(ylim)
        ax.set_yscale('log')
        ax.vlines(config.starttracking*dt,ylim[0],ylim[1],
                   color='k', linestyle=':')
        ax2.vlines(config.starttracking*dt,0,NA,
                   color='k', linestyle=':')
        ax.set_ylabel("SNR")
        ax2.set_ylabel("Rank of Igd")
        ax2.set_xlabel('Time (ms)')
        ax2.legend()
        ax.legend()
        if display:
            fig.show()
        if mergedPdf:
            pdf.savefig(fig)
            
    if mergedPdf:
        pdf.close()


def ShowPerformance(TimeBonds, SpectraForScience,DIT,FileInterferometer='',
                    CoherentFluxObject=[],SNR_SI=False,
                    R=140, p=10, magSI=-1,display=True, get=[],
                    verbose=False):
    """
    Processes the performance of the fringe-tracking starting at the StartingTime


    Parameters
    ----------
    TimeBonds : INT or ARRAY [ms]
        If int:
            The performance are processed from TimeBonds until the end
        If array [StartingTime,EndingTime]: 
            The performance are processed between StartingTime and EndingTime
    WavelengthOfInterest : ARRAY
        Wavelength when the Fringe Contrast needs to be calculated.
    DIT : INT
        Integration time of the science instrument [ms]
    p : INT
        Defines the maximal phase residuals RMS for conseidering a frame as exploitable.
        MaxRMSForLocked = WavelengthOfInterest/p
        MaxPhaseRMSForLocked = 2*Pi/p
    R : FLOAT
        Spectral resolution of the instrument whose performance are estimated.
        By default, R=140 (minimal spectral resolution of SPICA-VIS)
        
    Observables processed :
        - VarOPD                (Temporal Variance OPD [µm])
        - VarPDEst              (Temporal Variance PD [rad])
        - VarGDEst              Temporal Variance of GD [rad]
        - VarCPD                Temporal Variance of CPD [rad]
        - VarCGD                Temporal Variance of CGD [rad]
        - FringeContrast        Fringe Contrast [0,1] at given wavelengths

    Returns
    -------
    None.

    """
    from . import outputs
    from . import config
    
    ich = config.FS['ich']
    
    from .config import NA,NIN,NC,dt,NT
    
    NINmes = config.FS['NINmes']
    
    # if verbose:
    #     print("Entry parameters ShowPerformance function:",TimeBonds, SpectraForScience,DIT)
    """ Modify some outputs data for compatibility between simu and true data """
    
    trueTelemetries = False
    if outputs.OPDTrue.shape[0] != config.NT:
        trueTelemetries = True
        
    if trueTelemetries:
        outputs.OPDTrue = np.unwrap(outputs.PDEstimated,period=config.wlOfTrack/2)
        outputs.TrackedBaselines = np.ones([NT,NINmes])
        outputs.PistonTrue = np.zeros([NT,NA])
        
    """
    LOAD COHERENT FLUX IN SPECTRAL BAND FOR SNR COMPUTATION
    So far, I assume it is SPICA-VIS
    """
        
    if magSI<0:
        if 'SImag' not in config.Target.Star1.keys():
            config.Target.Star1['SImag'] = config.Target.Star1['Hmag']
    else:
        config.Target.Star1['SImag'] = magSI
        
    MeanWavelength = np.mean(SpectraForScience)
    if hasattr(SpectraForScience, "__len__"):
        MW=len(SpectraForScience)
        MultiWavelength=True
    else:
        MultiWavelength=False
        MW=1
        SpectraForScience = np.array([MeanWavelength])

    Lc = R*SpectraForScience      # Vector or float
    
    
    DIT_NumberOfFrames = int(DIT/dt)
    if verbose:
        print("deuxième:",dt,DIT,DIT/dt,DIT_NumberOfFrames)
    if TimeBonds[1]==-1:
        TimeBonds = (TimeBonds[0],outputs.timestamps[-1])
    if isinstance(TimeBonds,(float,int)):
        Period = int(NT - TimeBonds/dt)
        InFrame = round(TimeBonds/dt)
    elif isinstance(TimeBonds,(np.ndarray,list,tuple)):
        Period = int((TimeBonds[1]-TimeBonds[0])/dt)
        InFrame = round(TimeBonds[0]/dt)
    else:
        raise Exception('"TimeBonds" must be instance of (float,int,np.ndarray,list)')
       
    periodSeconds = Period*dt
    if not len(FileInterferometer):
        FileInterferometer = "data/interferometers/CHARAinterferometerR.fits"
      
    if SNR_SI:
        if not len(CoherentFluxObject):
            # The interferometer is "not the same" as for simulation, because not the same band.
            # In the future, both bands could be integrated in the same Interf class object.
            InterfArraySI = config.Interferometer(name=FileInterferometer)
            
            if MultiWavelength:
                CoherentFluxObject = ct.create_CfObj(SpectraForScience,
                                                     config.Obs,config.Target,InterfArraySI)
            else:
                CoherentFluxObject = ct.create_CfObj(MeanWavelength,
                                                     config.Obs,config.Target,InterfArraySI,R=R)
            CoherentFluxObject = CoherentFluxObject*dt  # [MW,:] whether it is multiWL or not
        
        
        from cophasim.SCIENTIFIC_INSTRUMENTS import SPICAVIS
        outputs.IntegrationTime, outputs.VarSquaredVis, outputs.SNR_E, outputs.SNR_E_perSC = SPICAVIS(CoherentFluxObject,outputs.OPDTrue[InFrame:],SpectraForScience,DIT=DIT)
        
   
    if MultiWavelength:
        outputs.FringeContrast=np.zeros([MW,NIN])  # Fringe Contrast at given wavelengths [0,1]
    else:
        outputs.FringeContrast=np.zeros(NIN)       # Fringe Contrast at given wavelengths [0,1]

    outputs.VarOPD=np.zeros(NIN)
    outputs.VarGDRes=np.zeros(NINmes)
    outputs.VarPDRes=np.zeros(NINmes)
    outputs.VarGDEst=np.zeros(NINmes)
    outputs.VarPDEst=np.zeros(NINmes)
    outputs.VarDispersion = np.zeros(NINmes)        # in microns
    outputs.VarPiston=np.zeros(NA)
    outputs.VarPistonGD=np.zeros(NA)
    outputs.VarPistonPD=np.zeros(NA)

    outputs.VarCPD =np.zeros(NC); outputs.VarCGD=np.zeros(NC)
    
    outputs.LockedRatio=np.zeros(NIN)          # sig_opd < lambda/p
    outputs.LR2 = np.zeros(NINmes)             # Mode TRACK
    outputs.LR3= np.zeros(NIN)                 # In Central Fringe
    outputs.LR4= np.zeros(NIN)                 # No fringe jump
    outputs.WLockedRatio = np.zeros(NIN)
    
    MaxPhaseVarForLocked = (2*np.pi/p)**2
    MaxVarOPDForLocked = (MeanWavelength/p)**2
    
    Ndit = Period//DIT_NumberOfFrames
    outputs.PhaseVar_atWOI = np.zeros([Ndit,NIN])
    outputs.PhaseStableEnough= np.zeros([Ndit,NIN])
    outputs.LR2 = np.mean(outputs.TrackedBaselines[InFrame:], axis=0)   # Array [NINmes]
    outputs.InCentralFringe = np.abs(outputs.OPDTrue-outputs.OPDrefObject) < MeanWavelength/2
    outputs.LR3 = np.mean(outputs.InCentralFringe[InFrame:], axis=0)    # Array [NIN]
    
    for it in range(Ndit):

        OutFrame=InFrame+DIT_NumberOfFrames
        timerange = range(InFrame,OutFrame)
        OPDVar = np.var(outputs.OPDTrue[timerange,:],axis=0)
        OPDptp = np.ptp(outputs.OPDTrue[timerange,:],axis=0)
        
        GDResVar = np.var(outputs.GDResidual2[timerange,:],axis=0)
        PDResVar = np.var(outputs.PDResidual2[timerange,:],axis=0)
        GDEstVar = np.var(outputs.GDEstimated[timerange,:],axis=0)
        PDEstVar = np.var(outputs.PDEstimated[timerange,:],axis=0)
        DispersionVar = np.var(outputs.GDEstimated[timerange,:]*config.FS['R']*config.wlOfTrack/2/np.pi-outputs.PDEstimated[timerange,:]*config.wlOfTrack/2/np.pi,axis=0)
        PistonVar = np.var(outputs.PistonTrue[timerange,:],axis=0)
        PistonVarGD = np.var(outputs.GDPistonResidual[timerange,:],axis=0)
        PistonVarPD = np.var(outputs.PDPistonResidual[timerange,:],axis=0)
        outputs.PhaseVar_atWOI[it] = np.var(2*np.pi*outputs.OPDTrue[timerange,:]/MeanWavelength,axis=0)
        outputs.PhaseStableEnough[it] = 1*(OPDVar < MaxVarOPDForLocked)
        NoFringeJumpDuringPose = 1*(OPDptp < 1.5*MeanWavelength)
        
        outputs.VarOPD += 1/Ndit*OPDVar
        outputs.LR4 += 1/Ndit*NoFringeJumpDuringPose
        
        outputs.VarPDRes += 1/Ndit*PDResVar
        outputs.VarGDRes += 1/Ndit*GDResVar
        outputs.VarPDEst += 1/Ndit*PDEstVar
        outputs.VarGDEst += 1/Ndit*GDEstVar
        outputs.VarDispersion += 1/Ndit*DispersionVar
        outputs.VarPiston += 1/Ndit*PistonVar
        outputs.VarPistonGD += 1/Ndit*PistonVarGD
        outputs.VarPistonPD += 1/Ndit*PistonVarPD
        if outputs.ClosurePhasePD.shape[-1] == config.NC:
            outputs.VarCPD += 1/Ndit*np.var(outputs.ClosurePhasePD[timerange,:],axis=0)
            outputs.VarCGD += 1/Ndit*np.var(outputs.ClosurePhaseGD[timerange,:],axis=0)
        else:
            outputs.VarCPD = np.zeros([NT,NC])
            outputs.VarCGD = np.zeros([NT,NC])
        
        # Fringe contrast
        if MultiWavelength:
            for iw in range(MW):
                wl = SpectraForScience[iw]
                for ib in range(NIN):
                    CoherenceEnvelopModulation = np.sinc(outputs.OPDTrue[timerange,ib]/Lc[iw])
                    Phasors = np.exp(1j*2*np.pi*outputs.OPDTrue[timerange,ib]/wl)
                    outputs.FringeContrast[iw,ib] += 1/Ndit*np.abs(np.mean(Phasors*CoherenceEnvelopModulation,axis=0))
        else:
            for ib in range(NIN):
                CoherenceEnvelopModulation = np.sinc(outputs.OPDTrue[timerange,ib]/Lc)
                Phasors = np.exp(1j*2*np.pi*outputs.OPDTrue[timerange,ib]/MeanWavelength)
                outputs.FringeContrast[ib] += 1/Ndit*np.abs(np.mean(Phasors*CoherenceEnvelopModulation,axis=0))

        InFrame += DIT_NumberOfFrames
        
    outputs.LockedRatio = np.mean(outputs.PhaseStableEnough,axis=0)
    outputs.WLockedRatio = np.mean(outputs.PhaseStableEnough*outputs.FringeContrast, axis=0)
    
    outputs.autreWlockedRatio = np.mean((MaxPhaseVarForLocked-outputs.PhaseVar_atWOI)/MaxPhaseVarForLocked, axis=0)

    outputs.WLR2 = np.mean(outputs.TrackedBaselines * outputs.SquaredSNRMovingAveragePD, axis=0)
    
    gdCmdDiff = outputs.GDCommand[1:] - outputs.GDCommand[:-1]
    pdCmdDiff = outputs.PDCommand[1:] - outputs.PDCommand[:-1]
    gdCmdDiff[-1] = 0 ; pdCmdDiff[-1] = 0
    
    nJumps = np.sum(np.abs(gdCmdDiff),axis=0)
    periodSecondsArr = np.ones_like(nJumps)*periodSeconds
    outputs.fringeJumpsPeriod = np.divide(periodSecondsArr,nJumps,out=periodSecondsArr,where=nJumps!=0)
    # outputs.fringeJumpsPeriod[outputs.fringeJumpsPeriod==np.inf] = periodSeconds
    
    gdCmdDiff = outputs.PistonGDCommand[1:] - outputs.PistonGDCommand[:-1]
    gdCmdDiff[-1] = 0
    
    nJumps = np.sum(np.abs(gdCmdDiff),axis=0)
    periodSecondsArr = np.ones_like(nJumps)*periodSeconds
    outputs.fringeJumpsPeriodTel = np.divide(periodSecondsArr,nJumps,out=periodSecondsArr,where=nJumps!=0)
    # outputs.fringeJumpsPeriodTel[outputs.fringeJumpsPeriodTel==np.inf] = periodSeconds
    
    
    # if 'ThresholdGD' in config.FT.keys():
    #     outputs.WLR3 = np.mean(outputs.TrackedBaselines * outputs.SquaredSNRMovingAveragePD, axis=0)

    if not display:
        return

    observable = outputs.VarOPD
    xrange = np.arange(NIN)
    
    plt.figure(f'Variance OPD @{round(config.wlOfTrack,2)}µm')    
    plt.ylim([np.min(observable),np.max(observable)])
    plt.scatter(np.arange(NIN),observable)
    plt.hlines(np.mean(observable), xrange[0],xrange[-1],linestyle='--')
    plt.xticks(ticks=np.arange(NIN),labels=ich, rotation='vertical')    
    plt.xlabel('Baseline')
    plt.ylabel('Variance [µm]')
    plt.grid()
    plt.show()
    config.newfig += 1
    
    observable = outputs.VarPDEst*(config.wlOfTrack/2/np.pi)
    
    plt.figure(f'Variance Estimated PD @{round(config.wlOfTrack,2)}µm')    
    plt.ylim([np.min(observable),np.max(observable)])
    plt.scatter(np.arange(NINmes),observable)
    plt.hlines(np.mean(observable), xrange[0],xrange[-1],linestyle='--')
    plt.xticks(ticks=np.arange(NINmes),labels=ich, rotation='vertical')    
    plt.xlabel('Baseline')
    plt.ylabel('Variance [µm]')
    plt.grid()
    plt.show()
    config.newfig += 1
    
    
    observable = outputs.VarGDEst*(config.wlOfTrack/2/np.pi)*config.FS['R']
    
    plt.figure(f'Variance Estimated GD @{round(config.wlOfTrack,2)}µm')    
    plt.ylim([np.min(observable),np.max(observable)])
    plt.scatter(np.arange(NINmes),observable)
    plt.hlines(np.mean(observable), xrange[0],xrange[-1],linestyle='--')
    plt.xticks(ticks=np.arange(NINmes),labels=ich, rotation='vertical')    
    plt.xlabel('Baseline')
    plt.ylabel('Variance [µm]')
    plt.grid()
    plt.show()
    config.newfig += 1
    
    
    observable = outputs.VarCPD
    xrange = np.arange(config.NC)
    
    plt.figure('Variance CPD')    
    plt.ylim([np.min(observable),np.max(observable)])
    plt.scatter(xrange,observable)
    plt.hlines(np.mean(observable), xrange[0],xrange[-1],linestyle='--')
    plt.xticks(ticks=xrange,labels=config.CPindex, rotation='vertical')    
    plt.xlabel('Triangle')
    plt.ylabel('Variance [rad]')
    plt.grid()
    plt.show()
    config.newfig += 1


    observable = outputs.VarCGD
    
    plt.figure('Variance CGD')    
    plt.ylim([np.min(observable),np.max(observable)])
    plt.scatter(xrange,observable)
    plt.hlines(np.mean(observable), xrange[0],xrange[-1],linestyle='--')
    plt.xticks(ticks=xrange,labels=config.CPindex, rotation='vertical')    
    plt.xlabel('Triangle')
    plt.ylabel('Variance [rad]')
    plt.grid()
    plt.show()
    config.newfig += 1

    
    plt.figure('Integrated Visibility')    
    plt.ylim([0.9*np.min(outputs.FringeContrast),1.1])
    for ib in range(NIN):
        plt.scatter(SpectraForScience,outputs.FringeContrast[:,ib], label=f'{ich[ib]}')
        
    plt.legend(), plt.grid()
    plt.show()
    config.newfig += 1
    
    return


'''
def ShowPerformance_multiDITs(TimeBonds,SpectraForScience,IntegrationTimes=[],
                              CoherentFluxObject=[],FileInterferometer='',
                              R=140, p=10, magSI=-1,display=True, get=[],criterias='light',
                              verbose=False, onlySNR=False,check_DITs=False, OnlyCheckDIT=False):
    """
    Process the performance of the fringe-tracking starting at the StartingTime.

    Observables processed:
        -VarOPD                 # Temporal Variance OPD [µm]
        -VarPDEst              # Temporal Variance PD [rad]
        -VarGDEst              # Temporal Variance of GD [rad]
        -VarCPD                 # Temporal Variance of CPD [rad]
        -VarCGD                 # Temporal Variance of CGD [rad]
        -FringeContrast         # Fringe Contrast [0,1] at given wavelengths
        
    Parameters
    ----------
    TimeBonds : INT or ARRAY [ms]
        If int:
            The performance are processed from TimeBonds until the end
        If array [StartingTime,EndingTime]: 
            The performance are processed between StartingTime and EndingTime
    MeanWavelength : ARRAY
        Wavelength when the Fringe Contrast needs to be calculated.
    DIT : INT
        Integration time of the science instrument [ms]
    p : INT
        Defines the maximal phase residuals RMS for conseidering a frame as exploitable.
        MaxRMSForLocked = MeanWavelength/p
        MaxPhaseRMSForLocked = 2*Pi/p
    R : FLOAT
        Spectral resolution of the instrument whose performance are estimated.
        By default, R=140 (minimal spectral resolution of SPICA-VIS)
    Returns
    -------
    None.

    """
    from . import outputs
    from . import config
    
    ich = config.FS['ich']
    
    from .config import NA,NIN,NC,dt,NT
    
    NINmes = config.FS['NINmes']
    
    
    """
    LOAD COHERENT FLUX IN SPECTRAL BAND FOR SNR COMPUTATION
    So far, I assume it is SPICA-VIS
    """
    
    MeanWavelength = np.mean(SpectraForScience)
    if hasattr(SpectraForScience, "__len__"):
        MW=len(SpectraForScience)
        MultiWavelength=True
    else:
        MultiWavelength=False
        MW=1
        SpectraForScience = np.array([MeanWavelength])

    Lc = R*SpectraForScience      # Vector or float

    if magSI<0:
        if 'SImag' not in config.Target.Star1.keys():
            config.Target.Star1['SImag'] = config.Target.Star1['Hmag']
    else:
        config.Target.Star1['SImag'] = magSI
        
    if not len(FileInterferometer):
        FileInterferometer = "data/interferometers/CHARAinterferometerR.fits"
    
    
    if isinstance(TimeBonds,(float,int)):
        Period = int(NT - TimeBonds/dt)
        InFrame = round(TimeBonds/dt)
    elif isinstance(TimeBonds,(np.ndarray,list)):
        Period = int((TimeBonds[1]-TimeBonds[0])/dt)
        InFrame = round(TimeBonds[0]/dt)
    else:
        raise Exception('"TimeBonds" must be instance of (float,int,np.ndarray,list)')

    if check_DITs:
        DITf=IntegrationTimes/dt
        #IntegrationTimes = IntegrationTimes//dt * dt
        Ndit=len(DITf)
        
        ObservingTime = Period*dt
        
        NewDITf=[]#DITf.copy()
        kbefore=[]
        for idit in range(Ndit):
            k = Period//DITf[idit]
            if k in kbefore: 
                k=np.min(kbefore)-1  # Avoid that two DITs be the same
            if k == -1:
                break # Stop if we reached the unique frame.
            r = Period%DITf[idit]
            if r > 0.05*DITf[idit]:
                NewDITf.append(Period//(k+1))
            kbefore.append(k)
            
        NewDITf = np.array(NewDITf)
        NewIntegrationTimes = NewDITf*dt
        ListNframes = Period//NewDITf
        ThrownFrames = Period%NewDITf
        LengthOfKeptSequence = ListNframes * Period
        
        if verbose:
            print(f"ObservingTimes:{ObservingTime}")
            print(f"Proposed DITs:{IntegrationTimes}")
            print(f"ListNframes :{ListNframes}")
            print(f"ThrownFrames :{ThrownFrames}")
            print(f"New DITs:{NewIntegrationTimes}")
            print(f"Percentage of loss: {np.round(ThrownFrames/LengthOfKeptSequence*100,2)}")
    
        
    else:
        NewIntegrationTimes = IntegrationTimes
        
    outputs.DITsForPerformance = NewIntegrationTimes
    Ndit = len(NewIntegrationTimes)
    
    if OnlyCheckDIT:
        return NewIntegrationTimes
    
    if not len(CoherentFluxObject):
        # The interferometer is "not the same" as for simulation, because not the same band.
        # In the future, both bands could be integrated in the same Interf class object.
        InterfArray = ct.get_array(name=FileInterferometer)
        
        if MultiWavelength:
            CoherentFluxObject = ct.create_CfObj(SpectraForScience,
                                                 config.Obs,config.Target,InterfArray)
        else:
            CoherentFluxObject = ct.create_CfObj(MeanWavelength,
                                                 config.Obs,config.Target,InterfArray,R=R)
        CoherentFluxObject = CoherentFluxObject*dt  # [MW,:] whether it is multiWL or not
    
    from cophasim.SCIENTIFIC_INSTRUMENTS import SPICAVIS
   

    outputs.VarSquaredVis=np.zeros([Ndit,MW,NIN])*np.nan
    outputs.SNR_E=np.zeros([Ndit,NIN])*np.nan
    outputs.SNR_E_perSC=np.zeros([Ndit,MW,NIN])*np.nan
    outputs.VarOPD=np.zeros([Ndit,NIN])
    outputs.LockedRatio=np.zeros([Ndit,NIN])    # sig_opd < lambda/p
    outputs.LR2 = np.zeros([Ndit,NINmes])          # Mode TRACK
    outputs.LR3= np.zeros([Ndit,NIN])           # In Central Fringe
    outputs.LR4 = np.zeros([Ndit,NIN])          # No Fringe Jump
    outputs.FringeContrast=np.zeros([Ndit,MW,NIN])  # Fringe Contrast at given wavelengths [0,1]
    
    outputs.VarPiston=np.zeros([Ndit,NA])
    outputs.VarPistonGD=np.zeros([Ndit,NA])
    outputs.VarPistonPD=np.zeros([Ndit,NA])
    
    if criterias!='light': # additional informations

        outputs.EnergyPicFrange=np.zeros([Ndit,MW,NIN])*np.nan
        outputs.PhotonNoise=np.zeros([Ndit,MW,NIN])*np.nan
        outputs.ReadNoise=np.zeros([Ndit,NIN])*np.nan
        outputs.CoupledTerms=np.zeros([Ndit,MW,NIN])*np.nan
        outputs.VarCf=np.zeros([Ndit,MW,NIN])*np.nan
    
        outputs.VarGDRes=np.zeros([Ndit,NINmes])
        outputs.VarPDRes=np.zeros([Ndit,NINmes])
    
        outputs.VarPDEst=np.zeros([Ndit,NINmes])
        outputs.VarGDEst=np.zeros([Ndit,NINmes])
        
        outputs.WLockedRatio = np.zeros([Ndit,NIN])
        outputs.WLR2 = np.zeros([Ndit,NINmes])
    
        outputs.VarCPD =np.zeros([Ndit,NC])
        outputs.VarCGD = np.zeros([Ndit,NC])
    
    outputs.IntegrationTime=NewIntegrationTimes

    MaxPhaseVarForLocked = (2*np.pi/p)**2
    MaxVarOPDForLocked = (MeanWavelength/p)**2
    
    InCentralFringe = np.abs(outputs.OPDTrue-outputs.OPDrefObject) < MeanWavelength/2
    if 'ThresholdGD' in config.FT.keys():
        TrackedBaselines = (outputs.SquaredSNRMovingAverage >= config.FT['ThresholdGD']**2) #Array [NT,NIN]
        
    FirstFrame = InFrame
    for idit in range(Ndit):
        
        DIT=NewIntegrationTimes[idit]
        
        # if config.NA ==6:
        #     # Calculation of SNR
        #     _, outputs.VarSquaredVis[idit], outputs.SNR_E[idit], outputs.SNR_E_perSC[idit] = SPICAVIS(CoherentFluxObject,outputs.OPDTrue[FirstFrame:], SpectraForScience,DIT=DIT)
        
        # if criterias!='light':
        #     outputs.EnergyPicFrange[idit] = outputs.SNRnum
        #     outputs.PhotonNoise[idit]= outputs.PhNoise
        #     outputs.ReadNoise[idit]=outputs.RNoise
        #     outputs.CoupledTerms[idit]=outputs.CTerms
        #     outputs.VarCf[idit]=outputs.var_cf
        
        if onlySNR:
            continue
        
        DIT_NumberOfFrames = int(DIT/dt)
        Nframes = Period//DIT_NumberOfFrames
        
        PhaseVar_atWOI = np.zeros([NIN])
        PhaseStableEnough= np.zeros([NIN])
        
        if 'ThresholdGD' in config.FT.keys():
            outputs.LR2[idit] = np.mean(TrackedBaselines[InFrame:], axis=0)   # Array [NIN]
        
        outputs.LR3[idit] = np.mean(InCentralFringe[InFrame:], axis=0)    # Array [NIN]
        
        InFrame = FirstFrame
        for iframe in range(Nframes):
            OutFrame=InFrame+DIT_NumberOfFrames
            
            OPDVar = np.var(outputs.OPDTrue[InFrame:OutFrame,:],axis=0)
            OPDptp = np.ptp(outputs.OPDTrue[InFrame:OutFrame,:],axis=0)
            
            PhaseVar_atWOI = np.var(2*np.pi*outputs.OPDTrue[InFrame:OutFrame,:]/MeanWavelength,axis=0)
            PhaseStableEnough = 1*(OPDVar < MaxVarOPDForLocked)
            NoFringeJumpDuringPose = 1*(OPDptp < 1.5*MeanWavelength)
            
            outputs.LockedRatio[idit] += 1/Nframes*np.mean(PhaseStableEnough,axis=0)
            outputs.LR4[idit] += 1/Nframes*np.mean(NoFringeJumpDuringPose,axis=0)
            outputs.VarOPD[idit] += 1/Nframes*OPDVar


            # Telescopes
            PistonVar = np.var(outputs.PistonTrue[InFrame:OutFrame,:],axis=0)
            PistonVarGD = np.var(outputs.GDPistonResidual[InFrame:OutFrame,:],axis=0)
            PistonVarPD = np.var(outputs.PDPistonResidual[InFrame:OutFrame,:],axis=0)
            outputs.VarPiston[idit] += 1/Nframes*PistonVar
            outputs.VarPistonGD[idit] += 1/Nframes*PistonVarGD
            outputs.VarPistonPD[idit] += 1/Nframes*PistonVarPD
            
            # Fringe contrast
            for iw in range(MW):
                wl = SpectraForScience[iw]
                for ib in range(NIN):
                    CoherenceEnvelopModulation = np.sinc(outputs.OPDTrue[InFrame:OutFrame,ib]/Lc[iw])
                    Phasors = np.exp(1j*2*np.pi*outputs.OPDTrue[InFrame:OutFrame,ib]/wl)
                    outputs.FringeContrast[idit,iw,ib] += 1/Nframes*np.abs(np.mean(Phasors*CoherenceEnvelopModulation,axis=0))


            if criterias!='light':
                GDResVar = np.var(outputs.GDResidual2[InFrame:OutFrame,:],axis=0)
                PDResVar = np.var(outputs.PDResidual2[InFrame:OutFrame,:],axis=0)
                outputs.VarPDRes[idit] += 1/Nframes*PDResVar
                outputs.VarGDRes[idit] += 1/Nframes*GDResVar

                outputs.VarPDEst[idit] += 1/Nframes*np.var(outputs.PDEstimated2[InFrame:OutFrame,:],axis=0)
                outputs.VarGDEst[idit] += 1/Nframes*np.var(outputs.GDEstimated2[InFrame:OutFrame,:],axis=0)
                outputs.VarCPD[idit] += 1/Nframes*np.var(outputs.ClosurePhasePD[InFrame:OutFrame,:],axis=0)
                outputs.VarCGD[idit] += 1/Nframes*np.var(outputs.ClosurePhaseGD[InFrame:OutFrame,:],axis=0)
                
                outputs.WLockedRatio[idit] += 1/Nframes*np.mean(PhaseStableEnough*outputs.FringeContrast[idit], axis=0)
    
            InFrame += DIT_NumberOfFrames
            
        # Don't depend on DIT but better for same treatment after.
        if criterias!='light':
            outputs.WLR2[idit] = np.mean(TrackedBaselines * outputs.SquaredSNRMovingAverage, axis=0)
        
        
    if onlySNR:
        return NewIntegrationTimes
    
    
    if display:

        observable = outputs.VarOPD
        xrange = np.arange(NIN)
        
        plt.figure(f'Variance OPD @{round(config.wlOfTrack,2)}µm')    
        plt.ylim([np.min(observable),np.max(observable)])
        plt.scatter(np.arange(NIN),observable)
        plt.hlines(np.mean(observable), xrange[0],xrange[-1],linestyle='--')
        plt.xticks(ticks=np.arange(NIN),labels=ich, rotation='vertical')    
        plt.xlabel('Baseline')
        plt.ylabel('Variance [µm]')
        plt.grid()
        plt.show()
        config.newfig += 1
        
        observable = outputs.VarPDEst*(config.wlOfTrack/2/np.pi)
        
        plt.figure(f'Variance Estimated PD @{round(config.wlOfTrack,2)}µm')    
        plt.ylim([np.min(observable),np.max(observable)])
        plt.scatter(np.arange(NIN),observable)
        plt.hlines(np.mean(observable), xrange[0],xrange[-1],linestyle='--')
        plt.xticks(ticks=np.arange(NIN),labels=ich, rotation='vertical')    
        plt.xlabel('Baseline')
        plt.ylabel('Variance [µm]')
        plt.grid()
        plt.show()
        config.newfig += 1
        
        
        observable = outputs.VarGDEst*(config.wlOfTrack/2/np.pi)*config.FS['R']
        
        plt.figure(f'Variance Estimated GD @{round(config.wlOfTrack,2)}µm')    
        plt.ylim([np.min(observable),np.max(observable)])
        plt.scatter(np.arange(NIN),observable)
        plt.hlines(np.mean(observable), xrange[0],xrange[-1],linestyle='--')
        plt.xticks(ticks=np.arange(NIN),labels=ich, rotation='vertical')    
        plt.xlabel('Baseline')
        plt.ylabel('Variance [µm]')
        plt.grid()
        plt.show()
        config.newfig += 1
        
        
        observable = outputs.VarCPD
        xrange = np.arange(config.NC)
        
        plt.figure('Variance CPD')    
        plt.ylim([np.min(observable),np.max(observable)])
        plt.scatter(xrange,observable)
        plt.hlines(np.mean(observable), xrange[0],xrange[-1],linestyle='--')
        plt.xticks(ticks=xrange,labels=config.CPindex, rotation='vertical')    
        plt.xlabel('Triangle')
        plt.ylabel('Variance [rad]')
        plt.grid()
        plt.show()
        config.newfig += 1
    
    
        observable = outputs.VarCGD
        
        plt.figure('Variance CGD')    
        plt.ylim([np.min(observable),np.max(observable)])
        plt.scatter(xrange,observable)
        plt.hlines(np.mean(observable), xrange[0],xrange[-1],linestyle='--')
        plt.xticks(ticks=xrange,labels=config.CPindex, rotation='vertical')    
        plt.xlabel('Triangle')
        plt.ylabel('Variance [rad]')
        plt.grid()
        plt.show()
        config.newfig += 1
    
        
        plt.figure('Integrated Visibility')    
        plt.ylim([0.9*np.min(outputs.FringeContrast),1.1])
        for ib in range(NIN):
            plt.scatter(MeanWavelength,outputs.FringeContrast[:,ib], label=f'{ich[ib]}')
            
        plt.legend(), plt.grid()
        plt.show()
        config.newfig += 1
    
    return NewIntegrationTimes
'''


def BodeDiagrams(Input,Output,Command,timestamps,
                  fbonds=[], details='', window='hanning',
                  display=True, figsave=False, figdir='',ext='pdf'):
    """
    Compute the Bode Diagrams of a close loop.
    
    Parameters
    ----------
    Input : ARRAY
        Input of the close loop.
    Output : ARRAT
        Residues after close loop.
    Command : ARRAY
        Commands of the close loop.
    timestamps : ARRAY
        Time (in seconds) sampling associated with entries.
    fbonds : LIST, optional
        Min and max frequency to display. The default is [].
    details : STRING, optional
        String which appears in the figure name, to define it. The default is ''.
    window : STRING, optional
        Filter window to apply before FFT. The default is 'hanning' to avoid
        aliasing. It's the only managed filter. If empty string, no filter used.
    display : BOOLEAN, optional
        Display or not the figures. The default is True.
    figsave : BOOLEAN, optional
        Save or not the figures. The default is False.
    figdir : STRING, optional
        Directory where saving the figures. The default is ''.
    ext : STRING, optional
        File extension. The default is 'pdf'.

    Returns
    -------
    frequencySampling : ARRAY
        Frequencies associated with the transfert function arrays.
    FTrej : ARRAY
        Rejection Tranfer Function.
    FTBO : ARRAY
        Open Loop Transfer Function.
    FTBF : ARRAY
        Close Loop Transfer Function.

    """
     
    nNT = len(timestamps) ; dt = np.mean(timestamps[1:]-timestamps[:-1])

    frequencySampling1 = np.fft.fftfreq(nNT, dt)
    if len(fbonds):
        fmin, fmax = fbonds
    else:
        fmin=0
        fmax=np.max(frequencySampling1)
    
    PresentFrequencies = (frequencySampling1 > fmin) \
        & (frequencySampling1 < fmax)
        
    frequencySampling = frequencySampling1[PresentFrequencies]
    
    if window =='hanning':
        windowsequence = np.hanning(nNT)
    else:
        windowsequence = np.ones(nNT)
        
    Output = Output*windowsequence
    Input = Input*windowsequence
    Command = Command*windowsequence
    
    FTResidues = np.fft.fft(Output)[PresentFrequencies]
    FTTurb = np.fft.fft(Input)[PresentFrequencies]
    FTCommand = np.fft.fft(Command)[PresentFrequencies]
    
    FTrej = FTResidues/FTTurb
    FTBO = FTCommand/FTResidues
    FTBF = FTCommand/FTTurb

    if display:
        plt.rcParams.update(rcParamsForBaselines)
        title = f'{details} - Transfer Functions'
        fig = plt.figure(title, clear=True)
        fig.suptitle(title)
        ax1,ax2,ax3 = fig.subplots(nrows=3,sharex=True)
        
        ax1.plot(frequencySampling, np.abs(FTrej),color='k')           

        # plt.plot(frequencySampling, frequencySampling*10**(-2), linestyle='--')
        ax1.set_yscale('log') #; ax1.set_ylim(1e-3,5)
        ax1.set_ylabel('FTrej')
        
        ax2.plot(frequencySampling, np.abs(FTBO),color='k')
        ax2.set_yscale('log') #; ax2.set_ylim(1e-3,5)
        ax2.set_ylabel("FTBO")
        
        ax3.plot(frequencySampling, np.abs(FTBF),color='k')
    
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

        fig.show()
        
        fig = plt.figure(f'{details} - Temporal sampling used', clear=True)
        ax1,ax2,ax3 = fig.subplots(nrows=3,sharex=True)
        
        ax1.plot(timestamps, Input,'k')
        # plt.plot(frequencySampling, frequencySampling*10**(-2), linestyle='--')
    
        ax1.set_ylabel('Open loop')
        
        ax2.plot(timestamps, Output,'k')
        ax2.set_ylabel("Close loop")
        
        ax3.plot(timestamps,Command,'k')
    
        ax3.set_xlabel('Timestamps [s]')
        ax3.set_ylabel('Command')
        
        fig.show()
        
    return frequencySampling, FTrej, FTBO, FTBF



def SpectralAnalysis(OPD = (1,2),TimeBonds=0, details='', window='hanning',
                      figsave=False, figdir='',ext='pdf'):
    """
    Plot the three Transfer Function of the servo loop controlling the OPD between
    the telescopes OPD[0] and OPD[1]

    Parameters
    ----------
    OPD : TYPE, optional
        DESCRIPTION. The default is (1,2).

    Returns
    -------
    None.

    """
    from . import outputs
    from .config import NA, NT, dt
    
    tel1 = OPD[0]-1
    tel2 = OPD[1]-1
    
    ib = ct.posk(tel1, tel2, NA)
    
    if isinstance(TimeBonds,(float,int)):
        BeginSample = round(TimeBonds/dt)
        EndSample = NT
    elif isinstance(TimeBonds,(np.ndarray,list)):
        BeginSample = round(TimeBonds[0]/dt)
        EndSample = round(TimeBonds[1]/dt)
    else:
        raise Exception('"TimeBonds" must be instance of (float,int,np.ndarray,list)')
    
    SampleIndices = range(BeginSample,EndSample) ; nNT = len(SampleIndices)
    
    frequencySampling = np.fft.fftfreq(nNT, dt)
    PresentFrequencies = (frequencySampling >= 0) & (frequencySampling < 200)
    frequencySampling = frequencySampling[PresentFrequencies]
    
    Residues = outputs.OPDTrue[SampleIndices,ib]
    Turb = outputs.OPDDisturbance[SampleIndices,ib]
    Command = outputs.OPDCommand[SampleIndices,ib]
    
    if window =='hanning':
        windowsequence = np.hanning(nNT)
    else:
        windowsequence = np.ones(nNT)
        
    Residues = Residues*windowsequence
    Turb = Turb*windowsequence
    Command = Command*windowsequence
    
    FTResidues = np.fft.fft(Residues)[PresentFrequencies]
    FTTurb = np.fft.fft(Turb)[PresentFrequencies]
    FTCommand = np.fft.fft(Command)[PresentFrequencies]
    
    FTrej = FTResidues/FTTurb
    FTBO = FTCommand/FTResidues
    FTBF = FTCommand/FTTurb

    fig = plt.figure('Rejection Transfer Function')
    ax1,ax2,ax3 = fig.subplots(nrows=3,sharex=True)
    
    ax1.plot(frequencySampling, 20*np.log10(np.abs(FTrej)))
    ax1.set_ylabel('FTrej\nGain [dB]')
    
    ax2.plot(frequencySampling, 20*np.log10(np.abs(FTBO)))
    ax2.set_ylabel("FTBO\nGain [dB]")
    
    ax3.plot(frequencySampling, 20*np.log10(np.abs(FTBF)))

    ax3.set_xlabel('Frequencies [Hz]')
    ax3.set_ylabel('FTBF\nGain [dB]')
    ax3.set_xscale('log')


    if figsave:
        prefix = details.replace("=","").replace(";","").replace(" ","").replace(".","").replace('\n','_').replace('Phase-delay','PD').replace('Group-delay','GD')
        figname = "BodeDiagrams"
        if isinstance(ext,list):
            for extension in ext:
                plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
        else:
            plt.savefig(figdir+f"{prefix}_{figname}.{ext}")

    pass



def FSdefault(NA, NW):
    """
    NEEDS TO BE UPDATE FOR BEING USED
    
    Simplest Fringe Sensor to be run with the COPHASIM
    This procedure implements the P2V matrix of the simplest fringe sensor for the coh library.
    The principles are that:
        - the raw image is equal to the coherence
        - so that the coherence can be directly estimated from the image
        - but since image is real & coherence is complex, a conversion is done
        thanks to the transfer matrix created with the lowlevel function coh__matcoher2real.
        - the final image is obtained by spectral reduction (until now, we don't take spectral
        dimension into account)
        - because the full coherence is estimated, demodulation (calculation of pistons and 
        amplitudes) is very simple and directly performed in coh_fs_default so that it 
        is a low-level self-contained FS used for debugging.
        
    In conclusion, it is just a non realistic, basic FS to close the loop.
    
    STRUCTURES OF IMAGE AND COHERENCE VECTORS:
            - image: [NW, NP] the mutual intensities on the detector. In this simple FS, \
        this is directly the real and imaginary parts of coherences. \ 
        (here NP = pixels number = NB = bases number). If this default parameter is given, \
        the function is run in INVERSE mode.
                    
            - coher: [NW, NB] the coherences of all basis. For k = i+NA*j, C_k(i,j) is the \
        complex conjugate of C_k(j,i). (and C_k(i,i) = 1). If this default parameter is given, \
        the function is run in DIRECT mode.

    Parameters
    ----------
    NA : int
        Number of telescopes.
    NW : int
        Number of wavelengths.

    Returns
    -------
    V2PM : [NP,NW] floats
        Visibility to Pixels matrix.
    P2VM : [NW,NP] floats
        Pixels to Visibilities matrix.
    ich : [NP, 3]
        Pixel sorting. There is no ABCD. We keep dimension 2 equal to 3 for 
        compatibility matter.
        If pixel k corresponds to base ij: ich[k] = [i,j,0]

    """
    
    NB = NA**2
    NP = NB
    
    matB = coh__matcoher2real(NA)                   # simplest FS: each pixel=coher converted2real
    V2PM = np.reshape(np.repeat(matB[np.newaxis,:],NW,0),[NW,NP,NB])
    matInvB = coh__matcoher2real(NA, 'inverse')      # simplest FS: each pixel=coher converted2real
    P2VM = np.reshape(np.repeat(matInvB[np.newaxis,:],NW,0),[NW,NP,NB])
    
    ich = np.zeros([NP,2])
    for i in range(NA):
            for j in range(NA):
                ich[NA*i+j] = (int(i),int(j))
                
    return V2PM, P2VM, ich


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
    
    # Nflux = config.FT['Nflux']
    # if it < Nflux:
    #     Nflux=it+1
    # timerange = range(it-Nflux,it)
    # for ia in range(NA):
    #     # Test if there were flux in the Nf last frames, before updating the average
    #     thereIsFlux = not outputs.noFlux[timerange,ia].any()
    #     if thereIsFlux:
    #         outputs.PhotometryAverage[it,ia] = np.mean(outputs.PhotometryEstimated[timerange,:,ia])
    #     else:
    #         outputs.PhotometryAverage[it,ia] = outputs.PhotometryAverage[it-1,ia]
            
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
        cfgd = outputs.CfPD[iot]*np.exp(-1j*outputs.PDEstimated[iot])/Ngd
        
        # If ClosurePhase correction before wrapping
        # cfgd = cfgd*np.exp(-1j*outputs.GDref[it])
        
        outputs.CfGD[it,:,:] += cfgd
    
    CfPD = outputs.CfPD[it]
    CfGD = outputs.CfGD[it]

    return CfPD, CfGD



# def ReadCf(currCfEstimated):
#     """
#     From measured coherent flux, estimates GD, PD, CP, Photometry, Visibilities
    
#     NAME: 
#         ReadCf - Calculates the group-delay, phase-delay, closure phase and 
#         complex visibility from the coherent flux estimated by the FS
    
        
#     INPUT: CfEstimated [MW,NB]
    
#     OUTPUT: 
        
#     UPDATE:
#         - outputs.CfPD: Coherent Flux Phase-Delay      [NT,MW,NIN]*1j
#         - outputs.CfGD: Coherent Flux GD               [NT,MW,NIN]*1j
#         - outputs.ClosurePhasePD                       [NT,MW,NC]
#         - outputs.ClosurePhaseGD                       [NT,MW,NC]
#         - outputs.PhotometryEstimated                  [NT,MW,NA]
#         - outputs.VisibilityEstimated                  [NT,MW,NIN]*1j
#         - outputs.SquaredCoherenceDegree               [NT,MW,NIN]
#     """

#     from . import outputs
    
#     from .config import NA,NIN,NC
#     from .config import MW,FS
    
#     it = outputs.it            # Time
     
#     """
#     Photometries extraction
#     [NT,MW,NA]
#     """
#     PhotEst = np.zeros([MW,NA])
#     for ia in range(NA):
#         PhotEst[:,ia] = np.real(currCfEstimated[:,ia*(NA+1)])
    
#     # Extract NIN-sized coherence vector from NB-sized one. 
#     # (eliminates photometric and conjugate terms)
#     currCfEstimatedNIN = np.zeros([MW, NIN])*1j
#     for imw in range(MW):    
#         from .coh_tools import NB2NIN
#         currCfEstimatedNIN[imw,:] = NB2NIN(currCfEstimated[imw,:])
        
#     # Save coherent flux and photometries in stack
#     outputs.PhotometryEstimated[it] = PhotEst
    
#     """
#     Visibilities extraction
#     [NT,MW,NIN]
#     """
    
#     for ia in range(NA):
#         for iap in range(ia+1,NA):
#             ib = ct.posk(ia,iap,NA)
#             Iaap = currCfEstimatedNIN[:,ib]                     # Interferometric intensity (a,a')
#             Ia = PhotEst[:,ia]                                  # Photometry pupil a
#             Iap = PhotEst[:,iap]                                # Photometry pupil a'
#             outputs.VisibilityEstimated[it,:,ib] = 2*Iaap/(Ia+Iap)          # Fringe VisibilityEstimated of the base (a,a')
#             outputs.SquaredCoherenceDegree[it,:,ib] = np.abs(Iaap)**2/(Ia*Iap)      # Spatial coherence of the source on the base (a,a')
 
#     """
#     Phase-delays extraction
#     PD_ is a global stack variable [NT, NIN]
#     Eq. 10 & 11
#     """
#     D = 0   # Dispersion correction factor: so far, put to zero because it's a 
#             # calibration term (to define in coh_fs?)
            
#     LmbdaTrack = config.wlOfTrack
    
#     # Coherent flux corrected from dispersion
#     for imw in range(MW):
#         outputs.CfPD[it,imw,:] = currCfEstimatedNIN[imw,:]*np.exp(1j*D*(1-LmbdaTrack/config.spectraM[imw])**2)
        
#         # If ClosurePhase correction before wrapping
#         # outputs.CfPD[it,imw] = outputs.CfPD[it,imw]*np.exp(-1j*outputs.PDref)
        
    # # Current Phase-Delay
    # currPD = np.angle(np.sum(outputs.CfPD[it,:,:], axis=0))*FS['active_ich']
        
    # """
    # Group-Delays extration
    # GD_ is a global stack variable [NT, NIN]
    # Eq. 15 & 16
    # """
    
    # if MW <= 1:
    #     raise ValueError('Tracking mode = GD but no more than one wavelength. \
    #                       Need several wavelengths for group delay')              # Group-delay calculation
    
    # Ngd = config.FT['Ngd']                 # Group-delay DIT
    # Ncross = config.FT['Ncross']           # Distance between wavelengths channels for GD calculation
    
    # if it < Ngd:
    #     Ngd = it+1
    
    # # Integrates GD with Ngd last frames (from it-Ngd to it)
    # timerange = range(it+1-Ngd,it+1)
    # for iot in timerange:
    #     cfgd = outputs.CfPD[iot]*np.exp(-1j*outputs.PDEstimated[iot])/Ngd
        
    #     # If ClosurePhase correction before wrapping
    #     # cfgd = cfgd*np.exp(-1j*outputs.GDref)
        
    #     outputs.CfGD[it,:,:] += cfgd


    # currGD = np.zeros(NIN)
    # for ib in range(NIN):
    #     # cs = 0*1j
    #     cfGDlmbdas = outputs.CfGD[it,:-Ncross,ib]*np.conjugate(outputs.CfGD[it,Ncross:,ib])
    #     cfGDmoy = np.sum(cfGDlmbdas)

            
    #     currGD[ib] = np.angle(cfGDmoy)*FS['active_ich'][ib]    # Group-delay on baseline (ib).
    
    # """
    # Closure phase calculation
    # cpPD_ is a global stack variable [NT, NC]
    # cpGD_ is a global stack variable [NT, NC]
    # Eq. 17 & 18
    # CORRECTION: Eq. 18 Lacour érronée --> 
    #     --> CPgd = arg[sum_{N_l-Ncross}(gamma''_{i,j,l+Ncross}*conj(gamma''_{i,j,l}))*sum(....)]
    # """
    
    # Ncp = config.FT['Ncp']
    
    # if it < Ncp:
    #     Ncp = it+1
        
    # bispectrumPD = np.zeros([NC])*1j
    # bispectrumGD = np.zeros([NC])*1j
    
    # timerange = range(it+1-Ncp,it+1) ; validcp=np.zeros(NC)
    # for ia in range(NA):
    #     for iap in range(ia+1,NA):
    #         ib = ct.posk(ia,iap,NA)      # coherent flux (ia,iap)  
    #         valid1=config.FS['active_ich'][ib]
    #         cs1 = np.sum(outputs.CfPD[timerange,:,ib], axis=1)     # Sum of coherent flux (ia,iap)
    #         cfGDlmbdas = outputs.CfGD[timerange,Ncross:,ib]*np.conjugate(outputs.CfGD[timerange,:-Ncross,ib])
    #         cfGDmoy1 = np.sum(cfGDlmbdas,axis=1)     # Sum of coherent flux (ia,iap)  
    #         for iapp in range(iap+1,NA):
    #             ib = ct.posk(iap,iapp,NA) # coherent flux (iap,iapp)    
    #             valid2=config.FS['active_ich'][ib]
    #             cs2 = np.sum(outputs.CfPD[timerange,:,ib], axis=1) # Sum of coherent flux (iap,iapp)    
    #             cfGDlmbdas = outputs.CfGD[timerange,Ncross:,ib]*np.conjugate(outputs.CfGD[timerange,:-Ncross,ib])
    #             cfGDmoy2 = np.sum(cfGDlmbdas,axis=1)
                
    #             ib = ct.posk(ia,iapp,NA) # coherent flux (iapp,ia)    
    #             valid3=config.FS['active_ich'][ib]
    #             cs3 = np.sum(np.conjugate(outputs.CfPD[timerange,:,ib]),axis=1) # Sum of 
    #             cfGDlmbdas = outputs.CfGD[timerange,Ncross:,ib]*np.conjugate(outputs.CfGD[timerange,:-Ncross,ib])
    #             cfGDmoy3 = np.sum(cfGDlmbdas,axis=1)
                
    #             # The bispectrum of one time and one triangle adds up to
    #             # the Ncp last times
    #             ic = ct.poskfai(ia,iap,iapp,NA)        # 0<=ic<NC=(NA-2)(NA-1) 
    #             validcp[ic]=valid1*valid2*valid3
    #             bispectrumPD[ic]=np.sum(cs1*cs2*cs3)
    #             bispectrumGD[ic]=np.sum(cfGDmoy1*cfGDmoy2*np.conjugate(cfGDmoy3))
    
    # outputs.ClosurePhasePD[it] = np.angle(bispectrumPD)*validcp
    # outputs.ClosurePhaseGD[it] = np.angle(bispectrumGD)*validcp
    
    # if config.FT['CPref'] and (it>Ncp):                     # At time 0, we create the reference vectors
    #     for ia in range(1,NA-1):
    #         for iap in range(ia+1,NA):
    #             k = ct.posk(ia,iap,NA)
    #             ic = ct.poskfai(0,ia,iap,NA)   # Position of the triangle (0,ia,iap)
    #             outputs.PDref[it,k] = outputs.ClosurePhasePD[it,ic]
    #             outputs.GDref[it,k] = outputs.ClosurePhaseGD[it,ic]

    
#     return currPD, currGD


def SimpleIntegrator(*args, init=False, Ngd=1, Npd=1, Ncp = 1, GainPD=0, GainGD=0,
                      Ncross = 1, CPref=True,roundGD='round',Threshold=True,
                      usePDref=True,verbose=False):
    """
    Calculates, from the measured coherent flux, the new positions to send to the delay lines.
    
    INPUT:
        - If init: all the below parameters.
        - If not init: CfEstimated - Measured Coherent Flux   [MW,NB]
    
    OUTPUT:
        - currCmd: Piston Command to send to the ODL     [NA]
    
    USED OBSERVABLES:
        - config.FT
    UPDATED OBSERVABLES:
        - outputs.PDEstimated: [MW,NIN] Estimated PD before subtraction of the reference
        - outputs.GDEstimated: [MW,NIN] Estimated GD before subtraction of the reference
        - outputs.CommandODL: Piston Command to send       [NT,NA]
        
    SUBROUTINES:
        - ReadCf
        - SimpleCommandCalc
 
        
    Parameters
    ----------
    *args : ARRAY [MW,NB]
        Expect current CfEstimated.
    init : BOOLEAN, optional
        If True, initialize the below parameters.
        Needs to be called before starting the simulation.
        The default is False.
    Npd : INT, optional
        Frame integration PD. The default is 1.
    Ngd : INT, optional
        Frame integration GD. The default is 1.
    Ncp : INT, optional
        Frame integration CP. The default is 1.
    GainPD : FLOAT, optional
        Gain PD. The default is 0.
    GainGD : FLOAT, optional
        Gain GD. The default is 0.
    Ncross : INT, optional
        Separation between two spectral channels for GD calculation. 
        The default is 1.
    CPref : BOOLEAN, optional
        If False, the Closure Phase is not subtracted for reference. 
        The default is True.
    roundGD : BOOLEAN, optional
        If True, the GD command is rounded to wavelength integers. 
        Advised to avoid a superposition of the PD and GD commands.
        The default is True.

    Returns
    -------
    currCmd : ARRAY [NA]
        Piston Command to send to the ODL.

    """
    
    if init:
        config.FT['Name'] = 'integrator'
        config.FT['func'] = SimpleIntegrator
        config.FT['GainPD'] = GainPD
        config.FT['Npd'] = Npd
        config.FT['Ngd'] = Ngd
        config.FT['Ncp'] = Ncp
        config.FT['GainGD'] = GainGD
        config.FT['GainPD'] = GainPD
        config.FT['Ncross'] = Ncross
        config.FT['CPref'] = CPref
        config.FT['roundGD'] = roundGD
        config.FT['Threshold'] = Threshold
        config.FT['cmdOPD'] = True
        config.FT['CPref'] = CPref
        config.FT['BestTel'] = 1
        
        if verbose:
            print(f"*** \n Inititilise Integrator: \n GainPD={GainPD} ; GainGD={GainGD} \n \
Ngd={Ngd} ; Type={config.FT['Name']} \n\ ***")
              
        return
    
    from . import outputs
    from .outputs import it
    
    currCfEstimated = args[0]
    
    CfPD, CfGD = ReadCf(currCfEstimated)
    
    currCmd = SimpleCommandCalc(CfPD,CfGD)
    
    return currCmd
    

def SimpleCommandCalc(CfPD,CfGD, verbose=False):
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
    
    from .config import NA,NIN,MW,NC
    from .config import FT,FS
    
    Ncross = FT['Ncross']
    
    it = outputs.it            # Frame number
    
    
    """ Current Phase-Delay """
    currPD = np.angle(np.sum(CfPD, axis=0))*FS['active_ich']
    
    """
    Group-Delays extration
    GD_ is a global stack variable [NT, NIN]
    Eq. 15 & 16
    """
    
    currGD = np.zeros(NIN)
    for ib in range(NIN):
        # cs = 0*1j
        cfGDlmbdas = CfGD[:-Ncross,ib]*np.conjugate(CfGD[Ncross:,ib])
        cfGDmoy = np.sum(cfGDlmbdas)

        currGD[ib] = np.angle(cfGDmoy)*FS['active_ich'][ib]    # Group-delay on baseline (ib).
    
    outputs.PDEstimated[it] = currPD
    outputs.GDEstimated[it] = currGD
    
    """
    Closure phase calculation
    cpPD_ is a global stack variable [NT, NC]
    cpGD_ is a global stack variable [NT, NC]
    Eq. 17 & 18
    CORRECTION: Eq. 18 Lacour érronée --> 
        --> CPgd = arg[sum_{N_l-Ncross}(gamma''_{i,j,l+Ncross}*conj(gamma''_{i,j,l}))*sum(....)]
    """
    
    Ncp = config.FT['Ncp']
    
    if it < Ncp:
        Ncp = it+1
        
    bispectrumPD = np.zeros([NC])*1j
    bispectrumGD = np.zeros([NC])*1j
    
    timerange = range(it+1-Ncp,it+1) ; validcp=np.zeros(NC)
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib = ct.posk(ia,iap,NA)      # coherent flux (ia,iap)  
            valid1=config.FS['active_ich'][ib]
            cs1 = np.sum(outputs.CfPD[timerange,:,ib], axis=1)     # Sum of coherent flux (ia,iap)
            cfGDlmbdas = outputs.CfGD[timerange,Ncross:,ib]*np.conjugate(outputs.CfGD[timerange,:-Ncross,ib])
            cfGDmoy1 = np.sum(cfGDlmbdas,axis=1)     # Sum of coherent flux (ia,iap)  
            for iapp in range(iap+1,NA):
                ib = ct.posk(iap,iapp,NA) # coherent flux (iap,iapp)    
                valid2=config.FS['active_ich'][ib]
                cs2 = np.sum(outputs.CfPD[timerange,:,ib], axis=1) # Sum of coherent flux (iap,iapp)    
                cfGDlmbdas = outputs.CfGD[timerange,Ncross:,ib]*np.conjugate(outputs.CfGD[timerange,:-Ncross,ib])
                cfGDmoy2 = np.sum(cfGDlmbdas,axis=1)
                
                ib = ct.posk(ia,iapp,NA) # coherent flux (iapp,ia)    
                valid3=config.FS['active_ich'][ib]
                cs3 = np.sum(np.conjugate(outputs.CfPD[timerange,:,ib]),axis=1) # Sum of 
                cfGDlmbdas = outputs.CfGD[timerange,Ncross:,ib]*np.conjugate(outputs.CfGD[timerange,:-Ncross,ib])
                cfGDmoy3 = np.sum(cfGDlmbdas,axis=1)
                
                # The bispectrum of one time and one triangle adds up to
                # the Ncp last times
                ic = ct.poskfai(ia,iap,iapp,NA)        # 0<=ic<NC=(NA-2)(NA-1) 
                validcp[ic]=valid1*valid2*valid3
                bispectrumPD[ic]=np.sum(cs1*cs2*cs3)
                bispectrumGD[ic]=np.sum(cfGDmoy1*cfGDmoy2*np.conjugate(cfGDmoy3))
    
    outputs.ClosurePhasePD[it] = np.angle(bispectrumPD)*validcp
    outputs.ClosurePhaseGD[it] = np.angle(bispectrumGD)*validcp
    
    if config.FT['CPref'] and (it>Ncp):                     # At time 0, we create the reference vectors
        for ia in range(1,NA-1):
            for iap in range(ia+1,NA):
                k = ct.posk(ia,iap,NA)
                ic = ct.poskfai(0,ia,iap,NA)   # Position of the triangle (0,ia,iap)
                outputs.PDref[it,k] = outputs.ClosurePhasePD[it,ic]
                outputs.GDref[it,k] = outputs.ClosurePhaseGD[it,ic]

    
    """
    Group-Delay tracking
    """
    
    currGDerr = currGD - outputs.GDref[it]
    
    # Keep the GD between [-Pi, Pi] (because the GDref could have make it
    # leave this interval)
    # Eq. 35
    
    for ib in range(NIN):
        if currGDerr[ib] > np.pi:
            currGDerr[ib] -= 2*np.pi
        elif currGDerr[ib] < -np.pi:
            currGDerr[ib] += 2*np.pi
    
    R = config.FS['R']
    
    # Store residual GD for display only [radians]
    outputs.GDResidual[it] = currGDerr*R

    
    if FT['Threshold']:             # Threshold function (eq.36)
        for ib in range(NIN):
            if currGDerr[ib] > np.pi/R:
                currGDerr[ib] -= np.pi/R
            elif currGDerr[ib] < -np.pi/R:
                currGDerr[ib] += np.pi/R
            else:
                currGDerr[ib] = 0
    
    
    # Integrator (Eq.37)
    """
    ATTENTION: The GD estimated doesn't have a telescope of reference. It's
    according to the average
    """
    if FT['cmdOPD']:     # integrator on OPD
        # Integrator
        outputs.GDCommand[it+1] = outputs.GDCommand[it] + FT['GainGD']*config.wlOfTrack*config.FS['R']/(2*np.pi)*currGDerr
        # From OPD to Pistons
        outputs.PistonGDCommand[it+1] = np.dot(FS['OPD2Piston'], outputs.GDCommand[it+1])
        
    else:                       # integrator on PD
        # From OPD to Piston
        currPistonGD = np.dot(FS['OPD2Piston'], currGDerr)
        # Integrator
        outputs.PistonGDCommand[it+1] = outputs.PistonGDCommand[it] + FT['GainPD']*currPistonGD
    
    uGD = outputs.PistonGDCommand[it+1]
    
    if config.FT['roundGD']=='round':
        for ia in range(NA):
            jumps = round(uGD[ia]/config.wlOfTrack)
            uGD[ia] = jumps*config.wlOfTrack

    elif config.FT['roundGD']=='int':
        for ia in range(NA):
            jumps = int(uGD[ia]/config.wlOfTrack)
            uGD[ia] = jumps*config.wlOfTrack
            
    elif config.FT['roundGD']=='no':
        pass

    else:
        if verbose:
            print("roundGD must be either 'round', 'int' or 'no'.")
       
    """
    Phase-Delay command
    """
    
    currPDerr = currPD - outputs.PDref[it]
 
    # Keep the PD between [-Pi, Pi]
    # Eq. 35
    for ib in range(NIN):
        if currPDerr[ib] > np.pi:
            currPDerr[ib] -= 2*np.pi
        elif currPDerr[ib] < -np.pi:
            currPDerr[ib] += 2*np.pi
    
    # Store residual PD for display only [radians]
    outputs.PDResidual[it] = currPDerr
    
    if config.FT['cmdOPD']:     # integrator on OPD
        # Integrator
        outputs.PDCommand[it+1] = outputs.PDCommand[it] + FT['GainPD']*config.wlOfTrack/(2*np.pi)*currPDerr
        # From OPD to Pistons
        outputs.PistonPDCommand[it+1] = np.dot(FS['OPD2Piston'], outputs.PDCommand[it+1])
        
    else:                       # integrator on PD
        # From OPD to Piston
        currPistonPD = np.dot(FS['OPD2Piston'], currPDerr)
        # Integrator
        outputs.PistonPDCommand[it+1] = outputs.PistonPDCommand[it] + FT['GainPD']*currPistonPD
    
    uPD = outputs.PistonPDCommand[it+1]

    """
    ODL command
    It is the addition of the GD, PD, SEARCH and modulation functions
    """
    
    CommandODL = uPD + uGD
    
    return CommandODL


def addnoiseADU(inputADU):
    """
    Add Noise to ADU image

    Parameters
    ----------
    inputADU : TYPE
        DESCRIPTION.
    sensitivity : TYPE, optional
        DESCRIPTION. The default is 5.88.
    dark_noise : TYPE, optional
        DESCRIPTION. The default is 2.
    floor : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    image : TYPE
        DESCRIPTION.

    """
    
    from .config import FS
    from .outputs import it
    
    ron, G, enf = config.FS['ron'],config.FS['G'],config.FS['enf']
    (seedPh, seedRon) = (config.seedPh+1,config.seedRon+1)
    
    # Add SHOT noise
    rs = np.random.RandomState(seedPh*it)
    photonADU = rs.poisson(inputADU/enf, size=inputADU.shape)*enf
    
    # Add DARK noise.
    rs = np.random.RandomState(seedRon*it)
    ronADU = rs.normal(scale=ron/G, size=photonADU.shape) + photonADU
        
    # Quantify ADU
    ADU_out = np.round(ronADU)
    
    
    return ADU_out


def addnoise(inPhotons):
    """
    Add Noise to photon image

    Parameters
    ----------
    inputADU : TYPE
        DESCRIPTION.
    sensitivity : TYPE, optional
        DESCRIPTION. The default is 5.88.
    dark_noise : TYPE, optional
        DESCRIPTION. The default is 2.
    floor : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    image : TYPE
        DESCRIPTION.

    """
    
    from .outputs import it
    
    ron, enf, qe, = config.FS['ron'], config.FS['enf'], config.FS['qe']
    
    (seedPh, seedRon) = (config.seedPh+1,config.seedRon+1)
    
    # Add SHOT noise (in the space of photons)
    rs = np.random.RandomState(seedPh*it)
    photons = rs.poisson(inPhotons/enf, size=inPhotons.shape)*enf
    
    # Converts photons to electrons
    electrons = photons*qe  # Works when qe is float and when qe is array of length MW
    
    # Add readout noise: here we assume DARK noise is only readout noise, so dark current is null.
    # That's why it can be modelised as a Gaussian noise.
    rs = np.random.RandomState(seedRon*it)
    electrons_with_darknoise = electrons + rs.normal(scale=ron, size=electrons.shape)
    
    # Convert back to photons
    outPhotons = electrons_with_darknoise/qe
    
    return outPhotons



def addnoise_morecomplex(inPhotons):
    """
    Add Noise to photon image
    /!\/!\ For using this function, the value of the parameter "ron"
    account for it real value before gains subtraction /!\/!\

    Parameters
    ----------
    inputADU : TYPE
        DESCRIPTION.
    sensitivity : TYPE, optional
        DESCRIPTION. The default is 5.88.
    dark_noise : TYPE, optional
        DESCRIPTION. The default is 2.
    floor : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    image : TYPE
        DESCRIPTION.

    """
    
    from .outputs import it
    
    ron, enf, qe, = config.FS['ron'], config.FS['enf'], config.FS['qe']
    darkCurrent = config.FS['darkCurrent'] ; G = config.FS['G']
    
    seedPh, seedRon, seedDark = config.seedPh+1,config.seedRon+1,config.seedDark+1
    
    # Add SHOT noise (in the space of photons)
    rs = np.random.RandomState(seedPh*it)
    photons = rs.poisson(inPhotons/enf, size=inPhotons.shape)*enf
    
    # Converts photons to electrons
    electrons = photons*qe  # Works when qe is float and when qe is array of length MW
    
    # Add DARK noise: Poisson noise of parameter darkCurrent
    rs = np.random.RandomState(seedDark*it)
    electrons_with_darknoise = electrons + rs.poisson(darkCurrent, size=electrons.shape)
        
    # Convert to ADU
    signalADU = electrons_with_darknoise * G
    
    # Add readout noise: Gaussian noise of parameter ron. 
    # /!\ ron value must account for value before gains subtraction /!\
    rs = np.random.RandomState(seedRon*it)
    signalADU_withron = signalADU + rs.normal(scale=ron, size=electrons.shape)
    
    # Quantification
    roundADU = np.round(signalADU_withron)
    electrons_with_dark_ron_and_quantification = roundADU/G
    
    # Convert back to photons
    outPhotons = electrons_with_dark_ron_and_quantification/qe
    
    return outPhotons

# =============================================================================
# Low-level functions:
# =============================================================================


def coh__pis2coh(pis, sigma, ampl=[]):
    """
    Used in the MakeAtmosphereCoherence function.
    From the two arrays (of dimension NA = number of pupils) containing the pistons 
    and the amplitudes and from the array containing the signal's spectra, the function 
    build a NAxNA array with all the mutual coherences. This enables to simulate 
    any kind of coherence pattern.
    INPUTS:
        - pis: vector containing the piston in each aperture
        - sigma: the working wavenumber (later, it will be a vector)
        - ampl: a vector containing the amplitude of the electric field in each aperture
                by default, the unitary vector.
    OUTPUT:
        - coher: [NW,NB] matrix containing the mutual coherences
    """
    
    
    NA = len(pis)
    NW = len(sigma)
    pis = np.reshape(pis, [NA,1])
    
    if len(ampl) == 0:
        ampl = np.ones([NW,NA])
    
    phasor = ampl * np.exp(2j*np.pi*np.dot(np.reshape(sigma, [NW,1]),\
                                           np.transpose(pis)))

    coher = coh__phasor2coher(phasor, NA, NW)
    
    return coher

def coh__phasor2coher(phasor, NA, NW):
    
    coher = np.zeros([NW, NA, NA])*1j
    for iw in range(NW):
        phasor_iw = np.reshape(phasor[iw,:],[1,NA])
        coher[iw,:] = np.dot(np.transpose(phasor_iw),np.conjugate(phasor_iw))
    
    coher = np.reshape(coher, [NW, NA*NA])
    return coher

def coh__matcoher2real(NA, *args):
    '''
    Create a transfer matrix which convert conjugate coherences (complex numbers) into 
    intensities (real numbers)
    INPUTS:
        - NA: number of apertures (arms)
        - *args: can enter 'inverse' to use inverse mode
    OUTPUT:
        - MatB: 
            if direct: coherences to intensities transfert matrix
            if indirect: intensities to coherences transfert matrix
    '''
    
    NB = NA**2
    MatB = np.zeros([NB,NB])*1j
    s2 = np.sqrt(0.5)
    MO = 1j*np.zeros([NA,NA,NA,NA])
    
    # first level (ia,iap) targets the real or imag domain.
    # Second level (ia,iap) targets the direct or conjugate coherence.
    for ia in range(NA):
        MO[ia,ia,ia,ia]=1
        for iap in np.arange(ia+1,NA):
            MO[ia,iap,ia,iap]=s2                # in (ia, iap) we store
            MO[ia,iap,iap,ia]=s2                # real parts
            MO[iap,ia,ia,iap]=1j*s2             # in (iap, ia) we store
            MO[iap,ia,iap,ia]=-1j*s2            # imaginary parts
    MatB = np.reshape(MO, (NB,NB))          # the matrix is unwrapped
    
    if 'inverse' in args:
        MatB = np.conjugate(np.transpose(MatB))
    
    return MatB


def repeat_sequence(sequence, newNT, verbose=False):
    NT = len(sequence)
    if NT > newNT:
        if verbose:
            print(f"The given sequence is longer than the desired length, we take only the {newNT} elements.")
        newseq = sequence[:newNT]
    elif NT==newNT:
        if verbose:
            print("The given sequence already has the desired length, we return the sequence without any modification.")
        newseq = sequence
    else:
        IntRepetitions, NumberOfRemainingElements = newNT//NT, newNT%NT
        newseq = sequence
        if NumberOfRemainingElements == 0:
            for i in range(IntRepetitions-1):
                newseq = np.concatenate([newseq,sequence])
        else:
            for i in range(IntRepetitions-1):
                newseq = np.concatenate([newseq,sequence])
            newseq = np.concatenate([newseq,sequence[:NumberOfRemainingElements]])
    return newseq

def populate_hdr(objet, hdr, prefix="",verbose=False):
    """
    Function that puts all attributes (or keys if objet is a dictionary) into a header.
    """
    
    if isinstance(objet, dict):
        names = objet.keys()
        
    else:
        names = dir(objet)
        
    names = [name for name in names if '_' not in name]
    
    DicoForImageHDU = {} # This dictionary will get all arrays that can't be inserted into header.
    
    supported_types = (str, float, int)
    arrnames=[] ; notsupportednames=[]
    for varname in names:

        if isinstance(objet, dict):
            value = objet.get(varname)
        else:
            value = getattr(objet, varname)

        if isinstance(value, (np.ndarray,list)):
            arrnames.append(varname)
            DicoForImageHDU[f"{prefix}{varname}"] = value
            
        elif isinstance(value, dict):
            if verbose:
                print(f"{varname} is a ditionary")
            _,dico_out = populate_hdr(value, hdr, f"{varname}.")
            for key in dico_out.keys():
                DicoForImageHDU[key] = dico_out[key]
                
        elif isinstance(value, str):
            if "µ" in value:
                value.replace("µ","micro")
                
            if len(value.encode('utf-8')) > 45:
                
                value = value.split('/')[-1]
                    
                # subfolders=value.split('/')
                # k=len(subfolders)
                # value = '/'.join(subfolders[-k:])
                # while len(value.encode('utf-8')) > 80:
                #     k-=1
                #     value = '/'.join(subfolders[-k:])
                # if k==0:
                #     value = value.split('/')[-1]
                    
                
            hdr[f"{prefix}{varname}"] = value
                
        elif isinstance(value, supported_types):
            hdr[f"{prefix}{varname}"] = value
            
        else:
            notsupportednames.append(varname)
            if verbose:
                print(f"{varname} is not of a supported type")

    if len(arrnames):
        if verbose:
            print(f"These arrays can't be saved in header but will be in other hdus:{arrnames}")
            
    if len(notsupportednames):
        if verbose:
            print(f"These values had not a suitable type for being saved into fitsfile:{notsupportednames}")
    
    return hdr, DicoForImageHDU


def save_data(simuobj, configobj, filepath, LightSave=True, overwrite=False, verbose=False):
    
    if verbose:
        if LightSave:
            print(f"Saving a part of the data into {filepath}")
        else:
            print(f"Saving all data into {filepath}")
        
    
    if os.path.exists(filepath):
        if not overwrite:
            if verbose:
                print(f"{filepath} already exists, and overwrite parameter is False.")
                print("DATA IS NOT SAVED.")
            return
    
    hdr = fits.Header()

    """ LOAD CONFIG DATA"""
    hdr, DicoForImageHDU = populate_hdr(configobj, hdr, verbose=verbose)
    
    primary = fits.PrimaryHDU(header=hdr)
    
    cols=[] ; imhdu=[]
    for key, array in DicoForImageHDU.items():
        
        if key=='FS.ichdetails':
            pass
            # cols.append(fits.Column(name=key+'.ich', format='I', array=np.array(array)[:,0]))
            # cols.append(fits.Column(name=key+'mod', format='A', array=np.array(array)[:,1]))
            
            
        elif np.ndim(array) == 1:
            if len(array):
                if isinstance(array[0], str):
                    form='A'
                else:
                    form='D'
                cols.append(fits.Column(name=key, format=form, array=array))
            
        elif np.ndim(array) == 2:
            if not isinstance(array,list):
                if not isinstance(array[0,0],complex):
                    imhdu.append(fits.ImageHDU(array, name=key))
            
    hdu = fits.BinTableHDU.from_columns(cols, name='config')
    
    hduL = fits.HDUList([primary,hdu])
    for newhdu in imhdu:
        hduL.append(newhdu)
        
        
    """ LOAD SIMU DATA"""
    names = dir(simuobj) ; names = [name for name in names if '_' not in name]
    
    excludeddata=[]
    cols=[] ; imhduL=[]
    #AchromOPDspaceData=[] # Will get all arrays of size [NINxNT] and the truncatur of commands of size [NINxNT+2]
    #MicChromOPDspaceData=[] # Will get all arrays of size [NINxNWxNT] 
    #MacChromOPDspaceData=[] # Will get all arrays of size [NINxMWxNT]
    #AchromPistonData=[] # Will get all arrays of size [NAxNT] and the truncatur of commands of size [NINxNT+2]
    #MicChromPistonData=[] # Will get all arrays of size [NAxNWxNT] 
    #MacChromPistonData=[] # Will get all arrays of size [NAxMWxNT] 
    #ChromCfData=[] # Will get all arrays of size [NBxMWxNT]
    
    if LightSave=='OPDTrue':
        NamesToSave = ['OPDTrue']
        names = [value for value in names if value in NamesToSave]
        if verbose:
            print(names)
    
    elif LightSave==True:
        NamesToSave = ['OPDTrue','GDEstimated','GDResidual',
                       'PDEstimated','PDResidual',
                       'ClosurePhaseGD','ClosurePhasePD',
                       'PistonGDCommand','PistonPDCommand','SearchCommand',
                       'PistonTrue','EffectiveMoveODL',
                       'PhotometryEstimated']
        names = [value for value in names if value in NamesToSave]

    
    for varname in names:
        value = getattr(simuobj, varname)

        if isinstance(value, (np.ndarray,list)):
            if np.ndim(value) == 1:
                if isinstance(value[0], str):
                    form='A'
                else:
                    form='1D'
                
                cols.append(fits.Column(name=varname, format=form, array=value))

            elif np.ndim(value) >= 2:
                if np.iscomplex(value).any():
                    imhduL.append(fits.ImageHDU(np.real(value), name=f"{varname}R"))
                    imhduL.append(fits.ImageHDU(np.imag(value), name=f"{varname}I"))

                else:
                    value = np.real(value)  # If the imaginary part is null, the test doesn't detect it as complex 
                    #whereas FITS detect it as complex so it doesn't work.

                    imhduL.append(fits.ImageHDU(value, name=varname))
                    
        else:
            excludeddata.append(varname)
    
    if len(excludeddata):
        if verbose:
            print(f"These data couldn't be saved into fits file: {excludeddata}")
    

    if len(cols):
        hdu = fits.BinTableHDU.from_columns(cols, name='simu')
        hduL.append(hdu)
    
    # t = PandasToTableHDU()
    # hdu = fits.BinTableHDU(timestamps, name="Perf")
    
    for newhdu in imhduL:
        hduL.append(newhdu)
        
    if '/' in filepath:
        filedir = '/'.join(filepath.split('/')[:-1])    # remove the filename to get the file directory only.
    
        if not os.path.exists(filedir):
            os.makedirs(filedir)
            
    if os.path.exists(filepath):
        os.remove(filepath)
        hduL.writeto(filepath)
        hduL.close()
        filedir=os.getcwd()
        if verbose:
            print("Overwrite on "+os.getcwd().replace('\\','/')+"/"+filepath)
            
    else:
        hduL.writeto(filepath)
        if verbose:
            print("Write in "+os.getcwd().replace('\\','/')+"/"+filepath)
        
    if verbose:
        print('Saved.')
    return
    
    
def PandasToTableHDU(DataFrame_or_Table, direct=True):
    from astropy.table import Table
    
    if direct:
        df = DataFrame_or_Table.copy()
        df.reset_index(level=0, inplace=True)
        df.columns = ["_".join([str(ind) for ind in col]) if isinstance(col,tuple) else col for col in df.columns.values]
        t = Table.from_pandas(df)
        return t
    
    else:
        t = DataFrame_or_Table
        df = t.to_pandas()
        
        L = []
        for colname in df.columns.values:
            if '.' in colname:
                L.append([float(value) if '.' in value else value for value in colname.split('_')])
            else:
                indval = df[colname].values
        
        arr = np.array(L)
        L2 = list(arr.T)
        
        MultiIndexdf = pd.DataFrame(data=df.T.iloc[1:].values.T, columns=L2, index=indval)
        MultiIndexdf.index.name = "Base"
        return MultiIndexdf