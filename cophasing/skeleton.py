# -*- coding: utf-8 -*-
#        import pdb; pdb.set_trace()

import os
import time
import pkg_resources

import numpy as np
# import cupy as cp # NumPy-equivalent module accelerated with NVIDIA GPU  
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from importlib import reload  # Python 3.4+ only.

from . import coh_tools as ct
from . import config

from astropy.io import fits

from .decorators import timer

from cophasing.tol_colors import tol_cset
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

# Change the display font
# plt.rc('font', **{'family' : 'serif', 'serif' : ['Computer Modern Roman']})
# plt.rc('text', usetex = True)

def initialize(Interferometer, ObsFile, DisturbanceFile, NT=512, OT=1, MW = 5, ND=1, 
             spectra = [], spectraM=[],PDspectra=0,
             spectrum = [], mode = 'search',
             fs='default', TELref=0, FSfitsfile='', R = 0.5, dt=1,sigmap=[],imsky=[],
             ft = 'integrator', state = 0,
             noise=True,ron=0, qe=0.5, phnoise = 0, G=1, enf=1.5, M=1,
             seedph=100, seedron=100, seeddist=100,
             starttracking=50, latencytime=0,
             piston_average=0,display=False,
             checktime=True, checkperiod=10):
    
    """
    NAME: initialize - Initializes a structure to simulate an interferometer by \
        COPHASING
    
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
    
    MT=int(NT/OT)                        # Total number of temporal samples
    timestamps = np.arange(NT)*dt       # Time sampling in [ms]
    
    # SPECTRAL PARAMETERS
    
    if len(spectra) == 0:
        raise ValueError('Lambda array required')      # Array which contains our signal's 
    NW = len(spectra)
    
    if len(spectraM) == 0:
        raise ValueError('Macro lambda array required')
    MW = len(spectraM)
    
    if PDspectra==0:
        PDspectra = np.median(spectraM) 
        

    if type(qe) == float:
        qe=qe*np.ones(MW)

    # Disturbance Pattern
    
    OW = int(NW/MW)
    if OW != NW/MW:
        raise ValueError('Oversampling might be integer.')
    

    # CONFIG PARAMETERS
    
    dyn=0.                                  # to be updtated later by FS (for ex via coh_algo)
      
    config.SimuTimeID=""
    
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
 
    # Fringe Sensor parameters
    config.NW=NW
    config.MW=MW
    config.OW=OW
    config.NX=0
    config.NY=0
    config.ND=ND
    config.dt=dt                    # ms

    # Noises
    config.qe = qe.reshape([MW,1])
    config.noise=noise
    config.ron=ron
    
    if imsky:
        config.FS['imsky'] = imsky
    if sigmap:
        config.FS['sigmap'] = sigmap
    
    if noise:
        np.random.seed(seedron+60)
        config.FS['sigmap'] = np.random.randn(MW,config.FS['NP'])*ron
    
    config.phnoise=phnoise
    config.G = G
    config.enf=enf
    config.M=M

    if latencytime == 0:
        config.latency = config.dt
    else:
        config.latency = round(latencytime/config.dt)
    
    
    # Random Statemachine seeds
    config.seedph=seedph
    config.seedron=seedron
    config.seeddist=seeddist
    
    # Fringe tracker
    config.starttracking = starttracking
    
    
    # Source description
    config.spectra=spectra
    config.spectraM=spectraM
    config.PDspectra=PDspectra
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
        from . import simu
        simu.timestamps = np.copy(config.timestamps)
        config.FT['state'] = np.zeros(config.NT)
        config.FT['usaw'] = np.zeros(config.NT)
        if verbose:
            print(' - New NT not equal to len(timestamps) so we change timestamps.')
            
    if 'ron' in kwargs.keys():
        ron = kwargs['ron']
        config.noise=True
        config.FS['sigmap'] = np.random.randn(config.MW,config.FS['NP'])*ron
        if verbose:
            print(f' - Updated sigmap for ron={ron}.')
        
    return

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
                            spectra=[], RefLambda=0, NT=1000,dt=1,
                            ampl=0, seed=100, dist='step', startframe = 10, 
                            f_fin=200, value_start=0, value_end=0,
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
        BaseData = hdu[2].data
    
    obstime = NT*dt                     # Observation time [ms]
    timestamps = np.arange(NT)*dt        # Time sampling [ms]
    
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
                if ia<NAfile:
                    injtemp = inj[:,:,ia]
                else:
                    injtemp = inj[NTfile//3:,:,ia-NAfile]
                if np.shape(injtemp)[0] < NT:
                    TransmissionDisturbance[:,:,ia] = repeat_sequence(injtemp, NT)
                else:
                    TransmissionDisturbance[:,:,ia] = injtemp
            
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
                print("ATTENTION: You gave 'value_end' and 'ampl', only value_end' is used.")
            ampl = (value_end-value_start)/NT
        
        itel = (tel-1 if tel else 0)
        PistonDisturbance[:,itel] = value_start + np.arange(NT) * ampl
        
    # The first telil sees a range of piston from -Lc to Lc
    elif dist == 'browse':
        PistonDisturbance[:,1] = (np.arange(NT)/NT - 1/2) * 2*Lc
    
    elif dist == 'random':
        
        if 'old' in kwargs.keys():
            rmsOPD = ampl
            rmsPiston = rmsOPD/np.sqrt(2)
            freq = np.fft.fftshift(np.fft.fftfreq(NT,d=dt*1e-3))
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

            # simu.DisturbancePSD = np.abs(newdsp[freqfft>=0])**2    # Save the PSD of the last disturbance for example
            # simu.FreqSampling = freqfft[freqfft>=0]

        elif 'new' in kwargs.keys():        # Correct: The DSP are true DSP

            if 'baselines' in kwargs.keys():
                baselines = kwargs['baselines']
            else:
                InterfArray = ct.get_array(config.Name)
                baselines = InterfArray.BaseNorms
                coords = InterfArray.TelCoordinates
            
            V = 0.31*r0/t0*1e3              # Average wind velocity in its direction [m/s]
            L0 = L0                         # Outer scale [m]
            direction = direction           # Orientation from the North [deg] (positive toward East)
            d = d                           # Telescopes diameter [m]
                
            if ampl==0:
                wl_r0 = 0.55                # Wavelength at which r0 is defined
                # rmsOPD = np.sqrt(6.88*(L0/r0)**(5/3))*wl_r0/(2*np.pi)    # microns
                rmsOPD = 30*np.sqrt((0.12/r0)**(5/3)) # on fixe rmsOPD = 15µm pour r0=12cm
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
    
                dfreq = np.min([0.008,1/(2.2*NT*dt*1e-3)]) # Minimal sampling wished
                freqmax = 1/(2*dt*1e-3)                  # Maximal frequency derived from given temporal sampling
                
                Npix = int(freqmax/dfreq)*2         # Array length (taking into account aliasing)
                
                freqfft = (np.arange(Npix)-Npix//2)*dfreq
                timefft = (np.arange(Npix)-Npix//2)*dt  #ms
            
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
                stdtime = (timefft>=0)#*(timefft<30000)              # We'll compute the standard deviation on a sample of 10s
                            
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
    
                dfreq = np.min([0.008,1/(2*NT*dt*1e-3)]) # Minimal sampling wished
                freqmax = 1/(2*dt*1e-3)                  # Maximal frequency derived from given temporal sampling
                
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
                stdtime = (timefft>=0)#*(timefft<30000)              # We'll compute the standard deviation on a sample of 10s
                            
                motif = motif0[keeptime]
    
                calibmotif = motif/np.std(motif0[stdtime])
    
                PistonDisturbance[:,ia] = rmsPiston*calibmotif
                
                # PistonDisturbance[:,ia] = PistonDisturbance[:,ia] - PistonDisturbance[startframe,ia]
                ElapsedTime = time.perf_counter() - tstart
                if verbose:
                    print(f'Done. Ellapsed time: {ElapsedTime}s')
        
    elif dist == 'chirp':
        f_fin = f_fin*1e-3   # Conversion to kHz
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
    
    if dist == 'random' or dist == 'chirp':

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
    
    from . import simu

    from .config import NT, NA, NW,timestamps, spectra, OW, MW, checktime, checkperiod
    
    config.SimuTimeID=time.strftime("%Y%m%d-%H%M%S")
    
    # Reload simu module for initialising the observables with their shape
    if verbose:
        print('Reloading simu for reinitialising the observables.')
    reload(simu)
    
    # Importation of the object 
    if NA>=3:
        CfObj, VisObj, CPObj = ct.get_CfObj(config.ObservationFile,spectra)
    else:
        CfObj, VisObj = ct.get_CfObj(config.ObservationFile,spectra)
        
    
    #scaling it to the spectral sampling  and integration time dt
    delta_wav = np.abs(spectra[1]-spectra[2])
    
    CfObj = CfObj * delta_wav           # Photons/spectralchannel/second at the entrance of the FS
    CfObj = CfObj * config.dt*1e-3      # Photons/spectralchannel/DIT at the entrance of the FS
    
    if NA>=3:
        simu.ClosurePhaseObject = CPObj
        
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
                    
                        simu.OPDrefObject[ib] = np.median(simu.ClosurePhaseObject[:,ic])/(2*np.pi)*config.PDspectra
        
    simu.CoherentFluxObject = CfObj
    simu.VisibilityObject = VisObj
    
    # Importation of the disturbance
    CfDist, PistonDist, TransmissionDist = ct.get_CfDisturbance(config.DisturbanceFile, spectra, timestamps,verbose=verbose)
    
    simu.CfDisturbance = CfDist
    simu.PistonDisturbance = PistonDist
    simu.TransmissionDisturbance = TransmissionDist    
    # simu.PhotometryDisturbance = np.zeros([config.NT,config.NW,config.NA])
    
    for ia in range(config.NA):
        PhotometryObject = np.abs(CfObj[:,ia*(config.NA+1)])
        simu.PhotometryDisturbance[:,:,ia] = simu.TransmissionDisturbance[:,:,ia]*PhotometryObject

    simu.FTmode[:config.starttracking] = np.zeros(config.starttracking)

    if verbose2:
        print("Processing simulation ...")
    
    simu.it = 0
    time0 = time.time()
    for it in range(NT):                        # We browse all the (macro)times
        simu.it = it
        
        # Coherence of the ODL
        CfODL = coh__pis2coh(-simu.EffectiveMoveODL[it,:],1/config.spectra)
        
        currCfTrue = CfObj * simu.CfDisturbance[it,:,:] * CfODL
        simu.CfTrue[it,:,:] = currCfTrue   #NB
        
        """
        Fringe Sensor: From oversampled true coherences to macrosampled 
        measured coherences
        """
        fringesensor = config.FS['func']
        currCfEstimated = fringesensor(currCfTrue)
        simu.CfEstimated[it,:,:] = currCfEstimated   # NBmes

        """
        FRINGE TRACKER: From measured coherences to ODL commands
        """
        GainGD = config.FT['GainGD']
        GainPD = config.FT['GainPD']
        
        if simu.FTmode[it] == 0:
            config.FT['GainGD'] = 0
            config.FT['GainPD'] = 0
            
        fringetracker = config.FT['func']
        CmdODL = fringetracker(currCfEstimated)
        
        config.FT['GainGD'] = GainGD
        config.FT['GainPD'] = GainPD
        
        simu.CommandODL[it+1,:] = CmdODL
        
        # Very simple step response of the delay lines: if latency==1, the odl response is instantaneous.
        simu.EffectiveMoveODL[it+config.latency] = CmdODL
        
        checkpoint = int(checkperiod/100 * NT)
        if (it>checkpoint) and (it%checkpoint == 0) and (it!=0) and checktime:
            processedfraction = it/NT
            # LeftProcessingTime = (time.time()-time0)*(1-processedfraction)/processedfraction
            if verbose:
                print(f'Processed: {processedfraction*100}%, Elapsed time: {round(time.time()-time0)}s')

    print(f"Done. (Total: {round(time.time()-time0)}s)")
    
    # Process observables for visualisation
    simu.PistonTrue = simu.PistonDisturbance - simu.EffectiveMoveODL[:-config.latency]

    # Save true OPDs in an observable
    for ia in range(config.NA):
        for iap in range(ia+1,config.NA):
            ib = ct.posk(ia,iap,config.NA)
            simu.OPDTrue[:,ib] = simu.PistonTrue[:,ia] - simu.PistonTrue[:,iap]
            simu.OPDDisturbance[:,ib] = simu.PistonDisturbance[:,ia] - simu.PistonDisturbance[:,iap]
            simu.OPDCommand[:,ib] = simu.CommandODL[:,ia] - simu.CommandODL[:,iap]    
            simu.EffectiveOPDMove[:,ib] = simu.EffectiveMoveODL[:,ia] - simu.EffectiveMoveODL[:,iap]    
            
            # if 'search' in config.FT.keys():
            #     simu.OPDSearchCommand[:,ib] = simu.SearchCommand[:,ia] - simu.SearchCommand[:,iap]
            
            for iow in range(MW):
                GammaObject = simu.CoherentFluxObject[iow*OW,ia*NA+iap]/np.sqrt(simu.CoherentFluxObject[iow*OW,ia*(NA+1)]*simu.CoherentFluxObject[iow*OW,iap*(NA+1)])
                
                Ia = np.abs(simu.CfTrue[:,iow*OW,ia*(NA+1)])    # Photometry pupil a
                Iap = np.abs(simu.CfTrue[:,iow*OW,iap*(NA+1)])  # Photometry pupil a'
                Iaap = np.abs(simu.CfTrue[:,iow*OW,ia*NA+iap])  # Mutual intensity aa'
                
                Lc = config.FS['R']*spectra[iow*OW]
                simu.VisibilityTrue[:,iow,ib] = Iaap/np.sqrt(Ia*Iap)*np.abs(GammaObject)*np.sinc(simu.OPDTrue[:,ib]/Lc)*np.exp(1j*2*np.pi*simu.OPDTrue[:,ib]/spectra[iow*OW])
    
    if len(args):
        filepath = args[0]+f"results_{config.SimuTimeID}.fits"
        save_data(simu, config, filepath, LightSave=LightSave, overwrite=overwrite, verbose=verbose)
    
    return


def display(*args, WLOfTrack=1.6,DIT=50,WLOfScience=0.75,
            Pistondetails=False,OPDdetails=False,
            OneTelescope=True, pause=False, display=True,verbose=False,
            start_pd_tracking = 50,
            savedir='',ext='pdf',infos={"details":''}):
    """
    
    Purpose
    ---------
        This procedure plots different results from the simulation.
        Available observables to display:
            - Photometries: 'phot'
            - Disturbances: 'disturbances'
            - Pistons: 'piston'
            - Piston details: 'Pistondetails'
            - GD, PD, SNR evolution: 'perftable'
            - OPDs: 'opd'
            - OPD commands: 'OPDcmd'
            - OPD details: 'OPDdetails'
            - All OPDs on the same figure: 'OPDgathered'
            - Closure phases: 'cp'
            - Visibilities: 'vis'
            - Detector: 'detector'
            - State-machine: 'state'"

    Parameters
    ----------
    *args : optional argument
        Write, as strings, the different observables you want to plot among those listed above.
    WLOfTrack : FLOAT, optional
        Wavelength at which the pistonic data are computed. The default is 1.6.
    DIT : FLOAT, optional
        Integration time for the variances, averages, etc... The default is 50.
    WLOfScience : FLOAT, optional
        Wavelength at which the science instrument is working, for SNR computation. The default is 0.75.
    Pistondetails : BOOLEAN, optional
        DESCRIPTION. The default is False.
    OPDdetails : BOOLEAN, optional
        DESCRIPTION. The default is False.
    OneTelescope : BOOLEAN, optional
        If true, stop opd displaying after the first telescope. The default is True.
    pause : BOOLEAN, optional
        Enable to show plots during a simulation. The default is False.
    display : BOOLEAN, optional
        If True, display plots. Else, generate but don't display. The default is True.
    verbose : BOOLEAN, optional
        If True, writes information in the terminal. The default is False.
    savedir : STRING, optional
        Directory path for saving the files. If empty, don't save plots. The default is ''.
    ext : STRING, optional
        Extension of the file. The default is 'pdf'.
    infos : DICTIONARY, optional
        Managed with few plots: enables to show a part of the baselines. The default is {"details":''}.

    Returns
    -------
    None.

    """
    
    from . import simu
    
    from .simu import timestamps
    from .config import NA,NT,NIN,NC,OW, SimuTimeID, InterfArray
    
    NINmes = config.FS['NINmes']
    
    #timestr = config.timestr
    
    if 'which' in args:
        print(f"Available observables to display:\n\
              - Photometries: 'phot'\n\
              - Disturbances: 'disturbances'\n\
              - Pistons: 'piston'\n\
              - Piston details: 'Pistondetails'\n\
              - GD, PD, SNR evolution: 'perftable'\n\
              - OPDs: 'opd'\n\
              - OPD commands: 'OPDcmd'\n\
              - OPD details: 'OPDdetails'\n\
              - All OPDs on the same figure: 'OPDgathered'\n\
              - Closure phases: 'cp'\n\
              - Visibilities: 'vis'\n\
              - Detector: 'detector'\n\
              - State-machine: 'state'\n")
        return
    
    # if not display:
        # 
        # currentGUI = plt.get_backend()
        # matplotlib.use('Qt5Agg')
    
    if (len(savedir)) and (not os.path.exists(savedir)):
        os.makedirs(savedir)
    
    WLIndex = np.argmin(np.abs(config.spectraM-WLOfTrack))
    wl = config.spectraM[WLIndex]
    
    timestr=SimuTimeID
    
    ich = config.FS['ich']
    R = config.FS['R']
    
    dt=config.dt
    ms=1e-3

    stationaryregim_start = config.starttracking+(config.NT-config.starttracking)*2//3
    if stationaryregim_start >= NT: stationaryregim_start=config.NT*1//3
    stationaryregim = np.arange(stationaryregim_start,NT)
    
    effDIT = min(DIT, config.NT - config.starttracking -1)
    
    if not ('opdcontrol' in args):
        ShowPerformance(float(timestamps[stationaryregim_start]), WLOfScience, effDIT, display=False)
    else:
        if verbose:
            print("don't compute performances")

    
    if verbose:
        print('Displaying observables...')
        print(f'First fig is Figure {config.newfig}')
    
    displayall = False
    if len(args)==0:
        displayall = True
        
    from matplotlib.ticker import AutoMinorLocator
    
    # Each figure only shows 15 baselines, distributed on two subplots
    # If there are more than 15 baselines, multiple figures will be created
    NINdisp = 15
    NumberOfBaseFiguresNIN = 1+NIN//NINdisp - 1*(NIN % NINdisp==0)
    NumberOfBaseFigures = 1+NINmes//NINdisp - 1*(NINmes % NINdisp==0)
    
    NAdisp = 10
    NumberOfTelFigures = 1+NA//NAdisp - 1*(NA % NAdisp==0)
    telcolors = colors[:NAdisp]*NumberOfTelFigures
    
    pis_max = 1.1*np.max([np.max(np.abs(simu.PistonDisturbance)),wl/2])
    pis_min = -pis_max
    ylim = [pis_min,pis_max]
    
    
    """
    HANDLE THE POSSILIBITY TO SHOW ONLY A PART OF THE TELESCOPES/BASELINES/CLOSURES
    """
    
    t = simu.timestamps*1e-3 # time in ms
    timerange = range(NT)
    
    TelConventionalArrangement = InterfArray.TelNames
    if 'TelescopeArrangement' in infos.keys():
        tels = infos['TelescopeArrangement']
    else:
        tels = TelConventionalArrangement
        
        
    beam_patches = []
    for ia in range(NA):
        beam_patches.append(mpatches.Patch(color=telcolors[ia],label=tels[ia]))
        
    # Tel2Beam = np.zeros([NA,NA])
    # for ia in range(NA):
    #     tel = tels[ia] ; tel0 = TelConventionalArrangement[ia]
    #     pos = np.argwhere(np.array(tels)==tel0)[0][0]
    #     Tel2Beam[pos,ia]=1
        
        
    
    baselinesNIN = []
    itel=0
    for tel1 in tels:
        for tel2 in tels[itel+1:]:
            baselinesNIN.append(f'{tel1}{tel2}')
        itel+=1
    baselinesNIN = np.array(baselinesNIN) 
    
    baselines = []
    itel=0
    for ib in range(NINmes):
        ia, iap = int(ich[ib][0])-1,int(ich[ib][1])-1
        tel1,tel2 = tels[ia],tels[iap]
        baselines.append(f'{tel1}{tel2}')
        
    baselines = np.array(baselines) 
    
    closures = []
    tel1=tels[0] ; itel1=0 
    for tel1 in tels:
        itel2 = itel1+1
        for tel2 in tels[itel1+1:]:
            itel3=itel2+1
            for tel3 in tels[itel2+1:]:
                closures.append(f'{tel1}{tel2}{tel3}')
                ib = ct.poskfai(itel1,itel2, itel3, NA)
                itel3+=1
            itel2+=1
    closures = np.array(closures)
    
    PlotTel = [False]*NA ; PlotTelOrigin=[False]*NA
    PlotBaselineNIN = [False]*NIN
    PlotBaseline = [False]*NINmes
    PlotClosure = [False]*NC
    TelNameLength = len(config.InterfArray.TelNames)
    
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
                baseline = baselinesNIN[ib]
                tel1,tel2=baseline[:TelNameLength],baseline[TelNameLength:]
                if (tel1 in TelsToDisplay) \
                    and (tel2 in TelsToDisplay):
                        PlotBaselineNIN[ib] = True
                        
            for ib in range(NINmes):
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
            baseline = baselinesNIN[ib]
            if (baseline in BaselinesToDisplay) or (baseline[2:]+baseline[:2] in BaselinesToDisplay):
                PlotBaselineNIN[ib] = True
        
        for ib in range(NINmes):
            baseline = baselines[ib]
            if (baseline in BaselinesToDisplay) or (baseline[2:]+baseline[:2] in BaselinesToDisplay):
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
            baseline = baselinesNIN[ib]
            for closure in ClosuresToDisplay:
                if baseline in closure:
                    PlotBaselineNIN[ib] = True
        
        for ib in range(NINmes):
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
        PlotBaselineNIN = [True]*NIN
        PlotBaseline = [True]*NINmes
        PlotClosure = [True]*NC
        
    PlotBaselineNINIndex = np.argwhere(PlotBaselineNIN).ravel()
    PlotBaselineIndex = np.argwhere(PlotBaseline).ravel()

    
    """
    COMPUTATION RMS
    """
    
    # Estimated, before patch (priority 1)
    GD = simu.GDEstimated ; PD=simu.PDEstimated 
    
    # Estimated, after patch
    GD2 = simu.GDEstimated2 ; PD2 = simu.PDEstimated2   
    
    # Residual, after subtraction of reference vectors (priority 2)
    GDerr = simu.GDResidual ; PDerr = simu.PDResidual
    
    # Residual, after Igd and Ipd
    GDerr2 = simu.GDResidual2 ; PDerr2 = simu.PDResidual2
    
    # Reference vectors
    GDrefmic = simu.GDref*R*wl/2/np.pi ; PDrefmic = simu.PDref*wl/2/np.pi
    
    
    GDmic = GD*R*wl/2/np.pi ; PDmic = PD*wl/2/np.pi
    GDmic2 = GD2*R*wl/2/np.pi ; PDmic2 = PD2*wl/2/np.pi
    GDerrmic = GDerr*R*wl/2/np.pi ; PDerrmic = PDerr*wl/2/np.pi
    GDerrmic2 = GDerr2*R*wl/2/np.pi ; PDerrmic2 = PDerr2*wl/2/np.pi
    
    
    # GDmic = GD*R*wl/2/np.pi ; PDmic = PD*wl/2/np.pi
    # GDrefmic = simu.GDref*R*wl/2/np.pi ; PDrefmic = simu.PDref*wl/2/np.pi
    
    # GDerr = simu.GDResidual2 ; PDerr =simu.PDResidual2
    # GDmic = GD*R*wl/2/np.pi ; PDmic = PD*wl/2/np.pi
    
    RMSgdmic = np.std(GDmic[start_pd_tracking:,:],axis=0)
    RMSpdmic = np.std(PDmic[start_pd_tracking:,:],axis=0)
    
    RMSgderrmic = np.std(GDerrmic[start_pd_tracking:,:],axis=0)
    RMSpderrmic = np.std(PDerrmic[start_pd_tracking:,:],axis=0)
    
    RMStrueOPD = np.sqrt(simu.VarOPD)
    
    VisObj = ct.NB2NIN(simu.VisibilityObject[WLIndex])
    VisibilityPhase = np.angle(simu.VisibilityEstimated[:,WLIndex,:])
    
    """
    SIGNAL TO NOISE RATIOS
    """
    
    
    SNR_pd = np.sqrt(simu.SquaredSNRMovingAveragePD)
    SNR_gd = np.sqrt(simu.SquaredSNRMovingAverageGD)
    SNRGD = np.sqrt(simu.SquaredSNRMovingAverageGDUnbiased)
    
    if config.FT['whichSNR'] == 'pd':
        SNR = SNR_pd
    else:
        SNR = SNR_gd
    
    
    """
    NOW YOU CAN DISPLAY
    """
    
    
    if displayall or ('disturbances' in args):
        
        fig = plt.figure("Disturbances")
        
        if not hasattr(simu, 'FreqSampling'):
            ax1 = fig.subplots(nrows=1,ncols=1)
            for ia in range(NA):
                # plt.subplot(NA,1,ia+1), plt.title('Beam {}'.format(ia+increment))
                ax1.plot(timestamps, simu.PistonDisturbance[:,ia],color=telcolors[ia])
            
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Piston [µm]')
            ax1.set_ylim(ylim)
            ax1.grid()
            ax1.set_title('Disturbance scheme at {:.2f}µm'.format(wl))
            ax1.legend(handles=beam_patches)

        else:
            ax1,ax2,ax3 = fig.subplots(nrows=3,ncols=1)
            
            ax1 = fig.subplots(nrows=1,ncols=1)
            for ia in range(NA):
                # plt.subplot(NA,1,ia+1), plt.title('Beam {}'.format(ia+increment))
                ax1.plot(timestamps, simu.PistonDisturbance[:,ia],color=telcolors[ia])
            
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Piston [µm]')
            ax1.set_ylim(ylim)
            ax1.grid()
            ax1.set_title('Disturbance scheme at {:.2f}µm'.format(wl))
            ax1.legend(handles=beam_patches)
            
            if simu.FreqSampling.size == simu.DisturbancePSD.size:
                ax2.plot(simu.FreqSampling, simu.DisturbancePSD)
                ax2.set_title('Power spectral distribution of the last pupil \
            (same shape for all)')
                ax2.set_xlabel('Frequency [Hz]')             
                ax2.set_xscale('log')
                ax2.set_yscale('log')
                
                ax3.plot(simu.FreqSampling, simu.DisturbanceFilter)
                ax3.set_title('Filter')
                ax3.set_xlabel('Frequency [Hz]')             
                ax3.set_xscale('log')
                ax3.set_yscale('log')    
        
        if display:
            plt.show()
        config.newfig+=1    
        
        
    if displayall or ('phot' in args):
        s=(0,1.1*np.max(simu.PhotometryEstimated))
        linestyles=[]
        linestyles.append(mlines.Line2D([], [], color='black',
                                        linestyle='solid',label='Estimated'))    
        linestyles.append(mlines.Line2D([], [], color='black',
                                        linestyle='dashed',label='Disturbance'))
        linestyles.append(mlines.Line2D([],[], color='black',
                                        linestyle=':', label='Start tracking'))
    
    
        plt.figure("Photometries")
        plt.suptitle('Photometries in the spectral channel containing {:.2f}µm'.format(wl))
        
        for ia in range(NA):
            plt.plot(t[timerange], np.sum(simu.PhotometryDisturbance[:,OW*WLIndex:OW*(WLIndex+1),ia],axis=1),
                     color=telcolors[ia],linestyle='dashed')#),label='Photometry disturbances')
            plt.plot(t[timerange], simu.PhotometryEstimated[:,WLIndex,ia],
                     color=telcolors[ia],linestyle='solid')#,label='Estimated photometries')
            
        plt.vlines(config.starttracking*dt*ms,s[0],s[1],
                   color='k', linestyle=':')
        plt.legend(handles=beam_patches+linestyles)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylim(s[0],s[1])
        if display:
            plt.show()    
        config.newfig+=1    
        
    
    if displayall or ('piston' in args):
        """
        PISTONS
        """
        linestyles=[]
        linestyles.append(mlines.Line2D([], [], color='black',
                                        linestyle='solid',label='Estimated'))    
        linestyles.append(mlines.Line2D([], [], color='black',
                                        linestyle='dashed',label='Disturbance'))
        linestyles.append(mlines.Line2D([],[], color='black',
                                        linestyle=':', label='Start tracking'))
        # linestyles.append(mlines.Line2D([], [], color='black',
        #                                 linestyle='dotted',label='Command'))
        
        ax2ymax = np.max([np.max(np.abs(simu.PistonTrue)),wl/2])
        ax2ylim = [-ax2ymax,ax2ymax]
        fig = plt.figure("Pistons")
        plt.suptitle('Piston time evolution at {:.2f}µm'.format(wl))
        ax1,axText = fig.subplots(ncols=2,gridspec_kw={"width_ratios":[4,1]})
        axText.axis("off")
        ax2 = ax1.twinx()
        
        if config.TELref:
            iTELref = config.TELref - 1
            PistonRef=simu.PistonTrue[:,iTELref]
            PistonDistRef = simu.PistonDisturbance[:,iTELref]
        else:
            PistonRef=0#np.mean(simu.PistonTrue, axis=1)
            PistonDistRef = 0
        
        for ia in range(NA):
            
            ax1.plot(timestamps, simu.PistonDisturbance[:,ia]-PistonDistRef,
                      color=telcolors[ia],linestyle='dashed')
            ax2.plot(timestamps, simu.PistonTrue[:,ia]-PistonRef,
                     color=telcolors[ia],linestyle='solid')
            
            axText.text(0.3,.9-ia*.05,f"$\sigma_{{{ia+1}}}={int(np.sqrt(simu.VarPiston[ia])*1e3)}$nm")
        
        ax1.vlines(config.starttracking*dt,ylim[0],ylim[1],
                   color='k', linestyle=':')
        # ax2.vlines(config.starttracking*dt,ax2ylim[0],ax2ylim[1],
        #                color='k', linestyle=':')
        ax2.set_ylabel('True Pistons [µm]')
        ax2.set_ylim(ax2ylim)
        ax1.set_ylabel('Disturbance Pistons [µm]')
        ax1.set_ylim(ylim)
        plt.xlabel('Time (ms)')
        axText.legend(handles=beam_patches+linestyles,loc="lower right")
        ax2.grid(True)
        if display:
            plt.show()
        config.newfig+=1
    
        if len(savedir):
            fig.savefig(savedir+f"Simulation{timestr}_piston.{ext}")
    
    
    if displayall or ('Pistondetails' in args):
        
        linestyles=[]
        # linestyles.append(mlines.Line2D([], [], color=colors[0],
        #                                 linestyle='solid',label='Estimated'))    
        linestyles.append(mlines.Line2D([], [], color=colors[0],
                                        label='Disturbance'))
        linestyles.append(mlines.Line2D([], [], color=colors[1],
                                        label='Total Command'))
        linestyles.append(mlines.Line2D([], [], color=colors[2],
                                        label='PD Command'))
        linestyles.append(mlines.Line2D([], [], color=colors[3],
                                        label='GD Command'))
        linestyles.append(mlines.Line2D([], [], color=colors[4],
                                        label='Search Command'))
        linestyles.append(mlines.Line2D([],[], color='black',
                                        linestyle=':', label='Start tracking'))
        # linestyles.append(mlines.Line2D([], [], color=colors[5],
        #                                 label='Modulation Command'))
    
    
        fig = plt.figure("Piston details")
        fig.suptitle('Piston time evolution at {:.2f}µm'.format(wl))
        axes = fig.subplots(nrows=NA,ncols=1, sharex=True)
        ax2ymax = np.max(np.abs(simu.PistonTrue))
        ax2ylim = [-ax2ymax,ax2ymax]
        for ia in range(NA):
            ax = axes[ia]
            ax.plot(timestamps, simu.PistonDisturbance[:,ia],
                     color=colors[0])
            ax.plot(timestamps, simu.CommandODL[:-1,ia],
                     color=colors[1])
            ax.plot(timestamps, simu.PistonPDCommand[:-1,ia],
                     color=colors[2])
            ax.plot(timestamps, simu.PistonGDCommand[:-1,ia],
                     color=colors[3])
            ax.plot(timestamps, simu.SearchCommand[:-1,ia],
                     color=colors[4])
            # ax2 = ax.twinx()
            # ax2.plot(timestamps, simu.PistonTrue[:,ia],
            #          color='blue',linestyle='solid')
            ax.set_ylim(ylim)
            # ax2.set_ylim(ax2ylim)
            ax.set_ylabel(f'Tel {ia+1} \n[µm]')
            # ax2.set_ylabel(f'Residual Piston {ia+increment} [µm]')
            ax.grid()
        
            ax.vlines(config.starttracking*dt,ylim[0],ylim[1],
                   color='k', linestyle=':')

        plt.xlabel('Time (ms)')
        plt.legend(handles=linestyles)
        if display:
            plt.show()
        config.newfig+=1    

        

    if displayall or ('perftable' in args) :
        plt.rcParams.update(rcParamsForBaselines)
        generaltitle = "GD and PD estimated"
        typeobs = "GDPDest"
        
        GDobs = GDmic
        PDobs = PDmic
        
        RMSgdobs = np.std(GDobs[start_pd_tracking:,:],axis=0)
        RMSpdobs = np.std(PDobs[start_pd_tracking:,:],axis=0)

        linestyles=[mlines.Line2D([],[], color='black',
                                        linestyle=':', label='Start tracking')]
        if 'ThresholdGD' in config.FT.keys():
            linestyles.append(mlines.Line2D([],[], color='black',
                                        linestyle='--', label='Threshold GD'))
        
        for iFig in range(NumberOfBaseFigures):
            NINtodisplay=NINdisp
            if iFig == NumberOfBaseFigures-1:
                if (NINmes%NINdisp < NINdisp) and (NINmes%NINdisp != 0):
                    NINtodisplay = NINmes%NINdisp
                    
            iFirstBase = NINdisp*iFig   # Index of first baseline to display
            iLastBase = iFirstBase + NINtodisplay - 1        # Index of last baseline to display
            
            len2 = NINtodisplay//2 ; len1 = NINtodisplay-len2
            basecolors = colors[:len1]+colors[:len2]
            basecolors = np.array(basecolors)
            
            rangeBases = f"{baselines[iFirstBase]}-{baselines[iLastBase]}"
            title=f'{generaltitle}: {rangeBases}'
            plt.close(title)
            fig=plt.figure(title, clear=True)
            fig.suptitle(title)
            (ax1,ax6),(ax2,ax7), (ax3,ax8),(ax11,ax12),(ax4,ax9),(ax5,ax10) = fig.subplots(nrows=6,ncols=2, gridspec_kw={"height_ratios":[1,4,4,0.5,1,1]})
            ax1.set_title(f"From {baselines[iFirstBase]} \
to {baselines[iFirstBase+len1-1]}")
            ax6.set_title(f"From {baselines[iFirstBase+len1]} \
to {baselines[iLastBase]}")
            
            FirstSet = range(iFirstBase,iFirstBase+len1)
            SecondSet = range(iFirstBase+len1,iLastBase+1)
            iColor=0
            for iBase in FirstSet:   # First serie
                ax1.plot(t[timerange],SNR[timerange,iBase],color=basecolors[iColor])
                if 'ThresholdGD' in config.FT.keys():
                    ax1.hlines(config.FT['ThresholdGD'][iBase], t[timerange[0]],t[timerange[-1]], color=basecolors[iColor], linestyle='dashed')
                ax2.plot(t[timerange],GDobs[timerange,iBase],color=basecolors[iColor])
                ax2.plot(t[timerange],GDrefmic[timerange,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
                ax3.plot(t[timerange],PDobs[timerange,iBase],color=basecolors[iColor])
                ax3.plot(t[timerange],PDrefmic[timerange,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
                iColor+=1
            for iBase in SecondSet:   # Second serie
                ax6.plot(t[timerange],SNR[timerange,iBase],color=basecolors[iColor])
                if 'ThresholdGD' in config.FT.keys():
                    ax6.hlines(config.FT['ThresholdGD'][iBase],t[timerange[0]],t[timerange[-1]],color=basecolors[iColor], linestyle='dashed')
                ax7.plot(t[timerange],GDobs[timerange,iBase],color=basecolors[iColor])
                ax7.plot(t[timerange],GDrefmic[timerange,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
                ax8.plot(t[timerange],PDobs[timerange,iBase],color=basecolors[iColor])
                ax8.plot(t[timerange],PDrefmic[timerange,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
                iColor+=1
            
            # ax2.vlines(config.starttracking*dt,-3*np.max(np.abs(GDmic)),3*np.max(np.abs(GDmic)),
            #             color='k', linestyle=':')
            # ax3.vlines(config.starttracking*dt,-wl/2,wl/2,
            #             color='k', linestyle=':')
            # ax7.vlines(config.starttracking*dt,-3*np.max(np.abs(GDmic)),3*np.max(np.abs(GDmic)),
            #             color='k', linestyle=':')
            # ax8.vlines(config.starttracking*dt,-wl/2,wl/2,
            #             color='k', linestyle=':')
            
            ax4.bar(baselines[FirstSet],RMSgdobs[FirstSet], color=basecolors[:len1])
            # ax4.bar(baselines[FirstSet],simu.LR4[FirstSet],fill=False,edgecolor='black',linestyle='-')
            ax5.bar(baselines[FirstSet],RMSpdobs[FirstSet], color=basecolors[:len1])
            # ax5.bar(baselines[FirstSet],RMStrueOPD[FirstSet],fill=False,edgecolor='black',linestyle='-')
            
            ax9.bar(baselines[SecondSet],RMSgdobs[SecondSet], color=basecolors[len1:])
            # ax9.bar(baselines[SecondSet],simu.LR4[SecondSet],fill=False,edgecolor='black',linestyle='-')
            ax10.bar(baselines[SecondSet],RMSpdobs[SecondSet], color=basecolors[len1:])
            # ax10.bar(baselines[SecondSet],RMStrueOPD[SecondSet],fill=False,edgecolor='black',linestyle='-')
            
            ax1.get_shared_x_axes().join(ax1,ax2,ax3)
            ax6.get_shared_x_axes().join(ax6,ax7,ax8)
            ax4.get_shared_x_axes().join(ax4,ax5)
            ax9.get_shared_x_axes().join(ax9,ax10)
            
            ax1.get_shared_y_axes().join(ax1,ax6)
            ax2.get_shared_y_axes().join(ax2,ax7)
            ax3.get_shared_y_axes().join(ax3,ax8)
            ax4.get_shared_y_axes().join(ax4,ax9)
            ax5.get_shared_y_axes().join(ax5,ax10)
            
            ax6.tick_params(labelleft=False) ; ct.setaxelim(ax1,ydata=SNR,ymin=0)
            ax7.tick_params(labelleft=False) ; ct.setaxelim(ax2,ydata=GDmic[stationaryregim],ylim_min=[-wl/2,wl/2])
            ax8.tick_params(labelleft=False) ; ax3.set_ylim([-wl/2,wl/2])
            ax9.tick_params(labelleft=False) ; ct.setaxelim(ax4,ydata=np.concatenate([np.stack(RMSgdmic),[1]]),ymin=0)
            ax10.tick_params(labelleft=False) ; ct.setaxelim(ax5,ydata=np.concatenate([np.stack(RMSpdmic)]),ymin=0)
            
            ax4.tick_params(labelbottom=False)
            ax9.tick_params(labelbottom=False)
            
            ax1.set_ylabel('SNR')
            ax2.set_ylabel('Group-Delays [µm]')
            ax3.set_ylabel('Phase-Delays [µm]')
            ax4.set_ylabel('GD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
            ax5.set_ylabel('PD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
            
            ax11.remove() ; ax12.remove()       # These axes are here to let space for ax3 and ax8 labels
            
            ax3.set_xlabel('Time [s]', labelpad=-10) ; ax8.set_xlabel('Time [s]', labelpad=-10)
            ax5.set_xlabel('Baselines') ; ax10.set_xlabel('Baselines')
    
            ax7.legend(handles=linestyles, loc='upper right')
            if display:
                fig.show()
    
            if len(savedir):
                if verbose:
                    print("Saving perftable figure.")
                plt.savefig(savedir+f"Simulation{timestr}_{typeobs}_{rangeBases}.{ext}")

        plt.rcParams.update(plt.rcParamsDefault)

    if displayall or ('perftable2' in args) :
        plt.rcParams.update(rcParamsForBaselines)
        generaltitle = "GD and PD after patch"
        typeobs = "GDPDest2"
        
        GDobs = GDmic2
        PDobs = PDmic2
        
        RMSgdobs = np.std(GDobs[start_pd_tracking:,:],axis=0)
        RMSpdobs = np.std(PDobs[start_pd_tracking:,:],axis=0)

        linestyles=[mlines.Line2D([],[], color='black',
                                        linestyle=':', label='Start tracking')]
        if 'ThresholdGD' in config.FT.keys():
            linestyles.append(mlines.Line2D([],[], color='black',
                                        linestyle='--', label='Threshold GD'))
        
        for iFig in range(NumberOfBaseFigures):
            NINtodisplay=NINdisp
            if iFig == NumberOfBaseFigures-1:
                if (NINmes%NINdisp < NINdisp) and (NINmes%NINdisp != 0):
                    NINtodisplay = NINmes%NINdisp
                    
            iFirstBase = NINdisp*iFig   # Index of first baseline to display
            iLastBase = iFirstBase + NINtodisplay - 1        # Index of last baseline to display
            
            len2 = NINtodisplay//2 ; len1 = NINtodisplay-len2
            basecolors = colors[:len1]+colors[:len2]
            basecolors = np.array(basecolors)
            
            rangeBases = f"{baselines[iFirstBase]}-{baselines[iLastBase]}"
            title=f'{generaltitle}: {rangeBases}'
            plt.close(title)
            fig=plt.figure(title, clear=True)
            fig.suptitle(title)
            (ax1,ax6),(ax2,ax7), (ax3,ax8),(ax11,ax12),(ax4,ax9),(ax5,ax10) = fig.subplots(nrows=6,ncols=2, gridspec_kw={"height_ratios":[1,4,4,0.5,1,1]})
            ax1.set_title(f"From {baselines[iFirstBase]} \
to {baselines[iFirstBase+len1-1]}")
            ax6.set_title(f"From {baselines[iFirstBase+len1]} \
to {baselines[iLastBase]}")
            
            FirstSet = range(iFirstBase,iFirstBase+len1)
            SecondSet = range(iFirstBase+len1,iLastBase+1)
            iColor=0
            for iBase in FirstSet:   # First serie
                ax1.plot(t[timerange],SNR[timerange,iBase],color=basecolors[iColor])
                if 'ThresholdGD' in config.FT.keys():
                    ax1.hlines(config.FT['ThresholdGD'][iBase], t[timerange[0]],t[timerange[-1]], color=basecolors[iColor], linestyle='dashed')
                ax2.plot(t[timerange],GDobs[timerange,iBase],color=basecolors[iColor])
                ax2.plot(t[timerange],GDrefmic[timerange,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
                ax3.plot(t[timerange],PDobs[timerange,iBase],color=basecolors[iColor])
                ax3.plot(t[timerange],PDrefmic[timerange,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
                iColor+=1
            for iBase in SecondSet:   # Second serie
                ax6.plot(t[timerange],SNR[timerange,iBase],color=basecolors[iColor])
                if 'ThresholdGD' in config.FT.keys():
                    ax6.hlines(config.FT['ThresholdGD'][iBase],t[timerange[0]],t[timerange[-1]],color=basecolors[iColor], linestyle='dashed')
                ax7.plot(t[timerange],GDobs[timerange,iBase],color=basecolors[iColor])
                ax7.plot(t[timerange],GDrefmic[timerange,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
                ax8.plot(t[timerange],PDobs[timerange,iBase],color=basecolors[iColor])
                ax8.plot(t[timerange],PDrefmic[timerange,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
                iColor+=1
            
            # ax2.vlines(config.starttracking*dt,-3*np.max(np.abs(GDmic)),3*np.max(np.abs(GDmic)),
            #             color='k', linestyle=':')
            # ax3.vlines(config.starttracking*dt,-wl/2,wl/2,
            #             color='k', linestyle=':')
            # ax7.vlines(config.starttracking*dt,-3*np.max(np.abs(GDmic)),3*np.max(np.abs(GDmic)),
            #             color='k', linestyle=':')
            # ax8.vlines(config.starttracking*dt,-wl/2,wl/2,
            #             color='k', linestyle=':')
            
            ax4.bar(baselines[FirstSet],RMSgdobs[FirstSet], color=basecolors[:len1])
            # ax4.bar(baselines[FirstSet],simu.LR4[FirstSet],fill=False,edgecolor='black',linestyle='-')
            ax5.bar(baselines[FirstSet],RMSpdobs[FirstSet], color=basecolors[:len1])
            # ax5.bar(baselines[FirstSet],RMStrueOPD[FirstSet],fill=False,edgecolor='black',linestyle='-')
            
            ax9.bar(baselines[SecondSet],RMSgdobs[SecondSet], color=basecolors[len1:])
            # ax9.bar(baselines[SecondSet],simu.LR4[SecondSet],fill=False,edgecolor='black',linestyle='-')
            ax10.bar(baselines[SecondSet],RMSpdobs[SecondSet], color=basecolors[len1:])
            # ax10.bar(baselines[SecondSet],RMStrueOPD[SecondSet],fill=False,edgecolor='black',linestyle='-')
            
            ax1.get_shared_x_axes().join(ax1,ax2,ax3)
            ax6.get_shared_x_axes().join(ax6,ax7,ax8)
            ax4.get_shared_x_axes().join(ax4,ax5)
            ax9.get_shared_x_axes().join(ax9,ax10)
            
            ax1.get_shared_y_axes().join(ax1,ax6)
            ax2.get_shared_y_axes().join(ax2,ax7)
            ax3.get_shared_y_axes().join(ax3,ax8)
            ax4.get_shared_y_axes().join(ax4,ax9)
            ax5.get_shared_y_axes().join(ax5,ax10)
            
            ax6.tick_params(labelleft=False) ; ct.setaxelim(ax1,ydata=SNR,ymin=0)
            ax7.tick_params(labelleft=False) ; ct.setaxelim(ax2,ydata=GDmic[stationaryregim],ylim_min=[-wl/2,wl/2])
            ax8.tick_params(labelleft=False) ; ax3.set_ylim([-wl/2,wl/2])
            ax9.tick_params(labelleft=False) ; ct.setaxelim(ax4,ydata=np.concatenate([np.stack(RMSgdmic),[1]]),ymin=0)
            ax10.tick_params(labelleft=False) ; ct.setaxelim(ax5,ydata=np.concatenate([np.stack(RMSpdmic)]),ymin=0)
            
            ax4.tick_params(labelbottom=False)
            ax9.tick_params(labelbottom=False)
            
            ax1.set_ylabel('SNR')
            ax2.set_ylabel('Group-Delays [µm]')
            ax3.set_ylabel('Phase-Delays [µm]')
            ax4.set_ylabel('GD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
            ax5.set_ylabel('PD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
            
            ax11.remove() ; ax12.remove()       # These axes are here to let space for ax3 and ax8 labels
            
            ax3.set_xlabel('Time [s]', labelpad=-10) ; ax8.set_xlabel('Time [s]', labelpad=-10)
            ax5.set_xlabel('Baselines') ; ax10.set_xlabel('Baselines')
    
            ax7.legend(handles=linestyles, loc='upper right')
            if display:
                fig.show()
    
            if len(savedir):
                if verbose:
                    print("Saving perftable figure.")
                plt.savefig(savedir+f"Simulation{timestr}_{typeobs}_{rangeBases}.{ext}")

        plt.rcParams.update(plt.rcParamsDefault)


    if displayall or ('perftableres' in args) :
        plt.rcParams.update(rcParamsForBaselines)
        generaltitle = "GD and PD residuals"
        typeobs = "GDPDres"
        
        GDobs = GDerrmic
        PDobs = PDerrmic
        
        RMSgdobs = np.std(GDobs[start_pd_tracking:,:],axis=0)
        RMSpdobs = np.std(PDobs[start_pd_tracking:,:],axis=0)

        linestyles=[mlines.Line2D([],[], color='black',
                                        linestyle=':', label='Start tracking')]
        if 'ThresholdGD' in config.FT.keys():
            linestyles.append(mlines.Line2D([],[], color='black',
                                        linestyle='--', label='Threshold GD'))
        
        for iFig in range(NumberOfBaseFigures):
            NINtodisplay=NINdisp
            if iFig == NumberOfBaseFigures-1:
                if (NINmes%NINdisp < NINdisp) and (NINmes%NINdisp != 0):
                    NINtodisplay = NINmes%NINdisp
                    
            iFirstBase = NINdisp*iFig   # Index of first baseline to display
            iLastBase = iFirstBase + NINtodisplay - 1        # Index of last baseline to display
            
            len2 = NINtodisplay//2 ; len1 = NINtodisplay-len2
            basecolors = colors[:len1]+colors[:len2]
            basecolors = np.array(basecolors)
            
            rangeBases = f"{baselines[iFirstBase]}-{baselines[iLastBase]}"
            title=f'{generaltitle}: {rangeBases}'
            plt.close(title)
            fig=plt.figure(title, clear=True)
            fig.suptitle(title)
            (ax1,ax6),(ax2,ax7), (ax3,ax8),(ax11,ax12),(ax4,ax9),(ax5,ax10) = fig.subplots(nrows=6,ncols=2, gridspec_kw={"height_ratios":[1,4,4,0.5,1,1]})
            ax1.set_title(f"From {baselines[iFirstBase]} \
to {baselines[iFirstBase+len1-1]}")
            ax6.set_title(f"From {baselines[iFirstBase+len1]} \
to {baselines[iLastBase]}")
            
            FirstSet = range(iFirstBase,iFirstBase+len1)
            SecondSet = range(iFirstBase+len1,iLastBase+1)
            iColor=0
            for iBase in FirstSet:   # First serie
                ax1.plot(t[timerange],SNR[timerange,iBase],color=basecolors[iColor])
                if 'ThresholdGD' in config.FT.keys():
                    ax1.hlines(config.FT['ThresholdGD'][iBase], t[timerange[0]],t[timerange[-1]], color=basecolors[iColor], linestyle='dashed')
                ax2.plot(t[timerange],GDobs[timerange,iBase],color=basecolors[iColor])
                ax2.plot(t[timerange],GDrefmic[timerange,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
                ax3.plot(t[timerange],PDobs[timerange,iBase],color=basecolors[iColor])
                ax3.plot(t[timerange],PDrefmic[timerange,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
                iColor+=1
            for iBase in SecondSet:   # Second serie
                ax6.plot(t[timerange],SNR[timerange,iBase],color=basecolors[iColor])
                if 'ThresholdGD' in config.FT.keys():
                    ax6.hlines(config.FT['ThresholdGD'][iBase],t[timerange[0]],t[timerange[-1]],color=basecolors[iColor], linestyle='dashed')
                ax7.plot(t[timerange],GDobs[timerange,iBase],color=basecolors[iColor])
                ax7.plot(t[timerange],GDrefmic[timerange,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
                ax8.plot(t[timerange],PDobs[timerange,iBase],color=basecolors[iColor])
                ax8.plot(t[timerange],PDrefmic[timerange,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
                iColor+=1
            
            # ax2.vlines(config.starttracking*dt,-3*np.max(np.abs(GDmic)),3*np.max(np.abs(GDmic)),
            #             color='k', linestyle=':')
            # ax3.vlines(config.starttracking*dt,-wl/2,wl/2,
            #             color='k', linestyle=':')
            # ax7.vlines(config.starttracking*dt,-3*np.max(np.abs(GDmic)),3*np.max(np.abs(GDmic)),
            #             color='k', linestyle=':')
            # ax8.vlines(config.starttracking*dt,-wl/2,wl/2,
            #             color='k', linestyle=':')
            
            ax4.bar(baselines[FirstSet],RMSgdobs[FirstSet], color=basecolors[:len1])
            # ax4.bar(baselines[FirstSet],simu.LR4[FirstSet],fill=False,edgecolor='black',linestyle='-')
            ax5.bar(baselines[FirstSet],RMSpdobs[FirstSet], color=basecolors[:len1])
            # ax5.bar(baselines[FirstSet],RMStrueOPD[FirstSet],fill=False,edgecolor='black',linestyle='-')
            
            ax9.bar(baselines[SecondSet],RMSgdobs[SecondSet], color=basecolors[len1:])
            # ax9.bar(baselines[SecondSet],simu.LR4[SecondSet],fill=False,edgecolor='black',linestyle='-')
            ax10.bar(baselines[SecondSet],RMSpdobs[SecondSet], color=basecolors[len1:])
            # ax10.bar(baselines[SecondSet],RMStrueOPD[SecondSet],fill=False,edgecolor='black',linestyle='-')
            
            ax1.get_shared_x_axes().join(ax1,ax2,ax3)
            ax6.get_shared_x_axes().join(ax6,ax7,ax8)
            ax4.get_shared_x_axes().join(ax4,ax5)
            ax9.get_shared_x_axes().join(ax9,ax10)
            
            ax1.get_shared_y_axes().join(ax1,ax6)
            ax2.get_shared_y_axes().join(ax2,ax7)
            ax3.get_shared_y_axes().join(ax3,ax8)
            ax4.get_shared_y_axes().join(ax4,ax9)
            ax5.get_shared_y_axes().join(ax5,ax10)
            
            ax6.tick_params(labelleft=False) ; ct.setaxelim(ax1,ydata=SNR,ymin=0)
            ax7.tick_params(labelleft=False) ; ct.setaxelim(ax2,ydata=GDmic[stationaryregim],ylim_min=[-wl/2,wl/2])
            ax8.tick_params(labelleft=False) ; ax3.set_ylim([-wl/2,wl/2])
            ax9.tick_params(labelleft=False) ; ct.setaxelim(ax4,ydata=np.concatenate([np.stack(RMSgdmic),[1]]),ymin=0)
            ax10.tick_params(labelleft=False) ; ct.setaxelim(ax5,ydata=np.concatenate([np.stack(RMSpdmic)]),ymin=0)
            
            ax4.tick_params(labelbottom=False)
            ax9.tick_params(labelbottom=False)
            
            ax1.set_ylabel('SNR')
            ax2.set_ylabel('Group-Delays [µm]')
            ax3.set_ylabel('Phase-Delays [µm]')
            ax4.set_ylabel('GD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
            ax5.set_ylabel('PD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
            
            ax11.remove() ; ax12.remove()       # These axes are here to let space for ax3 and ax8 labels
            
            ax3.set_xlabel('Time [s]', labelpad=-10) ; ax8.set_xlabel('Time [s]', labelpad=-10)
            ax5.set_xlabel('Baselines') ; ax10.set_xlabel('Baselines')
    
            ax7.legend(handles=linestyles, loc='upper right')
            if display:
                fig.show()
    
            if len(savedir):
                if verbose:
                    print("Saving perftable figure.")
                plt.savefig(savedir+f"Simulation{timestr}_{typeobs}_{rangeBases}.{ext}")

        plt.rcParams.update(plt.rcParamsDefault)



    if displayall or ('perftableres2' in args) :
        plt.rcParams.update(rcParamsForBaselines)
        generaltitle = "GD and PD residuals after least square"
        typeobs = "GDPDres2"
        
        GDobs = GDerrmic2
        PDobs = PDerrmic2
        
        RMSgdobs = np.std(GDobs[start_pd_tracking:,:],axis=0)
        RMSpdobs = np.std(PDobs[start_pd_tracking:,:],axis=0)

        linestyles=[mlines.Line2D([],[], color='black',
                                        linestyle=':', label='Start tracking')]
        if 'ThresholdGD' in config.FT.keys():
            linestyles.append(mlines.Line2D([],[], color='black',
                                        linestyle='--', label='Threshold GD'))
        
        for iFig in range(NumberOfBaseFigures):
            NINtodisplay=NINdisp
            if iFig == NumberOfBaseFigures-1:
                if (NINmes%NINdisp < NINdisp) and (NINmes%NINdisp != 0):
                    NINtodisplay = NINmes%NINdisp
                    
            iFirstBase = NINdisp*iFig   # Index of first baseline to display
            iLastBase = iFirstBase + NINtodisplay - 1        # Index of last baseline to display
            
            len2 = NINtodisplay//2 ; len1 = NINtodisplay-len2
            basecolors = colors[:len1]+colors[:len2]
            basecolors = np.array(basecolors)
            
            rangeBases = f"{baselines[iFirstBase]}-{baselines[iLastBase]}"
            title=f'{generaltitle}: {rangeBases}'
            plt.close(title)
            fig=plt.figure(title, clear=True)
            fig.suptitle(title)
            (ax1,ax6),(ax2,ax7), (ax3,ax8),(ax11,ax12),(ax4,ax9),(ax5,ax10) = fig.subplots(nrows=6,ncols=2, gridspec_kw={"height_ratios":[1,4,4,0.5,1,1]})
            ax1.set_title(f"From {baselines[iFirstBase]} \
to {baselines[iFirstBase+len1-1]}")
            ax6.set_title(f"From {baselines[iFirstBase+len1]} \
to {baselines[iLastBase]}")
            
            FirstSet = range(iFirstBase,iFirstBase+len1)
            SecondSet = range(iFirstBase+len1,iLastBase+1)
            iColor=0
            for iBase in FirstSet:   # First serie
                ax1.plot(t[timerange],SNR[timerange,iBase],color=basecolors[iColor])
                if 'ThresholdGD' in config.FT.keys():
                    ax1.hlines(config.FT['ThresholdGD'][iBase], t[timerange[0]],t[timerange[-1]], color=basecolors[iColor], linestyle='dashed')
                ax2.plot(t[timerange],GDobs[timerange,iBase],color=basecolors[iColor])
                ax2.plot(t[timerange],GDrefmic[timerange,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
                ax3.plot(t[timerange],PDobs[timerange,iBase],color=basecolors[iColor])
                ax3.plot(t[timerange],PDrefmic[timerange,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
                iColor+=1
            for iBase in SecondSet:   # Second serie
                ax6.plot(t[timerange],SNR[timerange,iBase],color=basecolors[iColor])
                if 'ThresholdGD' in config.FT.keys():
                    ax6.hlines(config.FT['ThresholdGD'][iBase],t[timerange[0]],t[timerange[-1]],color=basecolors[iColor], linestyle='dashed')
                ax7.plot(t[timerange],GDobs[timerange,iBase],color=basecolors[iColor])
                ax7.plot(t[timerange],GDrefmic[timerange,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
                ax8.plot(t[timerange],PDobs[timerange,iBase],color=basecolors[iColor])
                ax8.plot(t[timerange],PDrefmic[timerange,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
                iColor+=1
            
            ax4.bar(baselines[FirstSet],RMSgdobs[FirstSet], color=basecolors[:len1])
            # ax4.bar(baselines[FirstSet],simu.LR4[FirstSet],fill=False,edgecolor='black',linestyle='-')
            ax5.bar(baselines[FirstSet],RMSpdobs[FirstSet], color=basecolors[:len1])
            # ax5.bar(baselines[FirstSet],RMStrueOPD[FirstSet],fill=False,edgecolor='black',linestyle='-')
            
            ax9.bar(baselines[SecondSet],RMSgdobs[SecondSet], color=basecolors[len1:])
            # ax9.bar(baselines[SecondSet],simu.LR4[SecondSet],fill=False,edgecolor='black',linestyle='-')
            ax10.bar(baselines[SecondSet],RMSpdobs[SecondSet], color=basecolors[len1:])
            # ax10.bar(baselines[SecondSet],RMStrueOPD[SecondSet],fill=False,edgecolor='black',linestyle='-')
            
            ax1.get_shared_x_axes().join(ax1,ax2,ax3)
            ax6.get_shared_x_axes().join(ax6,ax7,ax8)
            ax4.get_shared_x_axes().join(ax4,ax5)
            ax9.get_shared_x_axes().join(ax9,ax10)
            
            ax1.get_shared_y_axes().join(ax1,ax6)
            ax2.get_shared_y_axes().join(ax2,ax7)
            ax3.get_shared_y_axes().join(ax3,ax8)
            ax4.get_shared_y_axes().join(ax4,ax9)
            ax5.get_shared_y_axes().join(ax5,ax10)
            
            ax6.tick_params(labelleft=False) ; ct.setaxelim(ax1,ydata=SNR,ymin=0)
            ax7.tick_params(labelleft=False) ; ct.setaxelim(ax2,ydata=GDmic[stationaryregim],ylim_min=[-wl/2,wl/2])
            ax8.tick_params(labelleft=False) ; ax3.set_ylim([-wl/2,wl/2])
            ax9.tick_params(labelleft=False) ; ct.setaxelim(ax4,ydata=np.concatenate([np.stack(RMSgdmic),[1]]),ymin=0)
            ax10.tick_params(labelleft=False) ; ct.setaxelim(ax5,ydata=np.concatenate([np.stack(RMSpdmic)]),ymin=0)
            
            ax4.tick_params(labelbottom=False)
            ax9.tick_params(labelbottom=False)
            
            ax1.set_ylabel('SNR')
            ax2.set_ylabel('Group-Delays [µm]')
            ax3.set_ylabel('Phase-Delays [µm]')
            ax4.set_ylabel('GD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
            ax5.set_ylabel('PD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
            
            ax11.remove() ; ax12.remove()       # These axes are here to let space for ax3 and ax8 labels
            
            ax3.set_xlabel('Time [s]', labelpad=-10) ; ax8.set_xlabel('Time [s]', labelpad=-10)
            ax5.set_xlabel('Baselines') ; ax10.set_xlabel('Baselines')
    
            ax7.legend(handles=linestyles, loc='upper right')
            if display:
                fig.show()
    
            if len(savedir):
                if verbose:
                    print("Saving perftable figure.")
                plt.savefig(savedir+f"Simulation{timestr}_{typeobs}_{rangeBases}.{ext}")

        plt.rcParams.update(plt.rcParamsDefault)


    if displayall or ('perfarray' in args):
        plt.rcParams.update(rcParamsForBaselines)
        
        from .tol_colors import tol_cmap as tc
        import matplotlib as mpl
        
        
        
        #visibilities, _,_,_=ct.VanCittert(WLOfScience,config.Obs,config.Target)
        #simu.VisibilityAtPerfWL = visibilities
        visibilities = ct.NB2NIN(simu.VisibilityObject[WLIndex])
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
        ax1.set_title(f"Target visibility and photometric balance ({wl:.3}µm)")
        ax2.set_title(f"Fringe contrast and Time on central fringe ({WLOfScience:.3}µm)")
        
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
                ls = (0,(10*simu.LR4[ib],np.max([0,10*(1-simu.LR4[ib])])))
                if PhotometricBalance[ib]>0:
                    im=ax2.plot([x1,x2],[y1,y2],linestyle=ls,
                            linewidth=3,
                            color=cm(int(simu.FringeContrast[ib]*Nnuances)))
                else:
                    im=ax2.plot([x1,x2],[y1,y2],linestyle=ls,
                            linewidth=1,
                            color=cm(int(simu.FringeContrast[ib]*Nnuances)))
        ax2.set_xlabel("X [m]")
        ax2.set_ylabel("Y [m]")
        ax2.set_xlim([-210,160]) ; ax2.set_ylim([-50,350])
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # mpl.colorbar.ColorbarBase(cbar_ax, cmap=cm,
        #                           orientation='vertical',
        #                           label=f"Fringe Contrast at {WLOfScience:.3}µm")


        
        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.1, 0.05, 0.85, 0.05])
        mpl.colorbar.ColorbarBase(cbar_ax, cmap=cm,
                                  orientation='horizontal')

        if len(savedir):
            if verbose:
                print("Saving perfarray figure.")
            plt.savefig(savedir+f"Simulation{timestr}_perfarray.{ext}")

    plt.rcParams.update(plt.rcParamsDefault)

        
    if displayall or ('opd' in args):
        plt.rcParams.update(rcParamsForBaselines)
        generaltitle = 'True OPD'
        linestyles=[mlines.Line2D([],[], color='black',
                                        linestyle=':', label='Start tracking')]
        
        for iFig in range(NumberOfBaseFiguresNIN):
            NINtodisplay=NINdisp
            if iFig == NumberOfBaseFiguresNIN-1:
                if (NIN%NINdisp < NINdisp) and (NIN%NINdisp != 0):
                    NINtodisplay = NIN%NINdisp
                    
            iFirstBase = NINdisp*iFig   # Index of first baseline to display
            iLastBase = iFirstBase + NINtodisplay - 1        # Index of last baseline to display
            
            len2 = NINtodisplay//2 ; len1 = NINtodisplay-len2
            basecolors = colors[:len1]+colors[:len2]
            basecolors = np.array(basecolors)
            
            rangeBases = f"{baselinesNIN[iFirstBase]}-{baselinesNIN[iLastBase]}"
            title=f'{generaltitle}: {rangeBases}'
    
            plt.close(title)
            fig=plt.figure(title, clear=True)
            fig.suptitle(title)
            (ax1,ax6),(ax2,ax7),(ax3,ax8),(ax4,ax9),(ax5,ax10)=fig.subplots(nrows=5,ncols=2, gridspec_kw={"height_ratios":[4,.5,1,.2,1]})
            ax1.set_title(f"From {baselinesNIN[iFirstBase]} \
to {baselinesNIN[iFirstBase+len1-1]}")
            ax6.set_title(f"From {baselinesNIN[iFirstBase+len1]} \
to {baselinesNIN[iLastBase]}")
            
            FirstSet = range(iFirstBase,iFirstBase+len1)
            SecondSet = range(iFirstBase+len1,iLastBase+1)
            iColor = 0
            for iBase in FirstSet:  # First serie
                ax1.plot(t[timerange],simu.OPDTrue[timerange,iBase],color=basecolors[iColor])
                iColor+=1
                
            for iBase in SecondSet:   # Second serie
                ax6.plot(t[timerange],simu.OPDTrue[timerange,iBase],color=basecolors[iColor])
                iColor+=1
                
            ax1.vlines(config.starttracking*dt*ms,-3*np.max(np.abs(simu.OPDTrue)),3*np.max(np.abs(simu.OPDTrue)),
                       color='k', linestyle=':')
            ax6.vlines(config.starttracking*dt*ms,-3*np.max(np.abs(simu.OPDTrue)),3*np.max(np.abs(simu.OPDTrue)),
                       color='k', linestyle=':')
            
            ax3.bar(baselinesNIN[FirstSet],RMStrueOPD[FirstSet], color=basecolors[:len1])
            ax5.bar(baselinesNIN[FirstSet],simu.LR4[FirstSet], color=basecolors[:len1])
            ax5.bar(baselinesNIN[FirstSet],np.abs(VisObj[FirstSet])**2, fill=False, edgecolor='black', linestyle='-',linewidth=1.5)
            
            ax8.bar(baselinesNIN[SecondSet],RMStrueOPD[SecondSet], color=basecolors[len1:])
            ax10.bar(baselinesNIN[SecondSet],simu.LR4[SecondSet], color=basecolors[len1:])
            ax10.bar(baselinesNIN[SecondSet],np.abs(VisObj[SecondSet])**2, fill=False, edgecolor='black', linestyle='-', linewidth=1.5)
    
            ax1.get_shared_x_axes().join(ax1,ax6)
            ax3.get_shared_x_axes().join(ax3,ax5)
            ax8.get_shared_x_axes().join(ax8,ax10)
            
            ax1.get_shared_y_axes().join(ax1,ax6)
            ax3.get_shared_y_axes().join(ax3,ax8)
            ax5.get_shared_y_axes().join(ax5,ax10)
            
            ax6.tick_params(labelleft=False)
            ax8.tick_params(labelleft=False)
            ax10.tick_params(labelleft=False)
            ax3.tick_params(labelbottom=False) ; ax8.tick_params(labelbottom=False)
            
            ax2.remove() ; ax4.remove();ax7.remove();ax9.remove()
            
            ax1.set_ylabel('OPD [µm]')
            ax3.set_ylabel('$\sigma_{OPD}$\n[µm]',rotation=1,labelpad=40,loc='bottom')
            ax5.set_ylabel('Lock\nratio\n|V|²',rotation=1,labelpad=40, loc='bottom')
            
            #ax11.remove() ; ax12.remove()       # These axes are here to let space for ax3 and ax8 labels
            
            ax1.set_xlabel('Time [s]') ; ax6.set_xlabel('Time [s]')
            ax5.set_xlabel('Baselines') ; ax10.set_xlabel('Baselines')
    
            ct.setaxelim(ax1,ydata=simu.OPDTrue,ylim_min=[-wl/2,wl/2])        
            ct.setaxelim(ax3,ydata=RMStrueOPD,ymin=0)
            ax5.set_ylim(0,1.1) ; ax5.grid(True) ; ax10.grid(True)
    
            if display:
                fig.show()
    
            if len(savedir):
                if verbose:
                    print("Saving opd figure.")
                plt.savefig(savedir+f"Simulation{timestr}_opd_{rangeBases}.{ext}")
            
    """
    OPD
    """
    
    if displayall or ("opd4" in args):
        plt.rcParams.update(rcParamsForBaselines)
        generaltitle = "OPD"
        title=f"{generaltitle} - {infos['details']}"
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        
        OneAxe = False ; NumberOfBaselinesToPlot = np.sum(PlotBaseline)
        if NumberOfBaselinesToPlot < len1:
            OneAxe=True
            ax1,ax2 = fig.subplots(nrows=2,gridspec_kw={"height_ratios":[3,1]})
        else:    
            #(ax1,ax3),(ax2,ax4),(ax5,ax6) = fig.subplots(nrows=3,ncols=2, sharey='row', gridspec_kw={'height_ratios':[4,4,1]})
            (ax1,ax6),(ax2,ax7),(ax3,ax8),(ax4,ax9),(ax5,ax10)=fig.subplots(nrows=5,ncols=2, gridspec_kw={"height_ratios":[4,.5,1,.2,1]})
            ax1.set_title("First serie of baselines, from 12 to 25")
            ax3.set_title("Second serie of baselines, from 26 to 56")
        
        if OneAxe:
            baselinestemp = [baselines[iBase] for iBase in PlotBaselineIndex]
            basecolorstemp = basecolors[:NumberOfBaselinesToPlot]
            baselinestyles=['-']*len1 + ['--']*len2
            baselinehatches=['']*len1 + ['/']*len2
            
            k=0
            for iBase in PlotBaselineIndex:   # First serie
                ax1.plot(t[timerange],simu.OPDTrue[timerange,iBase],color=basecolorstemp[k],label=baselines[iBase])
                # barbasecolors[iBase] = basecolorstemp[k]
                k+=1                
            
            p2=ax2.bar(baselinestemp,[RMStrueOPD[iBase] for iBase in PlotBaselineIndex], color=basecolorstemp)
            
            ax1.legend()
    
        else:
            for iBase in range(len1):   # First serie
                if PlotBaseline[iBase]:
                    ax1.plot(t[timerange],simu.OPDTrue[timerange,iBase],color=basecolors[iBase])
    
            for iBase in range(len1,NIN):   # Second serie
                if PlotBaseline[iBase]:
                    ax3.plot(t[timerange],simu.OPDTrue[timerange,iBase],color=basecolors[iBase])
    
            p1=ax5.bar(baselines[:len1],RMStrueOPD[:len1], color=basecolors[:len1])
            p2=ax6.bar(baselines[len1:],RMStrueOPD[len1:], color=basecolors[len1:])
            # ax4.sharey(ax2) ; ax4.tick_params(labelleft=False)
            # ax4.set_ylim(ylimGD)
            ct.setaxelim(ax1, ydata=simu.OPDTrue, ymargin=0.4,ymin=0)
            
            ax5.set_ylabel('OPD rms\n[µm]') ;
            ax5.bar_label(p1,label_type='edge',fmt='%.2f')
            ax6.bar_label(p2,label_type='edge',fmt='%.2f')
            ax5.set_anchor('S') ; ax6.set_anchor('S')
            ax5.set_box_aspect(1/15) ; ax6.set_box_aspect(1/15)
            
        ct.setaxelim(ax1,ydata=[simu.OPDTrue[timerange,iBase] for iBase in PlotBaselineIndex])
        ct.setaxelim(ax2,ydata=list(RMStrueOPD)+[wl/2], ymin=0)
        ax1.set_ylabel('True OPD [µm]')
        ax1.set_xlabel("Time [s]")#, labelpad=xlabelpad) ; ax4.set_xlabel("Time (s)", labelpad=xlabelpad)
        ax2.set_ylabel('RMS [µm]')
        ax2.set_xlabel("Baselines")#, labelpad=xlabelpad) ; ax4.set_xlabel("Time (s)", labelpad=xlabelpad)
            
        ax2.set_box_aspect(1/20)
    
        if display:
            if pause:
                plt.pause(0.1)
            else:
                plt.show()  
        if len(savedir):
            fig.savefig(savedir+f"Simulation{timestr}_opd.{ext}")

    
    if displayall or ('opdcontrol' in args):
    
        linestyles=[mlines.Line2D([],[], color='black',
                                        linestyle=':', label='Start tracking')]
        
        plt.rcParams.update(rcParamsForBaselines)
        title='True OPD'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        (ax1,ax6),(ax2,ax7),(ax3,ax8),(ax4,ax9),(ax5,ax10)=fig.subplots(nrows=5,ncols=2, gridspec_kw={"height_ratios":[4,.5,1,.2,1]})
        ax1.set_title("First serie of baselines, from 12 to 25")
        ax6.set_title("Second serie of baselines, from 26 to 56")
        
        for iBase in range(len1):   # First serie
            ax1.plot(t[timerange],simu.OPDTrue[timerange,iBase],color=basecolors[iBase])

        for iBase in range(len1,NIN):   # Second serie
            ax6.plot(t[timerange],simu.OPDTrue[timerange,iBase],color=basecolors[iBase])
        
        ax1.vlines(config.starttracking*dt,-3*np.max(np.abs(simu.OPDTrue)),3*np.max(np.abs(simu.OPDTrue)),
                   color='k', linestyle=':')
        ax6.vlines(config.starttracking*dt,-3*np.max(np.abs(simu.OPDTrue)),3*np.max(np.abs(simu.OPDTrue)),
                   color='k', linestyle=':')
        
        # Histogram of OPD rms (colored bars) and visibilities (black lines)
        ax3.bar(baselines[:len1],RMStrueOPD[:len1], color=basecolors[:len1])
        ax5.bar(baselines[:len1],np.abs(VisObj[:len1]), color=basecolors[:len1])
        
        ax8.bar(baselines[len1:],RMStrueOPD[len1:], color=basecolors[len1:])
        ax10.bar(baselines[len1:],np.abs(VisObj[len1:]), color=basecolors[len1:])
        
        ax1.get_shared_x_axes().join(ax1,ax6)
        ax3.get_shared_x_axes().join(ax3,ax5)
        ax8.get_shared_x_axes().join(ax8,ax10)
        
        ax1.get_shared_y_axes().join(ax1,ax6)
        ax3.get_shared_y_axes().join(ax3,ax8)
        ax5.get_shared_y_axes().join(ax5,ax10)
        
        ax6.tick_params(labelleft=False)
        ax8.tick_params(labelleft=False)
        ax10.tick_params(labelleft=False)
        ax3.tick_params(labelbottom=False) ; ax8.tick_params(labelbottom=False)
        
        ax2.remove() ; ax4.remove();ax7.remove();ax9.remove()
        
        ax1.set_ylabel('OPD [µm]')
        ax3.set_ylabel('$\sigma_{OPD}$\n[µm]',rotation=1,labelpad=40,loc='bottom')
        ax5.set_ylabel('|V|',rotation=1,labelpad=40, loc='bottom')
        
        #ax11.remove() ; ax12.remove()       # These axes are here to let space for ax3 and ax8 labels
        
        ax1.set_xlabel('Time [ms]') ; ax6.set_xlabel('Time [ms]')
        ax5.set_xlabel('Baselines') ; ax10.set_xlabel('Baselines')

        ct.setaxelim(ax1,ydata=simu.OPDTrue,ylim_min=[-wl/2,wl/2])        
        ct.setaxelim(ax3,ydata=RMStrueOPD,ymin=0)
        ax5.set_ylim(0,1.1) ; ax5.grid(True) ; ax10.grid(True)

        if display:
            fig.show()

        if len(savedir):
            if verbose:
                print("Saving opdcontrol figure.")
            plt.savefig(savedir+f"Simulation{timestr}_opdcontrol.{ext}")
    
    
    
    
    if 'opd2' in args:
        """
        OPD
        """
        
        OPD_max = 1.1*np.max([np.max(np.abs([simu.OPDDisturbance,
                              simu.OPDTrue,
                              simu.OPDCommand[:-1],simu.EffectiveOPDMove[:-config.latency]])),wl/2])
        OPD_min = -OPD_max
        ylim = [OPD_min,OPD_max]
    
        linestyles=[]
        linestyles.append(mlines.Line2D([], [], color='black',
                                        linestyle='solid',label='Residual'))    
        linestyles.append(mlines.Line2D([], [], color='black',
                                        linestyle='--',label='Disturbance'))
        linestyles.append(mlines.Line2D([], [], color='black',
                                        linestyle=':',label='Command'))
        linestyles.append(mlines.Line2D([], [], color='black',
                                        linestyle='-.',label='Effective Move ODL'))
        linestyles.append(mlines.Line2D([],[], color='red',
                                        linestyle=':', label='Start tracking'))
    
        linestyles2=[]
        
        NumberOfBaselinesToShow = np.min([NIN, NA-1])
        ShownBaselines = np.arange(NumberOfBaselinesToShow)
        
        for ia in range(NumberOfBaselinesToShow):
            
            BaselinesIndices=[]
            for iap in range(NA):
                if iap < ia:
                    BaselinesIndices.append(ct.posk(iap,ia,NA))
                elif ia < iap:
                    BaselinesIndices.append(ct.posk(iap,ia,NA))
            
            fig = plt.figure(f"OPD {ia+1}")

            axes = fig.subplots(nrows=NumberOfBaselinesToShow,ncols=2,sharex=True, gridspec_kw={'width_ratios': [4, 1]})
            iap,iax=0,0
            
            gs = axes[0, 0].get_gridspec()
            # remove the all axes of first column
            for ax in axes[:, 0]:
                ax.remove()
                
            # Add a unique axe on the first column
            ax1 = fig.add_subplot(gs[:, 0])
            ax2 = ax1.twinx()
            ax2ymax = 1.1*np.max([np.max(np.abs(simu.OPDTrue[stationaryregim][:,BaselinesIndices])),wl/2])
            ax2ylim = [-ax2ymax,ax2ymax]
            
            ax1.vlines(config.starttracking*dt,ylim[0],ylim[1],
               color='red', linestyle=':')
            #ax1.annotate('Big Axes \nGridSpec[1:, -1]', (0.1, 0.5),
            #               xycoords='axes fraction', va='center')
            
            if np.ndim(axes)==1:
                axes = [(axes[0], axes[1])]
                
            k=0
            for iap in range(NA):
                
                if iap<ia:
                    linestyles2.append(mlines.Line2D([],[],color=colors[iap],label=f'Baseline {iap+1}{ia+1}'))
                    ib = ct.posk(iap,ia,NA)
                    ax1.plot(timestamps, -simu.OPDDisturbance[:,ib],
                             color=colors[iap],linestyle="--")
                    ax1.plot(timestamps, -simu.OPDCommand[:-1,ib],
                            color=colors[iap],linestyle=':')   
                    ax1.plot(timestamps, -simu.EffectiveOPDMove[:-config.latency,ib],
                            color=colors[iap],linestyle='-.')
                    ax1.plot(timestamps, -simu.OPDTrue[:,ib],
                             color=colors[iap],linestyle='solid')
                    
                elif iap == ia:
                    continue
                
                else: # ia<iap
                    linestyles2.append(mlines.Line2D([],[],color=colors[iap],label=f'Baseline {ia+1}{iap+1}'))
                    ib = ct.posk(ia,iap,NA)
                    ax1.plot(timestamps, simu.OPDDisturbance[:,ib],
                             color=colors[iap],linestyle="--")
                    ax1.plot(timestamps, simu.OPDCommand[:-1,ib],
                             color=colors[iap],linestyle=':')
                    ax1.plot(timestamps, simu.EffectiveOPDMove[:-config.latency,ib],
                            color=colors[iap],linestyle='-.')
                    ax1.plot(timestamps, simu.OPDTrue[:,ib],
                             color=colors[iap],linestyle='solid')
                
                
                
                axes[0,1].text(0,0.9-0.2*k,f"$\sigma_{{{ia+1}{iap+1}}}={np.sqrt(simu.VarOPD[ib])*1e3:.0f}$nm RMS")
                axes[k,1].axis("off") ; k+=1
                
            ax1.set_ylim(ylim)
            ax2.set_ylim(ax2ylim)
            if ax2ymax > wl:
                ax2.set_yticks([-ax2ylim[0],-wl,0,wl,ax2ylim[1]])
            else:
                ax2.set_yticks([-wl,0,wl])
            ax1.set_ylabel(f'OPD {ia+1}# \n [µm]')
            ax2.set_ylabel('Residual \n [µm]')
            
            fig.tight_layout()
            # wlr = round(wl,2)
            # ax2.set_yticks([-wl,0,wl])
            # ax2.set_yticklabels([-wlr,0,wlr])
            ax2.tick_params(axis='y',which='major', length=7)
            ax2.tick_params(axis='y',which='minor', length=4)
            ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax2.grid(b=True,which='major')
            ax2.grid(b=True, which='minor')
                
            # plt.tight_layout()
            ax.set_xlabel('Time (ms)')
            axes[-2,1].legend(handles=linestyles,loc='lower right')
            axes[-1,1].legend(handles=linestyles2,loc='lower right')
            config.newfig+=1
            
            if display:
                plt.show()
            if OneTelescope:
                break
            
            
        if len(savedir):
            if verbose:
                print("Saving opd figure.")
            plt.savefig(savedir+f"Simulation{timestr}_opd.{ext}")

    
    if 'OPDcmd' in args:
        OPD_max = 1.1*np.max(np.abs([simu.OPDDisturbance,
                              simu.GDCommand[:-1,:],simu.OPDCommand[:-1,:]]))
        OPD_min = -OPD_max
        ylim = [OPD_min,OPD_max]
    
        linestyles=[]
        linestyles.append(mlines.Line2D([], [],label='Disturbance',
                                        color=colors[0],linestyle='-'))
        linestyles.append(mlines.Line2D([], [],label='Total command',
                                        color=colors[1],linestyle='-'))
        linestyles.append(mlines.Line2D([], [],label='PD command',
                                        color=colors[2],linestyle='-'))
        linestyles.append(mlines.Line2D([], [],label='GD command',
                                        color=colors[3],linestyle='-'))
        linestyles.append(mlines.Line2D([], [],label='Search command',
                                        color=colors[4],linestyle='-'))
        linestyles.append(mlines.Line2D([], [],label='Effective Move ODL',
                                        color=colors[5],linestyle='-'))
        linestyles.append(mlines.Line2D([],[], color='black',
                                        linestyle=':', label='Start tracking'))
        
        for ia in range(NA):
            fig = plt.figure(f"OPD commands {ia+1}")
            fig.suptitle(f"OPD evolution at {wl:.2f}µm for baselines \n\
    including telescope {ia+1}")
            axes = fig.subplots(nrows=NA-1,ncols=3,sharex=True,gridspec_kw={'width_ratios': [8, 1,1]})
            iap,iax=0,0
            for ax,axText,axLegend in axes:
                ax2 = ax.twinx()
                ax2ymax = 1.1*np.max(np.abs(simu.GDEstimated*config.FS['R']/config.FT['Ncross']*wl/(2*np.pi)))
                ax2ylim = [-ax2ymax,ax2ymax]
                if iap == ia:
                    iap+=1
                if ia < iap:
                    ib = ct.posk(ia,iap,NA)
                    ax.plot(timestamps, simu.OPDDisturbance[:,ib],
                            color=colors[0])
                    ax.plot(timestamps, simu.OPDCommand[:-1,ib],
                            color=colors[1])
                    ax.plot(timestamps, simu.PDCommand[:-1,ib],
                            color=colors[2])
                    ax.plot(timestamps, simu.GDCommand[:-1,ib],
                            color=colors[3])
                    ax.plot(timestamps, simu.OPDSearchCommand[:-1,ib],
                            color=colors[4])
                    ax.plot(timestamps, -simu.EffectiveOPDMove[:-config.latency,ib],
                            color=colors[5])

                else:
                    ib = ct.posk(iap,ia,NA)
                    ax2.plot(timestamps, -simu.OPDDisturbance[:,ib],
                            color=colors[0])
                    ax2.plot(timestamps, -simu.OPDCommand[:-1,ib],
                            color=colors[1])
                    ax2.plot(timestamps, -simu.PDCommand[:-1,ib],
                            color=colors[2])
                    ax2.plot(timestamps, -simu.GDCommand[:-1,ib],
                            color=colors[3])
                    ax2.plot(timestamps, -simu.OPDSearchCommand[:-1,ib],
                            color=colors[4])
                    ax2.plot(timestamps, -simu.EffectiveOPDMove[:-config.latency,ib],
                            color=colors[5])
                
                ax.vlines(config.starttracking*dt,ylim[0],ylim[1],
                   color='k', linestyle=':')

                ax.set_ylim(ylim)
                ax.set_ylabel(f'[{ia+1},{iap+1}] [µm]')
                
                ax2.set_ylim(ax2ylim)
                ax2.set_ylabel('OPD [µm]')
                ax2.set_yticks([-wl,0,wl])

                axText.text(0,0.70,f"PD:{np.std(simu.PDEstimated[stationaryregim,ib]*wl/(2*np.pi)*1e3):.0f}nm RMS")
                axText.text(0,0.30,f"GD:{np.std(simu.GDEstimated[stationaryregim,ib]*wl/(2*np.pi)*1e3):.0f}nm RMS")
                # axText.axis("off")
                # axLegend.axis("off")
                wlr = round(wl,2)
                ax2.set_yticklabels([-wlr,0,wlr])
                ax2.tick_params(axis='y',which='major', length=7)
                ax2.tick_params(axis='y',which='minor', length=4)
                ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax2.grid(b=True,which='major')
                ax2.grid(b=True, which='minor')
                
                iap += 1
                iax+=1
                # ax2.minorticks_on()
            
                
            plt.xlabel('Time (ms)')
            if display:
                plt.show()
            axLegend.legend(handles=linestyles)
            config.newfig+=1
    
            if OneTelescope:
                break

    
    
    if displayall or ('OPDdetails' in args):
        OPD_max = 1.1*np.max(np.abs([simu.OPDDisturbance,
                              simu.GDCommand[:-config.latency,:]]))
        OPD_min = -OPD_max
        ylim = [OPD_min,OPD_max]
        
        linestyles=[]
        linestyles.append(mlines.Line2D([], [],label='Disturbance',
                                        color='red',linestyle='solid'))
        linestyles.append(mlines.Line2D([], [], color='green',
                                    linestyle='dotted',label='GD Command'))
        linestyles.append(mlines.Line2D([], [],label='GD Residuals',
                                        color='black',linestyle='-.'))
        linestyles.append(mlines.Line2D([], [],label='PD Residuals',
                                        color='black',linestyle='-'))
        linestyles.append(mlines.Line2D([],[], color='black',
                                        linestyle=':', label='Start tracking'))

        for ia in range(NA):
            fig = plt.figure(f"OPD details {ia+1}")
            fig.suptitle(f"OPD evolution at {wl:.2f}µm for baselines \n\
    including telescope {ia+1}")
            axes = fig.subplots(nrows=NA-1,ncols=3,sharex=True,gridspec_kw={'width_ratios': [8, 1,1]})
            iap,iax=0,0
            for ax,axText,axLegend in axes:
                ax2 = ax.twinx()
                ax2ymax = 1.1*np.max(np.abs(simu.GDEstimated[stationaryregim,:]*config.FS['R']/config.FT['Ncross']*wl/(2*np.pi)))
                ax2ylim = [-ax2ymax,ax2ymax]
                if iap == ia:
                    iap+=1
                if ia < iap:
                    ib = ct.posk(ia,iap,NA)
                    ax.plot(timestamps, simu.OPDDisturbance[:,ib],
                            color='red')
                    ax2.plot(timestamps, simu.GDCommand[:-config.latency,ib],
                            color='green',linestyle='dotted')
                    ax2.plot(timestamps, simu.GDResidual[:,ib]*wl/(2*np.pi),
                             color='black',linestyle='-.')
                    ax2.plot(timestamps, simu.PDResidual[:,ib]*wl/(2*np.pi),
                             color='black',linestyle='-')
                else:
                    ib = ct.posk(iap,ia,NA)
                    ax.plot(timestamps, -simu.OPDDisturbance[:,ib],color='red')
                    ax.plot(timestamps, -simu.GDCommand[:-config.latency,ib],
                            color='green',linestyle='dotted')
                    ax2.plot(timestamps, -simu.GDResidual[:,ib]*wl/(2*np.pi), color='black',
                             linestyle='-.')
                    ax2.plot(timestamps, -simu.PDResidual[:,ib]*wl/(2*np.pi), color='black',
                             linestyle='-')
                
                ax.vlines(config.starttracking*dt,ylim[0],ylim[1],
                          color='k', linestyle=':')
                
                ax.set_ylim(ylim)
                ax.set_ylabel(f'[{ia+1},{iap+1}] [µm]')
                
                ax2.set_ylim(ax2ylim)
                ax2.set_ylabel('Residuals')
                if ax2ymax > wl:
                    ax2.set_yticks([-ax2ylim[0],-wl,0,wl,ax2ylim[1]])
                else:
                    ax2.set_yticks([-wl,0,wl])

                axText.text(0,0.70,f"PD:{np.std(simu.PDEstimated[stationaryregim,ib]*wl/(2*np.pi)*1e3):.0f}nm RMS")
                axText.text(0,0.30,f"GD:{np.std(simu.GDEstimated[stationaryregim,ib]*wl/(2*np.pi)*1e3):.0f}nm RMS")
                axText.axis("off")
                axLegend.axis("off")
                # wlr = round(wl,2)
                # ax2.set_yticklabels([-wlr,0,wlr])
                # ax2.tick_params(axis='y',which='major', length=7)
                # ax2.tick_params(axis='y',which='minor', length=4)
                # ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
                # ax2.grid(b=True,which='major')
                # ax2.grid(b=True, which='minor')
                
                iap += 1
                iax+=1
                # ax2.minorticks_on()
            fig.tight_layout()
            
            plt.xlabel('Time (ms)')
            # plt.show()
            axLegend.legend(handles=linestyles)
            config.newfig+=1
    
            if display:
                plt.show()
            if OneTelescope:
                break
    

    if 'OPDgathered' in args:

        OPD_max = 1.1*np.max(np.abs(simu.OPDTrue))
        OPD_min = -OPD_max
        s = [OPD_min,OPD_max]

        fig = plt.figure("OPD on one window")
        ax = fig.subplots(nrows=1)
        
        for ib in range(config.NIN):
            
            ax.plot(timestamps, simu.OPDTrue[:,ib], linestyle='-', label=f'{ich[ib]}: {np.sqrt(simu.VarOPD[ib])*1e3:.0f}nm RMS')
            
        ax.vlines(config.starttracking*dt,s[0],s[1],
               color='k', linestyle=':')
        ax.set_ylim(s)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('OPD [µm]')
        if display:
            plt.show()
        ax.legend()
        config.newfig+=1



    if displayall or ('cp' in args):
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
                
                ax1.plot(timestamps, simu.ClosurePhasePD[:,ic],
                         color=colors[ic])
                ax1.plot(timestamps, simu.ClosurePhaseGD[:,ic],
                         color=colors[ic],linestyle=':')
                ax1.hlines(simu.ClosurePhaseObject[WLIndex,ic], 0, timestamps[-1], 
                           color=colors[ic], linestyle='--')
                linestyles1.append(mlines.Line2D([],[], color=colors[ic],
                                                linestyle='-', label=f'{1}{ia+1}{iap+1}'))
                
        # Plot on ax2 the (NA-1)(NA-2)/2 (independant?) other Closure Phases
        for ia in range(1,NA):
            for iap in range(ia+1,NA):
                for iapp in range(iap+1,NA):
                    ic = ct.poskfai(ia,iap,iapp,NA)
                    colorindex = int(ic - config.NC//2)
                    ax2.plot(timestamps, simu.ClosurePhasePD[:,ic],
                             color=colors[colorindex])
                    ax2.plot(timestamps, simu.ClosurePhaseGD[:,ic],
                             color=colors[colorindex],linestyle=':')
                    ax2.hlines(simu.ClosurePhaseObject[WLIndex,ic], 0, timestamps[-1],
                               color=colors[colorindex], linestyle='--')
                    linestyles2.append(mlines.Line2D([],[], color=colors[colorindex],
                                                    linestyle='-', label=f'{ia+1}{iap+1}{iapp+1}'))
        
        ax1.vlines(config.starttracking*dt,ylim[0],ylim[1],
                   color='k', linestyle=':')
        # ax2.vlines(config.starttracking*dt,ylim[0],ylim[1],
        #            color='k', linestyle=':')
        plt.xlabel('Time [ms]')
        plt.ylabel('Closure Phase [rad]')
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)
        ax1.legend(handles=linestyles)
        ax3.legend(handles=linestyles1,loc='upper left') ; ax3.axis('off')
        ax4.legend(handles=linestyles2,loc='upper left') ; ax4.axis('off')
        
        if display:
            plt.show()
        config.newfig+=1
        
        if len(savedir):
            fig.savefig(savedir+f"Simulation{timestr}_cp.{ext}")

    if displayall or ('vis' in args):
        """
        VISIBILITIES
        """
    
        ylim =[0,1.1]
        # Squared Visibilities
        for ia in range(NA):
            fig = plt.figure(f"Squared Vis {ia+1}")
            fig.suptitle(f"Squared visibility |V|² at {wl:.2f}µm for baselines \n\
    including telescope {ia+1}")
            axes = fig.subplots(nrows=NA-1,ncols=1,sharex=True)
            iap=0
            for ax in axes:
                if iap == ia:
                    iap+=1
                
                ib = ct.posk(ia,iap,NA)
                ax.plot(timestamps, np.abs(simu.VisibilityEstimated[:,WLIndex,ib]), color='k')
                ax.plot(timestamps, np.abs(simu.VisibilityTrue[:,WLIndex,ib]),color='k',linestyle='--')
                ax.set_ylim(ylim)
                ax.set_ylabel(f'[{ia+1},{iap+1}] [µm]')
                iap += 1
                ax.vlines(config.starttracking*dt,ylim[0],ylim[1],
                       color='k', linestyle=':')
            plt.xlabel('Time (ms)')
            if display:
                plt.show()
            config.newfig+=1
            
            if OneTelescope:
                break
    
    
        # Phase of the visibilities
        ymax = np.pi #2*np.max(np.abs(VisibilityPhase))
        ylim = [-ymax,ymax]
        for ia in range(NA):
            fig = plt.figure(f"Phase Vis {ia+1}")
            fig.suptitle(f"Visibility phase \u03C6 at {wl:.2f}µm for baselines \n\
    including telescope {ia+1}")
            axes = fig.subplots(nrows=NA-1,ncols=1,sharex=True)
            iap=0
            for iax in range(len(axes)):
                ax = axes[iax]
                if iap == ia:
                    iap+=1
                
                ib = ct.posk(ia,iap,NA)
                ax.plot(timestamps, np.angle(simu.VisibilityEstimated[:,WLIndex,ib]), color='k')
                ax.plot(timestamps, np.angle(simu.VisibilityTrue[:,WLIndex,ib]),color='k',linestyle='--')
                ax.set_ylim(ylim)
                ax.set_ylabel(f'[{ia+1},{iap+1}] [µm]')
                
                iap += 1
                fig.subplots_adjust(right=0.8)
                RMS_ax = fig.add_axes([0.82, 1-1/NA*(iax+1), 0.1, 0.9/NA])
                RMS_ax.text(0,0,f"{np.std(VisibilityPhase[stationaryregim,ib])/(2*np.pi):.2f}\u03BB RMS")
                RMS_ax.axis("off")
                ax.vlines(config.starttracking*dt,ylim[0],ylim[1],
                       color='k', linestyle=':')
            plt.xlabel('Time (ms)')
            if display:
                plt.show()
            config.newfig+=1
    
            if OneTelescope:
                break

    if displayall or ('detector' in args):
        """
        DETECTOR VIEW
        """
        title="Detector&SNR"
        fig = plt.figure(title,clear=True)
        axes = fig.subplots()
        plt.suptitle(f'Sequence of intensities on pixels corresponding to {wl:.2f}µm')

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
                im = plt.imshow(np.transpose(np.dot(np.reshape(simu.MacroImages[:,WLIndex,ip],[NT,1]), \
                                                    np.ones([1,100]))), vmin=np.min(simu.MacroImages), vmax=np.max(simu.MacroImages))    
                    
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
            #     realimage[:,posp[ia]] = simu.MacroImages[:,:,ia]
            
            # realimage[:,:,posi_center//p-(NP-NA)//2:posi_center//p+(NP-NA)//2] = simu.MacroImages[:,:,NA:]
            ax = plt.subplot()
            im = plt.imshow(simu.MacroImages[:,WLIndex,:], vmin=np.min(simu.MacroImages), vmax=np.max(simu.MacroImages))
            
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
        if len(savedir):
            fig.savefig(savedir+f"Simulation{timestr}_detector.{ext}")
        
        
        
        
    if 'snr' in args:
        
        plt.rcParams.update(rcParamsForBaselines)
        generaltitle = "SNR used by SPICA-FT"
        
        linestyles=[mlines.Line2D([],[], color='black',
                                        linestyle='solid', label='Maximal SNR'),
                    mlines.Line2D([],[], color='black',
                                        linestyle=':', label='Start tracking')]
        if 'ThresholdGD' in config.FT.keys():
            linestyles.append(mlines.Line2D([],[], color='black',
                                        linestyle='--', label='Squared Threshold GD'))
        
        for iFig in range(NumberOfBaseFigures):
            NINtodisplay=NINdisp
            if iFig == NumberOfBaseFigures-1:
                if (NINmes%NINdisp < NINdisp) and (NINmes%NINdisp != 0):
                    NINtodisplay = NINmes%NINdisp
                    
            iFirstBase = NINdisp*iFig   # Index of first baseline to display
            iLastBase = iFirstBase + NINtodisplay - 1        # Index of last baseline to display
            
            len2 = NINtodisplay//2 ; len1 = NINtodisplay-len2
            basecolors = colors[:len1]+colors[:len2]
            basecolors = np.array(basecolors)
            
            rangeBases = f"{baselines[iFirstBase]}-{baselines[iLastBase]}"
            title=f'{generaltitle}: {rangeBases}'

            plt.close(title)
            fig=plt.figure(title, clear=True)
            fig.suptitle(title)
            (ax1,ax2),(ax3,ax4) = fig.subplots(nrows=2,ncols=2, gridspec_kw={"height_ratios":[4,1]})
            ax1.set_title(f"From {baselines[iFirstBase]} \
to {baselines[iFirstBase+len1-1]}")
            ax6.set_title(f"From {baselines[iFirstBase+len1]} \
to {baselines[iLastBase]}")
            
            FirstSet = range(iFirstBase,iFirstBase+len1)
            SecondSet = range(iFirstBase+len1,iLastBase+1)
            iColor = 0
            for iBase in FirstSet:   # First serie
                ax1.plot(t[timerange],SNR[timerange,iBase],color=basecolors[iColor])
                if 'ThresholdGD' in config.FT.keys():
                    ax1.hlines(config.FT['ThresholdGD'][iBase], t[timerange[0]],t[timerange[-1]], color=basecolors[iColor], linestyle='dashed')
                iColor+=1
                
            for iBase in SecondSet:   # Second serie
                ax2.plot(t[timerange],SNR[timerange,iBase],color=basecolors[iColor])
                if 'ThresholdGD' in config.FT.keys():
                    ax2.hlines(config.FT['ThresholdGD'][iBase],t[timerange[0]],t[timerange[-1]],color=basecolors[iColor], linestyle='dashed')
                iColor+=1
                
            ax1.vlines(config.starttracking*dt*ms,0.5,2*np.max(SNR),
                       color='k', linestyle=':')
            ax2.vlines(config.starttracking*dt*ms,0.5,2*np.max(SNR),
                       color='k', linestyle=':')
            
            maxSNR = np.nanmax(SNR,axis=0)
            ax3.bar(baselines[FirstSet],maxSNR[FirstSet], color=basecolors[:len1])
            ax3.bar(baselines[FirstSet],config.FT['ThresholdGD'][FirstSet], fill=False,edgecolor='k')
            ax4.bar(baselines[SecondSet],maxSNR[SecondSet], color=basecolors[len1:])
            ax4.bar(baselines[SecondSet],config.FT['ThresholdGD'][SecondSet], fill=False,edgecolor='k')
    
            # ax4.bar(list(baselines[SecondSet])+['']*(len1-len2),list(maxSNR[SecondSet])+[0]*(len1-len2), color=basecolors[len1:]+['k']*(len1-len2))
            # ax4.bar(list(baselines[SecondSet])+['']*(len1-len2),list(config.FT['ThresholdGD'][SecondSet])+[0]*(len1-len2), fill=False,edgecolor='k')
            ax3.hlines(config.FT['ThresholdPD'],-0.5,len1-0.5,color='r',linestyle='-.')
            ax4.hlines(config.FT['ThresholdPD'],-0.5,len1-0.5,color='r',linestyle='-.')
            
            ax1.get_shared_x_axes().join(ax1,ax2)
            ax1.get_shared_y_axes().join(ax1,ax2)
            ax3.get_shared_y_axes().join(ax3,ax4)
            
            ax2.tick_params(labelleft=False) 
            ax4.tick_params(labelleft=False)
            
            ax3.set_box_aspect(1/20)
            ax4.set_box_aspect(1/20)
            
            ax1.grid(True) ; ax2.grid(True)
            ax3.grid(True) ; ax4.grid(True)
            ax1.set_ylabel('SNR')
            ax3.set_ylabel('max(SNR) &\n Thresholds')
            ax1.set_xlabel('Time [ms]') ; ax2.set_xlabel('Time [ms]')
            ax3.set_xlabel('Baseline') ; ax4.set_xlabel('Baseline')
            ct.setaxelim(ax3,ydata=maxSNR,ymin=0.5)
            
            if display:
                if pause:
                    plt.pause(0.1)
                else:
                    plt.show()



    if 'snrpd' in args:

        plt.rcParams.update(rcParamsForBaselines)
        generaltitle = "SNR PD"
        
        linestyles=[mlines.Line2D([],[], color='black',
                                        linestyle='solid', label='Maximal SNR'),
                    mlines.Line2D([],[], color='black',
                                        linestyle=':', label='Start tracking')]
        if 'ThresholdGD' in config.FT.keys():
            linestyles.append(mlines.Line2D([],[], color='black',
                                        linestyle='--', label='Squared Threshold GD'))
        
        for iFig in range(NumberOfBaseFigures):
            NINtodisplay=NINdisp
            if iFig == NumberOfBaseFigures-1:
                if (NINmes%NINdisp < NINdisp) and (NINmes%NINdisp != 0):
                    NINtodisplay = NINmes%NINdisp
                    
            iFirstBase = NINdisp*iFig   # Index of first baseline to display
            iLastBase = iFirstBase + NINtodisplay - 1        # Index of last baseline to display
            
            len2 = NINtodisplay//2 ; len1 = NINtodisplay-len2
            basecolors = colors[:len1]+colors[:len2]
            basecolors = np.array(basecolors)
            
            rangeBases = f"{baselines[iFirstBase]}-{baselines[iLastBase]}"
            title=f'{generaltitle}: {rangeBases}'

            plt.close(title)
            fig=plt.figure(title, clear=True)
            fig.suptitle(title)
            (ax1,ax2),(ax3,ax4) = fig.subplots(nrows=2,ncols=2, gridspec_kw={"height_ratios":[4,1]})
            ax1.set_title(f"From {baselines[iFirstBase]} \
to {baselines[iFirstBase+len1-1]}")
            ax6.set_title(f"From {baselines[iFirstBase+len1]} \
to {baselines[iLastBase]}")
            
            FirstSet = range(iFirstBase,iFirstBase+len1)
            SecondSet = range(iFirstBase+len1,iLastBase+1)
            iColor = 0
            for iBase in FirstSet:   # First serie
                ax1.plot(t[timerange],SNR_pd[timerange,iBase],color=basecolors[iColor])
                if 'ThresholdGD' in config.FT.keys():
                    ax1.hlines(config.FT['ThresholdGD'][iBase], t[timerange[0]],t[timerange[-1]], color=basecolors[iColor], linestyle='dashed')
                iColor+=1
                
            for iBase in SecondSet:   # Second serie
                ax2.plot(t[timerange],SNR_pd[timerange,iBase],color=basecolors[iColor])
                if 'ThresholdGD' in config.FT.keys():
                    ax2.hlines(config.FT['ThresholdGD'][iBase],t[timerange[0]],t[timerange[-1]],color=basecolors[iColor], linestyle='dashed')
                iColor+=1
                
            ax1.vlines(config.starttracking*dt*ms,0.5,2*np.max(SNR_pd),
                       color='k', linestyle=':')
            ax2.vlines(config.starttracking*dt*ms,0.5,2*np.max(SNR_pd),
                       color='k', linestyle=':')
            
            maxSNR = np.nanmax(SNR_pd,axis=0)
            ax3.bar(baselines[FirstSet],maxSNR[FirstSet], color=basecolors[:len1])
            ax3.bar(baselines[FirstSet],config.FT['ThresholdGD'][FirstSet], fill=False,edgecolor='k')
    
            # ax4.bar(list(baselines[SecondSet])+['']*(len1-len2),list(maxSNR[SecondSet])+[0]*(len1-len2), color=basecolors[len1:]+['k']*(len1-len2))
            # ax4.bar(list(baselines[SecondSet])+['']*(len1-len2),list(config.FT['ThresholdGD'][SecondSet])+[0]*(len1-len2), fill=False,edgecolor='k')
            ax3.hlines(config.FT['ThresholdPD'],-0.5,len1-0.5,color='r',linestyle='-.')
            ax4.hlines(config.FT['ThresholdPD'],-0.5,len1-0.5,color='r',linestyle='-.')
            
            ax1.get_shared_x_axes().join(ax1,ax2)
            ax1.get_shared_y_axes().join(ax1,ax2)
            ax3.get_shared_y_axes().join(ax3,ax4)
            
            ax2.tick_params(labelleft=False) 
            ax4.tick_params(labelleft=False)
            
            ax1.grid(True) ; ax2.grid(True)
            ax3.grid(True) ; ax4.grid(True)
            ax1.set_ylabel('SNR')
            ax3.set_ylabel('max(SNR) &\n Thresholds')
            ax1.set_xlabel('Time [ms]') ; ax2.set_xlabel('Time [ms]')
            ax3.set_xlabel('Baseline') ; ax4.set_xlabel('Baseline')
            ct.setaxelim(ax3,ydata=maxSNR,ymin=0.5)
            
            if display:
                if pause:
                    plt.pause(0.1)
                else:
                    plt.show()


    if 'snrgd' in args:

        plt.rcParams.update(rcParamsForBaselines)
        generaltitle = "SNR GD"
        
        linestyles=[mlines.Line2D([],[], color='black',
                                        linestyle='solid', label='Maximal SNR'),
                    mlines.Line2D([],[], color='black',
                                        linestyle=':', label='Start tracking')]
        if 'ThresholdGD' in config.FT.keys():
            linestyles.append(mlines.Line2D([],[], color='black',
                                        linestyle='--', label='Squared Threshold GD'))
        
        for iFig in range(NumberOfBaseFigures):
            NINtodisplay=NINdisp
            if iFig == NumberOfBaseFigures-1:
                if (NINmes%NINdisp < NINdisp) and (NINmes%NINdisp != 0):
                    NINtodisplay = NINmes%NINdisp
                    
            iFirstBase = NINdisp*iFig   # Index of first baseline to display
            iLastBase = iFirstBase + NINtodisplay - 1        # Index of last baseline to display
            
            len2 = NINtodisplay//2 ; len1 = NINtodisplay-len2
            basecolors = colors[:len1]+colors[:len2]
            basecolors = np.array(basecolors)
            
            rangeBases = f"{baselines[iFirstBase]}-{baselines[iLastBase]}"
            title=f'{generaltitle}: {rangeBases}'

            plt.close(title)
            fig=plt.figure(title, clear=True)
            fig.suptitle(title)
            (ax1,ax2),(ax3,ax4) = fig.subplots(nrows=2,ncols=2, gridspec_kw={"height_ratios":[4,1]})
            ax1.set_title(f"From {baselines[iFirstBase]} \
to {baselines[iFirstBase+len1-1]}")
            ax6.set_title(f"From {baselines[iFirstBase+len1]} \
to {baselines[iLastBase]}")
            
            FirstSet = range(iFirstBase,iFirstBase+len1)
            SecondSet = range(iFirstBase+len1,iLastBase+1)
            iColor = 0
            for iBase in FirstSet:   # First serie
                ax1.plot(t[timerange],SNR_gd[timerange,iBase],color=basecolors[iColor])
                if 'ThresholdGD' in config.FT.keys():
                    ax1.hlines(config.FT['ThresholdGD'][iBase], t[timerange[0]],t[timerange[-1]], color=basecolors[iColor], linestyle='dashed')
                iColor+=1
                
            for iBase in SecondSet:   # Second serie
                ax2.plot(t[timerange],SNR_gd[timerange,iBase],color=basecolors[iColor])
                if 'ThresholdGD' in config.FT.keys():
                    ax2.hlines(config.FT['ThresholdGD'][iBase],t[timerange[0]],t[timerange[-1]],color=basecolors[iColor], linestyle='dashed')
                iColor+=1
                
            ax1.vlines(config.starttracking*dt*ms,0.5,2*np.max(SNR_gd),
                       color='k', linestyle=':')
            ax2.vlines(config.starttracking*dt*ms,0.5,2*np.max(SNR_gd),
                       color='k', linestyle=':')
            
            maxSNR = np.nanmax(SNR_gd,axis=0)
            ax3.bar(baselines[FirstSet],maxSNR[FirstSet], color=basecolors[:len1])
            ax3.bar(baselines[FirstSet],config.FT['ThresholdGD'][FirstSet], fill=False,edgecolor='k')
    
            ax4.bar(list(baselines[SecondSet])+['']*(len1-len2),list(maxSNR[SecondSet])+[0]*(len1-len2), color=basecolors[len1:]+['k']*(len1-len2))
            ax4.bar(list(baselines[SecondSet])+['']*(len1-len2),list(config.FT['ThresholdGD'][SecondSet])+[0]*(len1-len2), fill=False,edgecolor='k')
            ax3.hlines(config.FT['ThresholdPD'],-0.5,len1-0.5,color='r',linestyle='-.')
            ax4.hlines(config.FT['ThresholdPD'],-0.5,len1-0.5,color='r',linestyle='-.')
            
            ax1.get_shared_x_axes().join(ax1,ax2)
            ax1.get_shared_y_axes().join(ax1,ax2)
            ax3.get_shared_y_axes().join(ax3,ax4)
            
            ax2.tick_params(labelleft=False) 
            ax4.tick_params(labelleft=False)
            
            ax1.grid(True) ; ax2.grid(True)
            ax3.grid(True) ; ax4.grid(True)
            ax1.set_ylabel('SNR')
            ax3.set_ylabel('max(SNR) &\n Thresholds')
            ax1.set_xlabel('Time [ms]') ; ax2.set_xlabel('Time [ms]')
            ax3.set_xlabel('Baseline') ; ax4.set_xlabel('Baseline')
            ct.setaxelim(ax3,ydata=maxSNR,ymin=0.5)
            
            if display:
                if pause:
                    plt.pause(0.1)
                else:
                    plt.show()



    """
    GD ONLY
    """

    if displayall or ("GDonly" in args):
        plt.rcParams.update(rcParamsForBaselines)
        title=infos['details']
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        
        OneAxe = False ; NumberOfBaselinesToPlot = np.sum(PlotBaseline)
        if NumberOfBaselinesToPlot < len1:
            OneAxe=True
            ax1,ax2 = fig.subplots(nrows=2,sharex=True,sharey='row')
        else:    
            (ax1,ax3),(ax2,ax4),(ax5,ax6) = fig.subplots(nrows=3,ncols=2, sharey='row', gridspec_kw={'height_ratios':[4,4,1]})
            ax1.set_title("First serie of baselines, from 12 to 25")
            ax3.set_title("Second serie of baselines, from 26 to 56")

        # ax1.set_ylim(np.sqrt(ylimSNR)) 
        
        if OneAxe:
            baselinestemp = [baselines[iBase] for iBase in PlotBaselineIndex]
            basecolorstemp = basecolors[:NumberOfBaselinesToPlot]
            baselinestyles=['-']*len1 + ['--']*len2
            baselinehatches=['']*len1 + ['/']*len2
            
            k=0
            for iBase in PlotBaselineIndex:   # First serie
                ax1.plot(t[timerange],SNR[timerange,iBase],color=basecolorstemp[k],label=baselines[iBase])
                ax2.plot(t[timerange],GDerr[timerange,iBase],color=basecolorstemp[k])
                # barbasecolors[iBase] = basecolorstemp[k]
                k+=1                

            ax1.legend()

        else:
            for iBase in range(len1):   # First serie
                if PlotBaseline[iBase]:
                    ax1.plot(t[timerange],SNR[timerange,iBase],color=basecolors[iBase])
                    ax2.plot(t[timerange],GDerr[timerange,iBase],color=basecolors[iBase])

            for iBase in range(len1,NIN):   # Second serie
                if PlotBaseline[iBase]:
                    ax3.plot(t[timerange],SNR[timerange,iBase],color=basecolors[iBase])
                    ax4.plot(t[timerange],GDerr[timerange,iBase],color=basecolors[iBase])

            ax1.sharex(ax2)
            ax3.sharex(ax4)
            p1=ax5.bar(baselines[:len1],RMSgderrmic[:len1], color=basecolors[:len1])
            p2=ax6.bar(baselines[len1:],RMSgderrmic[len1:], color=basecolors[len1:])
            # ax4.sharey(ax2) ; ax4.tick_params(labelleft=False)
            # ax4.set_ylim(ylimGD)
            ct.setaxelim(ax5, ydata=RMSgderrmic, ymargin=0.4,ymin=0)
            ct.setaxelim(ax1, ydata=SNR, ymargin=0.4,ymin=0)
            
            ax5.set_ylabel('GD rms\n[µm]') ;
            ax5.bar_label(p1,label_type='edge',fmt='%.2f')
            ax6.bar_label(p2,label_type='edge',fmt='%.2f')
            ax5.set_anchor('S') ; ax6.set_anchor('S')
            ax5.set_box_aspect(1/15) ; ax6.set_box_aspect(1/15)
            
        ct.setaxelim(ax2,ydata=[GDerr[timerange,iBase] for iBase in PlotBaselineIndex])
        ax1.set_ylabel('SNR')
        ax2.set_ylabel('Group-Delays [µm]')
    
        ax2.set_xlabel("Time [ms]")#, labelpad=xlabelpad) ; ax4.set_xlabel("Time (s)", labelpad=xlabelpad)


        if display:
            if pause:
                plt.pause(0.1)
            else:
                plt.show()  
        if len(savedir):
            fig.savefig(savedir+f"Simulation{timestr}_GDonly.{ext}")








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
            
        ax.hlines(config.FT['ThresholdGD']**2,0,timestamps[-1],
                  linestyle='--',label='Threshold GD')
        
        ax2.plot(timestamps, config.FT['state'], 
                color='blue', label='State-machine')
        ax2.plot(timestamps, simu.IgdRank, 
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
            if pause:
                plt.pause(0.1)
            else:
                plt.show()
        config.newfig+=1
        
    # if not display:
    #     matplotlib.use(currentGUI)
        
    pass


def investigate(*args):
    from . import config
    from . import simu
    
    from cophasing.tol_colors import tol_cset
    
    if 'detail search' in args:
        fig=plt.figure('detail FS', clear=True)
        ax1,ax2,ax3 = fig.subplots(nrows=3)
        for ia in range(config.NA):
            ax1.plot(simu.timestamps, simu.last_usaw[:,ia], color=colors[ia], label=f"Tel{ia+1}")
            ax2.plot(simu.timestamps, simu.it_last[:,ia]*config.dt, color=colors[ia])
            ax3.plot(simu.timestamps, simu.eps[:,ia], color=colors[ia])
            ax3.plot(simu.timestamps, np.gradient(simu.SearchCommand[:-2,ia]), color=colors[ia], linestyle='-.')
        ax1.set_ylabel("last_usaw")
        ax2.set_ylabel("it_last")
        ax3.set_ylabel("eps")
        ax3.set_ylim(-2,2)
        ax1.legend()

    pass

def ShowPerformance(TimeBonds, SpectraForScience,DIT,FileInterferometer='',
                    CoherentFluxObject=[],SNR_SI=False,
                    R=140, p=10, magSI=-1,display=True, get=[],
                    verbose=False):
    """
    Processes the performance of the fringe-tracking starting at the StartingTime
    Observables processed:
        -VarOPD                 # Temporal Variance OPD [µm]
        -VarPDEst              # Temporal Variance PD [rad]
        -VarGDEst              # Temporal Variance of GD [rad]
        -VarCPD                 # Temporal Variance of CPD [rad]
        -VarCGD                 # Temporal Variance of CGD [rad]
        -FringeContrast         # Fringe Contrast [0,1] at given wavelengths
WavelengthOfInterest
        

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
    Returns
    -------
    None.

    """
    from . import simu
    from . import config
    
    ich = config.FS['ich']
    
    from .config import NA,NIN,NC,dt,NT
    
    NINmes = config.FS['NINmes']
    
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
    
    if isinstance(TimeBonds,(float,int)):
        Period = int(NT - TimeBonds/dt)
        InFrame = round(TimeBonds/dt)
    elif isinstance(TimeBonds,(np.ndarray,list)):
        Period = int((TimeBonds[1]-TimeBonds[0])/dt)
        InFrame = round(TimeBonds[0]/dt)
    else:
        raise '"TimeBonds" must be instance of (float,int,np.ndarray,list)'
       
        
    if not len(FileInterferometer):
        FileInterferometer = "data/interferometers/CHARAinterferometerR.fits"
      
    if SNR_SI:
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
            CoherentFluxObject = CoherentFluxObject*dt*1e-3  # [MW,:] whether it is multiWL or not
        
        
        from cophasing.SCIENTIFIC_INSTRUMENTS import SPICAVIS
        simu.IntegrationTime, simu.VarSquaredVis, simu.SNR_E, simu.SNR_E_perSC = SPICAVIS(CoherentFluxObject,simu.OPDTrue[InFrame:],SpectraForScience,DIT=DIT)
        
   
    if MultiWavelength:
        simu.FringeContrast=np.zeros([MW,NIN])  # Fringe Contrast at given wavelengths [0,1]
    else:
        simu.FringeContrast=np.zeros(NIN)       # Fringe Contrast at given wavelengths [0,1]

    simu.VarOPD=np.zeros(NIN)
    simu.VarGDRes=np.zeros(NINmes)
    simu.VarPDRes=np.zeros(NINmes)
    simu.VarGDEst=np.zeros(NINmes)
    simu.VarPDEst=np.zeros(NINmes)
    simu.VarPiston=np.zeros(NA)
    simu.VarPistonGD=np.zeros(NA)
    simu.VarPistonPD=np.zeros(NA)

    simu.VarCPD =np.zeros(NC); simu.VarCGD=np.zeros(NC)
    
    simu.LockedRatio=np.zeros(NIN)          # sig_opd < lambda/p
    simu.LR2 = np.zeros(NINmes)             # Mode TRACK
    simu.LR3= np.zeros(NIN)                 # In Central Fringe
    simu.LR4= np.zeros(NIN)                 # No fringe jump
    simu.WLockedRatio = np.zeros(NIN)
    
    MaxPhaseVarForLocked = (2*np.pi/p)**2
    MaxVarOPDForLocked = (MeanWavelength/p)**2
    
    Ndit = Period//DIT_NumberOfFrames
    simu.PhaseVar_atWOI = np.zeros([Ndit,NIN])
    simu.PhaseStableEnough= np.zeros([Ndit,NIN])
    simu.LR2 = np.mean(simu.TrackedBaselines[InFrame:], axis=0)   # Array [NINmes]
    simu.InCentralFringe = np.abs(simu.OPDTrue-simu.OPDrefObject) < MeanWavelength/2
    simu.LR3 = np.mean(simu.InCentralFringe[InFrame:], axis=0)    # Array [NIN]
    
    for it in range(Ndit):
        OutFrame=InFrame+DIT_NumberOfFrames
        OPDVar = np.var(simu.OPDTrue[InFrame:OutFrame,:],axis=0)
        OPDptp = np.ptp(simu.OPDTrue[InFrame:OutFrame,:],axis=0)
        
        GDResVar = np.var(simu.GDResidual2[InFrame:OutFrame,:],axis=0)
        PDResVar = np.var(simu.PDResidual2[InFrame:OutFrame,:],axis=0)
        GDEstVar = np.var(simu.GDEstimated[InFrame:OutFrame,:],axis=0)
        PDEstVar = np.var(simu.PDEstimated[InFrame:OutFrame,:],axis=0)
        PistonVar = np.var(simu.PistonTrue[InFrame:OutFrame,:],axis=0)
        PistonVarGD = np.var(simu.GDPistonResidual[InFrame:OutFrame,:],axis=0)
        PistonVarPD = np.var(simu.PDPistonResidual[InFrame:OutFrame,:],axis=0)
        simu.PhaseVar_atWOI[it] = np.var(2*np.pi*simu.OPDTrue[InFrame:OutFrame,:]/MeanWavelength,axis=0)
        simu.PhaseStableEnough[it] = 1*(OPDVar < MaxVarOPDForLocked)
        NoFringeJumpDuringPose = 1*(OPDptp < 1.5*MeanWavelength)
        
        simu.VarOPD += 1/Ndit*OPDVar
        simu.LR4 += 1/Ndit*NoFringeJumpDuringPose
        
        simu.VarPDRes += 1/Ndit*PDResVar
        simu.VarGDRes += 1/Ndit*GDResVar
        simu.VarPDEst += 1/Ndit*PDEstVar
        simu.VarGDEst += 1/Ndit*GDEstVar
        simu.VarPiston += 1/Ndit*PistonVar
        simu.VarPistonGD += 1/Ndit*PistonVarGD
        simu.VarPistonPD += 1/Ndit*PistonVarPD
        simu.VarCPD += 1/Ndit*np.var(simu.ClosurePhasePD[InFrame:OutFrame,:],axis=0)
        simu.VarCGD += 1/Ndit*np.var(simu.ClosurePhaseGD[InFrame:OutFrame,:],axis=0)
        
        
        # Fringe contrast
        if MultiWavelength:
            for iw in range(MW):
                wl = SpectraForScience[iw]
                for ib in range(NIN):
                    CoherenceEnvelopModulation = np.sinc(simu.OPDTrue[InFrame:OutFrame,ib]/Lc[iw])
                    Phasors = np.exp(1j*2*np.pi*simu.OPDTrue[InFrame:OutFrame,ib]/wl)
                    simu.FringeContrast[iw,ib] += 1/Ndit*np.abs(np.mean(Phasors*CoherenceEnvelopModulation,axis=0))
        else:
            for ib in range(NIN):
                CoherenceEnvelopModulation = np.sinc(simu.OPDTrue[InFrame:OutFrame,ib]/Lc)
                Phasors = np.exp(1j*2*np.pi*simu.OPDTrue[InFrame:OutFrame,ib]/MeanWavelength)
                simu.FringeContrast[ib] += 1/Ndit*np.abs(np.mean(Phasors*CoherenceEnvelopModulation,axis=0))

        InFrame += DIT_NumberOfFrames
        
    simu.LockedRatio = np.mean(simu.PhaseStableEnough,axis=0)
    simu.WLockedRatio = np.mean(simu.PhaseStableEnough*simu.FringeContrast, axis=0)
    
    simu.autreWlockedRatio = np.mean((MaxPhaseVarForLocked-simu.PhaseVar_atWOI)/MaxPhaseVarForLocked, axis=0)

    simu.WLR2 = np.mean(simu.TrackedBaselines * simu.SquaredSNRMovingAveragePD, axis=0)
    
    # if 'ThresholdGD' in config.FT.keys():
    #     simu.WLR3 = np.mean(simu.TrackedBaselines * simu.SquaredSNRMovingAveragePD, axis=0)


    if not display:
        return

    observable = simu.VarOPD
    xrange = np.arange(NIN)
    
    plt.figure(f'Variance OPD @{round(config.PDspectra,2)}µm')    
    plt.ylim([np.min(observable),np.max(observable)])
    plt.scatter(np.arange(NIN),observable)
    plt.hlines(np.mean(observable), xrange[0],xrange[-1],linestyle='--')
    plt.xticks(ticks=np.arange(NIN),labels=ich, rotation='vertical')    
    plt.xlabel('Baseline')
    plt.ylabel('Variance [µm]')
    plt.grid()
    plt.show()
    config.newfig += 1
    
    observable = simu.VarPDEst*(config.PDspectra/2/np.pi)
    
    plt.figure(f'Variance Estimated PD @{round(config.PDspectra,2)}µm')    
    plt.ylim([np.min(observable),np.max(observable)])
    plt.scatter(np.arange(NINmes),observable)
    plt.hlines(np.mean(observable), xrange[0],xrange[-1],linestyle='--')
    plt.xticks(ticks=np.arange(NINmes),labels=ich, rotation='vertical')    
    plt.xlabel('Baseline')
    plt.ylabel('Variance [µm]')
    plt.grid()
    plt.show()
    config.newfig += 1
    
    
    observable = simu.VarGDEst*(config.PDspectra/2/np.pi)*config.FS['R']
    
    plt.figure(f'Variance Estimated GD @{round(config.PDspectra,2)}µm')    
    plt.ylim([np.min(observable),np.max(observable)])
    plt.scatter(np.arange(NINmes),observable)
    plt.hlines(np.mean(observable), xrange[0],xrange[-1],linestyle='--')
    plt.xticks(ticks=np.arange(NINmes),labels=ich, rotation='vertical')    
    plt.xlabel('Baseline')
    plt.ylabel('Variance [µm]')
    plt.grid()
    plt.show()
    config.newfig += 1
    
    
    observable = simu.VarCPD
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


    observable = simu.VarCGD
    
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
    plt.ylim([0.9*np.min(simu.FringeContrast),1.1])
    for ib in range(NIN):
        plt.scatter(SpectraForScience,simu.FringeContrast[:,ib], label=f'{ich[ib]}')
        
    plt.legend(), plt.grid()
    plt.show()
    config.newfig += 1
    
    return



def ShowPerformance_multiDITs(TimeBonds,SpectraForScience,IntegrationTimes=[],
                              CoherentFluxObject=[],FileInterferometer='',
                              R=140, p=10, magSI=-1,display=True, get=[],criterias='light',
                              verbose=False, onlySNR=False,check_DITs=False, OnlyCheckDIT=False):
    """
    Processes the performance of the fringe-tracking starting at the StartingTime
    Observables processed:
        -VarOPD                 # Temporal Variance OPD [µm]
        -VarPDEst              # Temporal Variance PD [rad]
        -VarGDEst              # Temporal Variance of GD [rad]
        -VarCPD                 # Temporal Variance of CPD [rad]
        -VarCGD                 # Temporal Variance of CGD [rad]
        -FringeContrast         # Fringe Contrast [0,1] at given wavelengths
MeanWavelength
        

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
    from . import simu
    from . import config
    
    ich = config.FS['ich']
    
    from .config import NA,NIN,NC,dt,NT,NB
    
    
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
        raise '"TimeBonds" must be instance of (float,int,np.ndarray,list)'

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
        
    simu.DITsForPerformance = NewIntegrationTimes
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
        CoherentFluxObject = CoherentFluxObject*dt*1e-3  # [MW,:] whether it is multiWL or not
    
    from cophasing.SCIENTIFIC_INSTRUMENTS import SPICAVIS
   

    simu.VarSquaredVis=np.zeros([Ndit,MW,NIN])*np.nan
    simu.SNR_E=np.zeros([Ndit,NIN])*np.nan
    simu.SNR_E_perSC=np.zeros([Ndit,MW,NIN])*np.nan
    simu.VarOPD=np.zeros([Ndit,NIN])
    simu.LockedRatio=np.zeros([Ndit,NIN])    # sig_opd < lambda/p
    simu.LR2 = np.zeros([Ndit,NIN])          # Mode TRACK
    simu.LR3= np.zeros([Ndit,NIN])           # In Central Fringe
    simu.LR4 = np.zeros([Ndit,NIN])          # No Fringe Jump
    simu.FringeContrast=np.zeros([Ndit,MW,NIN])  # Fringe Contrast at given wavelengths [0,1]
    
    simu.VarPiston=np.zeros([Ndit,NA])
    simu.VarPistonGD=np.zeros([Ndit,NA])
    simu.VarPistonPD=np.zeros([Ndit,NA])
    
    if criterias!='light': # additional informations

        simu.EnergyPicFrange=np.zeros([Ndit,MW,NIN])*np.nan
        simu.PhotonNoise=np.zeros([Ndit,MW,NIN])*np.nan
        simu.ReadNoise=np.zeros([Ndit,NIN])*np.nan
        simu.CoupledTerms=np.zeros([Ndit,MW,NIN])*np.nan
        simu.VarCf=np.zeros([Ndit,MW,NIN])*np.nan
    
        simu.VarGDRes=np.zeros([Ndit,NIN])
        simu.VarPDRes=np.zeros([Ndit,NIN])
    
        simu.VarPDEst=np.zeros([Ndit,NIN])
        simu.VarGDEst=np.zeros([Ndit,NIN])
        
        simu.WLockedRatio = np.zeros([Ndit,NIN])
        simu.WLR2 = np.zeros([Ndit,NIN])
    
        simu.VarCPD =np.zeros([Ndit,NC])
        simu.VarCGD=np.zeros([Ndit,NC])
    
    simu.IntegrationTime=NewIntegrationTimes

    MaxPhaseVarForLocked = (2*np.pi/p)**2
    MaxVarOPDForLocked = (MeanWavelength/p)**2
    
    InCentralFringe = np.abs(simu.OPDTrue-simu.OPDrefObject) < MeanWavelength/2
    if 'ThresholdGD' in config.FT.keys():
        TrackedBaselines = (simu.SquaredSNRMovingAveragePD >= config.FT['ThresholdGD']**2) #Array [NT,NIN]
        
    FirstFrame = InFrame
    for idit in range(Ndit):
        
        DIT=NewIntegrationTimes[idit]
        
        # Calculation of SNR
        _, simu.VarSquaredVis[idit], simu.SNR_E[idit], simu.SNR_E_perSC[idit] = SPICAVIS(CoherentFluxObject,simu.OPDTrue[FirstFrame:], SpectraForScience,DIT=DIT)
        
        if criterias!='light':
            simu.EnergyPicFrange[idit] = simu.SNRnum
            simu.PhotonNoise[idit]= simu.PhNoise
            simu.ReadNoise[idit]=simu.RNoise
            simu.CoupledTerms[idit]=simu.CTerms
            simu.VarCf[idit]=simu.var_cf
        
        if onlySNR:
            continue
        
        DIT_NumberOfFrames = int(DIT/dt)
        Nframes = Period//DIT_NumberOfFrames
        
        PhaseVar_atWOI = np.zeros([NIN])
        PhaseStableEnough= np.zeros([NIN])
        
        if 'ThresholdGD' in config.FT.keys():
            simu.LR2[idit] = np.mean(TrackedBaselines[InFrame:], axis=0)   # Array [NIN]
        
        simu.LR3[idit] = np.mean(InCentralFringe[InFrame:], axis=0)    # Array [NIN]
        
        InFrame = FirstFrame
        for iframe in range(Nframes):
            OutFrame=InFrame+DIT_NumberOfFrames
            
            OPDVar = np.var(simu.OPDTrue[InFrame:OutFrame,:],axis=0)
            OPDptp = np.ptp(simu.OPDTrue[InFrame:OutFrame,:],axis=0)
            
            PhaseVar_atWOI = np.var(2*np.pi*simu.OPDTrue[InFrame:OutFrame,:]/MeanWavelength,axis=0)
            PhaseStableEnough = 1*(OPDVar < MaxVarOPDForLocked)
            NoFringeJumpDuringPose = 1*(OPDptp < 1.5*MeanWavelength)
            
            simu.LockedRatio[idit] += 1/Nframes*np.mean(PhaseStableEnough,axis=0)
            simu.LR4[idit] += 1/Nframes*np.mean(NoFringeJumpDuringPose,axis=0)
            simu.VarOPD[idit] += 1/Nframes*OPDVar


            # Telescopes
            PistonVar = np.var(simu.PistonTrue[InFrame:OutFrame,:],axis=0)
            PistonVarGD = np.var(simu.GDPistonResidual[InFrame:OutFrame,:],axis=0)
            PistonVarPD = np.var(simu.PDPistonResidual[InFrame:OutFrame,:],axis=0)
            simu.VarPiston[idit] += 1/Nframes*PistonVar
            simu.VarPistonGD[idit] += 1/Nframes*PistonVarGD
            simu.VarPistonPD[idit] += 1/Nframes*PistonVarPD
            
            # Fringe contrast
            for iw in range(MW):
                wl = SpectraForScience[iw]
                for ib in range(NIN):
                    CoherenceEnvelopModulation = np.sinc(simu.OPDTrue[InFrame:OutFrame,ib]/Lc[iw])
                    Phasors = np.exp(1j*2*np.pi*simu.OPDTrue[InFrame:OutFrame,ib]/wl)
                    simu.FringeContrast[idit,iw,ib] += 1/Nframes*np.abs(np.mean(Phasors*CoherenceEnvelopModulation,axis=0))


            if criterias!='light':
                GDResVar = np.var(simu.GDResidual2[InFrame:OutFrame,:],axis=0)
                PDResVar = np.var(simu.PDResidual2[InFrame:OutFrame,:],axis=0)
                simu.VarPDRes[idit] += 1/Nframes*PDResVar
                simu.VarGDRes[idit] += 1/Nframes*GDResVar

                simu.VarPDEst[idit] += 1/Nframes*np.var(simu.PDEstimated2[InFrame:OutFrame,:],axis=0)
                simu.VarGDEst[idit] += 1/Nframes*np.var(simu.GDEstimated2[InFrame:OutFrame,:],axis=0)
                simu.VarCPD[idit] += 1/Nframes*np.var(simu.ClosurePhasePD[InFrame:OutFrame,:],axis=0)
                simu.VarCGD[idit] += 1/Nframes*np.var(simu.ClosurePhaseGD[InFrame:OutFrame,:],axis=0)
                
                simu.WLockedRatio[idit] += 1/Nframes*np.mean(PhaseStableEnough*simu.FringeContrast[idit], axis=0)
    
            InFrame += DIT_NumberOfFrames
            
        # Don't depend on DIT but better for same treatment after.
        if criterias!='light':
            simu.WLR2[idit] = np.mean(InCentralFringe * simu.SquaredSNRMovingAveragePD, axis=0)
        
        
    if onlySNR:
        return NewIntegrationTimes
    
    
    if display:

        observable = simu.VarOPD
        xrange = np.arange(NIN)
        
        plt.figure(f'Variance OPD @{round(config.PDspectra,2)}µm')    
        plt.ylim([np.min(observable),np.max(observable)])
        plt.scatter(np.arange(NIN),observable)
        plt.hlines(np.mean(observable), xrange[0],xrange[-1],linestyle='--')
        plt.xticks(ticks=np.arange(NIN),labels=ich, rotation='vertical')    
        plt.xlabel('Baseline')
        plt.ylabel('Variance [µm]')
        plt.grid()
        plt.show()
        config.newfig += 1
        
        observable = simu.VarPDEst*(config.PDspectra/2/np.pi)
        
        plt.figure(f'Variance Estimated PD @{round(config.PDspectra,2)}µm')    
        plt.ylim([np.min(observable),np.max(observable)])
        plt.scatter(np.arange(NIN),observable)
        plt.hlines(np.mean(observable), xrange[0],xrange[-1],linestyle='--')
        plt.xticks(ticks=np.arange(NIN),labels=ich, rotation='vertical')    
        plt.xlabel('Baseline')
        plt.ylabel('Variance [µm]')
        plt.grid()
        plt.show()
        config.newfig += 1
        
        
        observable = simu.VarGDEst*(config.PDspectra/2/np.pi)*config.FS['R']
        
        plt.figure(f'Variance Estimated GD @{round(config.PDspectra,2)}µm')    
        plt.ylim([np.min(observable),np.max(observable)])
        plt.scatter(np.arange(NIN),observable)
        plt.hlines(np.mean(observable), xrange[0],xrange[-1],linestyle='--')
        plt.xticks(ticks=np.arange(NIN),labels=ich, rotation='vertical')    
        plt.xlabel('Baseline')
        plt.ylabel('Variance [µm]')
        plt.grid()
        plt.show()
        config.newfig += 1
        
        
        observable = simu.VarCPD
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
    
    
        observable = simu.VarCGD
        
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
        plt.ylim([0.9*np.min(simu.FringeContrast),1.1])
        for ib in range(NIN):
            plt.scatter(MeanWavelength,simu.FringeContrast[:,ib], label=f'{ich[ib]}')
            
        plt.legend(), plt.grid()
        plt.show()
        config.newfig += 1
    
    return NewIntegrationTimes


def BodeDiagrams(Input,Output,Command,timestamps,
                 fbonds=[], gain=0, details='', window='hanning',
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
        title = f'{details} - Bode diagrams'
        fig = plt.figure(title, clear=True)
        fig.suptitle(title)
        ax1,ax2,ax3 = fig.subplots(nrows=3,sharex=True)
        
        ax1.plot(FrequencySampling, np.abs(FTrej))
        
        if gain:
            gains = [gain] ; delays=np.arange(10,60,10)
            styles=['-','--',':']
            linestyles = []
            for ig in range(len(gains)):
                gain=gains[ig]
                linestyles.append(mlines.Line2D([],[],color='k',linestyle=styles[ig],label=f"Gain={gain}"))
                for idel in range(len(delays)):
                    gain=gains[ig];delay=delays[idel]
                    # ftr=model(FrequencySampling,delay,gain)
                    # ax1.plot(FrequencySampling, ftr, color=colors[idel], linestyle=styles[ig])
                    if ig==len(gains)-1:
                        linestyles.append(mlines.Line2D([],[],color=colors[idel],linestyle='-',label=f'\u03C4={delay}'))
            ax1.legend(handles=linestyles)
            

        # plt.plot(FrequencySampling, FrequencySampling*10**(-2), linestyle='--')
        ax1.set_yscale('log') #; ax1.set_ylim(1e-3,5)
        ax1.set_ylabel('FTrej')
        
        ax2.plot(FrequencySampling, np.abs(FTBO))
        ax2.set_yscale('log') #; ax2.set_ylim(1e-3,5)
        ax2.set_ylabel("FTBO")
        
        ax3.plot(FrequencySampling, np.abs(FTBF))
    
        ax3.set_xlabel('Frequencies [Hz]')
        ax3.set_ylabel('FTBF')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax1.grid(True) ; ax2.grid(True) ; ax3.grid(True)

        if figsave:
            prefix = details.replace("=","").replace(";","").replace(" ","").replace(".","").replace('\n','_').replace('Phase-delay','PD').replace('Group-delay','GD')
            figname = "BodeDiagrams"
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
            else:
                plt.savefig(figdir+f"{prefix}_{figname}.{ext}")


        fig = plt.figure(f'{details} - Temporal sampling used', clear=True)
        ax1,ax2,ax3 = fig.subplots(nrows=3,sharex=True)
        
        ax1.plot(timestamps, Input)
        # plt.plot(FrequencySampling, FrequencySampling*10**(-2), linestyle='--')
    
        ax1.set_ylabel('Open loop')
        
        ax2.plot(timestamps, Output)
        ax2.set_ylabel("Close loop")
        
        ax3.plot(timestamps,Command)
    
        ax3.set_xlabel('Timestamps [s]')
        ax3.set_ylabel('Command')
        
        
    return FrequencySampling, FTrej, FTBO, FTBF



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
    from . import simu
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
        raise '"TimeBonds" must be instance of (float,int,np.ndarray,list)'
    
    SampleIndices = range(BeginSample,EndSample) ; nNT = len(SampleIndices)
    
    FrequencySampling = np.fft.fftfreq(nNT, dt*1e-3)
    PresentFrequencies = (FrequencySampling >= 0) & (FrequencySampling < 200)
    FrequencySampling = FrequencySampling[PresentFrequencies]
    
    Residues = simu.OPDTrue[SampleIndices,ib]
    Turb = simu.OPDDisturbance[SampleIndices,ib]
    Command = simu.OPDCommand[SampleIndices,ib]
    
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
    
    ax1.plot(FrequencySampling, 20*np.log10(np.abs(FTrej)))
    ax1.set_ylabel('FTrej\nGain [dB]')
    
    ax2.plot(FrequencySampling, 20*np.log10(np.abs(FTBO)))
    ax2.set_ylabel("FTBO\nGain [dB]")
    
    ax3.plot(FrequencySampling, 20*np.log10(np.abs(FTBF)))

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
    
    Simplest Fringe Sensor to be run with the COPHASING
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
        ReadCf - Calculates the group-delay, phase-delay, closure phase and 
        complex visibility from the coherent flux estimated by the FS
    
        
    INPUT: CfEstimated [MW,NB]
    
    OUTPUT: 
        
    UPDATE:
        - simu.CfPD: Coherent Flux Phase-Delay      [NT,MW,NIN]*1j
        - simu.CfGD: Coherent Flux GD               [NT,MW,NIN]*1j
        - simu.ClosurePhasePD                       [NT,MW,NC]
        - simu.ClosurePhaseGD                       [NT,MW,NC]
        - simu.PhotometryEstimated                  [NT,MW,NA]
        - simu.VisibilityEstimated                  [NT,MW,NIN]*1j
        - simu.SquaredCoherenceDegree               [NT,MW,NIN]
    """

    from . import simu
    
    from .config import NA,NIN,NC
    from .config import MW,FS
    
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
        from .coh_tools import NB2NIN
        currCfEstimatedNIN[imw,:] = NB2NIN(currCfEstimated[imw,:])
        
    # Save coherent flux and photometries in stack
    simu.PhotometryEstimated[it] = PhotEst
    
    """
    Visibilities extraction
    [NT,MW,NIN]
    """
    
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib = ct.posk(ia,iap,NA)
            Iaap = currCfEstimatedNIN[:,ib]                     # Interferometric intensity (a,a')
            Ia = PhotEst[:,ia]                                  # Photometry pupil a
            Iap = PhotEst[:,iap]                                # Photometry pupil a'
            simu.VisibilityEstimated[it,:,ib] = 2*Iaap/(Ia+Iap)          # Fringe VisibilityEstimated of the base (a,a')
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
    currPD = np.angle(np.sum(simu.CfPD[it,:,:], axis=0))*FS['active_ich']
        
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
        # cs = 0*1j
        cfGDlmbdas = simu.CfGD[it,:-Ncross,ib]*np.conjugate(simu.CfGD[it,Ncross:,ib])
        cfGDmoy = np.sum(cfGDlmbdas)

            
        currGD[ib] = np.angle(cfGDmoy)*FS['active_ich'][ib]    # Group-delay on baseline (ib).
    
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
            cs1 = np.sum(simu.CfPD[timerange,:,ib], axis=1)     # Sum of coherent flux (ia,iap)
            cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
            cfGDmoy1 = np.sum(cfGDlmbdas,axis=1)     # Sum of coherent flux (ia,iap)  
            for iapp in range(iap+1,NA):
                ib = ct.posk(iap,iapp,NA) # coherent flux (iap,iapp)    
                valid2=config.FS['active_ich'][ib]
                cs2 = np.sum(simu.CfPD[timerange,:,ib], axis=1) # Sum of coherent flux (iap,iapp)    
                cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
                cfGDmoy2 = np.sum(cfGDlmbdas,axis=1)
                
                ib = ct.posk(ia,iapp,NA) # coherent flux (iapp,ia)    
                valid3=config.FS['active_ich'][ib]
                cs3 = np.sum(np.conjugate(simu.CfPD[timerange,:,ib]),axis=1) # Sum of 
                cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
                cfGDmoy3 = np.sum(cfGDlmbdas,axis=1)
                
                # The bispectrum of one time and one triangle adds up to
                # the Ncp last times
                ic = ct.poskfai(ia,iap,iapp,NA)        # 0<=ic<NC=(NA-2)(NA-1) 
                validcp[ic]=valid1*valid2*valid3
                bispectrumPD[ic]=np.sum(cs1*cs2*cs3)
                bispectrumGD[ic]=np.sum(cfGDmoy1*cfGDmoy2*np.conjugate(cfGDmoy3))
    
    simu.ClosurePhasePD[it] = np.angle(bispectrumPD)*validcp
    simu.ClosurePhaseGD[it] = np.angle(bispectrumGD)*validcp
    
    if config.FT['CPref'] and (it>Ncp):                     # At time 0, we create the reference vectors
        for ia in range(1,NA-1):
            for iap in range(ia+1,NA):
                k = ct.posk(ia,iap,NA)
                ic = ct.poskfai(0,ia,iap,NA)   # Position of the triangle (0,ia,iap)
                simu.PDref[it,k] = simu.ClosurePhasePD[it,ic]
                simu.GDref[it,k] = simu.ClosurePhaseGD[it,ic]

    
    return currPD, currGD




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
        - simu.PDEstimated: [MW,NIN] Estimated PD before subtraction of the reference
        - simu.GDEstimated: [MW,NIN] Estimated GD before subtraction of the reference
        - simu.CommandODL: Piston Command to send       [NT,NA]
        
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
    
    from . import simu
    from .simu import it
    
    currCfEstimated = args[0]
    
    currPD, currGD = ReadCf(currCfEstimated)
    
    simu.PDEstimated[it] = currPD
    simu.GDEstimated[it] = currGD#*config.FS['R']
    
    currCmd = SimpleCommandCalc(currPD,currGD)
    
    return currCmd
    

def SimpleCommandCalc(currPD,currGD, verbose=False):
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
    
    from .config import NA,NIN
    from .config import FT,FS
    
    it = simu.it            # Frame number
    
    """
    Group-Delay tracking
    """
    
    currGDerr = currGD - simu.GDref[it]
    
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
    simu.GDResidual[it] = currGDerr*R

    
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
        simu.GDCommand[it+1] = simu.GDCommand[it] + FT['GainGD']*config.PDspectra*config.FS['R']/(2*np.pi)*currGDerr
        # From OPD to Pistons
        simu.PistonGDCommand[it+1] = np.dot(FS['OPD2Piston'], simu.GDCommand[it+1])
        
    else:                       # integrator on PD
        # From OPD to Piston
        currPistonGD = np.dot(FS['OPD2Piston'], currGDerr)
        # Integrator
        simu.PistonGDCommand[it+1] = simu.PistonGDCommand[it] + FT['GainPD']*currPistonGD
    
    uGD = simu.PistonGDCommand[it+1]
    
    if config.FT['roundGD']=='round':
        for ia in range(NA):
            jumps = round(uGD[ia]/config.PDspectra)
            uGD[ia] = jumps*config.PDspectra

    elif config.FT['roundGD']=='int':
        for ia in range(NA):
            jumps = int(uGD[ia]/config.PDspectra)
            uGD[ia] = jumps*config.PDspectra
            
    elif config.FT['roundGD']=='no':
        pass

    else:
        if verbose:
            print("roundGD must be either 'round', 'int' or 'no'.")
       
    """
    Phase-Delay command
    """
    
    currPDerr = currPD - simu.PDref[it]
 
    # Keep the PD between [-Pi, Pi]
    # Eq. 35
    for ib in range(NIN):
        if currPDerr[ib] > np.pi:
            currPDerr[ib] -= 2*np.pi
        elif currPDerr[ib] < -np.pi:
            currPDerr[ib] += 2*np.pi
    
    # Store residual PD for display only [radians]
    simu.PDResidual[it] = currPDerr
    
    if config.FT['cmdOPD']:     # integrator on OPD
        # Integrator
        simu.PDCommand[it+1] = simu.PDCommand[it] + FT['GainPD']*config.PDspectra/(2*np.pi)*currPDerr
        # From OPD to Pistons
        simu.PistonPDCommand[it+1] = np.dot(FS['OPD2Piston'], simu.PDCommand[it+1])
        
    else:                       # integrator on PD
        # From OPD to Piston
        currPistonPD = np.dot(FS['OPD2Piston'], currPDerr)
        # Integrator
        simu.PistonPDCommand[it+1] = simu.PistonPDCommand[it] + FT['GainPD']*currPistonPD
    
    uPD = simu.PistonPDCommand[it+1]

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
    
    from .config import ron, G, enf
    from .simu import it
    
    (seedph, seedron) = (config.seedph+1,config.seedron+1)
    
    # Add SHOT noise
    rs = np.random.RandomState(seedph*it)
    photonADU = rs.poisson(inputADU/enf, size=inputADU.shape)*enf
    
    # Add DARK noise.
    rs = np.random.RandomState(seedron*it)
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
    
    from .config import ron, enf, qe
    from .simu import it
    
    (seedph, seedron) = (config.seedph+1,config.seedron+1)
    
    # Add SHOT noise (in the space of photons)
    rs = np.random.RandomState(seedph*it)
    photons = rs.poisson(inPhotons/enf, size=inPhotons.shape)*enf
    
    # Converts photons to electrons
    electrons = photons*qe  # Works when qe is float and when qe is array of length MW
    
    # Add DARK noise: here we assume DARK noise is only readout noise, so dark current is null.
    # That's why it can be modelised as a Gaussian noise.
    rs = np.random.RandomState(seedron*it)
    electrons_with_darknoise = electrons + rs.normal(scale=ron, size=electrons.shape)
        
    # Quantify ADU
    # ronADU = ron*G
    # roundADU = np.round(ronADU)
    # electrons_with_darknoise_and_quantification = roundADU/G
    
    outPhotons = electrons_with_darknoise/qe
    
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
            if len(value.encode('utf-8')) > 45:
                
                if ('CHARA' in value) and not ('interferometer' in value):
                    value = value.split('CHARA/')[-1].replace('µ','micro')
                else:
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
            cols.append(fits.Column(name=key+'.ich', format='I', array=np.array(array)[:,0]))
            cols.append(fits.Column(name=key+'mod', format='A', array=np.array(array)[:,1]))
            
        elif np.ndim(array) == 1:
                if isinstance(array[0], str):
                    form='A'
                else:
                    form='D'
                cols.append(fits.Column(name=key, format=form, array=array))
            
        elif np.ndim(array) == 2:
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
    # hdu = fits.BinTableHDU(t, name="Perf")
    
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