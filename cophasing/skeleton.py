# -*- coding: utf-8 -*-
#        import pdb; pdb.set_trace()

import os
import time

import numpy as np
# import cupy as cp # NumPy-equivalent module accelerated with NVIDIA GPU  

import matplotlib.pyplot as plt

from importlib import reload  # Python 3.4+ only.

from . import coh_tools
from . import config

from astropy.io import fits

from .decorators import timer


# Change the display font
# plt.rc('font', **{'family' : 'serif', 'serif' : ['Computer Modern Roman']})
# plt.rc('text', usetex = True)

def initialize(Interferometer, ObsFile, DisturbanceFile, NT=512, OT=1, MW = 5, ND=1, 
             spectra = [], spectraM=[],PDspectra=0,
             spectrum = [], mode = 'search',
             fs='default', TELref=0, FSfitsfile='', R = 0.5, dt=1,sigsky=[],imsky=[],
             ft = 'integrator', state = 0,
             noise=False,ron=0, qe=0.5, phnoise = 0, G=1, enf=1.5, M=1,
             seedph=100, seedron=100, seeddist=100,
             starttracking=100, latencytime=0,
             start_at_zero=True,display=False,
             **kwargs):
    
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

    filepath = Interferometer
    if not os.path.exists(filepath):
        raise Exception(f"{filepath} doesn't exist.")
    
    with fits.open(filepath) as hdu:
        ArrayParams = hdu[0].header
        NA, NIN = ArrayParams['NA'], ArrayParams['NIN']
        # TelData = hdu[1].data
        # BaseData = hdu[2].data
        
        # TelNames = TelData['TelNames']
        # TelCoordinates = TelData['TelCoordinates']
        # TelTransmissions = TelData['TelTransmissions']
        # TelSurfaces = TelData['TelSurfaces']
        # BaseNames = BaseData['BaseNames']
        # BaseCoordinates = BaseData['BaseCoordinates']
        
        
    # Redundant number of baselines
    NB = NA**2
    
    # Non-redundant number of Closure Phase
    NC = int((NA-2)*(NA-1))
   
    # NP = config.FS['NP']
    
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
      
    # Observation parameters
    config.ObservationFile = ObsFile
    config.DisturbanceFile = DisturbanceFile
    config.start_at_zero = start_at_zero
    config.NA=NA
    config.NB=NB
    config.NC=NC
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
    if sigsky:
        config.FS['sigsky'] = sigsky
    
    if noise:
        np.random.seed(seedron+60)
        config.FS['sigsky'] = np.random.randn([MW,config.FS['NP']])*ron
    
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
    
    # config.ich = np.zeros([NIN,2])
    # for ia in range(NA):
    #     for iap in range(ia+1,NA):
    #         config.ich[coh_tools.posk(ia,iap,NA)] = [ia,iap]
    
    config.CPindex = np.zeros([NC,3])
    for ia in range(NA):
        for iap in range(ia+1,NA):
            for iapp in range(iap+1,NA):
                config.CPindex[coh_tools.poskfai(ia,iap,iapp,NA)] = [ia,iap,iapp]

    return 


def update_params(DisturbanceFile):
    """
    Update any parameter the same way than first done in initialise function.

    Parameters
    ----------
    DisturbanceFile : STRING
        Disturbance file path.

    Returns
    -------
    None.

    """
    config.DisturbanceFile = DisturbanceFile
    
    return

def MakeAtmosphereCoherence(filepath, InterferometerFile, overwrite=False,
                            spectra=[], RefLambda=0, NT=1000,dt=1,
                            ampl=0, seed=100, dist='step', startframe = 10, 
                            f_fin=200, value_start=0, value_end=0,
                            r0=0.15,t0=10, L0=25, direction=0, d=1,
                            Levents=[],
                            TransDisturb=[],
                            debug=False, tel=0, highCF=True,pows=[],**kwargs):
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
        print(f'Disturbance file {filepath} exists.')
        if overwrite:
            os.remove(filepath)
            print(f'Parameter OVERWRITE is {overwrite}.')
        else:
            print(f'Parameter OVERWRITE is {overwrite}. Loading the disturbance scheme.')
            
            with fits.open(filepath) as hdu:
                CoherentFlux = hdu['RealCf'].data + hdu['ImagCf'].data*1j
                timestamps = hdu['TimeSampling'].data['timestamps']
                spectra = hdu['SpectralSampling'].data['spectra']
                
            return CoherentFlux, timestamps, spectra
       
    else:
        print(f'Creating the disturbance pattern and saving it in {filepath}')
        

    if not os.path.exists(InterferometerFile):
        raise Exception(f"{InterferometerFile} doesn't exist.")
    
    with fits.open(InterferometerFile) as hdu:
        ArrayParams = hdu[0].header
        NA, NIN = ArrayParams['NA'], ArrayParams['NIN']
        # ArrayName = ArrayParams['NAME']
        # TelData = hdu[1].data
        BaseData = hdu[2].data
        
        # TelNames = TelData['TelNames']
        # TelCoordinates = TelData['TelCoordinates']
        # BaseNames = BaseData['BaseNames']
        BaseCoordinates = BaseData['BaseCoordinates']
    
    # NB = NA**2
    NW = len(spectra)
    Lc = np.abs(1/(spectra[0]-spectra[1]))      # Coherence length
    
    # if not RefLambda:
    #     RefLmbda = np.mean(spectra)
    
    obstime = NT*dt                     # Observation time [ms]
    timestamps = np.arange(NT)*dt        # Time sampling [ms]
    
    # lmbdamin = 1/np.max(spectra)
    
    
# =============================================================================
#       TRANSMISSION DISTURBANCE
# =============================================================================

    TransmissionDisturbance = np.ones([NT,NW,NA])

    if TransDisturb:        # TransDisturb not empty
    
        typeinfo = TransDisturb['type'] # can be "sample" or "manual"
        if typeinfo == "sample":
            TransmissionDisturbance = TransDisturb['values']
            
        elif typeinfo == "manual":  # Format: TransDisturb['TELi']=[[time, duration, amplitude],...]
            for telescope in TransDisturb['tels']:
                itel = telescope-1
                tab = TransDisturb[f'TEL{telescope}']
                Nevents = np.shape(tab)[0]
                for ievent in range(Nevents):
                    tstart, duration, amplitude = tab[ievent]
                    istart = tstart//dt ; idur = duration//dt
                    TransmissionDisturbance[istart:istart+idur+1,:,itel] = amplitude
                    
        elif typeinfo == "fileMIRCx":
            
            file = TransDisturb["file"]
            d=fits.open(file) 
            p=d['PHOTOMETRY'].data
            
#             print("A file for the photometries has been given. It defines the spectral sampling of \
# the DisturbanceFile.")
            filespectra = d['WAVELENGTH'].data
            NWfile = len(spectra)
            
            
            
            filespectra[-1]
            
            
            Lc = np.abs(1/(spectra[0]-spectra[1]))      # Coherence length
            
            
            NT1,NT2,NWfile,NAfile = p.shape     # MIRCx data have a particular shape due to the camera reading mode
            inj = np.reshape(p[:,:,:,:],[NT1*NT2,NWfile,NAfile], order='C')
            inj = inj - np.min(inj)         # Because there are negative values
            inj = inj/np.max(inj)
            print(f"Max value: {np.max(inj)}, Moy: {np.mean(inj)}")
            NTfile = NT1*NT2
            
            if NTfile < NT:
                TransmissionDisturbance = repeat_sequence(inj, NT)
            else:
                TransmissionDisturbance = inj
            
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
    
    
    
# =============================================================================
#     PISTON DISTURBANCE
# =============================================================================
    
    PistonDisturbance = np.zeros([NT,NA])

    if dist == 'coherent':
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
            # Fc = 100            # Maximal frequency of the atmosphere
            Fmax = np.max(freq)
            # Nc = int(Fc*NT/2/Fmax)
            filtre = np.zeros(NT)
    
            # Atmospheric disturbance from Conan et al 1995
            for i in range(NT):
                if freq[i] < 0.02:
                    filtre[i] = 0
                elif freq[i] >= 0.02 or freq[i] < 3:    # Low frequencies regime 
                    filtre[i] = freq[i]**(-4/3)
                else:                                   # High frequencies regime
                    filtre[i] = freq[i]**(-8.5/3)
            print(Fmax,freq[1]-freq[0],dt*1e-3,NT)
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
                _,_,basedist,coords = coh_tools.get_array(config.Name, getcoords=True)
                
                
                
                baselines = np.zeros(NIN)
                for ia in range(NA):
                    for iap in range(ia+1,NA):
                        ib=coh_tools.posk(ia,iap,NA)
                        baselines[ib] = np.abs(basedist[ia*NA+iap])
                
                baselines = np.linalg.norm(BaseCoordinates, axis=1)
            
            V = 0.31*r0/t0*1e3              # Average wind velocity in its direction [m/s]
            L0 = L0                         # Outer scale [m]
            direction = direction           # Orientation from the North [deg] (positive toward East)
            d = d                           # Telescopes diameter [m]
                
            if ampl==0:
                wl_r0 = 0.55                # Wavelength at which r0 is defined
                rmsOPD = np.sqrt(6.88*(L0/r0)**(5/3))*wl_r0/(2*np.pi)    # microns
                print(f'RMS OPD={rmsOPD}')
                
            else:
                rmsOPD = ampl
                
            rmsPiston = rmsOPD/np.sqrt(2)
            for ia in range(NA):
                if tel:                 # Disturbances on only one pupil
                    itel = tel - 1
                    if ia != itel:
                        continue
    
                print(f'Piston on pupil {ia}')
    
                dfreq = np.min([0.008,1/(2.2*NT*dt*1e-3)]) # Minimal sampling wished
                freqmax = 1/(2*dt*1e-3)                  # Maximal frequency derived from given temporal sampling
                
                Npix = int(freqmax/dfreq)*2         # Array length (taking into account aliasing)
                
                freqfft = (np.arange(Npix)-Npix//2)*dfreq
                timefft = (np.arange(Npix)-Npix//2)*dt  #ms
            
                nu1 = 0.2*V/L0                 # Low cut-off frequency
                nu2 = 0.3*V/d                  # High cut-off frequency
            
                if not pows:
                    pow1, pow2, pow3 = (-2/3, -8/3, -17/3)  # Conan et al
                else:
                    pow1, pow2, pow3 = pows
                    
                b0 = nu1**(pow1-pow2)           # offset for continuity
                b1 = b0*nu2**(pow2-pow3)        # offset for continuity
                
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
                _,_,basedist,coords = coh_tools.get_array(config.Name, getcoords=True)
                
                
                
                baselines = np.zeros(NIN)
                for ia in range(NA):
                    for iap in range(ia+1,NA):
                        ib=coh_tools.posk(ia,iap,NA)
                        baselines[ib] = np.abs(basedist[ia*NA+iap])
                
                baselines = np.linalg.norm(BaseCoordinates, axis=1)
            
            V = 0.31*r0/t0*1e3              # Average wind velocity in its direction [m/s]
            L0 = L0                         # Outer scale [m]
            direction = direction           # Orientation from the North [deg] (positive toward East)
            d = d                           # Telescopes diameter [m]
                
            if ampl==0:
                wl_r0 = 0.55                # Wavelength at which r0 is defined
                rmsOPD = np.sqrt(6.88*(L0/r0)**(5/3))*wl_r0/(2*np.pi)    # microns
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
                
                print(f'Atmospheric cutoff frequencies: {nu1:.2}Hz and {nu2:.2}Hz')
                
                if highCF:
                    filtre = np.zeros(Npix)
                
                    for i in range(Npix):
                        checkpoint = int(Npix/10)
                        if i%checkpoint == 0:
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
                print(f'Done. Ellapsed time: {ElapsedTime}s')
        
    elif dist == 'chirp':
        f_fin = f_fin*1e-3   # Conversion to kHz
        omega_fin = 2*np.pi*f_fin
        t_fin = timestamps[-1]
        a = omega_fin/(2*t_fin)
        chirp = lambda phi0,t : np.sin(phi0 + a*t**2)
        if tel:
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

    
    print(f'Saving file into {filepath}')
    filedir = '/'.join(filepath.split('/')[:-1])    # remove the filename to get the file directory only
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    hdu.writeto(filepath)
    hdu.close()
    print('Saved.')
    
    return

      
# @timer
def loop(*args):
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

    from .config import NT, NA, timestamps, spectra, OW, MW
    
    # Reload simu module for initialising the observables with their shape
    print('Reloading simu for reinitialising the observables.')
    reload(simu)
    
    # Importation of the object 
    CfObj, CPObj = coh_tools.get_CfObj(config.ObservationFile,spectra)
    
    #scaling it to the spectral sampling  and integration time dt
    delta_wav = np.abs(spectra[1]-spectra[2])
    
    CfObj = CfObj * delta_wav           # Photons/spectralchannel/second at the entrance of the FS
    CfObj = CfObj * config.dt*1e-3      # Photons/spectralchannel/DIT at the entrance of the FS
    
    simu.ClosurePhaseObject = CPObj
    simu.CoherentFluxObject = CfObj
    
    # Importation of the disturbance
    CfDist, PistonDist, TransmissionDist = coh_tools.get_CfDisturbance(config.DisturbanceFile, spectra, timestamps)
    
    simu.CfDisturbance = CfDist
    simu.PistonDisturbance = PistonDist
    simu.TransmissionDisturbance = TransmissionDist    
    # simu.PhotometryDisturbance = np.zeros([config.NT,config.NW,config.NA])
    
    for ia in range(config.NA):
        PhotometryObject = np.abs(CfObj[:,ia*(config.NA+1)])
        simu.PhotometryDisturbance[:,:,ia] = simu.TransmissionDisturbance[:,:,ia]*PhotometryObject

    simu.FTmode[:config.starttracking] = np.zeros(config.starttracking)

    print("Processing simulation ...")
    
    simu.it = 0
    time0 = time.time()
    for it in range(NT):                        # We browse all the (macro)times
        simu.it = it
        
        # Coherence of the ODL
        CfODL = coh__pis2coh(-simu.CommandODL[it,:],1/config.spectra)
        
        currCfTrue = CfObj * simu.CfDisturbance[it,:,:] * CfODL
        simu.CfTrue[it,:,:] = currCfTrue
        
        """
        Fringe Sensor: From oversampled true coherences to macrosampled 
        measured coherences
        """
        fringesensor = config.FS['func']
        currCfEstimated = fringesensor(currCfTrue)
        simu.CfEstimated[it,:,:] = currCfEstimated

        """
        FRINGE TRACKER: From measured coherences to ODL commands
        """
        if simu.FTmode[it] == 0:
            GainGD = config.FT['GainGD']
            GainPD = config.FT['GainPD']
            config.FT['GainGD'] = 0
            config.FT['GainPD'] = 0
        fringetracker = config.FT['func']
        CmdODL = fringetracker(currCfEstimated)
        config.FT['GainGD'] = GainGD
        config.FT['GainPD'] = GainPD
        
        simu.CommandODL[it+config.latency,:] = CmdODL
        
        checkpoint = int(NT/10)
        if (it%checkpoint == 0) and (it!=0):
            processedfraction = it/NT
            # LeftProcessingTime = (time.time()-time0)*(1-processedfraction)/processedfraction
            print(f'Processed: {processedfraction*100}%, Elapsed time: {round(time.time()-time0)}s')

    
    # Process observables for visualisation
    simu.PistonTrue = simu.PistonDisturbance - simu.CommandODL[:-config.latency]

    # Save true OPDs in an observable
    for ia in range(config.NA):
        for iap in range(ia+1,config.NA):
            ib = coh_tools.posk(ia,iap,config.NA)
            simu.OPDTrue[:,ib] = simu.PistonTrue[:,ia] - simu.PistonTrue[:,iap]
            simu.OPDDisturbance[:,ib] = simu.PistonDisturbance[:,ia] - simu.PistonDisturbance[:,iap]
            simu.OPDCommand[:,ib] = simu.CommandODL[:,ia] - simu.CommandODL[:,iap]    
    
            for iow in range(MW):
                GammaObject = simu.CoherentFluxObject[iow*OW,ia*NA+iap]/np.sqrt(simu.CoherentFluxObject[iow*OW,ia*(NA+1)]*simu.CoherentFluxObject[iow*OW,iap*(NA+1)])
                
                Ia = np.abs(simu.CfTrue[:,iow*OW,ia*(NA+1)])    # Photometry pupil a
                Iap = np.abs(simu.CfTrue[:,iow*OW,iap*(NA+1)])  # Photometry pupil a'
                Iaap = np.abs(simu.CfTrue[:,iow*OW,ia*NA+iap])  # Mutual intensity aa'
                
                Lc = config.FS['R']*spectra[iow*OW]
                simu.VisibilityTrue[:,iow,ib] = Iaap/np.sqrt(Ia*Iap)*np.abs(GammaObject)*np.sinc(simu.OPDTrue[:,ib]/Lc)*np.exp(1j*2*np.pi*simu.OPDTrue[:,ib]/spectra[iow*OW])
    
    
    # print(args)
    if len(args):
        filepath = args[0]
        print(f'Saving infos in {filepath}')
        
        fileexists = os.path.exists(filepath)
        
        if fileexists:
            if 'overwrite' in args:
                os.remove(filepath)
            else:
                overwrite = (input(f'{filepath} already exists. Do you want to overwrite it? (y/n)') == 'y')
                if overwrite:
                    os.remove(filepath)
                else:
                    return
        
        
        hdr = fits.Header()
        hdr['Date'] = time.strftime("%a, %d %b %Y %H:%M", time.localtime())
        hdr['ObservationFile'] = config.ObservationFile
        hdr['DisturbanceFile'] = config.DisturbanceFile
        for key in ['NA', 'dt', 'ron', 'enf']:
            print(key)     
            hdr[key] = getattr(config, key)
            
        primary = fits.PrimaryHDU(header=hdr, data= config.spectra)
        col1 = fits.Column(name='OPDTrue', format='15D', array=simu.OPDTrue)
        col2 = fits.Column(name='PistonTrue', format='6D', array=simu.PistonTrue)
        
        coldefs = fits.ColDefs([col1, col2])
        hdu1 = fits.BinTableHDU.from_columns(coldefs, name='Observables' )

        hdu = fits.HDUList([primary, hdu1])
        
        hdu.writeto(filepath)
    
    return
        

# def SaveSimulation():
    
#     from . import simu
    
#     # infosimu = pd.DataFrame(coh)
    
#     # observables = pd.DataFrame({'timestamp':simu.timestamps,'PistonDisturbance':simu.PistonDisturbance,'pis_res':simu.PistonTrue,'cmd_odl':simu.CommandODL,\
#     #                            'PD_res':simu.PDResidual,'GD_res':simu.GDResidual,'PDCommand':simu.PDCommand,'GDCommand':simu.GDCommand,\
#     #                            'rmsPD':simu.rmsPD_,'rmsGD':simu.rmsGD_,\
#     #                                'phot_per':simu.TransmissionDisturbance,'phot_est':simu.PhotometryEstimated,'SquaredCoherenceDegree':simu.SquaredCoherenceDegree})
    
#     t2 = Table.from_pandas(df)
    
#     currentDT = datetime.datetime.now()
#     suffix=currentDT.strftime("%Y%m%d%H%M")
    
#     fits.writeto(prefix+suffix,np.array(t2))
    
    
        
def display(*args, wl=1.6,Pistondetails=False,OPDdetails=False,OneTelescope=True):
    
    '''
    NAME:
    COH__PLOT - Plots different interesting results from the simulation
    
    CALLING SEQUENCE:
        coh__plot(coh, pis_display=True, coher_display=True)
        
    PURPOSE:
        This procedure plots different results from the simulation, like \
        pistons commands and amplitudes of wavefronts function of time. \
        It is called at the end of a coh_turn routine to show these results\
        which have been stored in global variables during the loop.
        
    INPUTS:
        - args: write, as strings, the different observables you want to plot
        among: 'disturbance', phot', 'piston', 'opd','vis','detector'
        - wl: wavelength 
        
    '''
          
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    from cophasing.tol_colors import tol_cset
    
    from . import simu
    # import config
    
    from .simu import timestamps
    
    ind = np.argmin(np.abs(config.spectraM-wl))
    wl = config.spectraM[ind]
    
    from .config import NA,NT,NIN,OW
       
    ich = config.FS['ich']
    
    dt=config.dt

    increment=0
    if np.min(ich) == 1:
        increment = 1

    stationaryregim_start = config.starttracking+(config.starttracking-config.NT)*2//3
    if stationaryregim_start >= NT: stationaryregim_start=config.NT*1//3
    stationaryregim = range(stationaryregim_start,NT)
    
    print('Displaying observables...')
    print(f'First fig is Figure {config.newfig}')
    
    displayall = False
    if len(args)==0:
        print(args)
        displayall = True
        
    from matplotlib.ticker import AutoMinorLocator
    
    # Define the list of baselines
    baselines = []
    for ia in range(1,7):
        for iap in range(ia+1,7):
            baselines.append(f'{ia}{iap}')
            
    # Define the list of closures
    closures = []
    for iap in range(2,7):
        for iapp in range(iap+1,7):
            closures.append(f'{1}{iap}{iapp}')
    
    colors = tol_cset('muted')
    telcolors = tol_cset('bright')
    
    beam_patches = []
    for ia in range(NA):
        beam_patches.append(mpatches.Patch(color=telcolors[ia+1],label=f"Telescope {ia+increment}"))
    
    pis_max = 1.1*np.max([np.max(np.abs(simu.PistonDisturbance)),wl/2])
    pis_min = -pis_max
    ylim = [pis_min,pis_max]
    
    if displayall or ('disturbances' in args):
        
        fig = plt.figure("Disturbances")
        ax1,ax2,ax3 = fig.subplots(nrows=3,ncols=1)
        for ia in range(NA):
            # plt.subplot(NA,1,ia+1), plt.title('Beam {}'.format(ia+increment))
            ax1.plot(timestamps, simu.PistonDisturbance[:,ia],color=telcolors[ia+1])
        
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Piston [µm]')
        ax1.set_ylim(ylim)
        ax1.grid()
        ax1.set_title('Disturbance scheme at {:.2f}µm'.format(wl))
        ax1.legend(handles=beam_patches)

        if hasattr(simu, 'FreqSampling'):
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
                
        plt.show()
        config.newfig+=1    
        
        
    if displayall or ('phot' in args):
        s=(0,1.1*np.max(simu.PhotometryEstimated))
        linestyles=[]
        linestyles.append(mlines.Line2D([], [], color='black',
                                        linestyle='solid',label='Estimated'))    
        linestyles.append(mlines.Line2D([], [], color='black',
                                        linestyle='dashed',label='Disturbance'))
    
    
        plt.figure("Photometries")
        plt.suptitle('Photometries at {:.2f}µm'.format(wl))
        
        for ia in range(NA):
            plt.plot(timestamps, np.sum(simu.PhotometryDisturbance[:,OW*ind:OW*(ind+1),ia],axis=1),
                     color=telcolors[ia],linestyle='dashed')#),label='Photometry disturbances')
            plt.plot(timestamps, simu.PhotometryEstimated[:,ind,ia],
                     color=telcolors[ia],linestyle='solid')#,label='Estimated photometries')
            
        plt.vlines(config.starttracking*dt,s[0],s[1],
                   color='k', linestyle='--')
        plt.legend(handles=beam_patches+linestyles)
        plt.grid()
        plt.xlabel('Time (ms)')
        plt.ylim(s[0],s[1])
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
        # linestyles.append(mlines.Line2D([], [], color='black',
        #                                 linestyle='dotted',label='Command'))
        
        ax2ymax = np.max([np.max(np.abs(simu.PistonTrue)),wl/2])
        ax2ylim = [-ax2ymax,ax2ymax]
        fig = plt.figure("Pistons")
        plt.suptitle('Piston time evolution at {:.2f}µm'.format(wl))
        ax1 = fig.subplots()
        ax2 = ax1.twinx()
        
        if config.TELref:
            iTELref = config.TELref - 1
            PistonRef=simu.PistonTrue[:,iTELref]
        else:
            PistonRef=0
        
        for ia in range(NA):
            
            ax1.plot(timestamps, simu.PistonDisturbance[:,ia]-PistonRef,
                      color=telcolors[ia+1],linestyle='dashed')
            ax2.plot(timestamps, simu.PistonTrue[:,ia]-PistonRef,
                     color=telcolors[ia+1],linestyle='solid')
            plt.grid()
        
        ax1.vlines(config.starttracking*dt,ylim[0],ylim[1],
                   color='k', linestyle='--')
        ax2.vlines(config.starttracking*dt,ax2ylim[0],ax2ylim[1],
                       color='k', linestyle='--')
        ax2.set_ylabel('True Pistons [µm]')
        ax2.set_ylim(ax2ylim)
        ax1.set_ylabel('Disturbance Pistons [µm]')
        ax1.set_ylim(ylim)
        plt.xlabel('Time (ms)')
        plt.legend(handles=beam_patches+linestyles)
        plt.show()
        config.newfig+=1
    
        if Pistondetails:
            
            linestyles=[]
            linestyles.append(mlines.Line2D([], [], color='blue',
                                            linestyle='solid',label='Estimated'))    
            linestyles.append(mlines.Line2D([], [], color='red',
                                            linestyle='solid',label='Disturbance'))
            linestyles.append(mlines.Line2D([], [], color='green',
                                            linestyle='dotted',label='Command'))
            linestyles.append(mlines.Line2D([], [], color='green',
                                            linestyle=(0,(3,5,1,5)),label='PD Command'))
            linestyles.append(mlines.Line2D([], [], color='green',
                                            linestyle=(0,(3,5,1,5,1,5)),label='GD Command'))
            # linestyles.append(mlines.Line2D([], [], color='black',
            #                                 linestyle='dashdot',label='Search Command'))
            # linestyles.append(mlines.Line2D([], [], color='black',
            #                                 linestyle='dashdot',label='Modulation Command'))
        
        
            fig = plt.figure("Piston details")
            fig.suptitle('Piston time evolution at {:.2f}µm'.format(wl))
            axes = fig.subplots(nrows=NA,ncols=1, sharex=True)
            ax2ymax = np.max(np.abs(simu.PistonTrue))
            ax2ylim = [-ax2ymax,ax2ymax]
            for ia in range(NA):
                ax = axes[ia]
                ax.plot(timestamps, simu.PistonDisturbance[:,ia],
                         color='red',linestyle='solid')
                ax.plot(timestamps, simu.CommandODL[:-config.latency,ia],
                         color='green',linestyle='dashed')
                ax.plot(timestamps, simu.PistonPDCommand[:-config.latency,ia],
                         color='green',linestyle='dotted')
                ax.plot(timestamps, simu.PistonGDCommand[:-config.latency,ia],
                         color='green',linestyle=(0,(3,5,1,5,1,5)))
                ax2 = ax.twinx()
                ax2.plot(timestamps, simu.PistonTrue[:,ia],
                         color='blue',linestyle='solid')
                ax.set_ylim(ylim)
                ax2.set_ylim(ax2ylim)
                ax.set_ylabel(f'All Pistons except residual {ia+increment} [µm]')
                ax2.set_ylabel(f'Residual Piston {ia+increment} [µm]')
                ax.grid()
            
                ax.vlines(config.starttracking*dt,ylim[0],ylim[1],
                       color='k', linestyle='--')

            plt.xlabel('Time (ms)')
            plt.legend(handles=linestyles)
            plt.show()
            config.newfig+=1    


    if displayall or ('perftable' in args):
        
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
    
        from mypackage.plot_tools import setaxelim
        
        t = simu.timestamps ; timerange = range(NT)
        # NA=6 ; NIN = 15 ; NC = 10
        # nrows=int(np.sqrt(NA)) ; ncols=NA%nrows
        len2 = NIN//2 ; len1 = NIN-len2
        
        basecolors = colors[:len1]+colors[:len2]
        # basestyles = len1*['solid'] + len2*['dashed']
        # closurecolors = colors[:NC]
        
        R=config.FS['R']
        
        GD = simu.GDEstimated ; PD=simu.PDEstimated
        GDmic = GD*R*wl/2/np.pi ; PDmic = PD*wl/2/np.pi
        SquaredSNR = simu.SquaredSNRMovingAverage
        # gdClosure = simu.ClosurePhaseGD ; pdClosure = simu.ClosurePhasePD
        
        
        start_pd_tracking = 100
        
        RMSgdmic = np.std(GDmic[start_pd_tracking:,:],axis=0)
        RMSpdmic = np.std(PDmic[start_pd_tracking:,:],axis=0)
        # RMSgdc = np.std(gdClosure[start_pd_tracking:,:],axis=0)
        # RMSpdc = np.std(pdClosure[start_pd_tracking:,:],axis=0)
        
        
        
        plt.rcParams.update(rcParamsForBaselines)
        title='GD and PD'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        (ax1,ax6),(ax2,ax7), (ax3,ax8),(ax11,ax12),(ax4,ax9),(ax5,ax10) = fig.subplots(nrows=6,ncols=2, gridspec_kw={"height_ratios":[1,4,4,0.5,1,1]})
        ax1.set_title("First serie of baselines, from 12 to 25")
        ax6.set_title("Second serie of baselines, from 26 to 56")
        
        for iBase in range(len1):   # First serie
            ax1.plot(t[timerange],SquaredSNR[timerange,iBase],color=basecolors[iBase])
            ax2.plot(t[timerange],GDmic[timerange,iBase],color=basecolors[iBase])
            ax3.plot(t[timerange],PDmic[timerange,iBase],color=basecolors[iBase])
            
        for iBase in range(len1,NIN):   # Second serie
            ax6.plot(t[timerange],SquaredSNR[timerange,iBase],color=basecolors[iBase])
            ax7.plot(t[timerange],GDmic[timerange,iBase],color=basecolors[iBase])
            ax8.plot(t[timerange],PDmic[timerange,iBase],color=basecolors[iBase])
        
        
        ax4.bar(baselines[:len1],RMSgdmic[:len1], color=basecolors[:len1])
        ax5.bar(baselines[:len1],RMSpdmic[:len1], color=basecolors[:len1])
        
        ax9.bar(baselines[len1:],RMSgdmic[len1:], color=basecolors[len1:])
        ax10.bar(baselines[len1:],RMSpdmic[len1:], color=basecolors[len1:])
        
        ax1.sharex(ax3) ; ax2.sharex(ax3); ax6.sharex(ax8) ; ax7.sharex(ax8)
        ax6.sharey(ax1) ; ax6.tick_params(labelleft=False) ; setaxelim(ax1,ydata=SquaredSNR,ymin=0)
        ax7.sharey(ax2) ; ax7.tick_params(labelleft=False) ; setaxelim(ax2,ydata=GDmic)
        ax8.sharey(ax3) ; ax8.tick_params(labelleft=False) ; ax3.set_ylim([-wl/2,wl/2])
        ax9.sharey(ax4) ; ax9.tick_params(labelleft=False) ; setaxelim(ax4,ydata=RMSgdmic,ymin=0)
        ax10.sharey(ax5) ; ax10.tick_params(labelleft=False) ; setaxelim(ax5,ydata=np.concatenate([np.stack(RMSpdmic),[wl/5]]),ymin=0)
        
        ax4.sharex(ax5) ; ax4.tick_params(labelbottom=False)
        ax9.sharex(ax10) ; ax9.tick_params(labelbottom=False)
        
        ax1.set_ylabel('SNR²')
        ax2.set_ylabel('Group-Delays [µm]')
        ax3.set_ylabel('Phase-Delays [µm]')
        ax4.set_ylabel('GD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
        ax5.set_ylabel('PD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
        
        ax11.remove() ; ax12.remove()       # These axes are here to let space for ax3 and ax8 labels
        
        ax3.set_xlabel('Frames', labelpad=-10) ; ax8.set_xlabel('Frames', labelpad=-10)
        ax5.set_xlabel('Baselines') ; ax10.set_xlabel('Baselines')
        # figname = '_'.join(title.split(' ')[:3])
        # figname = 'GD&PD'
        # plt.savefig(datasave+f"{figname}.pdf")
        fig.show()


    if displayall or ('opd' in args):
        """
        OPD 
        """
        
        OPD_max = 1.1*np.max(np.abs([simu.OPDDisturbance,
                              simu.OPDTrue,
                              simu.OPDCommand[:-config.latency,:]]))
        OPD_min = -OPD_max
        ylim = [OPD_min,OPD_max]
    
        linestyles=[]
        linestyles.append(mlines.Line2D([], [], color='blue',
                                        linestyle='solid',label='Residual'))    
        linestyles.append(mlines.Line2D([], [], color='red',
                                        linestyle='solid',label='Disturbance'))
        linestyles.append(mlines.Line2D([], [], color='green',
                                        linestyle='dotted',label='Command'))
    
        DIT = min(50, config.NT - config.starttracking -1)
        ShowPerformance(float(timestamps[stationaryregim_start]), wl, DIT, display=False)
        NumberOfBaselinesToShow = 3
        for ia in range(NumberOfBaselinesToShow):
            fig = plt.figure(f"OPD {ia+increment}")
    #         fig.suptitle(f"OPD evolution at {wl:.2f}µm for baselines \n\
    # including telescope {ia+increment}")
            axes = fig.subplots(nrows=NumberOfBaselinesToShow,ncols=2,sharex=True, gridspec_kw={'width_ratios': [4, 1]})
            iap,iax=0,0
            for ax,axText in axes:
                ax2 = ax.twinx()
                ax2ymax = 1.1*np.max(np.abs(simu.OPDTrue[stationaryregim]))
                ax2ylim = [-ax2ymax,ax2ymax]
                # ax2ylim = [-wl/2,wl/2]
                if iap == ia:
                    iap+=1
                if ia < iap:
                    ib = coh_tools.posk(ia,iap,NA)
                    ax.plot(timestamps, simu.OPDDisturbance[:,ib],color='red')
                    ax.plot(timestamps, simu.OPDCommand[:-config.latency,ib],
                            color='green',linestyle='dotted')
                    ax2.plot(timestamps, simu.OPDTrue[:,ib],color='blue')
                else:
                    ib = coh_tools.posk(iap,ia,NA)
                    ax.plot(timestamps, -simu.OPDDisturbance[:,ib],color='red')
                    ax.plot(timestamps, -simu.OPDCommand[:-config.latency,ib],
                            color='green',linestyle='dotted')                
                    ax2.plot(timestamps, -simu.OPDTrue[:,ib],color='blue')
                
                ax2.hlines(np.mean(simu.OPDTrue[stationaryregim,ib]),0,NT*dt,
                           linestyle='-.',color='blue')
                ax.vlines(config.starttracking*dt,ylim[0],ylim[1],
                   color='k', linestyle='--')
                
                axText.text(0.5,0.5,f"{np.sqrt(simu.VarOPD[ib])*1e3:.0f}nm RMS")
                axText.axis("off")
                
                ax.set_ylim(ylim)
                ax2.set_ylim(ax2ylim)
                if ax2ymax > wl:
                    ax2.set_yticks([-ax2ylim[0],-wl,0,wl,ax2ylim[1]])
                else:
                    ax2.set_yticks([-wl,0,wl])
                ax.set_ylabel(f'OPD ({ia+1},{iap+1})\n [µm]')
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
                
                
                iap += 1
                iax+=1
            # plt.tight_layout()
            ax.set_xlabel('Time (ms)')
            plt.show()
            plt.legend(handles=linestyles)
            config.newfig+=1
            
            if OneTelescope:
                break
        
    
        if OPDdetails:
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
                                            color='black',linestyle=':'))
            linestyles.append(mlines.Line2D([], [],label='PD Residuals',
                                            color='black',linestyle='-'))

            for ia in range(NA):
                fig = plt.figure(f"OPD details {ia+increment}")
                fig.suptitle(f"OPD evolution at {wl:.2f}µm for baselines \n\
        including telescope {ia+increment}")
                axes = fig.subplots(nrows=NA-1,ncols=3,sharex=True,gridspec_kw={'width_ratios': [8, 1,1]})
                iap,iax=0,0
                for ax,axText,axLegend in axes:
                    ax2 = ax.twinx()
                    ax2ymax = 1.1*np.max(np.abs(simu.GDEstimated[stationaryregim,:]*config.FS['R']/config.FT['Ncross']*wl/(2*np.pi)))
                    ax2ylim = [-ax2ymax,ax2ymax]
                    if iap == ia:
                        iap+=1
                    if ia < iap:
                        ib = coh_tools.posk(ia,iap,NA)
                        ax.plot(timestamps, simu.OPDDisturbance[:,ib],
                                color='red')
                        ax2.plot(timestamps, simu.GDCommand[:-config.latency,ib],
                                color='green',linestyle='dotted')
                        ax2.plot(timestamps, simu.GDResidual[:,ib]*wl/(2*np.pi),
                                 color='black',linestyle=':')
                        ax2.plot(timestamps, simu.PDResidual[:,ib]*wl/(2*np.pi),
                                 color='black',linestyle='-')
                    else:
                        ib = coh_tools.posk(iap,ia,NA)
                        ax.plot(timestamps, -simu.OPDDisturbance[:,ib],color='red')
                        ax.plot(timestamps, -simu.GDCommand[:-config.latency,ib],
                                color='green',linestyle='dotted')
                        ax2.plot(timestamps, -simu.GDResidual[:,ib]*wl/(2*np.pi), color='black',
                                 linestyle=':')
                        ax2.plot(timestamps, -simu.PDResidual[:,ib]*wl/(2*np.pi), color='black',
                                 linestyle='-')
                    
                    ax.vlines(config.starttracking*dt,ylim[0],ylim[1],
                              color='k', linestyle='--')
                    
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
        
                if OneTelescope:
                    break
            
        if 'ODPcmd' in args:
            OPD_max = 1.1*np.max(np.abs([simu.OPDDisturbance,
                                  simu.GDCommand[:-config.latency,:]]))
            OPD_min = -OPD_max
            ylim = [OPD_min,OPD_max]
        
            linestyles=[]
            linestyles.append(mlines.Line2D([], [],label='Disturbance',
                                            color='red',linestyle='solid'))
            linestyles.append(mlines.Line2D([], [],label='Command OPD',
                                            color='blue',linestyle='solid'))
            linestyles.append(mlines.Line2D([], [],label='GD Estimated',
                                            color='black',linestyle='--'))
            linestyles.append(mlines.Line2D([], [],label='PD Estimated',
                                            color='black',linestyle=':'))

            for ia in range(NA):
                fig = plt.figure(f"OPD details {ia+increment}")
                fig.suptitle(f"OPD evolution at {wl:.2f}µm for baselines \n\
        including telescope {ia+increment}")
                axes = fig.subplots(nrows=NA-1,ncols=3,sharex=True,gridspec_kw={'width_ratios': [8, 1,1]})
                iap,iax=0,0
                for ax,axText,axLegend in axes:
                    ax2 = ax.twinx()
                    ax2ymax = 1.1*np.max(np.abs(simu.GDEstimated*config.FS['R']/config.FT['Ncross']*wl/(2*np.pi)))
                    ax2ylim = [-ax2ymax,ax2ymax]
                    if iap == ia:
                        iap+=1
                    if ia < iap:
                        ib = coh_tools.posk(ia,iap,NA)
                        ax.plot(timestamps, simu.OPDDisturbance[:,ib],
                                color='red')
                        # ax2.plot(timestamps, simu.GDCommand[:-config.latency,ib],
                        #         color='blue')
                        ax2.plot(timestamps, simu.OPDCommand[:-config.latency,ib],
                                color='blue')
                        ax2.plot(timestamps, simu.GDResidual[:,ib]*wl/(2*np.pi),
                                 color='black',linestyle='-.')
                        ax2.plot(timestamps, simu.PDResidual[:,ib]*wl/(2*np.pi),
                                 color='black',linestyle='-')
                    else:
                        ib = coh_tools.posk(iap,ia,NA)
                        ax.plot(timestamps, -simu.OPDDisturbance[:,ib],color='red')
                        ax.plot(timestamps, -simu.GDCommand[:-config.latency,ib],
                                color='green',linestyle=(0,(3,5,1,5,1,5)))
                        ax2.plot(timestamps, -simu.GDResidual[:,ib]*wl/(2*np.pi), color='black',
                                 linestyle='--')
                        ax2.plot(timestamps, -simu.PDResidual[:,ib]*wl/(2*np.pi), color='black',
                                 linestyle=':')
                    
                    ax.vlines(config.starttracking*dt,ylim[0],ylim[1],
                       color='k', linestyle='--')

                    ax.set_ylim(ylim)
                    ax.set_ylabel(f'[{ia+1},{iap+1}] [µm]')
                    
                    ax2.set_ylim(ax2ylim)
                    ax2.set_ylabel('Residuals')
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
                plt.show()
                axLegend.legend(handles=linestyles)
                config.newfig+=1
        
                if OneTelescope:
                    break


        if 'OPDgathered' in args:

            OPD_max = 1.1*np.max(np.abs(simu.OPDTrue[:,ib]))
            OPD_min = -OPD_max
            s = [OPD_min,OPD_max]

            fig = plt.figure("OPD on one window")
            ax = fig.subplots(nrows=1)
            
            for ib in range(config.NIN):
                
                ax.plot(timestamps, simu.OPDTrue[:,ib], linestyle='-', label=f'{ich[ib]}: {np.sqrt(simu.VarOPD[ib])*1e3:.0f}nm RMS')
                
            ax.vlines(config.starttracking*dt,s[0],s[1],
                   color='k', linestyle='--')
            ax.set_ylim(s)
            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('OPD [µm]')
            plt.show()
            ax.legend()
            config.newfig+=1



    if displayall or ('cp' in args):
        """
        CLOSURE PHASES
        """
        
        
        linestyles=[]
        linestyles.append(mlines.Line2D([], [],linestyle='solid',
                                        label='Estimated'))    
        linestyles.append(mlines.Line2D([], [],linestyle='dashed',
                                        label='Object'))
        
        
        ymax = np.pi
        ylim = [-ymax, ymax]
        fig = plt.figure('Closure Phases')
        fig.suptitle('Closure Phases')
        ax1,ax2 = fig.subplots(nrows=2, ncols=1)
        
        # Plot on ax1 the (NA-1)(NA-2)/2 independant Closure Phases
        for ia in range(1,NA):
            for iap in range(ia+1,NA):
                ic = coh_tools.poskfai(0,ia,iap,NA)
                # if ia == 0
                ax1.plot(timestamps, simu.ClosurePhasePD[:,ic],
                         color=colors[ic])
                ax1.hlines(simu.ClosurePhaseObject[ind,ic], 0, timestamps[-1], 
                           color=colors[ic], linestyle='--')
                
        # Plot on ax2 the (NA-1)(NA-2)/2 (independant?) other Closure Phases
        for ia in range(1,NA):
            for iap in range(ia+1,NA):
                for iapp in range(iap+1,NA):
                    ic = coh_tools.poskfai(ia,iap,iapp,NA)
                    colorindex = int(ic - config.NC//2)
                    ax2.plot(timestamps, simu.ClosurePhasePD[:,ic],
                             color=colors[colorindex])
                    ax2.hlines(simu.ClosurePhaseObject[ind,ic], 0, timestamps[-1],
                               color=colors[colorindex], linestyle='--')
        
        ax1.vlines(config.starttracking*dt,ylim[0],ylim[1],
                   color='k', linestyle='--')
        ax2.vlines(config.starttracking*dt,ylim[0],ylim[1],
                   color='k', linestyle='--')
        plt.xlabel('Time [ms]')
        plt.ylabel('Closure Phase [rad]')
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)
        ax1.grid()
        ax2.grid()
        plt.show()
        plt.legend(handles=linestyles)
        config.newfig+=1
        

    if displayall or ('vis' in args):
        """
        VISIBILITIES
        """
    
        ylim =[0,1.1]
        # Squared Visibilities
        for ia in range(NA):
            fig = plt.figure(f"Squared Vis {ia+increment}")
            fig.suptitle(f"Squared visibility |V|² at {wl:.2f}µm for baselines \n\
    including telescope {ia+increment}")
            axes = fig.subplots(nrows=NA-1,ncols=1,sharex=True)
            iap=0
            for ax in axes:
                if iap == ia:
                    iap+=1
                
                ib = coh_tools.posk(ia,iap,NA)
                ax.plot(timestamps, np.abs(simu.VisibilityEstimated[:,ind,ib]), color='k')
                ax.plot(timestamps, np.abs(simu.VisibilityTrue[:,ind,ib]),color='k',linestyle='--')
                ax.set_ylim(ylim)
                ax.set_ylabel(f'[{ia+1},{iap+1}] [µm]')
    
                ax.grid()
                iap += 1
                ax.vlines(config.starttracking*dt,ylim[0],ylim[1],
                       color='k', linestyle='--')
            plt.xlabel('Time (ms)')
            plt.show()
            config.newfig+=1
            
            if OneTelescope:
                break
    
    
        # Phase of the visibilities
        VisibilityPhase = np.angle(simu.VisibilityEstimated[:,ind,:])
        ymax = np.pi #2*np.max(np.abs(VisibilityPhase))
        ylim = [-ymax,ymax]
        for ia in range(NA):
            fig = plt.figure(f"Phase Vis {ia+increment}")
            fig.suptitle(f"Visibility phase \u03C6 at {wl:.2f}µm for baselines \n\
    including telescope {ia+increment}")
            axes = fig.subplots(nrows=NA-1,ncols=1,sharex=True)
            iap=0
            for iax in range(len(axes)):
                ax = axes[iax]
                if iap == ia:
                    iap+=1
                
                ib = coh_tools.posk(ia,iap,NA)
                ax.plot(timestamps, np.angle(simu.VisibilityEstimated[:,ind,ib]), color='k')
                ax.plot(timestamps, np.angle(simu.VisibilityTrue[:,ind,ib]),color='k',linestyle='--')
                ax.set_ylim(ylim)
                ax.set_ylabel(f'[{ia+1},{iap+1}] [µm]')
                ax.grid()
                iap += 1
                fig.subplots_adjust(right=0.8)
                RMS_ax = fig.add_axes([0.82, 1-1/NA*(iax+1), 0.1, 0.9/NA])
                RMS_ax.text(0,0,f"{np.std(VisibilityPhase[stationaryregim,ib])/(2*np.pi):.2f}\u03BB RMS")
                RMS_ax.axis("off")
                ax.vlines(config.starttracking*dt,ylim[0],ylim[1],
                       color='k', linestyle='--')
            plt.xlabel('Time (ms)')
            plt.show()
            config.newfig+=1
    
            if OneTelescope:
                break

    if displayall or ('detector' in args):
        """
        DETECTOR VIEW
        """
        fig = plt.figure("Detector intensities")
        axes = fig.subplots()
        plt.suptitle('Intensities recorded by the detector at {:.2f}µm'.\
                     format(wl))
        
        if config.fs == 'default':
            NMod = 1
        if 'spica' in config.fs or 'abcd' in config.fs:
            NMod = config.FS['NM']
            
        NIN = config.NIN
        NP = config.FS['NP']
        NMod = config.FS['NMod']
        
        for ip in range(NP):
            
            ax = plt.subplot(NIN,NMod,ip+1)
            if ip < NMod:
                ax.set_title(config.FS['Modulations'][ip])
            im = plt.imshow(np.transpose(np.dot(np.reshape(simu.MacroImages[:,ind,ip],[NT,1]), \
                                                np.ones([1,100]))), vmin=np.min(simu.MacroImages), vmax=np.max(simu.MacroImages))    
                
            plt.tick_params(axis='y',left='off')
            if ip//NMod == ip/NMod:
                plt.ylabel(str(int(ich[ip//NMod,0]))+str(int(ich[ip//NMod,1])))
                
            if ip>=NP-NMod:
                plt.xticks([0,NT],[0,NT*dt])
                plt.xlabel('Time (ms)') 
            else:
                plt.xticks([],[])
            plt.yticks([],[])
                
    
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()
        config.newfig+=1

    if ('state' in args):
        # ylim=[-0.1,2*config.FT['ThresholdGD']**2]
        ylim=[1e-1,np.max(simu.SquaredSNRMovingAverage[:,ib])]
        # State-Machine and SNR
        fig = plt.figure("SNR²")
        fig.suptitle("SNR² and State-Machine")
        ax,ax2 = fig.subplots(nrows=2,ncols=1, sharex=True)
        for ib in range(NIN):
            ax.plot(timestamps, simu.SquaredSNRMovingAverage[:,ib],
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
                   color='k', linestyle='--')
        ax2.vlines(config.starttracking*dt,0,NA,
                   color='k', linestyle='--')
        ax.set_ylabel(f"<SNR>_{config.FT['Ngd']}")
        ax2.set_ylabel("Rank of Igd")
        ax2.set_xlabel('Time (ms)')
        ax2.legend()
        ax2.grid()
        ax.legend()
        plt.show()
        config.newfig+=1



def ShowPerformance(TimeBonds, WavelengthOfInterest,DIT, display=True, get=[]):
    """
    Processes the performance of the fringe-tracking starting at the StartingTime
    Observables processed:
        -VarOPD                 # Temporal Variance OPD [µm]
        -TempVarPD              # Temporal Variance PD [rad]
        -TempVarGD              # Temporal Variance of GD [rad]
        -VarCPD                 # Temporal Variance of CPD [rad]
        -VarCGD                 # Temporal Variance of CGD [rad]
        -FringeContrast         # Fringe Contrast [0,1] at given wavelengths
WavelengthOfInterest
        

    Parameters
    ----------
    TimeBonds : INT or ARRAY [ms]
        If int:
            The performance are processed from StartingTime until the end
        If array [StartingTime,EndingTime]: 
            The performance are processed between StartingTime and EndingTime
    WavelengthOfInterest : ARRAY
        Wavelength when the Fringe Contrast needs to be calculated.
    DIT : INT
        Integration time of the science instrument [ms]
    Returns
    -------
    None.

    """
    from . import simu
    from . import config
    
    ich = config.FS['ich']
    
    from .config import NIN,dt,NT
    
    WOI = WavelengthOfInterest
    if isinstance(WOI, (float,np.float32,np.float64)):
        WOI = [WOI]    
    NW = len(WOI) 
    
    DIT_NumberOfFrames = int(DIT/dt)
    
    if isinstance(TimeBonds,(float,int)):
        Period = int(NT - TimeBonds/dt)
        InFrame = round(TimeBonds/dt)
    elif isinstance(TimeBonds,(np.ndarray,list)):
        Period = int((TimeBonds[1]-TimeBonds[0])/dt)
        InFrame = round(TimeBonds[0]/dt)
    else:
        raise '"TimeBonds" must be instance of (float,int,np.ndarray,list)'
        
   
    simu.FringeContrast=np.zeros([NW,NIN])      # Fringe Contrast at given wavelengths [0,1]
    simu.VarOPD=0
    simu.TempVarPD=0 ; simu.TempVarGD=0
    simu.VarCPD =0; simu.VarCGD=0
    
    Ndit = Period//DIT_NumberOfFrames
    
    for it in range(Ndit):
        OutFrame=InFrame+DIT_NumberOfFrames
        simu.VarOPD += 1/Ndit*np.var(simu.OPDTrue[InFrame:OutFrame,:],axis=0)
        simu.TempVarPD += 1/Ndit*np.var(simu.PDEstimated[InFrame:OutFrame,:],axis=0)
        simu.TempVarGD += 1/Ndit*np.var(simu.GDEstimated[InFrame:OutFrame,:],axis=0)
        simu.VarCPD += 1/Ndit*np.var(simu.ClosurePhasePD[InFrame:OutFrame,:],axis=0)
        simu.VarCGD += 1/Ndit*np.var(simu.ClosurePhaseGD[InFrame:OutFrame,:],axis=0)
        # Fringe contrast
        for iwl in range(NW):
            wl = WOI[iwl]
            for ib in range(NIN):   
                simu.FringeContrast[iwl,ib] += 1/Ndit*np.abs(np.mean(np.exp(1j*2*np.pi*simu.OPDTrue[InFrame:OutFrame,ib]/wl)))
    
        InFrame += DIT_NumberOfFrames
        
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
    
    
    observable = simu.TempVarPD*(config.PDspectra/2/np.pi)
    
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
    
    
    observable = simu.TempVarGD*(config.PDspectra/2/np.pi)*config.FS['R']
    
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
        plt.scatter(WOI,simu.FringeContrast[:,ib], label=f'{ich[ib]}')
        
    plt.legend(), plt.grid()
    plt.show()
    config.newfig += 1
    
    return    


def SpectralAnalysis(OPD = (1,2)):
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
    from .config import NA, NT, dt, latency
    
    tel1 = OPD[0]-1
    tel2 = OPD[1]-1
    
    ib = coh_tools.posk(tel1, tel2, NA)
    
    FrequencySampling = np.fft.fftfreq(NT, dt*1e-3)
    PresentFrequencies = (FrequencySampling >= 0) & (FrequencySampling < 200)
    FrequencySampling = FrequencySampling[PresentFrequencies]
    
    Residues = simu.OPDTrue[:,ib]
    Turb = simu.OPDDisturbance[:,ib]
    Command = simu.OPDCommand[:,ib]
    
    FTResidues = np.fft.fft(Residues)[PresentFrequencies]
    FTTurb = np.fft.fft(Turb)[PresentFrequencies]
    FTCommand = np.fft.fft(Command[:-latency])[PresentFrequencies]
    
    FTrej = FTResidues/FTTurb
    FTBO = FTCommand/FTResidues
    FTBF = FTCommand/FTTurb


    plt.figure('Rejection Transfer Function')
    plt.plot(FrequencySampling, np.abs(FTrej))
    # plt.plot(FrequencySampling, FrequencySampling*10**(-2), linestyle='--')
    plt.xlabel('Frequencies [Hz]')
    plt.ylabel('Normalised')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    plt.figure('FTBO')
    plt.plot(FrequencySampling, np.abs(FTBO))
    plt.xlabel('Frequencies [Hz]')
    plt.ylabel('Normalised')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    plt.figure('FTBF')
    plt.plot(FrequencySampling, np.abs(FTBF))
    plt.xlabel('Frequencies [Hz]')
    plt.ylabel('Normalised')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


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
        - simu.VisibilityEstimated                    [NT,MW,NIN]*1j
        - simu.SquaredCoherenceDegree                      [NT,MW,NIN]
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
            ib = coh_tools.posk(ia,iap,NA)
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
        # cs = 0*1j
        cfGDlmbdas = simu.CfGD[it,:-Ncross,ib]*np.conjugate(simu.CfGD[it,Ncross:,ib])
        cfGDmoy = np.sum(cfGDlmbdas)

            
        currGD[ib] = np.angle(cfGDmoy)    # Group-delay on baseline (ib).
    
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
    
    timerange = range(it+1-Ncp,it+1)
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib = coh_tools.posk(ia,iap,NA)      # coherent flux (ia,iap)    
            cs1 = np.sum(simu.CfPD[timerange,:,ib], axis=1)     # Sum of coherent flux (ia,iap)
            cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
            cfGDmoy1 = np.sum(cfGDlmbdas,axis=1)     # Sum of coherent flux (ia,iap)  
            for iapp in range(iap+1,NA):
                ib = coh_tools.posk(iap,iapp,NA) # coherent flux (iap,iapp)    
                cs2 = np.sum(simu.CfPD[timerange,:,ib], axis=1) # Sum of coherent flux (iap,iapp)    
                cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
                cfGDmoy2 = np.sum(cfGDlmbdas,axis=1)
                
                ib = coh_tools.posk(ia,iapp,NA) # coherent flux (iapp,ia)    
                cs3 = np.sum(np.conjugate(simu.CfPD[timerange,:,ib]),axis=1) # Sum of 
                cfGDlmbdas = simu.CfGD[timerange,Ncross:,ib]*np.conjugate(simu.CfGD[timerange,:-Ncross,ib])
                cfGDmoy3 = np.sum(cfGDlmbdas,axis=1)
                
                # The bispectrum of one time and one triangle adds up to
                # the Ncp last times
                ic = coh_tools.poskfai(ia,iap,iapp,NA)        # 0<=ic<NC=(NA-2)(NA-1) 
                
                bispectrumPD[ic]=np.sum(cs1*cs2*cs3)
                bispectrumGD[ic]=np.sum(cfGDmoy1*cfGDmoy2*np.conjugate(cfGDmoy3))
    
    
    
    
    # for iot in range(it+1-Ncp,it+1):          # integration on time Ncp
    #     # cs = 0*1j
    #     for ia in range(NA):
    #         for iap in range(ia+1,NA):
    #             ib = coh_tools.posk(ia,iap,NA)      # coherent flux (ia,iap)    
    #             cs1 = np.sum(simu.CfPD[iot,:,ib])     # Sum of coherent flux (ia,iap)
    #             cfGDlmbdas = simu.CfGD[iot,Ncross:,ib]*np.conjugate(simu.CfGD[iot,:-Ncross,ib])
    #             cfGDmoy1 = np.sum(cfGDlmbdas)     # Sum of coherent flux (ia,iap)  
    #             for iapp in range(iap+1,NA):
    #                 ib = coh_tools.posk(iap,iapp,NA) # coherent flux (iap,iapp)    
    #                 cs2 = np.sum(simu.CfPD[iot,:,ib]) # Sum of coherent flux (iap,iapp)    
    #                 cfGDlmbdas = simu.CfGD[iot,Ncross:,ib]*np.conjugate(simu.CfGD[iot,:-Ncross,ib])
    #                 cfGDmoy2 = np.sum(cfGDlmbdas)
                    
    #                 ib = coh_tools.posk(ia,iapp,NA) # coherent flux (iapp,ia)    
    #                 cs3 = np.sum(np.conjugate(simu.CfPD[iot,:,ib])) # Sum of 
    #                 cfGDlmbdas = simu.CfGD[iot,Ncross:,ib]*np.conjugate(simu.CfGD[iot,:-Ncross,ib])
    #                 cfGDmoy3 = np.sum(cfGDlmbdas)
                    
    #                 # The bispectrum of one time and one triangle adds up to
    #                 # the Ncp last times
    #                 ic = coh_tools.poskfai(ia,iap,iapp,NA)        # 0<=ic<NC=(NA-2)(NA-1) 
                    
    #                 bispectrumPD[ic]+=cs1*cs2*cs3
    #                 bispectrumGD[ic]+=cfGDmoy1*cfGDmoy2*np.conjugate(cfGDmoy3)
                    
    simu.ClosurePhasePD[it] = np.angle(bispectrumPD)
    simu.ClosurePhaseGD[it] = np.angle(bispectrumGD)
    
    if it%Ncp == 0:                     # At time 0, we create the reference vectors
        for ia in range(1,NA-1):
            for iap in range(ia+1,NA):
                k = coh_tools.posk(ia,iap,NA)
                ic = coh_tools.poskfai(0,ia,iap,NA)   # Position of the triangle (0,ia,iap)
                if config.FT['usePDref']:
                    simu.PDref[k] = simu.ClosurePhasePD[it,ic]
                    simu.GDref[k] = simu.ClosurePhaseGD[it,ic]
                else:
                    simu.PDref[k] = 0
                    simu.GDref[k] = 0
    
    return currPD, currGD


def SimpleIntegrator(*args, init=False, Ngd=1, Npd=1, Ncp = 1, GainPD=0, GainGD=0,
                      Ncross = 1, CPref=True,roundGD=True,Threshold=True, usePDref=True):
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
        config.FT['usePDref'] = usePDref
        
        from .config import NIN,NA
        config.FT['Piston2OPD'] = np.zeros([NIN,NA])    # Piston to OPD matrix
        config.FT['OPD2Piston'] = np.zeros([NA,NIN])    # OPD to Piston matrix
        
        from .coh_tools import posk
        
        for ia in range(NA):
            for iap in range(ia+1,NA):
                ib = posk(ia,iap,NA)
                config.FT['Piston2OPD'][ib,ia] = 1
                config.FT['Piston2OPD'][ib,iap] = -1
                config.FT['OPD2Piston'][ia,ib] = 1
                config.FT['OPD2Piston'][iap,ib] = -1

        config.FT['OPD2Piston'] = config.FT['OPD2Piston']/NA
        
        if config.TELref:
            iTELref = config.TELref - 1
            L_ref = config.FT['OPD2Piston'][iTELref,:]
            config.FT['OPD2Piston'] = config.FT['OPD2Piston'] - L_ref
        
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
    

def SimpleCommandCalc(currPD,currGD):
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
    from .config import FT
    
    it = simu.it            # Frame number
    
    """
    Group-Delay tracking
    """
    
    currGDerr = currGD - simu.GDref
    
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
        simu.GDCommand[it] = simu.GDCommand[it-1] + FT['GainGD']*currGDerr
        # From OPD to Pistons
        simu.PistonGDCommand[it] = np.dot(FT['OPD2Piston'], simu.GDCommand[it])
        
    else:                       # integrator on PD
        # From OPD to Piston
        currPistonGD = np.dot(FT['OPD2Piston'], currGDerr)
        # Integrator
        simu.PistonGDCommand[it] = simu.PistonGDCommand[it-1] + FT['GainPD']*currPistonGD
    
    uGD = simu.PistonGDCommand[it]
    
    if config.FT['roundGD']:
        for ia in range(NA):
            jumps = round(uGD[ia]/config.PDspectra)
            uGD[ia] = jumps*config.PDspectra
            
    
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
    
    # Store residual PD for display only [radians]
    simu.PDResidual[it] = currPDerr
    
    if config.FT['cmdOPD']:     # integrator on OPD
        # Integrator
        simu.PDCommand[it] = simu.PDCommand[it-1] + FT['GainPD']*currPDerr
        # From OPD to Pistons
        simu.PistonPDCommand[it] = np.dot(FT['OPD2Piston'], simu.PDCommand[it])
        
    else:                       # integrator on PD
        # From OPD to Piston
        currPistonPD = np.dot(FT['OPD2Piston'], currPDerr)
        # Integrator
        simu.PistonPDCommand[it] = simu.PistonPDCommand[it-1] + FT['GainPD']*currPistonPD
    
    uPD = simu.PistonPDCommand[it]

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



def repeat_sequence(sequence, newNT):
    NT = len(sequence)
    if NT > newNT:
        print(f"The given sequence is longer than the desired length, we take only the {newNT} elements.")
        newseq = sequence[:newNT]
    elif NT==newNT:
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