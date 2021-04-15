# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:14:42 2020

@author: cpannetier
"""

import os

datadir = 'data/'

import pandas as pd
import numpy as np
from scipy.special import jv
from scipy import interpolate

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from mypackage.tol_colors import tol_cset # colorblind-friendly library
import matplotlib.patches as mpatches

from astropy.io import fits

import matplotlib.pyplot as plt

from . import config
import cophasing.decorators as deco

global h_, c_, k_

h_ = 6.626e-34  # Planck's constant
c_ = 3.0e+8     # Light velocity
k_ = 1.38e-23    # Boltzmann's constant


class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


def BBstar(wav, T):
    """
    Returns the spectral luminance of a black-body given its temperature.
    Will be useful to give the shape of the spectra of a star.

    Parameters
    ----------
    wav : [NW] float [meter]
        Wavelength in meter.
    T : float [K]
        Temperature of the Black-Body in Kelvin.

    Returns
    -------
    luminance : [NW] floats [W/s/m²/sr]
        Luminance of the source per steradian.

    """
    global h_, c_, k_
    
    a = 2.0*h_*c_**2
    b = h_*c_/(wav*k_*T)
    luminance = a / ( (wav**5) * (np.exp(b) - 1.0) ) # W/m²/sr/µm
    
    # nu = c_/spectra   # Electromagnetic frequency [Hz]
    # a = 2.*h_/c_**2
    # b = h_/k_/T
    # B_sig = a*nu**3/(np.exp(b*nu)-1)  # W/m²/sr/s
    
    return luminance


def info_array(array, band):
    """
    Stores the transmission efficiencies of the CHARA interferometer in the
    expected bands: so far, R and H.
    Gives the same transmission for all telescopes.
    Take into account:
        - Telescope mirror transmission
        - Injection loss: in fibers
        - Strehl ratio: 
        - Beam splitters - Part of the flux is used for:
            - For AO in R band
            - For image control in R band
            - For photometry calibration in R band

    Parameters
    ----------
    array : STRING
        Array name.
    band : STRING
        Observation band: H or R.

    Raises
    ------
    ValueError
        If array is not CHARA.

    Returns
    -------
    transmission : FLOAT
        Total transmission.

    """
    if array == 'chara': # information in SPICA JOSAA paper
        if band == 'H':     
            T_tel = 0.1
            T_inj = 0.65
            T_strehl = 0.8
            T_BS = 1            # No beam splitter in H ? OA?
        if band == 'R':
            T_tel = 0.03
            T_inj = 0.5
            T_strehl = 0.8
            T_BS = 0.88         # Transmission beam-splitter before injection
        
        # Diameter of a telescope and collecting surface [meter²]
        diameter = 1                            
        surface = np.pi*diameter**2/4            
        
        surface = 0.74 #MAM thesis - diameter=1m and occultation=0.25m
                                
    else:
        raise ValueError('So far, there is data for "chara" only.')
        
    transmission = T_tel*T_inj*T_strehl*T_BS

    return transmission, surface
    

def get_array(name,getcoords=False):
    """
    Returns the coordinates, baselines and base names of a given array

    Parameters
    ----------
    name : STRING
        Name of the array.
    getcoords : BOOLEAN, optional
        If True, add the array coordinates to the output.
        The default is False.

    Raises
    ------
    NameError
        Name must be 'CHARA'.

    Returns
    -------
    TelNames : [NA] list strings
        Names of the telescopes
    BaseNames : [NB] strings
        Name of each baseline.
    BaseNorms : [NB] floats [meter]
        Baselines.
    coords, OPTIONAL: [NA,3] FLOAT ARRAY [meter]
        Coordinates of the array, following the format:
            (XOFFSET,YOFFSET,ZOFFSET) where:
                XOFFSET - East offset in microns from S1
                YOFFSET - North offset in microns from S1
                ZOFFSET - vertical (+ is up) offset in microns from S1,,)
        Take S1 as reference.
    
    """
    
    print(name)
    if name == 'CHARA':         
            
        #official coordinates in [µm]
        coords = np.array([[125333989.819,305932632.737,-5909735.735],\
                                 [70396607.118,269713282.258,-2796743.645],\
                                     [0,0,0],\
                                         [-5746854.437,33580641.636,636719.086],\
                                             [-175073332.211,216320434.499,-10791111.235],\
                                                 [-69093582.796,199334733.235,467336.023]])
        coords=coords*1e-6      # [m]
        
    else:
        raise NameError('Array name must be "CHARA"')
        
    NA = np.shape(coords)[0]
    
    TelNames = ['E1','E2','S1','S2','W1','W2']
    BaseNames = []
    BaseNorms =[]
    config.BasesVectors = np.array([])
    for ia in range(NA):
        for iap in range(NA):
            
            BaseNames.append(str(TelNames[ia]+TelNames[iap]))
            BaseNorms.append(np.sqrt((coords[ia,0]-coords[iap,0])**2 + \
                                    (coords[ia,1]-coords[iap,1])**2 + \
                                  (coords[ia,2]-coords[iap,2])**2))
               
            
    if getcoords:
        return TelNames, BaseNames, BaseNorms, coords
        
    config.BasesVectors
    return TelNames, BaseNames, BaseNorms
    

def get_Bproj(baseline, theta):
    """
    Calculates projected bases acording to the viewed angle in radian

    Parameters
    ----------
    baseline : float [meter]
        Distance between telescopes in meters.
    theta : float [radian]
        Angle viewed by the base (depend on declination and azimuth).
        
    Returns
    -------
    Bproj : float [meter]
        Projected base.

    """
    
    Bproj = baseline * np.sin(theta)
    return Bproj


def get_visibility(alpha, baseline, spectra, model='disk'):
    """
    Using the Bessel expression of the visibility of a centered circular star,
    returns the spectral absolute visibility curve of a centered star of 
    angular diameter alpha as seen with the given baseline distance.
    Replaced by VanCittert function that calculates complex visibilities
    with a more realistic calculation.

    Parameters
    ----------
    alpha : float [radian]
        Angular diameter
    base : float [meter]
        Baseline
    spectra : [MW] floats [µm]
        Wavelength.
    model : string, optional
        The model of the object. The default is 'disk'.

    Returns
    -------
    V : [MW]
        Spectral visibilities of the object.

    """
    spectra = spectra*1e-6
    if model == 'disk':
        if base == 0:
            V = 1
        else:
            V = np.abs(2*jv(1, np.pi*alpha*baseline/spectra)/(np.pi*alpha*baseline/spectra))

    return V


@deco.timer
def VanCittert(spectra, Obs, Target,display=False):
    """
    Create the Coherent flux matrix of an object in the (u,v) plane according
    to:
        - the Observation parameters: array, date and star coords.
        - the Science object: angular diameters, positions, relative luminosity


    Parameters
    ----------
    spectra : TYPE
        DESCRIPTION.
    starname : TYPE, optional
        DESCRIPTION. The default is 'Deneb'.
    angdiameters : TYPE, optional
        DESCRIPTION. The default is (0.3,0.3).
    relative_L : TYPE, optional
        DESCRIPTION. The default is (1,1).
    display : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    visibilities : ARRAY [NW,NIN]*1j
        DESCRIPTION.

    """
        
    print("Calculation of object's visibilities from Van Cittert theorem\n ...")

    # Relative positions of the two stars
    pos_star1 = Target.Star1['Position']
    angular_diameter1 = Target.Star1['AngDiameter']
    H1 = Target.Star1['Hmag']
    
    BinaryObject = True
    try:
        pos_star2 = Target.Star2['Position']
        angular_diameter2 = Target.Star2['AngDiameter']
        H2 = Target.Star2['Hmag']
    except:
        print('Simple centered star1')
        BinaryObject = False
        
        # angular_diameter2 = angular_diameter1
        # H2 = 0
        
    # pos_star1 = np.array([1,-1])
    # pos_star2 = np.array([-1,1])
    
    # BinaryObject = (type(angdiameters) == tuple)
    
    if BinaryObject:
        # Calculation of a relative luminosity
        (luminosity1,luminosity2) = (10**(-0.4*H1),10**(-0.4*H2))
        # (angular_diameter1,angular_diameter2) = angdiameters
        
        # Pixel unit luminance ratio between both stars
        LuminanceRatio = luminosity1/angular_diameter1*angular_diameter2/luminosity2
    # else:
    #     # angular_diameter1 = angdiameters
    #     # angular_diameter2 = angdiameters
    #     luminosity1 = 1
    #     # luminosity2 = 0
    #     # pos_star1 = np.array([0,0])
    
    
    # Npix = 1024
    # thetamax = 10
    # dtheta = 2*thetamax/Npix
    
    # dtheta must well sample the smallest star
    if BinaryObject:
        dtheta = np.min((angular_diameter1,angular_diameter2))/5
    else:
        dtheta = angular_diameter1/5
    
    # thetamax must be sufficiently high compared to the widest star
    # if BinaryObject:
    #     thetamax = 100*np.max((angular_diameter1,angular_diameter2))
    # else:
    #     thetamax = 100*angular_diameter1
    
    Npix = 5000     #2*int(thetamax/dtheta)

    thetamax = dtheta*Npix/2

    obj_plane = np.zeros([Npix,Npix])
    
    coords = (np.arange(Npix)- Npix/2)*dtheta
    (alpha,beta) = np.meshgrid(coords,-coords)
    dist1 = np.sqrt((alpha-pos_star1[0])**2 + (beta-pos_star1[1])**2)
    obj_plane[dist1<angular_diameter1] = 1
    NpixStar1 = np.sum(obj_plane)
    
    if BinaryObject:
        dist2 = np.sqrt((alpha-pos_star2[0])**2 + (beta-pos_star2[1])**2)
        obj_plane[dist2<angular_diameter2] = 1/LuminanceRatio
        
    
    # Normalisation of the luminosity, with first star as reference
    obj_plane = obj_plane/NpixStar1

    Nticks = 7
    ticks = np.linspace(0,Npix,Nticks)
        
    if display:            

        spaceticks = (ticks-round(Npix/2))*dtheta
        spaceticks = spaceticks.round(decimals=1)
        
        plt.figure(config.newfig), plt.title('Object')
        plt.imshow(obj_plane)
        plt.xticks(ticks,spaceticks)
        plt.yticks(ticks,-spaceticks)
        plt.xlabel('\u03B1 [mas]')
        plt.ylabel('\u03B2 [mas]')
        plt.show()
        config.newfig +=1
    
    # Van-Cittert theorem (calculation of the visibility)
    
    uv_plane = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(obj_plane)))
    uv_plane /= np.max(np.abs(uv_plane))
    
    # (u,v) sampling
    freqs = np.fft.fftshift(np.fft.fftfreq(Npix, dtheta))
    dfreq = freqs[1] - freqs[0]
    (ucoords,vcoords) = np.meshgrid(freqs, -freqs)
    freqticks = (ticks-round(Npix/2))*dfreq
    freqticks = freqticks.round(decimals=1)
    
    if display:        
        plt.figure(config.newfig), plt.title('Module of the visibility')
        plt.imshow(np.abs(uv_plane))
        plt.xticks(ticks,freqticks)
        plt.yticks(ticks,-freqticks)
        plt.xlabel('u [mas-1]')
        plt.ylabel('v [mas-1]')
        plt.show()
        config.newfig+=1
        
        plt.figure(config.newfig), plt.title('Phase of the visibility')
        plt.imshow(np.angle(uv_plane))
        plt.xticks(ticks,freqticks)
        plt.yticks(ticks,-freqticks)
        plt.xlabel('u [mas-1]')
        plt.ylabel('v [mas-1]')
        plt.show()
        config.newfig+=1
    
    """
    Projection of the interferometer on the (u,v) plane
    """
    
    # Get telescopes coordinates and names
    TelNames,_,_,CHARAcoords = get_array(config.Name, getcoords=True)    
    
    CHARAcoords *= 1e6          # Convert to [µm]
    NA = len(CHARAcoords)
    
    CHARAaltaz = np.zeros([NA,2])
    for ia in range(NA):
        if np.linalg.norm(CHARAcoords[ia]) == 0:
            CHARAaltaz[ia,0] = 0
            CHARAaltaz[ia,1] = 0    
        else:
            CHARAaltaz[ia,1] = np.arctan(CHARAcoords[ia,0]/CHARAcoords[ia,1])
            CHARAaltaz[ia,0] = np.arcsin(CHARAcoords[ia,2]/np.linalg.norm(CHARAcoords[ia]))
    
    NIN = int(NA*(NA-1)/2)
    basealtaz = np.zeros([NIN,2])       # Altazimuthal coordinates of the baselines [radians]
    basedist = np.zeros([NIN,1])        # Baselines lengths in [µm]
    basecoords = np.zeros([NIN,3])      # Baselines coordinates [µm]
    basenames = []                      # Baselines names [string]
    
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib = int(ia*NA-ia*(ia+3)/2+iap-1)
            
            basealtaz[ib] = CHARAaltaz[iap]-CHARAaltaz[ia]
            
            basenames.append(TelNames[ia]+TelNames[iap])
            basecoords[ib] = CHARAcoords[iap] - CHARAcoords[ia]
            basedist[ib] = np.linalg.norm(basecoords[ib])
    
    
    # First case: the name of the object has been given. We search it in Simbad
    # database and its AltAzimutal coordinates if a Date has been given.
    if Target.Name not in ('Simple','Binary'):
        starttime = Time(Obs.Date)
        print(f"Observation date: {Obs.Date}")
    
        starcoords = SkyCoord.from_name(Target.Name)
        staraltaz = starcoords.transform_to(AltAz(obstime=starttime,location=Obs.ArrayName))
        
        (altitude, azimuth) = (staraltaz.alt.radian,staraltaz.az.radian)
        
    else:
        (altitude, azimuth) = (theta*np.pi/180 for theta in Obs.AltAz)
        print(f"User defined {Target.Name} object with AltAz={Obs.AltAz}")
        
    
    # Projection baselines on (u,v) plane
    baselines = np.transpose(basedist)*np.sin(altitude-basealtaz[:,0])
    
    NW = len(spectra)
    # Projection baselines on the u,v coordinates (oriented with the star north-east)
    chara_uv = np.zeros([NW,NIN,2])
    for iw in range(NW):
        lmbda=spectra[iw]
        chara_uv[iw,:,0] = baselines*np.cos(azimuth-basealtaz[:,1])/lmbda
        chara_uv[iw,:,1] = baselines*np.sin(azimuth-basealtaz[:,1])/lmbda
    
    # Conversion functions
    mas2rad = lambda mas : 1/3600*1e-3*np.pi/180*mas
    rad2mas = lambda rad : 3600*1e3*180/np.pi*rad

    #Convert (u,v) plane from radian to mas
    chara_uv_direct = 1/chara_uv
    chara_uv_direct_mas = rad2mas(chara_uv_direct)
    chara_uv = 1/chara_uv_direct_mas
    
    if display:         # Display (u,v) plane with interferometer projections (first wavelength)
        print(f'Plot CHARA (u,v) coverage on figure {config.newfig}')    
        chara_uv_complete = np.concatenate((chara_uv[0], -chara_uv[0]),axis=0)
        
        uvmax = np.max(chara_uv[0]/dfreq)+10
        
        Ndisplay = 2*int(uvmax+10)
        
        uv_crop = uv_plane[(Npix-Ndisplay)//2:(Npix+Ndisplay)//2,(Npix-Ndisplay)//2:(Npix+Ndisplay)//2]
        chara_plot = chara_uv_complete/dfreq+Ndisplay/2
        
        Nticks = 11
        ticks = np.linspace(0,Ndisplay-1,Nticks)
        freqticks = (ticks+1-round(Ndisplay/2))*dfreq
        freqticks = freqticks.round(decimals=1)
        
        # Display the Visibility and the interferometer baselines on the (u,v) plane
        fig = plt.figure(config.newfig)
        ax = fig.subplots()
        fig.suptitle('(u,v) coverage')
        ax.imshow(np.abs(uv_crop))
        ax.plot(chara_plot[:,0], chara_plot[:,1], '.', linewidth=1, color='firebrick')
        plt.xticks(ticks,freqticks)
        plt.yticks(ticks,-freqticks)
        plt.xlabel('u [mas-1]')
        plt.ylabel('v [mas-1]')
        config.newfig += 1
        
    # Return the complex visibility vector of the source
    visibilities = np.zeros([NW,NIN])*1j
    for iw in range(NW):
        for ib in range(NIN):
            baselinecoords = chara_uv[iw,ib]
            closestfreq = ((ucoords - baselinecoords[0])**2+(vcoords - baselinecoords[1])**2).argmin()
            Nx = closestfreq//Npix
            Ny = closestfreq%Npix
            visibilities[iw,ib] = uv_plane[Nx,Ny]
    
    print("Visibilities calculated.")
    return visibilities

def create_obsfile(spectra, Obs, Target, savingfilepath='', overwrite=False, display=False):
    """
    Creates the coherent flux matrix of the object at the entrance of the 
    fringe sensor and save it into a fitsfile.
    It takes into account:
        - star spectral distribution (BlackBody approx, or flat)
        - Array transmission
        - Visibility on the different baselines
    
    These parameters are given by the objects Obs and Target.

    Parameters
    ----------
    spectra : FLOAT ARRAY [NW]
        Spectral sampling.
    Obs : OBSERVATION CLASS OBJECT
        Contains the information on the interferometer and other things.
    Target : OBJECT CLASS OBJECT
        Contains the information on the object.
    savingfilepath : STRING, optional
        File name. The default is ''.
    overwrite : BOOLEAN, optional
        If True, overwrite the existing file. The default is False.
    display : BOOLEAN, optional
        If True, display some plots. The default is False.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    CohIrradiance : COMPLEX ARRAY [NW,NB]
        Spectral photon distribution including visibility and star photometry
        on all baselines. [ph/s].
    UncohIrradiance : FLOAT ARRAY [NW]
        Spectral photon distribution [pht/s/telescope/µm] of the star without visibility..
    VisObj : COMPLEX ARRAY [NW,NB]
        Visibility of the object, normalised between 0 and 1.
    BaseNorms : FLOAT ARRAY [NIN or NB ?]
        Norm of the different baselines.
    TelNames : STRING ARRAY [NA]
        Names of the telescopes.
    """

    fileexists = os.path.exists(savingfilepath)
    if fileexists:
        print(f'{savingfilepath} already exists.')
        if overwrite:
            os.remove(savingfilepath)
        else:
            raise Exception("You didn't ask to overwrite it.")          

    # spectra = 1/sigma     # [µm] Wavelengths
    NW = len(spectra)     # [int] Wavelengths number
    
    # First case: Temperature of the first star is given
    # We compute a BlackBody model
    if 'T' in Target.Star1.keys():    
        T=Target.Star1['T']
        Irradiance = BBstar(spectra*1e-6, T)       # Create BB star model in wavelengths    
        Irradiance = Irradiance[::-1]             # (Reverse) the spectra according to sigmas
        norm = np.mean(Irradiance[(spectra>=1.4)*(spectra<=1.75)])    # Normalization of the flux on the whole H band
        Irradiance/=norm

    else:
        Irradiance=np.ones_like(spectra)
        
    magH = Target.Star1['Hmag']
    # Luminance according to apparent magnitude    
    
    L0_ph = 93.3e8        # Photons.m-2.s-1.µm-1 at 1.63µm
    
    Lph_H = L0_ph*10**(-0.4*magH)
    
    # delta_wav = np.abs(spectra[0] - spectra[1])     # Width of a spectral channel
    
    # Preference for photons directly to keep a uniform spectra
    UncohIrradiance = Irradiance*Lph_H             # [phot/s/m²/deltalmbda] Source Irradiance 
    
    # Using Watt as reference and converting in photons: drawback=non uniform spectra
    # L0_w = 7 * 10**(-10)                      # [W/m²/µm] Reference luminance at 1.65µm (Lena)
    # L0_w = 11.38 * 10**(-10)                  # [W/m²/µm] Reference luminance at 1.63µm (Bessel)
    # Lw_H = L0_w*10**(-0.4*magH)                        # Definition of the magnitude
    # UncohIrradiance_w = luminance*Lw_H*delta_wav / (h_*c_/spectra*1e6)          # [phot/s/m²/deltalmbda]


    filepath = Obs.Filepath
    if not os.path.exists(filepath):
        raise Exception(f"{filepath} doesn't exist.")
    
    with fits.open(filepath) as hdu:
        ArrayParams = hdu[0].header
        NA, NIN = ArrayParams['NA'], ArrayParams['NIN']
        ArrayName = ArrayParams['NAME']
        TelData = hdu[1].data
        BaseData = hdu[2].data
        
        TelNames = TelData['TelNames']
        TelCoordinates = TelData['TelCoordinates']
        TelTransmissions = TelData['TelTransmissions']
        TelSurfaces = TelData['TelSurfaces']
        BaseNames = BaseData['BaseNames']
        BaseCoordinates = BaseData['BaseCoordinates']
        
    NB = NA**2
    NC = (NA-2)*(NA-1)
    
    # Transportation of the star light into the interferometer
    Throughput = np.reshape(TelSurfaces*TelTransmissions,[1,NA])
    ThroughputMatrix = np.sqrt(np.dot(np.transpose(Throughput), Throughput))
    ThroughputMatrix = ThroughputMatrix.reshape([1,NB])
    # Matrix where the element at the (ia*NA+iap) position is Ta*Tap
    # UncohIrradianceAfterTelescopes = np.dot(np.diag(TelTransmissions),np.transpose(UncohIrradiance))
        
    BaseNorms = np.linalg.norm(BaseCoordinates, axis=1)
    
    # Projection of the base on the (u,v) plane
    BaseNorms = get_Bproj(np.array(BaseNorms), Obs.AltAz[0])
    
    VisObj = np.zeros([NW,NB])*1j       # Object Visibility [normalised]
    CohIrradiance = np.zeros([NW,NB])*1j        # Source coherent Irradiance [phot/s/m²/deltalmbda]    
      
    # UncohIrradianceAfterTelescopes = UncohIrradiance*TelTransmissions*TelSurfaces
    

    if Target.Name == 'Unresolved':
        VisObj = np.ones([NW,NB])
        CPObj = np.zeros([NW,NC])
        
    elif Target.Name == 'Manual':
        phi = np.zeros([NW,NB])
        for ia in range(NA):
            for iap in range(ia+1,NA):
                ib = posk(ia,iap,NA)
                phi[:,ia*NA+iap] = Target.Phases[ib]
                phi[:,iap*NA+ia] = -Target.Phases[ib]
        VisObj = np.exp(1j*phi)
        # VisObj = np.repeat(VisObj[np.newaxis,:],NW,axis=0)
        
        bispectrum = np.zeros([NW,NC])*1j
        for ia in range(NA):
            for iap in range(ia+1,NA):
                for iapp in range(iap+1,NA):
                    ci1 = VisObj[:,ia*NA+iap]
                    ci2 = VisObj[:,iap*NA+iapp]
                    ci3 = VisObj[:,iapp*NA+ia]
                    ic = poskfai(ia,iap,iapp,NA)
                    bispectrum[:,ic] = ci1*ci2*ci3
        CPObj = np.angle(bispectrum)
        
    else:           # Van-Cittert theorem visibility
        visibilities = VanCittert(spectra, Obs, Target,display=display)
        
        
        bispectrum = np.zeros([NW,NC])*1j
        
        for ia in range(NA):
            VisObj[:,ia*(NA+1)] = np.ones(NW)
            for iap in range(ia+1,NA):
                ib = posk(ia,iap,NA)
                VisObj[:,ia*NA+iap] = visibilities[:,ib]
                VisObj[:,iap*NA+ia] = np.conj(visibilities[:,ib])
                for iapp in range(iap+1,NA):
                    cs1 = VisObj[:,ia*NA+iap]     # Coherent flux (ia,iap) 
                    cs2 = VisObj[:,iap*NA+iapp]  # Coherent flux (iap,iapp) 
                    cs3 = VisObj[:,iapp*NA+ia] # Coherent flux (iapp,ia) 
                    ic = poskfai(ia,iap,iapp,NA) 
                    bispectrum[:,ic]+=cs1*cs2*cs3
                    
        CPObj = np.angle(bispectrum)
        
    for iw in range(NW):    
        CohIrradiance[iw] = UncohIrradiance[iw] * ThroughputMatrix * VisObj[iw]
            
    # hdr = ArrayParams
    hdr = fits.Header()
    
    hdr['Filepath'] = savingfilepath.split('/')[-1]
    hdr['ARRAY'] = Obs.ArrayName
    hdr['Target'] = Target.Name
    hdr['NA'] = NA
    hdr['NIN'] = NIN
    hdr['MINWAVE'] = spectra[0]
    hdr['MAXWAVE'] = spectra[-1]
    hdr['DWAVE'] = spectra[1] - spectra[0]

    for attr in vars(Target).keys():
        if type(getattr(Target, attr)) is not dict:            
            if attr not in ['Name','Filepath', 'Phases']:
                hdr[attr] = getattr(Target, attr)
        else:
            for key in getattr(Target, attr).keys():
                if key == 'Position':
                    hdr[f"{attr}_alpha"] = getattr(Target, attr)[key][0]
                    hdr[f"{attr}_beta"] = getattr(Target, attr)[key][1]
                else:
                    hdr[f"{attr}_{key}"] = getattr(Target, attr)[key]
                
    for attr in vars(Obs).keys():
        if isinstance(getattr(Obs, attr),
                      (str, int, float, complex, bool,
                       np.floating, np.integer, np.complexfloating,
                       np.bool_)):
            
            if attr != 'Filepath':
                if attr != ArrayName:
                    hdr[attr] = getattr(Obs, attr)
            else:
                hdr['ArrayFile'] = getattr(Obs, attr).split('/')[-1]
        
    
    primary = fits.PrimaryHDU(header=hdr)
    
    im1 = fits.ImageHDU(np.real(VisObj), name='VReal')
    im2 = fits.ImageHDU(np.imag(VisObj), name='VImag')
    im3 = fits.ImageHDU(np.real(CohIrradiance), name='CfReal')
    im4 = fits.ImageHDU(np.imag(CohIrradiance), name='CfImag')
    im5 = fits.ImageHDU(CPObj, name='Closure Phase')
    
    col1 = fits.Column(name='WLsampling', format='1D', array=spectra)
    hdu1 = fits.BinTableHDU.from_columns([col1], name='spectra' )
    
    hdu = fits.HDUList([primary,hdu1,im1,im2,im3,im4,im5])
    
    print(f'Saving file into {savingfilepath}')
    hdu.writeto(savingfilepath)


    if display:
        plt.figure(config.newfig), plt.title('Absolute Visibility of the star on different baselines')
        plt.plot(spectra, np.abs(VisObj[:,0]), label='{0:}:{1:.1f}m'.format(TelNames[0]+TelNames[0],BaseNorms[0]))  
        for ia in range(NA):
            for iap in range(ia+1,NA):
                k = posk(ia,iap,NA)
                plt.plot(spectra, np.abs(VisObj[:,k]), label='{0:}:{1:.1f}m'.format(TelNames[ia]+TelNames[iap],BaseNorms[k]))  
        plt.xlabel('Wavelengths [µm]')
        plt.ylabel('Coherence Degree Module')
        plt.legend()
        plt.grid()
        plt.show()
        config.newfig+=1
        
        plt.figure(config.newfig), plt.title('Phase Visibility of the star on different baselines')
        plt.plot(spectra, np.angle(VisObj[:,0]), label='{0:}'.format('Uncoherent flux'))  
        for ia in range(NA):
            for iap in range(ia+1,NA):
                k = posk(ia,iap,NA)
                plt.plot(spectra, np.angle(VisObj[:,k]), label='{0:}:{1:.1f}m'.format(TelNames[ia]+TelNames[iap],BaseNorms[k]))  
        plt.xlabel('Wavelengths [µm]')
        plt.ylabel('Coherence Degree Phase [rad]')
        plt.ylim([-1.1*np.pi,1.1*np.pi])
        plt.legend()
        plt.grid()
        plt.show()
        config.newfig += 1
    
        plt.figure(config.newfig), plt.title('Irradiance of the star')
        plt.plot(spectra, UncohIrradiance)  
        plt.xlabel('Wavelengths [µm]')
        plt.ylabel('Irradiance [photons/s/m²/µm]')
        plt.grid()
        plt.show()
        config.newfig += 1
        
        
        plt.figure(config.newfig), plt.title('Coherent Irradiance of the star')
        plt.plot(spectra, np.abs(VisObj[:,0])*UncohIrradiance, label='{0:}'.format('Uncoherent flux'))  
        for ia in range(NA):
            for iap in range(ia+1,NA):
                k = posk(ia,iap,NA)
                plt.plot(spectra, np.abs(VisObj[:,k])*UncohIrradiance, label='{0:}:{1:.1f}m'.format(TelNames[ia]+TelNames[iap],BaseNorms[k]))  
        plt.xlabel('Wavelengths [µm]')
        plt.ylabel('Irradiance [photons/s/m²/µm]')
        plt.legend()
        plt.grid()
        plt.show()
        config.newfig += 1
    
    return CohIrradiance, UncohIrradiance, VisObj, BaseNorms, TelNames


def get_CfObj(filepath, spectra):
    """
    Reads data of an observation contained in a FITSfile.
    Adapt the spectral sampling to the FS spectral sampling.
    Return it

    Parameters
    ----------
    filepath : TYPE
        DESCRIPTION.
    spectra : TYPE
        DESCRIPTION.

    Returns
    -------
    CohIrradiance : ARRAY[NW,NB]
        Coherence Flux of the object in photons/s.
    ClosurePhase : ARRAY[NW,NC]
        Closure Phases of the object in radian.

    """

    fileexists = os.path.exists(filepath)
    if not fileexists:
        raise Exception(f"{filepath} doesn't exists.")          
    with fits.open(filepath) as hdu:

        ObsParams = hdu[0].header
        WLsampling = hdu['SPECTRA'].data['WLsampling']
        
        real = hdu['VReal'].data
        imag = hdu['VImag'].data

        ComplexVisObj = real + imag*1j
        
        real = hdu['CfReal'].data
        imag = hdu['CfImag'].data


    f = interpolate.interp1d(WLsampling, real, axis=0)
    NewReal = f(spectra)
    f = interpolate.interp1d(WLsampling, imag, axis=0)
    NewImag = f(spectra)
    CoherentIrradiance = NewReal + NewImag*1j
    
    NW, NB = CoherentIrradiance.shape
    NA = int(np.sqrt(NB))
    NC = (NA-2)*(NA-1)
    ClosurePhase = np.zeros([NW,NC])
    for ia in range(NA):
        for iap in range(ia+1,NA):
            for iapp in range(iap+1,NA):
                ic = poskfai(ia,iap,iapp,NA)
                ci1 = CoherentIrradiance[:,ia*NA+iap]
                ci2 = CoherentIrradiance[:,iap*NA+iapp]
                ci3 = CoherentIrradiance[:,iapp*NA+ia]
                ClosurePhase[:,ic] = np.angle(ci1*ci2*ci3)
        # ClosurePhase = hdu['Closure Phase'].data
        
    return CoherentIrradiance, ClosurePhase


def get_infos(file):
    
    with fits.open(file) as hdu:
        hdr = hdu[0].header
        dt = hdr['dt']
        NT = hdr['NT']
        piston = hdu['Piston'].data
        transmission = hdu['transmission'].data
        filetimestamps = np.arange(NT)*dt
        filelmbdas = hdu['LambdaSampling'].data['lambdas']
        try:
            PSD = hdu['Last tel PSD'].data
            Filter = hdu['Disturbance Filter'].data
            
            df = hdr['df']
            NF = len(PSD)
            FreqSampling = np.arange(NF)*df
            print('Disturbance PSD present in FITSfile.')
            
        except:
            print('No Disturbance PSD in FITSfile. The arrays FreqSampling, PSD and Filter are put to zero.')
            FreqSampling = np.zeros(NT); PSD = np.zeros(NT); Filter = np.zeros(NT)

    return filetimestamps,filelmbdas, piston, transmission, FreqSampling, PSD, Filter,hdr
    
    
def get_CfDisturbance(DisturbanceFile, spectra, timestamps):
    from .config import start_at_zero
    filetimestamps, filespectra, PistonDisturbance, TransmissionDisturbance,_,_,_,_ = get_infos(DisturbanceFile)

    # Interpolate on the time axis   
    newobservables = []
    for observable in [PistonDisturbance, TransmissionDisturbance]:
        f = interpolate.interp1d(filetimestamps,observable, axis=0)
        newobservables.append(f(timestamps))
        
    
    PistonDisturbance, TempTransmissionDisturbance = newobservables
    
    # Interpolate transmission on the spectral axis (Piston is not chromatic)    
    f = interpolate.interp1d(filespectra,TempTransmissionDisturbance, axis=1)
    TransmissionDisturbance = f(spectra)
    
    NT = len(timestamps) ; NW = len(spectra)
    NB = np.shape(PistonDisturbance)[1]**2

    if start_at_zero:
        print("We subtract the average piston to the piston of all telescopes.")
        PistonDisturbance = PistonDisturbance-np.mean(PistonDisturbance[0])

    CfDisturbance = np.zeros([NT,NW,NB])*1j
    from cophasing import skeleton
    for it in range(NT):
        CfDisturbance[it,:,:] = skeleton.coh__pis2coh(PistonDisturbance[it,:], 1/spectra, ampl=TransmissionDisturbance[it,:])


    return CfDisturbance, PistonDisturbance, TransmissionDisturbance
    

def get_subarray(pup_used, dsp_all, TelNames, base):
    """
    From the visibility matrix of the whole array (dependant on the object)
    returns, the visibility matrix of the array made of the given pupils.

    Parameters
    ----------
    pup_used : string LIST [NA]
        Names of the pupils of the subarray.
    dsp_all : float [NW,NA0**2]
        The visibility matrix of the whole array.
    TelNames : string LIST [NA0]
        Names of the pupils composing the whole array.

    Returns
    -------
    dsp : float [NW,NA**2]
        The visibility matrix of the whole array.
    base : string list [NA**2]
        Base names of the subarray.

    """
    
    NA = len(pup_used)
    NA0 = len(TelNames)
    NW = np.shape(dsp_all)[0]
    dsp = np.zeros([NW,NA**2])
    tmp = base
    base=[]
    for ia in range(NA):
        a = TelNames.index(pup_used[ia])
        # dsp[:,ia*(NA+1)] = dsp_all[:,a*(NA0+1)]
        # base.append(tmp[a*(NA0+1)])
        for iap in range(NA):
            k = ia*NA+iap
            ap = TelNames.index(pup_used[iap])
            kp = a*NA0+ap
            dsp[:,k] = dsp_all[:,kp]
            base.append(tmp[kp])
    # import pdb; pdb.set_trace()
   
    return dsp, base, NA
    

def generate_spectra0(lmbda1,lmbda2,R, sampling='linearsig'):
    """
    Generates a spectra of a grism of resolution R linearly spaced on wavenumbers
    or wavelength

    Parameters
    ----------
    lmbda1 : float
        Minimal wavelength.
    lmbda2 : float
        Maximal wavelength.
    R : int
        Spectral resolution.
    sampling : TYPE, optional
        Method of sampling. The default is 'linearsig'.
        OPTIONS:
            - 'linearsig': linear in wavenumbers [advised]
            - 'linearlambda': linear in wavelengths

    Returns
    -------
    spectra : ARRAY 1D
        Ascending order.
    sigma : ARRAY 1D
        Descending order.

    """
    
    if sampling == 'linearsig':
        sigma1 = 1/lmbda2
        sigma2 = 1/lmbda1
        deltasig = np.min([sigma1,sigma2])/R
        sigma = np.arange(sigma1,sigma2,deltasig)
        sigma = np.sort(sigma)[::-1]
        spectra = 1/sigma
        
    elif sampling == 'linearlambda':
        deltalmbda = np.min([lmbda1,lmbda2])/R
        spectra = np.arange(lmbda1, lmbda2, deltalmbda)
        sigma = 1/spectra
        
    return spectra, sigma


def generate_spectra(lmbda1, lmbda2, OW, MW=0, R=0, spectraband=[], mode='linear_sig'):
    """
    Returns the micro and macro wavenumbers to use for the simulation

    Parameters
    ----------
    sigma : vector [MW]
        Wavenumbers
    OW : int
        Oversampling wavenumbers.
    sigmaband : vector [MW], optional
        Channel bandwidth
    mode : string, optional
        Way of generating the spectras. The default is 'regular'.

    Raises
    ------
    ValueError
        Mode needs to be in acceptable values.

    Returns
    -------
    sigma : [MW*OW] floats
        Micro Wavenumbers.
    sigmaM : [MW] floats
        Macro wavenumbers.

    """
    possible_modes = ['linear_wl','linear_sig']
    
    if R:
        if MW:
            raise Exception("MW can't be given with R.")
        deltalmbda = np.mean([lmbda1,lmbda2])/R
        MW = int(round((lmbda2-lmbda1)/deltalmbda))
        
    elif not MW:
        raise Exception("MW or R must be given.")
    
    if mode == 'linear_sig':
        sig1, sig2 = np.min([1/lmbda1,1/lmbda2]), np.max([1/lmbda1,1/lmbda2])
        sigma = np.linspace(sig1, sig2, MW)
        spectra = np.sort(1/sigma)

        deltasig = sigma[1] - sigma[0]
        # sigmaband = spectraband*spectra**(-2)
        
        sigma_os = np.array([])
        for i in range(MW):
            sigbottom = sigma[i]-deltasig/2
            sigtop = sigma[i]+deltasig/2
            sigma_temp = np.linspace(sigbottom, sigtop, OW)
            sigma_os = np.concatenate((sigma_os,sigma_temp))
        spectra_os = np.sort(1/sigma_os)
        
    elif mode == 'linear_wl':
        spectra_os = np.array([])
        spectra = np.linspace(lmbda1, lmbda2, MW)
        deltalmbda = spectra[1] - spectra[0]
        for i in range(MW):
            wlbottom = spectra[i]-deltalmbda/2
            wltop = spectra[i]+deltalmbda/2
            spectra_temp = np.linspace(wlbottom, wltop, OW)
            spectra_os = np.concatenate((spectra_os,spectra_temp))
    
    else:
        raise ValueError(f'mode={mode}. Unknown mode, needs to be in {possible_modes}.')
    
    return spectra_os, spectra


def oversample_wv(spectra, OW, spectraband=[], mode='linear_sig'):

# def oversample_wv(lmbda1, lmbda2, OW, MW=0, R=0, spectraband=[], mode='linear_sig'):
    """
    Returns the micro and macro wavenumbers to use for the simulation

    Parameters
    ----------
    spectra : vector [MW]
        Wavelengths
    OW : int
        Oversampling wavenumbers.
    spectraband : vector [MW], optional
        Channel bandwidth
    mode : string, optional
        Way of generating the spectras. The default is 'linear_sig'.

    Raises
    ------
    ValueError
        Mode needs to be in acceptable values.

    Returns
    -------
    spectra_os : [MW*OW] floats
        Oversampled micro Wavelengths.
    spectra : [MW] floats
        Macro Wavelengths.

    """
    possible_modes = ['linear_wl','linear_sig']
    
    
    MW = len(spectra)
    
    spectra = np.sort(spectra)
    sigma = np.sort(1/spectra)
    
    
    if len(spectraband)==0:
        spectraband = np.zeros_like(spectra)
        sigmaband = np.zeros_like(spectra)
        for i in range(1,MW):
            spectraband[i] = spectra[i] - spectra[i-1]
            sigmaband[i] = sigma[i] - sigma[i-1]
        spectraband[0] = spectraband[1]
        sigmaband[0] = sigmaband[1]
    
    if mode == 'linear_sig':
        sigma = np.sort(1/spectra)
        sigmaband = spectraband*spectra**(-2)
        
        sigma_os = np.array([])
        for i in range(MW):
            sigbottom = sigma[i]-sigmaband[i]/2
            sigtop = sigma[i]+sigmaband[i]/2
            sigma_temp = np.linspace(sigbottom, sigtop, OW)
            sigma_os = np.concatenate((sigma_os,sigma_temp))
        spectra_os = np.sort(1/sigma_os)
        
    elif mode == 'linear_wl':
        spectra_os = np.array([])
        for i in range(MW):
            wlbottom = spectra[i]-spectraband[i]/2
            wltop = spectra[i]+spectraband[i]/2
            spectra_temp = np.linspace(wlbottom, wltop, OW)
            spectra_os = np.concatenate((spectra_os,spectra_temp))
    
    else:
        raise ValueError(f'mode={mode}. Unknown mode, needs to be in {possible_modes}.')
    
    return spectra_os, spectra



def coh__GRAV2simu(gravmatrix):
    """
    Adapt the GRAVITY's matrix formalism in the simulator one

    Parameters
    ----------
    matrix : TYPE
        DESCRIPTION.

    Returns
    -------
    simuV2PM : TYPE
        DESCRIPTION.
    simuP2VM : TYPE
        DESCRIPTION.

    """    
    shape = np.shape(gravmatrix)
    
    if len(shape)==3:    # It's a V2PM
        (NW,NP,NB) = shape
        simuV2PM = np.zeros([NW,NP,NB])*1j
        simuP2VM = np.zeros([NW,NB,NP])*1j
        NA = int(np.sqrt(NB))
        for ia in range(NA):
            ksim = ia*(NA+1)
            simuV2PM[:,:,ksim] = gravmatrix[:,:,ia]
            for iap in range(ia+1,NA):
                kp = ia*NA - ia*(ia+3)/2 + iap-1

                # Real and Imaginary parts of the coherence vectors
                k = int(NA + kp)
                Real = gravmatrix[:,:,k]
                k = int(NA*(NA+1)/2 + kp)
                Imag = gravmatrix[:,:,k]
                
                # Direct and Conjugate coherence vectors
                ksim = ia*NA+iap
                simuV2PM[:,:,ksim] = 1/2*(Real + Imag*1j)
                ksim = iap*NA+ia          # 
                simuV2PM[:,:,ksim] = 1/2*(Real - Imag*1j)
        
        for iw in range(NW):
            simuP2VM[iw,:,:] = np.linalg.pinv(simuV2PM[iw,:,:])
        
    else:
        (NP,NB) = shape
        simuV2PM = np.zeros([NP,NB])*1j
        simuP2VM = np.zeros([NB,NP])*1j
        NA = int(np.sqrt(NB))
        for ia in range(NA):
            ksim = ia*(NA+1)
            simuV2PM[:,ksim] = gravmatrix[:,ia]
            for iap in range(ia+1,NA):
                kp = ia*NA - ia*(ia+3)/2 + iap-1

                # Real and Imaginary parts of the coherence vectors
                k = int(NA + kp)
                Real = gravmatrix[:,k]
                k = int(NA*(NA+1)/2 + kp)
                Imag = gravmatrix[:,k]
                
                # Direct and Conjugate coherence vectors
                ksim = ia*NA+iap
                simuV2PM[:,ksim] = 1/2*(Real + Imag*1j)
                ksim = iap*NA+ia          # 
                simuV2PM[:,ksim] = 1/2*(Real - Imag*1j)
        
        simuP2VM = np.linalg.pinv(simuV2PM)
        
    return simuV2PM, simuP2VM



def simu2GRAV(simumatrix, direction='v2pm'):
    
    Ndim = np.ndim(simumatrix)
    
    if direction=='p2vm':
        # Invert the second and third axis: for having the same code for both situations
        if Ndim==2:
            simumatrix = np.transpose(simumatrix)
        elif Ndim ==3:
            simumatrix = np.transpose(simumatrix, [0,2,1])  
        
    shape = np.shape(simumatrix)    
    
    if Ndim==3:    # There's several wavelengths
        NW,NP,NB = shape
        GRAVmatrix = np.zeros([NW,NP,NB])
        NA = int(np.sqrt(NB))
        for ia in range(NA):
            ksim = ia*(NA+1)
            GRAVmatrix[:,:,ia] = np.real(simumatrix[:,:,ksim])
            for iap in range(ia+1,NA):
                kp = ia*NA - ia*(ia+3)/2 + iap-1
                
                # Direct and Conjugate coherence vectors
                direct = simumatrix[:,:,ia*NA+iap]
                # conj = simumatrix[:,:,iap*NA+ia]
                
                # Real and Imaginary parts of the coherence vectors
                k = int(NA + kp)                    # Real part location
                GRAVmatrix[:,:,k] = 2*np.real(direct)
                k = int(NA*(NA+1)/2 + kp)           # Imaginary part location
                GRAVmatrix[:,:,k] = 2*np.imag(direct)
        
    else:        # Ndim == 2
        NP,NB = shape
        GRAVmatrix = np.zeros([NP,NB])
        NA = int(np.sqrt(NB))
        for ia in range(NA):
            ksim = ia*(NA+1)
            GRAVmatrix[:,ia] = np.real(simumatrix[:,ksim])
            for iap in range(ia+1,NA):
                kp = ia*NA - ia*(ia+3)/2 + iap-1
                
                # Direct and Conjugate coherence vectors
                direct = simumatrix[:,ia*NA+iap]
                # conj = simumatrix[:,iap*NA+ia]
                
                # Real and Imaginary parts of the coherence vectors
                k = int(NA + kp)                    # Real part location
                GRAVmatrix[:,k] = 2*np.real(direct)#np.real((direct+conj)/2)
                k = int(NA*(NA+1)/2 + kp)           # Imaginary part location
                GRAVmatrix[:,k] = 2*np.imag(direct)#np.real((direct-conj)/2/1j)
            
    if direction=='p2vm':
        # Put back the axis at the right locations
        if Ndim==2:
            GRAVmatrix = np.transpose(GRAVmatrix)
        elif Ndim ==3:
            GRAVmatrix = np.transpose(GRAVmatrix, [0,2,1]) 
        
    return GRAVmatrix


def sortmatrix(matrix,ich,ABCDindex,direction='v2pm'):
    """
    From the known sorting of the baselines and modulations, reorder the pixels
    to get the conventional ordering: 12, 13, ...,23, ...,46,56

    Parameters
    ----------
    matrix : [NW,NP,NB] or [NW,NB,NP] ARRAY
        V2PM or P2VM.
    ich : [NINx2] INT ARRAY
        INterferometric channels.
    ABCDindex : [Nmod] INT ARRAY
        Positions of the ABCD modulations. 
        For example, if BDAC then ABCDindex=[2,0,3,1]
    direction : STRING, optional
        v2pm or p2vm. The default is 'v2pm'.

    Returns
    -------
    matrix_sorted : FLOAT ARRAY
        Same size as matrix, but reordered.

    """
    if direction=='v2pm':
        NW,NP,NB=np.shape(matrix)
    else:
        NW,NB,NP = np.shape(matrix)
        
    NA=int(np.sqrt(NB))
    Nmod=len(ABCDindex) ; NIN = len(ich)
    baseline_sorted = np.zeros_like(matrix)
    # baseline_resorted[:NA] = demod[:NA]
    
    for ib in range(NIN):

        # Resort the positions in the conventional order
        detectorpositions = range(ib*Nmod,(ib+1)*Nmod)
        ia, iap = ich[ib]-1
        ibn = posk(ia,iap,NA)
        conventionalpositions = range(ibn*Nmod,(ibn+1)*Nmod)
        
        if direction=='v2pm':
            baseline_sorted[:,conventionalpositions,:] = matrix[:,detectorpositions,:]
        else:
            baseline_sorted[:,:,conventionalpositions] = matrix[:,:,detectorpositions]
        
    matrix_sorted = np.zeros_like(matrix)
    # Resort the ABCD modulations
    for ib in range(NIN):
        for k in range(Nmod):
            if direction=='v2pm':
                matrix_sorted[:,ib*Nmod+k,:] = baseline_sorted[:,ib*Nmod+ABCDindex[k],:]
            else:
                matrix_sorted[:,:,ib*Nmod+k] = baseline_sorted[:,:,ib*Nmod+ABCDindex[k]]

    return matrix_sorted

def studyP2VM(nfig=0):
    """
    Show the V2PM and the V2PM with grids that enable to see clearly the baselines

    Parameters
    ----------
    nfig : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """
    # Load data
    v2pm = config.FS['V2PMgrav']
    p2vm = config.FS['P2VMgrav']
    
    NW,NP,NB = np.shape(v2pm)
    NA = int(np.sqrt(NB))
    NIN = NA*(NA-1)//2
    Modulations = config.FS['Modulations']
    ABCDindex = config.FS['ABCDind']
    Nmod=len(Modulations)
    
    ich=config.FS['ich']
    
    v2pm_sorted = sortmatrix(v2pm, ich, ABCDindex)
    p2vm_sorted = sortmatrix(p2vm, ich, ABCDindex, direction='p2vm')
    
    conventionalorder=[]
    for ia in range(1,NA+1):
        for iap in range(ia+1,NA+1):
            conventionalorder.append(f"B{ia}{iap}")
    
    mod = v2pm_sorted[0]
    
    newfig=nfig
    fig = plt.figure(newfig,clear=True)
    fig.suptitle("V2PM and P2VM")
    
    ax1,ax2 = fig.subplots(ncols=2)
    ax1.set_title("V2PM")
    ax2.set_title("P2VM")
    
    # Show the V2PM matrix
    
    ax1.imshow(mod)
    
    # Create a Rectangle patch
    # rect = Rectangle((-0.5,11.5),1,1,linewidth=1,edgecolor='r',facecolor='none')
    xticks=[] ; xticklabels=[]
    xticks.append(NA//2) ; xticklabels.append('Photometries')
    x = NA-0.5 ; xticks.append(NA+NIN//2) ; xticklabels.append('Real parts')
    ax1.axvline(x = x, color = 'k', linestyle = '-')
    # x = NA+0.5 ; xticks.append(x) ; xticklabels.append('B1X')
    x = NA+NIN-0.5 ; xticks.append(NA+3*NIN//2) ; xticklabels.append('Imaginary parts')
    ax1.axvline(x = NA+NIN-0.5, color = 'k', linestyle = '-')
    # x = NA+NIN+0.5 ; xticks.append(x) ; xticklabels.append('B1X')
    for ia in range(1,NA-1):
        ib = posk(ia,ia+1,NA)
        x = NA+ib-0.5# ; xticks.append(x) ; xticklabels.append(f'B{ia+1}X')
        ax1.axvline(x = x, color = 'k', linestyle = '-', linewidth=.5)
        x = NA+NIN+ib-0.5# ; xticks.append(x) ; xticklabels.append(f'B{ia+1}X')
        ax1.axvline(x = x, color = 'k', linestyle = '-', linewidth=.5)
    
    for ib in range(NIN):
        ax1.axhline(y = ib*4-0.5, color = 'k', linestyle = '-', linewidth=.5)
    # Add the patch to the Axes
    # ax1.add_patch(rect)
    
    
    # xticklabels2=['Photometries','Real parts','Imaginary parts']
    
    
    # ax2.set_xticks(xticks2)
    # ax2.set_xticklabels(xticklabels2)
    
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels)
    ax1.xaxis.set_ticks_position('none')
    # ax1.set_yticks(np.arange(NP))
    # ax1.set_yticklabels(['A','B','C','D']*NIN)
    
    # ax2 = ax1.twiny()
    ax1.set_yticks(np.arange(NIN)*Nmod+Nmod//2-.5)
    ax1.set_yticklabels(conventionalorder)
    ax1.set_ylabel('Pixels sorted in ABCD')

    ax1.set_xlabel("Coherent flux")
    
    # ax1.tick_params(labelbottom='off',labeltop='on',bottom="off")
    
    
    # Show the P2VM matrix
    
    demod = p2vm_sorted[0]
    
    ax2.imshow(demod)
    
    # Create a Rectangle patch
    # rect = Rectangle((-0.5,11.5),1,1,linewidth=1,edgecolor='r',facecolor='none')
    yticks=[] ; yticklabels=[]
    yticks.append(-0.5) ; yticklabels.append('Photometries')
    y = NA-0.5 ; yticks.append(y) ; yticklabels.append('Real parts')
    ax2.axhline(y = y, color = 'k', linestyle = '-')
    y = NA+0.5 ; yticks.append(y) ; yticklabels.append('B1X')
    y = NA+NIN-0.5 ; yticks.append(y) ; yticklabels.append('Imaginary parts')
    ax2.axhline(y = NA+NIN-0.5, color = 'k', linestyle = '-')
    y = NA+NIN+0.5 ; yticks.append(y) ; yticklabels.append('B1X')
    for ia in range(1,NA-1):
        ib = posk(ia,ia+1,NA)
        y = NA+ib-0.5 ; yticks.append(y) ; yticklabels.append(f'B{ia+1}X')
        ax2.axhline(y = y, color = 'k', linestyle = '-', linewidth=.5)
        y = NA+NIN+ib-0.5 ; yticks.append(y) ; yticklabels.append(f'B{ia+1}X')
        ax2.axhline(y = y, color = 'k', linestyle = '-', linewidth=.5)
    
    for ib in range(NIN):
        ax2.axvline(x = ib*4-0.5, color = 'k', linestyle = '-', linewidth=.5)
    # Add the patch to the Axes
    # ax2.add_patch(rect)
    
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels)
    # ax2.set_xticks(np.arange(NP))
    # ax2.set_xticklabels(['A','B','C','D']*NIN)
    
    
    ax2.set_xticks(np.arange(NIN)*Nmod+Nmod//2-.5)
    ax2.set_xticklabels(conventionalorder)
    ax2.set_xlabel('Pixels sorted in ABCD')
    
# =============================================================================
#     Photometry repartition in the P2VM
# =============================================================================
    
    cset = tol_cset('bright')

    newfig+=1
    fig=plt.figure(newfig, clear=True)
    fig.suptitle("Photometry repartitions and phaseshift of the beams")
    ax1,ax2,ax3,ax4 = fig.subplots(nrows=4,gridspec_kw={'height_ratios': [.2,1,.2,1]})
    
    photometries = np.zeros([NW,NIN,Nmod,2])
    phases = np.zeros([NW,NIN,Nmod])
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib=posk(ia,iap,NA)
            for k in range(Nmod):
                photometries[:,ib,k,0] = v2pm_sorted[:,4*ib+k,ia]*100
                photometries[:,ib,k,1] = v2pm_sorted[:,4*ib+k,iap]*100
                R = v2pm_sorted[:,4*ib+k,NA+ib] ; I = v2pm_sorted[:,4*ib+k,NA+NIN+ib] 
                phases[:,ib,k] = np.arctan2(I,R)
                
    NINtemp = NIN//2+1
    xtop = np.arange(NINtemp)  # the label locations
    xbot = np.arange(NINtemp,NIN)
    width=0.8
    barwidth=width/8
    bar_patches1=[] ; bar_patches2=[]
    bar_patches1.append(mpatches.Patch(facecolor='gray',edgecolor='black',hatch='///',label="First beam"))
    bar_patches1.append(mpatches.Patch(facecolor='gray',edgecolor='black', label="Second beam"))
    for k in range(Nmod):
        
        firstbar_pos = xtop-width/2+barwidth/2
        rects1 = ax2.bar(firstbar_pos + 2*k*barwidth, photometries[0,:NINtemp,k,0], 
                         barwidth, hatch='///',color=cset[k],edgecolor='black').patches
        rects2 = ax2.bar(firstbar_pos + (2*k+1)*barwidth, photometries[0,:NINtemp,k,1], 
                         barwidth, color=cset[k],edgecolor='black').patches
        rects_moy1 = ax2.bar(firstbar_pos + (2*k+1/2)*barwidth, np.mean(photometries[0,:NINtemp,k,:],axis=-1), 
                         barwidth*2, color=cset[k],edgecolor='black',fill=False,linestyle='--').patches
        rects3 = ax1.bar(firstbar_pos + 2*k*barwidth+barwidth/2, phases[0,:NINtemp,k], 
                             barwidth,color=cset[k],edgecolor='black')
        firstbar_pos = xbot-width/2+barwidth/2
        rects4 = ax4.bar(firstbar_pos + 2*k*barwidth, photometries[0,NINtemp:,k,0],
                         barwidth, hatch='///',color=cset[k],edgecolor='black').patches
        rects5 = ax4.bar(firstbar_pos + (2*k+1)*barwidth, photometries[0,NINtemp:,k,1],
                         barwidth,color=cset[k],edgecolor='black').patches
        rects_moy2 = ax4.bar(firstbar_pos + (2*k+1/2)*barwidth, np.mean(photometries[0,NINtemp:,k,:],axis=-1), 
                         barwidth*2, color=cset[k],edgecolor='black',fill=False,linestyle='--').patches
        rects6 = ax3.bar(firstbar_pos + 2*k*barwidth+barwidth/2, phases[0,NINtemp:,k], 
                             barwidth,color=cset[k],edgecolor='black')
        
        # rects1 = ax2.patches[:NINtemp] ; rects2 = ax2.patches[NINtemp:2*NINtemp]
        # rects_moy = ax2.patches[2*NINtemp:]
        
        for rect1,rect2,rect_moy in zip(rects1,rects2,rects_moy1):
            height = np.max([rect1.get_height(),rect2.get_height(),rect_moy.get_height()])
            ax2.text(rect_moy.get_x() + rect_moy.get_width() / 2, height+0.1, round(height,1),
                    ha='center', va='bottom')
            
        for rect1,rect2,rect_moy in zip(rects4,rects5,rects_moy2):
            height = np.max([rect1.get_height(),rect2.get_height(),rect_moy.get_height()])
            ax4.text(rect_moy.get_x() + rect_moy.get_width() / 2, height+0.1, round(height,1),
                    ha='center', va='bottom')
            # ax4.text(rect.get_x() + rect.get_width() / 2, height + 5, height,
            #         ha='center', va='bottom')
            
        bar_patches2.append(mpatches.Patch(color=cset[k],label="ABCD"[k]))
        # rects3 = ax1.bar(x - width/2+0.25, photometries[0,:NINtemp,0,0], barwidth, label='C',color='b')
        # rects4 = ax1.bar(x - width/2+0.35, photometries[0,:NINtemp,0,1], barwidth, label='D',color='g')
    
    # ax1.bar_label(rects1)
    # ax1.bar_label(rects2)
    # xlim = (xtop[0]-width , xtop[-1]+width)
        
    # ax3.plot(npxtop,0.12*np.ones_like(xtop), color='black', label="Phase")
    # ax3.hlines(0.12, xlim[0],xlim[1], color='black', label="Phase")
    # ax3.hlines(0.08, xlim[0],xlim[1], color='black', label="Phase")
    # ax3.hlines(0.16, xlim[0],xlim[1], color='black', label="Phase")
    # ax3.plot(phases[0,NINtemp:,])
    
    
    xticklabels=[]
    for ia in range(NA):
        for iap in range(ia+1,NA):
            xticklabels.append(f"B{ia+1}{iap+1}")
    
    ax2.set_xticks(xtop)
    ax2.set_xticklabels(xticklabels[:NINtemp])
    
    ax2.set_ylim(0,10)
    ax4.set_ylim(0,10)
    
    ax1.set_ylim(-np.pi,np.pi)
    ax1.grid()

    ax3.set_ylim(-np.pi,np.pi)
    ax3.grid()
    ax4.set_xticks(xbot)
    ax4.set_xticklabels(xticklabels[NINtemp:])
    
    first_legend = ax4.legend(handles=bar_patches2, loc='lower right')
    ax4.add_artist(first_legend)
    ax4.legend(handles=bar_patches1, loc='upper right')
    
    
    ax1.set_ylabel("Phase [rad]")
    ax2.set_ylabel("Transmission")
    
    # ax4.bar_label(rects4, label_type='center')
    
    fig.show()
    
    
    
    
    
    newfig+=1
    fig=plt.figure(newfig, clear=True)
    fig.suptitle("Photometry repartitions and phaseshift of the beams")
    ax2,ax4 = fig.subplots(nrows=2)
    
    photometries = np.zeros([NW,NIN,Nmod,2])
    phases = np.zeros([NW,NIN,Nmod])
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib=posk(ia,iap,NA)
            for k in range(Nmod):
                photometries[:,ib,k,0] = v2pm_sorted[:,4*ib+k,ia]*100
                photometries[:,ib,k,1] = v2pm_sorted[:,4*ib+k,iap]*100
                R = v2pm_sorted[:,4*ib+k,NA+ib] ; I = v2pm_sorted[:,4*ib+k,NA+NIN+ib] 
                phases[:,ib,k] = np.arctan2(I,R)
                
    NINtemp = NIN//2+1
    xtop = np.arange(NINtemp)  # the label locations
    xbot = np.arange(NINtemp,NIN)
    
    ax2_firstbar_positions = xtop-width/2+barwidth/2
    ax4_firstbar_positions = xbot-width/2+barwidth/2
    width=0.8
    barwidth=width/8
    bar_patches1=[] ; bar_patches2=[]
    bar_patches1.append(mpatches.Patch(facecolor='gray',edgecolor='black',hatch='///',label="First beam"))
    bar_patches1.append(mpatches.Patch(facecolor='gray',edgecolor='black', label="Second beam"))
    for k in range(Nmod):
        
        rects1 = ax2.bar(ax2_firstbar_positions + 2*k*barwidth, photometries[0,:NINtemp,k,0], 
                         barwidth, hatch='///',color=cset[k],edgecolor='black').patches
        rects2 = ax2.bar(ax2_firstbar_positions + (2*k+1)*barwidth, photometries[0,:NINtemp,k,1], 
                         barwidth, color=cset[k],edgecolor='black').patches
        rects_moy1 = ax2.bar(ax2_firstbar_positions + (2*k+1/2)*barwidth, np.mean(photometries[0,:NINtemp,k,:],axis=-1), 
                         barwidth*2, color=cset[k],edgecolor='black',fill=False,linestyle='--').patches

        
        # rects3 = ax1.bar(firstbar_pos + 2*k*barwidth+barwidth/2, phases[0,:NINtemp,k], 
        #                      barwidth,color=cset[k],edgecolor='black')
        rects4 = ax4.bar(ax4_firstbar_positions + 2*k*barwidth, photometries[0,NINtemp:,k,0],
                         barwidth, hatch='///',color=cset[k],edgecolor='black').patches
        rects5 = ax4.bar(ax4_firstbar_positions + (2*k+1)*barwidth, photometries[0,NINtemp:,k,1],
                         barwidth,color=cset[k],edgecolor='black').patches
        rects_moy2 = ax4.bar(ax4_firstbar_positions + (2*k+1/2)*barwidth, np.mean(photometries[0,NINtemp:,k,:],axis=-1), 
                         barwidth*2, color=cset[k],edgecolor='black',fill=False,linestyle='--').patches
        # rects6 = ax3.bar(firstbar_pos + 2*k*barwidth+barwidth/2, phases[0,NINtemp:,k], 
        #                      barwidth,color=cset[k],edgecolor='black')
        
        # rects1 = ax2.patches[:NINtemp] ; rects2 = ax2.patches[NINtemp:2*NINtemp]
        # rects_moy = ax2.patches[2*NINtemp:]
        
        for rect1,rect2,rect_moy in zip(rects1,rects2,rects_moy1):
            height = np.max([rect1.get_height(),rect2.get_height(),rect_moy.get_height()])
            ax2.text(rect_moy.get_x() + rect_moy.get_width() / 2, height+0.1, round(height,1),
                    ha='center', va='bottom')
            
        for rect1,rect2,rect_moy in zip(rects4,rects5,rects_moy2):
            height = np.max([rect1.get_height(),rect2.get_height(),rect_moy.get_height()])
            ax4.text(rect_moy.get_x() + rect_moy.get_width() / 2, height+0.1, round(height,1),
                    ha='center', va='bottom')
            # ax4.text(rect.get_x() + rect.get_width() / 2, height + 5, height,
            #         ha='center', va='bottom')
            
        bar_patches2.append(mpatches.Patch(color=cset[k],label="ABCD"[k]))
        # rects3 = ax1.bar(x - width/2+0.25, photometries[0,:NINtemp,0,0], barwidth, label='C',color='b')
        # rects4 = ax1.bar(x - width/2+0.35, photometries[0,:NINtemp,0,1], barwidth, label='D',color='g')
    
    ax2_xmin,ax2_xmax = ax2.get_xlim() ; ax4_xmin,ax4_xmax = ax4.get_xlim()
    ax2_ymin,ax2_ymax = ax2.get_ylim() ; ax4_ymin,ax4_ymax = ax4.get_ylim()
    ax2_normalised_width = 1/(ax2_xmax-ax2_xmin)*width*.7
    ax2_normalised_height = 1/(ax2_ymax-ax2_ymin)*width
    ax4_normalised_width = 1/(ax4_xmax-ax4_xmin)*width*.7
    ax4_normalised_height = 1/(ax4_ymax-ax4_ymin)*width
    ax2_bottomleft = 1/(ax2_xmax-ax2_xmin)*(ax2_firstbar_positions+barwidth/2-ax2_xmin)
    ax4_bottomleft = 1/(ax4_xmax-ax4_xmin)*(ax4_firstbar_positions+barwidth/2-ax4_xmin)
    
    for ib in range(NINtemp):
        subpos=(ax2_bottomleft[ib],0.75,ax2_normalised_width,ax2_normalised_height)#firstbar_pos[0] + 2*k*barwidth,
        label = True if ib==0 else False
        ax=add_subplot_axes(ax2,subpos,polar=True,label=label)
        
        for k in range(Nmod):
            phase = phases[0,ib,k]
            ax.arrow(phase, 0, 0, 1, width = 0.05,
                     edgecolor=cset[k],facecolor = cset[k], lw = 2, zorder = 5)
        # ax.set_thetagrids(phases[0,ib,:]*180/np.pi,labels=np.round(phases[0,ib,:]*180/np.pi))
    for ib in range(NIN-NINtemp):
        subpos=(ax4_bottomleft[ib],0.75,ax4_normalised_width,ax4_normalised_height)#firstbar_pos[0] + 2*k*barwidth,
        label = False
        ax=add_subplot_axes(ax4,subpos,polar=True,label=label)
        
        for k in range(Nmod):
            phase = phases[0,NINtemp+ib,k]
            ax.arrow(phase, 0, 0, 1, width = 0.05,
                     edgecolor=cset[k],facecolor = cset[k], lw = 2, zorder = 5)
    # axin1 = add_subplot_axes(ax4,subpos)

    # ax1.bar_label(rects1)
    # ax1.bar_label(rects2)
    # xlim = (xtop[0]-width , xtop[-1]+width)
        
    # ax3.plot(npxtop,0.12*np.ones_like(xtop), color='black', label="Phase")
    # ax3.hlines(0.12, xlim[0],xlim[1], color='black', label="Phase")
    # ax3.hlines(0.08, xlim[0],xlim[1], color='black', label="Phase")
    # ax3.hlines(0.16, xlim[0],xlim[1], color='black', label="Phase")
    # ax3.plot(phases[0,NINtemp:,])
    
    
    xticklabels=[]
    for ia in range(NA):
        for iap in range(ia+1,NA):
            xticklabels.append(f"B{ia+1}{iap+1}")
    
    ax2.set_xticks(xtop)
    ax2.set_xticklabels(xticklabels[:NINtemp])
    
    ax2.set_ylim(0,14)
    ax4.set_ylim(0,14)
    
    ax1.set_ylim(-np.pi,np.pi)
    ax1.grid()

    ax3.set_ylim(-np.pi,np.pi)
    ax3.grid()
    ax4.set_xticks(xbot)
    ax4.set_xticklabels(xticklabels[NINtemp:])
    
    first_legend = ax4.legend(handles=bar_patches2, loc='lower right')
    ax4.add_artist(first_legend)
    ax4.legend(handles=bar_patches1, loc='upper right')
    
    
    ax1.set_ylabel("Phase [rad]")
    ax2.set_ylabel("Transmission")
    
    # ax4.bar_label(rects4, label_type='center')
    
    fig.show()
    
        
    
    
# =============================================================================
#     Noise propagation through the matrix
# =============================================================================
    
    photons = 1e3
    # Coherent vector of a perfectly coherent pair of fields.
    CoherentFlux = np.zeros(NB)
    CoherentFlux[:NA+NIN]=photons # In Cassaing formalism: N=photons ; C=photons ; S=0
    
    DetectionCoherent = np.dot(v2pm_sorted, CoherentFlux)
    
    QuadratureCoherentFlux = np.zeros(NB)
    QuadratureCoherentFlux[:NA] = photons
    QuadratureCoherentFlux[NA+NIN:] = photons # In Cassaing formalism: N=photons ; C=0 ; S=photons
    
    DetectionQuadrature = np.dot(v2pm_sorted, QuadratureCoherentFlux)
    
    # Covariance matrix
    CovMatrixCoherent = np.zeros([NW,NB,NB])
    CovMatrixQuadrature = np.zeros([NW,NB,NB])
    
    for iw in range(NW):
        CovMatrixCoherent[iw] = np.dot(p2vm_sorted[iw], np.dot(np.diag(DetectionCoherent[iw]),np.transpose(p2vm_sorted[iw])))
        CovMatrixQuadrature[iw] = np.dot(p2vm_sorted[iw], np.dot(np.diag(DetectionQuadrature[iw]),np.transpose(p2vm_sorted[iw])))
    
    newfig+=1
    fig = plt.figure(newfig, clear=True)
    fig.suptitle("Covariance matrices in photon noise regime")
    ax1, ax2 = fig.subplots(ncols=2)
    ax1.set_title('Covariance matrix when cophased')
    ax2.set_title('Covariance matrix when phase in quadrature')
    
    ax1.imshow(CovMatrixCoherent[0])
    ax2.imshow(CovMatrixQuadrature[0])
    
    xticks=[] ; xticklabels=[]
    xticks.append(NA//2) ; xticklabels.append('Photometries')
    x = NA-0.5 ; xticks.append(NA+NIN//2) ; xticklabels.append('Real parts')
    ax1.axvline(x = x, color = 'k', linestyle = '-')
    ax2.axvline(x = x, color = 'k', linestyle = '-')
    # x = NA+0.5 ; xticks.append(x) ; xticklabels.append('B1X')
    x = NA+NIN-0.5 ; xticks.append(NA+3*NIN//2) ; xticklabels.append('Imaginary parts')
    ax1.axvline(x = NA+NIN-0.5, color = 'k', linestyle = '-')
    ax2.axvline(x = NA+NIN-0.5, color = 'k', linestyle = '-')
        
    yticks=[] ; yticklabels=[]
    yticks.append(-0.5) ; yticklabels.append('Photometries')
    y = NA-0.5 ; yticks.append(y) ; yticklabels.append('Real parts')
    ax1.axhline(y = y, color = 'k', linestyle = '-')
    ax2.axhline(y = y, color = 'k', linestyle = '-')
    ax1.axvline(x = y, color = 'k', linestyle = '-')
    ax2.axvline(x = y, color = 'k', linestyle = '-')
    y = NA+0.5 ; yticks.append(y) ; yticklabels.append('B1X')
    y = NA+NIN-0.5 ; yticks.append(y) ; yticklabels.append('Imaginary parts')
    ax1.axhline(y = y, color = 'k', linestyle = '-')
    ax2.axhline(y = y, color = 'k', linestyle = '-')
    ax1.axvline(x = y, color = 'k', linestyle = '-')
    ax2.axvline(x = y, color = 'k', linestyle = '-')
    y = NA+NIN+0.5 ; yticks.append(y) ; yticklabels.append('B1X')
    for ia in range(1,NA-1):
        ib = posk(ia,ia+1,NA)
        y = NA+ib-0.5 ; yticks.append(y) ; yticklabels.append(f'B{ia+1}X')
        ax1.axhline(y = y, color = 'k', linestyle = '-', linewidth=.5)
        ax2.axhline(y = y, color = 'k', linestyle = '-', linewidth=.5)
        ax1.axvline(x = y, color = 'k', linestyle = '-', linewidth=.5)
        ax2.axvline(x = y, color = 'k', linestyle = '-', linewidth=.5)
        y = NA+NIN+ib-0.5 ; yticks.append(y) ; yticklabels.append(f'B{ia+1}X')
        ax1.axhline(y = y, color = 'k', linestyle = '-', linewidth=.5)
        ax2.axhline(y = y, color = 'k', linestyle = '-', linewidth=.5)
        ax1.axvline(x = y, color = 'k', linestyle = '-', linewidth=.5)
        ax2.axvline(x = y, color = 'k', linestyle = '-', linewidth=.5)

    ax1.set_yticks(xticks)
    # ax1.set_yticklabels(xticklabels)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)

    ax2.set_yticks(xticks)
    # ax2.set_yticklabels(xticklabels)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels)

    ax1.set_xticks(xticks) ; ax1.set_xticklabels(xticklabels)
    ax2.set_xticks(xticks) ; ax2.set_xticklabels(xticklabels)

    dico = {'v2pm_sorted':v2pm_sorted, 
            'p2vm_sorted':p2vm_sorted, 
            'photometries':photometries, 
            'phases':phases,
            'CovMatrixCoherent':CovMatrixCoherent, 
            'CovMatrixQuadrature':CovMatrixQuadrature}

    return dico

def poskfai(ia,iap,iapp,N):
    """
    k-Position of the PhaseClosure for non redundant combination of three pupils
    in a total of N pupils.

    Parameters
    ----------
    ia : TYPE
        DESCRIPTION.
    iap : TYPE
        DESCRIPTION.
    iapp : TYPE
        DESCRIPTION.
    N : Integer
        Number of telescopes.

    Returns
    -------
    Integer
        k-position

    """
    
    from scipy.special import binom
    k0 = np.sum(binom(np.arange(N-ia,N),2))
    k1 = posk(iap-ia-1,iapp-iap,N-ia)
    
    return int(k0+k1)


def posk(ia,iap,N):
    """
    k-Position of the OPD[ia,iap] for non redundant combination of two pupils in
    a total of N pupils

    Parameters
    ----------
    ia : TYPE
        DESCRIPTION.
    iap : TYPE
        DESCRIPTION.
    iapp : TYPE
        DESCRIPTION.
    N : Integer
        Number of telescopes

    Returns
    -------
    Integer
        k-position on the most logical order: 
            for 6T: 01,02,03,04,05,12,13,14,15,23,24,25,34,35,45

    """
    
    return int(ia*N-ia*(ia+3)/2+iap-1)

def NB2NIN(vector):
    """
    Returns the entry vector of length NA**2 (which might the expected quantities) 
    on a non-redundant form of length NIN=NA(NA-1)/2 sorted as follow:
        12,..,1NA,23,..,2NA,34,..,3NA,..,(NA-1)(NA)

    Parameters
    ----------
    vector : FLOAT COMPLEX ARRAY [NB]
        Vector of general complex coherences.

    Returns
    -------
    ninvec : FLOAT COMPLEX ARRAY [NIN]
        Complex Vector of non-redundant mutual coherences.
    """

    
    NB = len(vector)
    NA = int(np.sqrt(NB))
    NIN = int(NA*(NA-1)/2)
    
    ninvec = np.zeros([NIN])*1j
    for ia in range(NA):
        for iap in range(ia+1,NA):
            k = posk(ia,iap,NA)
            ninvec[k] = vector[ia*NA+iap]
    
    return ninvec


def add_subplot_axes(ax,rect,polar=False,label=False,facecolor='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = ax.figure.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height = width*box.width/box.height
    # height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],polar=polar)
    subax.set_rticks([])
    
    if label:
        subax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2])
    else:
        subax.set_thetagrids([0,90,180,270],labels=[])
    # x_labelsize = subax.get_xticklabels()[0].get_size()
    # y_labelsize = subax.get_yticklabels()[0].get_size()
    # x_labelsize *= rect[2]**0.5
    # y_labelsize *= rect[3]**0.5
    # subax.xaxis.set_tick_params(labelsize=x_labelsize)
    # subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


    