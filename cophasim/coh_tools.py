# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:14:42 2020

@author: cpannetier
"""

import os, pkg_resources

datadir = 'data/'

import numpy as np
from scipy.special import jv,binom
from scipy import interpolate

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from cophasim.tol_colors import tol_cset # colorblind-riendly and contrastful library
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from astropy.io import fits

import matplotlib.pyplot as plt

from cophasim import config
import cophasim.decorators as deco

global h_, c_, k_

h_ = 6.626e-34  # Planck's constant
c_ = 3.0e+8     # Light velocity
k_ = 1.38e-23    # Boltzmann's constant


colors = tol_cset('bright')*20

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
    if array in ['chara','CHARA']: # information in SPICA JOSAA paper
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
    


def get_array(name='',band='H',getcoords=False,
              verbose=False):
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
    
    class Interferometer:
        def __init__(self):
            pass
    InterfArray = Interferometer()
    
    if "fits" in name:
        filepath = name
        if not os.path.exists(filepath):
            try:
                if verbose:
                    print("Looking for the interferometer file into the package's data")
                filepath = pkg_resources.resource_stream(__name__,filepath)
            except:
                raise Exception(f"{filepath} doesn't exist.")

        with fits.open(filepath) as hdu:
            ArrayParams = hdu[0].header
            NA, NIN = ArrayParams['NA'], ArrayParams['NIN']
            InterfArray.NA = NA
            InterfArray.NIN = NIN
            TelData = hdu[1].data
            BaseData = hdu[2].data
        
            InterfArray.TelNames = TelData['TelNames']
            InterfArray.TelCoordinates = TelData['TelCoordinates']
            InterfArray.TelTransmissions = TelData['TelTransmissions']
            InterfArray.TelSurfaces = TelData['TelSurfaces']
            InterfArray.BaseNames = BaseData['BaseNames']
            InterfArray.BaseCoordinates = BaseData['BaseCoordinates']
            
        InterfArray.BaseNorms = np.linalg.norm(InterfArray.BaseCoordinates[:,:2],axis=1)
    
        
    elif name == 'CHARA':         
            
        #official coordinates in [µm]
        TelCoordinates= np.array([[0,0,0],\
                                  [-5746854.437,33580641.636,636719.086],\
                                      [125333989.819,305932632.737,-5909735.735],\
                                          [70396607.118,269713282.258,-2796743.645],\
                                              [-175073332.211,216320434.499,-10791111.235],\
                                                  [-69093582.796,199334733.235,467336.023]])
        TelCoordinates=TelCoordinates*1e-6      # [m]
        
        NA = np.shape(TelCoordinates)[0]
        
        TelNames = ['S1','S2','E1','E2','W1','W2']
        BaseNames = []
        BaseCoordinates =[]

        for ia in range(NA):
            for iap in range(ia+1,NA):
                BaseNames.append(str(TelNames[ia]+TelNames[iap]))
                BaseCoordinates.append([TelCoordinates[ia,0]-TelCoordinates[iap,0],
                                        TelCoordinates[ia,1]-TelCoordinates[iap,1],
                                        TelCoordinates[ia,2]-TelCoordinates[iap,2]])
        BaseCoordinates = np.array(BaseCoordinates)        
        BaseNorms = np.linalg.norm(BaseCoordinates[:,:2],axis=1)

        InterfArray.NA = NA
        InterfArray.TelNames=TelNames
        InterfArray.telNameLength=2
        InterfArray.BaseNorms=BaseNorms
        InterfArray.BaseNames = BaseNames
        InterfArray.TelCoordinates = TelCoordinates
        InterfArray.BaseCoordinates = BaseCoordinates
        
        transmission, surface = info_array(name,band)
        InterfArray.TelSurfaces = np.ones(NA)*surface
        InterfArray.TelTransmissions = np.ones(NA)*transmission
        
    else:
        raise Exception("For defining the array, you must give a file \
or a name (currently only CHARA is available).")
        
    return InterfArray
    

def get_Bproj(baseline, theta):
    """
    Calculates projected bases according to the viewed angle in radian

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
        if baseline == 0:
            V = 1
        else:
            V = np.abs(2*jv(1, np.pi*alpha*baseline/spectra)/(np.pi*alpha*baseline/spectra))

    return V


#@deco.timer
def VanCittert(spectra, Obs, Target, plottrace=60, display=False,
               savedir='',ext='pdf',verbose=False):
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
        
    if (len(savedir)) and (not os.path.exists(savedir)):
        os.makedirs(savedir)
        
    if verbose:
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
        if verbose:
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
    #if BinaryObject:
    #    dtheta = np.min((angular_diameter1,angular_diameter2))/5
    #else:
    #    dtheta = angular_diameter1/5
        
    du = 0.005
    Npix = 5000     #2*int(thetamax/dtheta)
    dtheta = 1/(du*Npix)   # 0.04mrad
    
    thetamax = dtheta*Npix/2

    obj_plane = np.zeros([Npix,Npix])
    
    coords = (np.arange(Npix)- Npix/2)*dtheta
    (alpha,beta) = np.meshgrid(coords,-coords)
    dist1 = np.sqrt((alpha-pos_star1[0])**2 + (beta-pos_star1[1])**2)
    obj_plane[dist1<angular_diameter1/2] = 1
    NpixStar1 = np.sum(obj_plane)
    
    if BinaryObject:
        dist2 = np.sqrt((alpha-pos_star2[0])**2 + (beta-pos_star2[1])**2)
        obj_plane[dist2<angular_diameter2] = 1/LuminanceRatio
        
    
    # Normalisation of the luminosity, with first star as reference
    obj_plane = obj_plane/NpixStar1

    Nticks = 7
    ticks = np.linspace(0,Npix,Nticks)
        
    if display:            
        title = "Zernike Van-Cittert"
        plt.close(title) ; fig = plt.figure(title, figsize=(16,9))
        (ax1,ax2,ax3) = fig.subplots(ncols=3)
        
        spaceticks = (ticks-round(Npix/2))*dtheta
        spaceticks = spaceticks.round(decimals=1)
        
        ax1.set_title('Object in direct plane')
        ax1.imshow(obj_plane)
        ax1.set_xticks(ticks) ; ax1.set_xticklabels(spaceticks)
        ax1.set_yticks(ticks) ; ax1.set_yticklabels(-spaceticks)
        ax1.set_xlabel('\u03B1 [mas]')
        ax1.set_ylabel('\u03B2 [mas]')
    
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

        ax2.set_title('Module of the visibility')
        ax2.imshow(np.abs(uv_plane),vmin=0,vmax=1)
        ax2.set_xticks(ticks) ; ax2.set_xticklabels(freqticks)
        ax2.set_yticks(ticks) ; ax2.set_yticklabels(-freqticks)
        ax2.set_xlabel('u [mas-1]')
        ax2.set_ylabel('v [mas-1]')
        
        ax3.set_title('Phase of the visibility')
        ax3.imshow(np.angle(uv_plane),vmin=-np.pi,vmax=np.pi)
        ax3.set_xticks(ticks) ; ax3.set_xticklabels(freqticks)
        ax3.set_yticks(ticks) ; ax3.set_yticklabels(-freqticks)
        ax3.set_xlabel('u [mas-1]')
        ax3.set_ylabel('v [mas-1]')
        
        fig.show()
        
        if len(savedir):
            import time
            timestr = time.strftime("%Y%m%d-%H%M%S")
            plt.savefig(savedir+f"UVplane{timestr}.{ext}")
    
    """
    Projection of the interferometer on the (u,v) plane
    """
    
    # Get telescopes coordinates and names
    InterfArray = get_array(config.InterferometerFile, getcoords=True)    
    
    TelNames = InterfArray.TelNames
    TelCoords = InterfArray.TelCoordinates
    basecoords = InterfArray.BaseCoordinates
    
    TelCoords *= 1e6          # Convert to [µm]
    basecoords *= 1e6          # Convert to [µm]
    
    NA = len(TelCoords)
    
    TelAltaz = np.zeros([NA,2])
    for ia in range(NA):
        if np.linalg.norm(TelCoords[ia]) == 0:
            TelAltaz[ia,0] = 0
            TelAltaz[ia,1] = 0    
        else:
            TelAltaz[ia,1] = np.arctan(TelCoords[ia,0]/TelCoords[ia,1])
            TelAltaz[ia,0] = np.arcsin(TelCoords[ia,2]/np.linalg.norm(TelCoords[ia]))
    
    NIN = int(NA*(NA-1)/2)
    basealtaz = np.zeros([NIN,2])       # Altazimuthal coordinates of the baselines [radians]
    basedist = np.zeros([NIN,1])        # Baselines lengths in [µm]
    basecoords = np.zeros([NIN,3])      # Baselines coordinates [µm]
    basenames = []                      # Baselines names [string]
    
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib = int(ia*NA-ia*(ia+3)/2+iap-1)
            
            basealtaz[ib] = TelAltaz[iap]-TelAltaz[ia]
            
            basenames.append(TelNames[ia]+TelNames[iap])
            basecoords[ib] = TelCoords[iap] - TelCoords[ia]
            basedist[ib] = np.linalg.norm(basecoords[ib])
    
    
    # First case: the name of the object has been given. We search it in Simbad
    # database and its AltAzimutal coordinates if a Date has been given.
    if 'AltAz' not in vars(Obs):
        if Target.Name not in ('Simple','Binary','Unresolved'):
        
            starttime = Time(Obs.DATE)
            if verbose:
                print(f"Observation date: {Obs.DATE}")
        
            starcoords = SkyCoord.from_name(Target.Name)
            ArrayLocation=EarthLocation.of_site(Obs.ArrayName)
            staraltaz = starcoords.transform_to(AltAz(obstime=starttime,location=ArrayLocation))
            
            (altitude, azimuth) = (staraltaz.alt.radian,staraltaz.az.radian)
            Obs.AltAz = (180/np.pi*altitude, 180/np.pi*azimuth)
            if verbose:
                print(f"Object AltAz coordinates: ({round(Obs.AltAz[0],1)},{round(Obs.AltAz[1],1)})")
        else:
            Obs.AltAz = (90,0)
            
    else:
        (altitude, azimuth) = (theta*np.pi/180 for theta in Obs.AltAz)
        if verbose:
            print(f"User defined {Target.Name} object with AltAz={Obs.AltAz}")
        
        
    
    """
    Altazimuthal coordinates definition:
        - azimuth: angular distance between the intersection of the target meridian with 
        the horizon and the north horizon: East = 90° Azimuth.
        - altitude: angular distance between the horizon and the target, along its meridian.
    
    Coordinates definition (from CHARA convention):
        - X: toward East
        - Y: toward North
        - Z: toward Zenith
    """
    
    
    # Coordinates of the (u,v) plane in the (E,N,Z) referential
    u_Ep = np.array([np.cos(azimuth),np.sin(azimuth),0])
    u_Np = np.array([-np.sin(altitude)*np.sin(azimuth),np.sin(altitude)*np.cos(azimuth),np.cos(altitude)])
    u_Zp = np.array([np.cos(altitude)*np.sin(azimuth),-np.cos(altitude)*np.cos(azimuth),np.sin(altitude)])
    
    
    # Projection baselines on the target's (u,v) plane
    B_Ep = np.dot(basecoords, np.transpose(u_Ep))
    B_Np = -np.dot(basecoords, np.transpose(u_Np))
    B_Zp = np.dot(basecoords, np.transpose(u_Zp))
    
    # baselines = np.transpose(basedist)*np.sin(altitude-basealtaz[:,0])
    
    if isinstance(spectra,(float,int)):
        spectra = [spectra]
    
    NW = len(spectra)
    
    # Projection baselines on the u,v coordinates (oriented with the star north-east)
    interf_uv = np.zeros([NW,NIN,2])
    for iw in range(NW):
        lmbda=spectra[iw]
        interf_uv[iw,:,0] = B_Ep/lmbda
        interf_uv[iw,:,1] = B_Np/lmbda
    
    
    # # Matrice de projection
    
    # M = np.array([[np.sin(h),np.cos(h),0],
    #               [-np.sin(delta)*np.cos(h),np.sin(delta)*np.sin(h),np.cos(delta)],
    #               [np.cos(delta)*np.cos(h),-np.cos(delta)*np.sin(h),np.sin(delta)]])
    
    
    
    # Conversion functions
    mas2rad = lambda mas : 1/3600*1e-3*np.pi/180*mas
    rad2mas = lambda rad : 3600*1e3*180/np.pi*rad

    #Convert (u,v) plane from radian to mas
    interf_uv_direct = 1/interf_uv
    interf_uv_direct_mas = rad2mas(interf_uv_direct)
    interf_uv = 1/interf_uv_direct_mas
    
    # Return the complex visibility vector of the source
    visibilities = np.zeros([NW,NIN])*1j
    for ib in range(NIN):
        for iw in range(NW):     
            ub=interf_uv[iw,ib,0] ; vb=interf_uv[iw,ib,1]
            Nu = int(round(ub/dfreq)+Npix/2)
            Nv = int(round(vb/dfreq)+Npix/2)
            visibilities[iw,ib] = uv_plane[Nu,Nv]
        
    UVcoords = (ucoords,vcoords)
    #UVcoordsMeters = [1/mas2rad(1/coord)*np.median(lmbda) for coord in UVcoords]
    
    if verbose:
        print("Visibilities calculated.")
    
    
    if display:         # Display (u,v) plane with interferometer projections (first wavelength)
        
        if 'active_ich' in config.FS.keys():
            active_ich = config.FS['active_ich']
            PhotSNR = config.FS['PhotometricSNR']
        else:
            active_ich=np.arange(NIN)
            PhotSNR = np.ones(NIN)
            
        actindNIN = np.array(active_ich)>=0
        actind = np.concatenate([actindNIN[::-1], actindNIN])
        
        PhotometricSNR = np.concatenate([PhotSNR[::-1],PhotSNR])
        
        if verbose:
            print(f"Plot interferometer's (u,v) coverage on figure {config.newfig}")  
        interf_uv_complete = np.concatenate((interf_uv[NW//2], -interf_uv[NW//2]),axis=0)
        
        uvmax = np.max(interf_uv_complete/dfreq)+10
        
        Ndisplay = 2*int(uvmax+10)
        
        uv_crop = uv_plane[(Npix-Ndisplay)//2:(Npix+Ndisplay)//2,(Npix-Ndisplay)//2:(Npix+Ndisplay)//2]
        interf_plot = interf_uv_complete/dfreq+Ndisplay/2
        
        Nticks = 11
        ticks = np.linspace(0,Ndisplay-1,Nticks)
        freqticks = (ticks+1-round(Ndisplay/2))*dfreq
        freqticks = freqticks.round(decimals=1)
        # Display the Visibility and the interferometer baselines on the (u,v) plane
        title = "UV plane - module"
        plt.close(title); fig = plt.figure(title)
        ax = fig.subplots()
        #fig.suptitle('(u,v) coverage')
        im=ax.imshow(np.abs(uv_crop),vmin=0,vmax=1)
        
        ax.scatter(interf_plot[actind,0], interf_plot[actind,1], marker='x', s=50*np.sqrt(PhotometricSNR[actind]),linewidth=1, color='firebrick')
        ax.scatter(interf_plot[actind==False,0], interf_plot[actind==False,1], marker='x', s=50, linewidth=1,color='black')
        plt.xticks(ticks,freqticks)
        plt.yticks(ticks,-freqticks)
        ax.set_xlabel('u (W-E) [$mas^{-1}$]')
        ax.set_ylabel('v (S-N) [$mas^{-1}$]')
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        
        if len(savedir):
            import time
            timestr = time.strftime("%Y%m%d-%H%M%S")
            plt.savefig(savedir+f"UVplane_mod{timestr}.{ext}")
            
        title = "UV plane - argument"
        plt.close(title)
        fig = plt.figure(title)
        ax = fig.subplots()
        #fig.suptitle('(u,v) coverage')
        im=ax.imshow(np.angle(uv_crop),vmin=-np.pi,vmax=np.pi)
        
        ax.scatter(interf_plot[actind,0], interf_plot[actind,1], marker='x', s=50*np.sqrt(PhotometricSNR[actind]),linewidth=1, color='firebrick')
        ax.scatter(interf_plot[actind==False,0], interf_plot[actind==False,1], marker='x', s=50, linewidth=1,color='black')
        plt.xticks(ticks,freqticks)
        plt.yticks(ticks,-freqticks)
        ax.set_xlabel('u (W-E) [$mas^{-1}$]')
        ax.set_ylabel('v (S-N) [$mas^{-1}$]')
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        
        if len(savedir):
            import time
            timestr = time.strftime("%Y%m%d-%H%M%S")
            plt.savefig(savedir+f"UVplane_angle{timestr}.{ext}")
    
    # if NW==1:
    #     visibilities = visibilities[0]
    
    return visibilities, interf_uv, uv_plane, UVcoords


def create_obsfile(spectra, Obs, Target, savingfilepath='',
                   savedir='', ext='pdf',overwrite=False, display=False,
                   verbose=True):
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
    
    # Preference for photons directly to keep a uniform spectra
    UncohIrradiance = Irradiance*Lph_H             # [phot/s/m²/µm] Source Irradiance 
    
    # Using Watt as reference and converting in photons: drawback=non uniform spectra
    # L0_w = 7 * 10**(-10)                      # [W/m²/µm] Reference luminance at 1.65µm (Lena)
    # L0_w = 11.38 * 10**(-10)                  # [W/m²/µm] Reference luminance at 1.63µm (Bessel)
    # Lw_H = L0_w*10**(-0.4*magH)                        # Definition of the magnitude
    # UncohIrradiance_w = luminance*Lw_H / (h_*c_/spectra*1e6)          # [phot/s/m²/µm]

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
        telNameLength = 2
        TelCoordinates = TelData['TelCoordinates']
        TelTransmissions = TelData['TelTransmissions']
        TelSurfaces = TelData['TelSurfaces']
        BaseNames = BaseData['BaseNames']
        BaseCoordinates = BaseData['BaseCoordinates']
        
        
    InterfArray = get_array(name=filepath)
    
    NB = NA**2
    NC = int(binom(NA,3))
    
    # Transportation of the star light into the interferometer
    Throughput = np.reshape(InterfArray.TelSurfaces*InterfArray.TelTransmissions,[1,NA])
    ThroughputMatrix = np.sqrt(np.dot(np.transpose(Throughput), Throughput))
    ThroughputMatrix = ThroughputMatrix.reshape([1,NB])
    # Matrix where the element at the (ia*NA+iap) position is Ta*Tap
    # UncohIrradianceAfterTelescopes = np.dot(np.diag(TelTransmissions),np.transpose(UncohIrradiance))
    
    # Projection of the base on the (u,v) plane
    #BaseNorms = get_Bproj(np.array(InterfArray.BaseNorms), Obs.AltAz[0])
    
    VisObj = np.zeros([NW,NB])*1j               # Object Visibility [normalised]
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
                    bispec = ci1*ci2*ci3
                    bispec[np.abs(bispec)<0.05] = 0
                    bispectrum[:,ic] = bispec
        
    else:           # Van-Cittert theorem visibility
        visibilities,_,_,_ = VanCittert(spectra, Obs, Target,
                                        display=display, 
                                        savedir=savedir, ext=ext,
                                        verbose=verbose)
        
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

    BaseNorms, TelNames = InterfArray.BaseNorms, InterfArray.TelNames
    
    if display:
        from . import config
        
        fig = plt.figure("Visibility of the star",figsize=(12,9))
        linestyles=[]
        (ax1,ax2,ax5),(ax3,ax4,ax6) = fig.subplots(nrows=2,ncols=3,gridspec_kw={'width_ratios':[3,3,1]})
        
        gs=ax3.get_gridspec()
        ax3.remove() ; ax4.remove()
        # Add a unique axe on the last row
        ax3 = fig.add_subplot(gs[1,:])
        
        ax5.axis('off') ; ax6.axis('off')
        
        ax1.set_title('|V|')
        ax2.set_title('arg(V)')
        ax3.set_title('Coherent Irradiance $\Gamma$')
        
        ax1.plot(spectra, np.abs(VisObj[:,0]),color='red')
        ax3.plot(spectra, UncohIrradiance,color='red')  # Same for all baselines
        linestyles.append(mlines.Line2D([],[],
                                        color='red',
                                        label=f"Uncoherent flux"))

        for ia in range(NA):
            for iap in range(ia+1,NA):
                ib = posk(ia,iap,NA)
                if ib<8:
                    cl=colors[ib] ; ls='solid'
                else:
                    cl=colors[ib-8] ; ls='--'
                ax1.plot(spectra, np.abs(VisObj[:,ib]),
                         color=cl,linestyle=ls)
                ax2.plot(spectra, np.angle(VisObj[:,ib]),
                         color=cl,linestyle=ls)
                ax3.plot(spectra, np.abs(CohIrradiance[:,ib]),
                         color=cl,linestyle=ls)
                linestyles.append(mlines.Line2D([],[],
                                                color=cl,linestyle=ls,
                                                label=f"{BaseNames[ib]}:{round(BaseNorms[ib])}m"))
        
        ax1.set_ylim(0,1.1) ; ax2.set_ylim(-np.pi,np.pi)
        ax1.set_xlabel('Wavelengths [µm]')
        ax2.set_xlabel('Wavelengths [µm]')
        ax3.set_xlabel('Wavelengths [µm]')
        
        ax5.legend(handles=linestyles, loc='upper right')
        
        ax3.set_ylabel('Coherent Irradiance \n [photons/s/m²/µm]')
        ax1.grid(True);ax2.grid(True);ax3.grid(True)
        
        config.newfig+=1       
        
        
    if savingfilepath=='no':
        if verbose:
            print("Not saving the data.")
        return CohIrradiance, UncohIrradiance, VisObj, BaseNorms, TelNames

    else:
        fileexists = os.path.exists(savingfilepath)
        if fileexists:
            if verbose:
                print(f'{savingfilepath} already exists.')
            if overwrite:
                os.remove(savingfilepath)
            else:
                if verbose:
                    print("The file already exists and you didn't ask to overwrite it. I don't save the file.")          
                return CohIrradiance, UncohIrradiance, VisObj, BaseNorms, TelNames
    
        filedir = '/'.join(savingfilepath.split('/')[:-1])+'/'

    
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        
        
        # hdr = ArrayParams
        hdr = fits.Header()
        
        hdr['Filepath'] = savingfilepath.split('/')[-1]
        hdr['ARRAY'] = Obs.ArrayName
        hdr['AltAz'] = Obs.AltAz
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
        
        if verbose:
            print(f'Saving file into {savingfilepath}')
        hdu.writeto(savingfilepath)
    
    return CohIrradiance, UncohIrradiance, VisObj, BaseNorms, TelNames

def create_CfObj(spectra,Obs,Target,InterfArray,R=140):
    """
    Returns the coherent flux (photometries and mutual intensities) of the
    object along the given spectra for the given interferometric array.

    Parameters
    ----------
    spectra : ARRAY
        Spectral sampling.
    Obs : OBS CLASS OBJECT
        Contains information on the object position in the sky.
    Target : TARGET CLASS OBJECT
        Contains information on the target geometry, magnitude, etc...
    InterfArray : INTERFARRAY CLASS OBJECT
        Contains information on the interferometer geometry, transmission, etc...

    Returns
    -------
    CoherentFluxObject : ARRAY [dMW,NB]
        Coherent flux sorted like follows:
            - 0 ... NA: photometries
            - NA ... NA+NIN: Real(Cf)
            - NA ... NIN:NB: Imag(Cf)

    """
    
    MeanWavelength = np.mean(spectra)
    if hasattr(spectra, "__len__"):
        MW=len(spectra)
        MultiWavelength=True
    else:
        MultiWavelength=False
        spectra=np.array([spectra])
        MW=1

    
    VisObject, _,_,_=VanCittert(spectra,Obs,Target)
    mag = Target.Star1['SImag']
    # Luminance according to apparent magnitude
    Irradiance = np.ones_like(spectra)
    L0_ph = 702e8        # Photons.m-2.s-1.µm-1 at 0.7µm
    Lph_H = L0_ph*10**(-0.4*mag)
    
    if MultiWavelength:
        delta_wav = np.abs(spectra[0]-spectra[1])
    else:
        delta_wav = MeanWavelength/R
        
    UncohIrradiance = Irradiance*Lph_H*delta_wav             # [phot/m²/deltalmbda/s] Source Irradiance 

    Throughput = InterfArray.TelSurfaces*InterfArray.TelTransmissions
    
    NA=len(Throughput)
    NB=NA**2 ; NIN = int(NA*(NA-1)/2)
    
    CoherentFluxObject = np.zeros([MW,NB])
    for ia in range(NA):
        CoherentFluxObject[:,ia] = Throughput[ia]*UncohIrradiance
    
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib = posk(ia,iap,NA)
            CoherentFluxObject[:,NA+ib] = np.sqrt(CoherentFluxObject[:,ia]*CoherentFluxObject[:,iap])*np.real(VisObject[:,ib])
            CoherentFluxObject[:,NA+NIN+ib] = np.sqrt(CoherentFluxObject[:,ia]*CoherentFluxObject[:,iap])*np.imag(VisObject[:,ib])

    return CoherentFluxObject

def get_ObsInformation(ObservationFile,verbose=False):
    
    from .config import ScienceObject, Observation
    from astropy.io import fits
    
    if not os.path.exists(ObservationFile):
        try:
            if verbose:
                print("Looking for the observation file into the package's data")
            ObservationFile = pkg_resources.resource_stream(__name__, ObservationFile)
        except:
            raise Exception(f"{ObservationFile} doesn't exist.")
            
    hdul = fits.open(ObservationFile)
    hdr=hdul['PRIMARY'].header
    
    Target = ScienceObject()
    Obs = Observation()
    
    Objects = list(dict.fromkeys([x[4] for x in hdr.keys() if 'Star' in x]))
    for No in Objects:
        currentobject = f"Star{No}"
        attributes = [x.split('_')[1] for x in hdr.keys() if currentobject in x]
        StarCharacteristics={}
        for attr in attributes:
            if attr=='alpha':
                if 'Position' not in StarCharacteristics.keys():
                    StarCharacteristics['Position']=[0,0]
                StarCharacteristics['Position'][0]=hdr[currentobject+f"_{attr}"]
            elif attr=='beta':
                if 'Position' not in StarCharacteristics.keys():
                    StarCharacteristics['Position']=(0,0)
                StarCharacteristics['Position'][1]=hdr[currentobject+f"_{attr}"]
            else:
                StarCharacteristics[attr]=hdr[currentobject+f"_{attr}"]
                #setattr(dico,attr,hdr[currentobject+f"_{attr}"])
        setattr(Target, f"Star{No}",StarCharacteristics)
    
    Target.Name = hdr['TARGET']
    Obs.ArrayName = hdr['ARRAY']
    Obs.Filepath = hdr['Filepath']
    for key,val in hdr.items():
        if key not in ['TARGET','ARRAY']:
            setattr(Obs,key,val)   
        
    return Obs, Target


def get_CfObj(filepath, spectra,verbose=False):
    """
    Reads data of an observation contained in a FITSfile.
    Adapt the spectral sampling to the FS spectral sampling.
    The coherent and uncoherent flux are given in Photons/µm/second at 
    the entrance of the fringe-sensor
    Return the interpolated coherent flux.

    Parameters
    ----------
    filepath : STRING
        Filepath of the file that contains target information.
    spectra : LIST or ARRAY
        Spectral sampling of the output data (for interpolation).

    Returns
    -------
    FinalCoherentIrradiance, FinalComplexVisObj, ClosurePhase
    
    FinalCoherentIrradiance : ARRAY[NW,NB]
        Coherence Flux of the object in photons/s.
        
    FinalComplexVisObj:
        Complex degree of mutual coherence. (between 0 and 1)
        
    ClosurePhase : ARRAY[NW,NC]
        Closure phases of the object in radian.
    """

    fileexists = os.path.exists(filepath)
    if not fileexists:
        try:
            if verbose:
                print("Looking for the observation file into the package's data")
            filepath = pkg_resources.resource_stream(__name__,filepath)
        except:
            raise Exception(f"{filepath} doesn't exists.")          
            
    with fits.open(filepath) as hdu:

        ObsParams = hdu[0].header
        WLsampling = hdu['SPECTRA'].data['WLsampling']
        
        realV = hdu['VReal'].data
        imagV = hdu['VImag'].data
        
        realCf = hdu['CfReal'].data
        imagCf = hdu['CfImag'].data


    f = interpolate.interp1d(WLsampling, realCf, axis=0)
    NewRealCf = f(spectra)
    f = interpolate.interp1d(WLsampling, imagCf, axis=0)
    NewImagCf = f(spectra)
    NewImagCf[np.abs(NewImagCf)<np.abs(NewRealCf)*1e-6]=0
    CoherentIrradiance = NewRealCf + NewImagCf*1j
    
    f = interpolate.interp1d(WLsampling, realV, axis=0)
    NewRealV = f(spectra)
    f = interpolate.interp1d(WLsampling, imagV, axis=0)
    NewImagV = f(spectra)
    NewImagV[np.abs(NewImagV)<np.abs(NewRealV)*1e-6]=0
    ComplexVisObj = NewRealV + NewImagV*1j
    
    if isinstance(spectra,float):
        NW=1
        NBfile=len(CoherentIrradiance)
    else:
        NW, NBfile = CoherentIrradiance.shape
    NAfile = int(np.sqrt(NBfile))

    from .config import NA, NB, NC
    
    if NW!=1:
        ClosurePhase = np.zeros([NW,NC])
        FinalCoherentIrradiance = np.zeros([NW,NB])*1j
        FinalComplexVisObj = np.zeros([NW,NB])*1j

        for ia in range(NA):
            for iap in range(NA):
                ib = ia*NA+iap                    
                FinalCoherentIrradiance[:,ib] = CoherentIrradiance[:,ia*NAfile+iap]
                FinalComplexVisObj[:,ib] = ComplexVisObj[:,ia*NAfile+iap]
                
    else:
        ClosurePhase = np.zeros([NC])
        FinalCoherentIrradiance = np.zeros([NB])*1j
        FinalComplexVisObj = np.zeros([NB])*1j
        for ia in range(NA):
            for iap in range(NA):
                ib = ia*NA+iap
                FinalCoherentIrradiance[ib] = CoherentIrradiance[ia*NAfile+iap]
                FinalComplexVisObj[ib] = ComplexVisObj[ia*NAfile+iap]
                
                        
    if NA < 3:
        return FinalCoherentIrradiance, FinalComplexVisObj
    
    else:
        if NW!=1:
            for ia in range(NA):
                for iap in range(ia+1,NA):
                    for iapp in range(iap+1,NA):
                        ic = poskfai(ia,iap,iapp,NA)
                        ci1 = CoherentIrradiance[:,ia*NAfile+iap]
                        ci2 = CoherentIrradiance[:,iap*NAfile+iapp]
                        ci3 = CoherentIrradiance[:,iapp*NAfile+ia]
                        ClosurePhase[:,ic] = np.angle(ci1*ci2*ci3)
        else:
            for ia in range(NA):
                for iap in range(ia+1,NA):
                    for iapp in range(iap+1,NA):
                        ic = poskfai(ia,iap,iapp,NA)
                        ci1 = CoherentIrradiance[ia*NAfile+iap]
                        ci2 = CoherentIrradiance[iap*NAfile+iapp]
                        ci3 = CoherentIrradiance[iapp*NAfile+ia]
                        ClosurePhase[ic] = np.angle(ci1*ci2*ci3)
        
    return FinalCoherentIrradiance, FinalComplexVisObj, ClosurePhase


def get_infos(file,verbose=False):
    
    if not os.path.exists(file):
        try:
            if verbose:
                print("Looking for the disturbance file into the package's data")
            file = pkg_resources.resource_stream(__name__,file)
        except:
            raise Exception(f"{file} doesn't exist.")
    
    with fits.open(file) as hdu:
        hdr = hdu[0].header
        dt = hdr['DT']
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
            
        except:
            FreqSampling = np.zeros(NT); PSD = np.zeros(NT); Filter = np.zeros(NT)

    return filetimestamps,filelmbdas, piston, transmission, FreqSampling, PSD, Filter,hdr
    
    
def get_CfDisturbance(DisturbanceFile, spectra, timestamps,foreground=[], verbose=False):
    
    from .config import piston_average,NA,NB
    NT = len(timestamps) ; NW = len(spectra)
    
    if 'fits' not in DisturbanceFile:
        if DisturbanceFile == 'NoDisturbance':
            PistonDisturbance = np.zeros([NT,NA])
            TransmissionDisturbance = np.ones([NT,NW,NA])
            
        elif DisturbanceFile == 'Foreground':
            Lc = config.FS['R']*config.wlOfTrack
            PistonDisturbance = np.zeros([NT,NA])
            for ia in range(NA):  # décoherencing of all telescopes
                PistonDisturbance[:,ia]=(ia-2)*2*Lc
            TransmissionDisturbance = np.ones([NT,NW,NA])
            
        elif DisturbanceFile == 'CophasedThenForeground':
            Lc = config.FS['R']*config.wlOfTrack
            PistonDisturbance = np.zeros([NT,NA])
            for ia in range(NA):  # décoherencing of all telescopes from it=100.
                PistonDisturbance[100:,ia]=(ia-2)*2*Lc
            TransmissionDisturbance = np.ones([NT,NW,NA])
    
        else:
            raise Exception("DisturbanceFile doesn't correspond to any valid case.")
    
    else:
        filetimestamps, filespectra, PistonDisturbance, TransmissionDisturbance,_,_,_,_ = get_infos(DisturbanceFile)
    
        PistonDisturbance = PistonDisturbance[:,:NA]
        TransmissionDisturbance = TransmissionDisturbance[:,:,:NA]
    
        # Interpolate on the time axis   
        newobservables = []
        for observable in [PistonDisturbance, TransmissionDisturbance]:
            f = interpolate.interp1d(filetimestamps,observable, axis=0, bounds_error=False, fill_value=(observable[0],observable[-1]))
            newobservables.append(f(timestamps))
            
        
        PistonDisturbance, TempTransmissionDisturbance = newobservables
        
        # Interpolate transmission on the spectral axis (Piston is not chromatic)    
        f = interpolate.interp1d(filespectra*1e6,TempTransmissionDisturbance, axis=1, bounds_error=False,fill_value=(TempTransmissionDisturbance[:,0,:],TempTransmissionDisturbance[:,-1,:]))
        TransmissionDisturbance = np.abs(f(spectra))

    if piston_average==1:
        if verbose:
            print("We subtract to the piston of each telescope its first value")
        PistonDisturbance = PistonDisturbance-PistonDisturbance[0]
    if piston_average==2:
        if verbose:
            print("We subtract the average of first piston to the piston of all telescopes.")
        PistonDisturbance = PistonDisturbance-np.mean(PistonDisturbance[0])
    elif piston_average==3:
        if verbose:
            print("We subtract to the piston of each telescope its temporal average.")
        PistonDisturbance = PistonDisturbance-np.mean(PistonDisturbance, axis=0)
        
    if len(foreground):     # Add a huge piston to be sure being far from fringes
        PistonDisturbance -= np.mean(PistonDisturbance,axis=0)  # Subtract to the piston of each telescope its temporal average.
        PistonDisturbance = PistonDisturbance + np.array(foreground)
    
    CfDisturbance = np.zeros([NT,NW,NB])*1j
    from cophasim import skeleton
    for it in range(NT):
        CfDisturbance[it,:,:] = skeleton.coh__pis2coh(PistonDisturbance[it,:], 1/spectra, ampl=np.sqrt(TransmissionDisturbance[it,:]))


    return CfDisturbance, PistonDisturbance, TransmissionDisturbance
    

def get_subarray(pup_used, dsp_all, TelNames, base):
    """
    From the visibility matrix of the whole array (dependant on the object),
    returns the visibility matrix of the array made of the given pupils.

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
            sigma_temp = np.linspace(sigbottom, sigtop, OW+1)[1:]
            sigma_os = np.concatenate((sigma_os,sigma_temp))
        spectra_os = np.sort(1/sigma_os)
        
    elif mode == 'linear_wl':
        spectra_os = np.array([])
        spectra = np.linspace(lmbda1, lmbda2, MW)
        deltalmbda = spectra[1] - spectra[0]
        for i in range(MW):
            wlbottom = spectra[i]-deltalmbda/2
            wltop = spectra[i]+deltalmbda/2
            spectra_temp = np.linspace(wlbottom, wltop, OW+1)[1:]
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
                kp = posk(ia,iap,NA)

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
                kp = posk(ia,iap,NA)

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
                kp = posk(ia,iap,NA)
                
                # Direct and Conjugate coherence vectors
                direct = simumatrix[:,:,ia*NA+iap]
                # conj = simumatrix[:,:,iap*NA+ia]
                
                # Real and Imaginary parts of the coherence vectors
                k = int(NA + kp)                    # Real part location
                GRAVmatrix[:,:,k] = np.real(direct)
                k = int(NA*(NA+1)/2 + kp)           # Imaginary part location
                GRAVmatrix[:,:,k] = np.imag(direct)
        
    else:        # Ndim == 2
        NP,NB = shape
        GRAVmatrix = np.zeros([NP,NB])
        NA = int(np.sqrt(NB))
        for ia in range(NA):
            ksim = ia*(NA+1)
            GRAVmatrix[:,ia] = np.real(simumatrix[:,ksim])
            for iap in range(ia+1,NA):
                kp = posk(ia,iap,NA)
                
                # Direct and Conjugate coherence vectors
                direct = simumatrix[:,ia*NA+iap]
                # conj = simumatrix[:,iap*NA+ia]
                
                # Real and Imaginary parts of the coherence vectors
                k = int(NA + kp)                    # Real part location
                GRAVmatrix[:,k] = np.real(direct)
                k = int(NA*(NA+1)/2 + kp)           # Imaginary part location
                GRAVmatrix[:,k] = np.imag(direct)
            
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
    
    if np.ndim(ich) == 1:
        ichtemp = ich
        ich = np.zeros([len(ich), 2])
        for ib in range(len(ichtemp)):
            ich[ib] = [char for char in str(ichtemp[ib])]
    
    for ib in range(NIN):
        # Sort the positions in the conventional order
        detectorpositions = range(ib*Nmod,(ib+1)*Nmod)
        ia, iap = int(ich[ib][0])-1, int(ich[ib][1])-1
        ibn = posk(ia,iap,NA)
        conventionalpositions = range(ibn*Nmod,(ibn+1)*Nmod)
        
        if direction=='v2pm':
            baseline_sorted[:,conventionalpositions,:] = matrix[:,detectorpositions,:]
        else:
            baseline_sorted[:,:,conventionalpositions] = matrix[:,:,detectorpositions]
        
    matrix_sorted = np.zeros_like(matrix)
    
    # Sort the ABCD modulations
    for ib in range(NIN):
        for k in range(Nmod):
            if direction=='v2pm':
                matrix_sorted[:,ib*Nmod+k,:] = baseline_sorted[:,ib*Nmod+ABCDindex[k],:]
            else:
                matrix_sorted[:,:,ib*Nmod+k] = baseline_sorted[:,:,ib*Nmod+ABCDindex[k]]

    return matrix_sorted

def studyP2VM(*args,savedir='',ext='pdf',nfig=0):
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
    ich=config.FS['ich']
    
    if 'NMod' in config.FS.keys():
        pairwise=True
        Modulation = config.FS['Modulation']
        ABCDindex = config.FS['ABCDind']
        Nmod=config.FS['NMod']
        v2pm_sorted = sortmatrix(v2pm, ich, ABCDindex)
        p2vm_sorted = sortmatrix(p2vm, ich, ABCDindex, direction='p2vm')
    else:
        pairwise=False
        v2pm_sorted = np.copy(v2pm)
        p2vm_sorted = np.copy(p2vm)
    
    conventionalorder=[]
    for ia in range(1,NA+1):
        for iap in range(ia+1,NA+1):
            conventionalorder.append(f"{ia}{iap}")
    
    
    """Photometries and phaseshifts"""
    if pairwise:
        photometries = np.zeros([NW,NIN,Nmod,2])
        phases = np.zeros([NW,NIN,Nmod]) ; normphasors = np.zeros([NW,NIN,Nmod])
        for ia in range(NA):
            for iap in range(ia+1,NA):
                ib=posk(ia,iap,NA)
                for k in range(Nmod):
                    photometries[:,ib,k,0] = v2pm_sorted[:,Nmod*ib+k,ia]*100
                    photometries[:,ib,k,1] = v2pm_sorted[:,Nmod*ib+k,iap]*100
                    R = v2pm_sorted[:,Nmod*ib+k,NA+ib] ; I = v2pm_sorted[:,Nmod*ib+k,NA+NIN+ib] 
                    phases[:,ib,k] = np.arctan2(I,R) ; normphasors[:,ib,k] = np.sqrt(R**2+I**2)
        
        ModuleCoherentFlux = normphasors/2
    
    
    """ Noise propagation"""
    photons = 1e3
    # Coherent vector of a perfectly coherent pair of fields.
    CoherentFlux = np.zeros([NW,NB])
    CoherentFlux[:,:NA+NIN]=photons # In Cassaing formalism: N=photons ; C=photons ; S=0
    
    DetectionCoherent = np.zeros([NW,NP])
    for iw in range(NW):
        DetectionCoherent[iw] = np.dot(v2pm_sorted[iw], CoherentFlux[iw])
    
    QuadratureCoherentFlux = np.zeros([NW,NB])
    QuadratureCoherentFlux[:,:NA] = photons
    QuadratureCoherentFlux[:,NA+NIN:] = photons # In Cassaing formalism: N=photons ; C=0 ; S=photons
    
    DetectionQuadrature = np.zeros([NW,NP])
    for iw in range(NW):
        DetectionQuadrature[iw] = np.dot(v2pm_sorted[iw], QuadratureCoherentFlux[iw])
    
    # Covariance matrix
    CovMatrixCoherent = np.zeros([NW,NB,NB])
    CovMatrixQuadrature = np.zeros([NW,NB,NB])
    
    for iw in range(NW):
        CovMatrixCoherent[iw] = np.dot(p2vm_sorted[iw], np.dot(np.diag(DetectionCoherent[iw]),np.transpose(p2vm_sorted[iw])))
        CovMatrixQuadrature[iw] = np.dot(p2vm_sorted[iw], np.dot(np.diag(DetectionQuadrature[iw]),np.transpose(p2vm_sorted[iw])))
            
    if pairwise:
        dico = {'v2pm_sorted':v2pm_sorted, 
                'p2vm_sorted':p2vm_sorted, 
                'photometries':photometries, 
                'phases':phases,
                'CovMatrixCoherent':CovMatrixCoherent, 
                'CovMatrixQuadrature':CovMatrixQuadrature}
    else:
        dico = {'v2pm_sorted':v2pm_sorted, 
                'p2vm_sorted':p2vm_sorted,
                'CovMatrixCoherent':CovMatrixCoherent, 
                'CovMatrixQuadrature':CovMatrixQuadrature}
        
    
    if ('p2vm' in args) or ('displayall' in args):
        mod = v2pm_sorted[0]
        
        newfig=nfig
        fig = plt.figure('V2PM & P2VM',clear=True)
        fig.suptitle("V2PM and P2VM")
        
        ax1,ax2 = fig.subplots(ncols=2)
        ax1.set_title("V2PM")
        ax2.set_title("P2VM")
        
        """Show the V2PM matrix"""
        
        ax1.imshow(mod)
        
        xticks=[] ; xticklabels=[]
        xticks.append(NA//2-1) ; xticklabels.append('Photometries')
        x = NA-0.5 ; xticks.append(NA+NIN//2-1) ; xticklabels.append('Real parts')
        ax1.axvline(x = x, color = 'w', linestyle = '-')
        x = NA+NIN-0.5 ; xticks.append(NA+3*NIN//2-1) ; xticklabels.append('Imaginary parts')
        ax1.axvline(x = NA+NIN-0.5, color = 'w', linestyle = '-')
    
        for ia in range(1,NA-1):
            ib = posk(ia,ia+1,NA)
            ax1.axvline(x = x, color = 'w', linestyle = '-', linewidth=.5)
            ax1.axvline(x = x, color = 'w', linestyle = '-', linewidth=.5)
        
        if pairwise:
            for ib in range(NIN):
                ax1.axhline(y = ib*Nmod-0.5, color = 'w', linestyle = '-')
            
            ax1.set_yticks(np.arange(NIN)*Nmod+Nmod//2-.5)
            ax1.set_yticklabels(conventionalorder)
            ax1.set_ylabel('Pixels sorted in ABCD')
            
            
        else:
            ax1.axhline(y = NA, color = 'w', linestyle = '-', linewidth=.5)
            
            ax1.set_yticks([NA//2, (NA+NP)//2])
            ax1.set_yticklabels(['Photometries', 'Interferogram'])
            
        
        ax1.set_xlabel("Coherent flux")
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticklabels, rotation=45)
        ax1.xaxis.set_ticks_position('none')
    
    
        """ Show the P2VM """
    
        demod = p2vm_sorted[0]
        
        ax2.imshow(demod)
        
        # Create a Rectangle patch
        # rect = Rectangle((-0.5,11.5),1,1,linewidth=1,edgecolor='r',facecolor='none')
        yticks=[] ; yticklabels=[]
        yticks=[] ; yticklabels=[]
        yticks.append(NA//2) ; yticklabels.append('Photometries')
        y = NA-0.5 ; yticks.append(y) ; yticklabels.append('Real parts')
        ax2.axhline(y = y, color = 'w', linestyle = '-')
        y = NA+0.5 ; yticks.append(y) ; yticklabels.append('B1#')
        y = NA+NIN-0.5 ; yticks.append(y) ; yticklabels.append('Imaginary parts')
        ax2.axhline(y = NA+NIN-0.5, color = 'w', linestyle = '-')
        y = NA+NIN+0.5 ; yticks.append(y) ; yticklabels.append('B1#')
        for ia in range(1,NA-1):
            ib = posk(ia,ia+1,NA)
            y = NA+ib-0.5 ; yticks.append(y+0.5) ; yticklabels.append(f'B{ia+1}#')
            ax2.axhline(y = y, color = 'w', linestyle = '-', linewidth=.5)
            y = NA+NIN+ib-0.5 ; yticks.append(y+0.5) ; yticklabels.append(f'B{ia+1}#')
            ax2.axhline(y = y, color = 'w', linestyle = '-', linewidth=.5)
        
        
        if pairwise:
            for ib in range(NIN):
                ax2.axvline(x = ib*Nmod-0.5, color = 'w', linestyle = '-', linewidth=.5)
                
            ax2.set_xticks(np.arange(NIN)*Nmod+Nmod//2-.5)
            ax2.set_xticklabels(conventionalorder,rotation=45)
            ax2.set_xlabel('Pixels sorted in ABCD')
            
        else:
            ax2.axvline(x = NA, color = 'w', linestyle = '-', linewidth=.5)
            ax2.set_xticks([NA,(NA+NP)//2])
            ax2.set_xticklabels(['Photometries', 'Interferogram'], rotation=45)
        
        ax2.yaxis.tick_right()
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(yticklabels)
        ax1.yaxis.set_ticks_position('none')
        ax2.yaxis.set_ticks_position('none')
        ax1.xaxis.set_ticks_position('none')
        ax2.xaxis.set_ticks_position('none')

        ax1.grid(False)
        ax2.grid(False)
    
# =============================================================================
#     Photometry and phaseshifts repartition in the P2VM
# =============================================================================
    
    # colors = tol_cset('bright')

    # newfig+=1
    # fig=plt.figure(newfig, clear=True)
    # fig.suptitle("Photometry repartitions and phaseshift of the beams")
    # ax1,ax2,ax3,ax4 = fig.subplots(nrows=4,gridspec_kw={'height_ratios': [.2,1,.2,1]})
    
    # photometries = np.zeros([NW,NIN,Nmod,2])
    # phases = np.zeros([NW,NIN,Nmod])
    # normphasors = np.zeros([NW,NIN,Nmod])
    # for ia in range(NA):
    #     for iap in range(ia+1,NA):
    #         ib=posk(ia,iap,NA)
    #         for k in range(Nmod):
    #             photometries[:,ib,k,0] = v2pm_sorted[:,4*ib+k,ia]*100
    #             photometries[:,ib,k,1] = v2pm_sorted[:,4*ib+k,iap]*100
    #             R = v2pm_sorted[:,4*ib+k,NA+ib] ; I = v2pm_sorted[:,4*ib+k,NA+NIN+ib] 
    #             phases[:,ib,k] = np.arctan2(I,R) ; normphasors = np.sqrt(R**2+I**2)
                
    # NINtemp = NIN//2+1
    # xtop = np.arange(NINtemp)  # the label locations
    # xbot = np.arange(NINtemp,NIN)
    # width=0.8
    # barwidth=width/8
    # bar_patches1=[] ; bar_patches2=[]
    # bar_patches1.append(mpatches.Patch(facecolor='gray',edgecolor='black',hatch='///',label="First beam"))
    # bar_patches1.append(mpatches.Patch(facecolor='gray',edgecolor='black', label="Second beam"))
    # for k in range(Nmod):
        
    #     firstbar_pos = xtop-width/2+barwidth/2
    #     rects1 = ax2.bar(firstbar_pos + 2*k*barwidth, photometries[0,:NINtemp,k,0], 
    #                      barwidth, hatch='///',color=colors[k],edgecolor='black').patches
    #     rects2 = ax2.bar(firstbar_pos + (2*k+1)*barwidth, photometries[0,:NINtemp,k,1], 
    #                      barwidth, color=colors[k],edgecolor='black').patches
    #     rects_moy1 = ax2.bar(firstbar_pos + (2*k+1/2)*barwidth, np.mean(photometries[0,:NINtemp,k,:],axis=-1), 
    #                      barwidth*2, color=colors[k],edgecolor='black',fill=False,linestyle='--').patches
    #     rects3 = ax1.bar(firstbar_pos + 2*k*barwidth+barwidth/2, phases[0,:NINtemp,k], 
    #                          barwidth,color=colors[k],edgecolor='black')
    #     firstbar_pos = xbot-width/2+barwidth/2
    #     rects4 = ax4.bar(firstbar_pos + 2*k*barwidth, photometries[0,NINtemp:,k,0],
    #                      barwidth, hatch='///',color=colors[k],edgecolor='black').patches
    #     rects5 = ax4.bar(firstbar_pos + (2*k+1)*barwidth, photometries[0,NINtemp:,k,1],
    #                      barwidth,color=colors[k],edgecolor='black').patches
    #     rects_moy2 = ax4.bar(firstbar_pos + (2*k+1/2)*barwidth, np.mean(photometries[0,NINtemp:,k,:],axis=-1), 
    #                      barwidth*2, color=colors[k],edgecolor='black',fill=False,linestyle='--').patches
    #     rects6 = ax3.bar(firstbar_pos + 2*k*barwidth+barwidth/2, phases[0,NINtemp:,k], 
    #                          barwidth,color=colors[k],edgecolor='black')
        
    #     # rects1 = ax2.patches[:NINtemp] ; rects2 = ax2.patches[NINtemp:2*NINtemp]
    #     # rects_moy = ax2.patches[2*NINtemp:]
        
    #     for rect1,rect2,rect_moy in zip(rects1,rects2,rects_moy1):
    #         height = np.max([rect1.get_height(),rect2.get_height(),rect_moy.get_height()])
    #         ax2.text(rect_moy.get_x() + rect_moy.get_width() / 2, height+0.1, round(height,1),
    #                 ha='center', va='bottom')
            
    #     for rect1,rect2,rect_moy in zip(rects4,rects5,rects_moy2):
    #         height = np.max([rect1.get_height(),rect2.get_height(),rect_moy.get_height()])
    #         ax4.text(rect_moy.get_x() + rect_moy.get_width() / 2, height+0.1, round(height,1),
    #                 ha='center', va='bottom')
    #         # ax4.text(rect.get_x() + rect.get_width() / 2, height + 5, height,
    #         #         ha='center', va='bottom')
            
    #     bar_patches2.append(mpatches.Patch(color=colors[k],label="ABCD"[k]))
    #     # rects3 = ax1.bar(x - width/2+0.25, photometries[0,:NINtemp,0,0], barwidth, label='C',color='b')
    #     # rects4 = ax1.bar(x - width/2+0.35, photometries[0,:NINtemp,0,1], barwidth, label='D',color='g')
    
    # # ax1.bar_label(rects1)
    # # ax1.bar_label(rects2)
    # # xlim = (xtop[0]-width , xtop[-1]+width)
        
    # # ax3.plot(npxtop,0.12*np.ones_like(xtop), color='black', label="Phase")
    # # ax3.hlines(0.12, xlim[0],xlim[1], color='black', label="Phase")
    # # ax3.hlines(0.08, xlim[0],xlim[1], color='black', label="Phase")
    # # ax3.hlines(0.16, xlim[0],xlim[1], color='black', label="Phase")
    # # ax3.plot(phases[0,NINtemp:,])
    
    
    # xticklabels=[]
    # for ia in range(NA):
    #     for iap in range(ia+1,NA):
    #         xticklabels.append(f"B{ia+1}{iap+1}")
    
    # ax2.set_xticks(xtop)
    # ax2.set_xticklabels(xticklabels[:NINtemp])
    
    # ax2.set_ylim(0,10)
    # ax4.set_ylim(0,10)
    
    # ax1.set_ylim(-np.pi,np.pi)
    # ax1.grid()

    # ax3.set_ylim(-np.pi,np.pi)
    # ax3.grid()
    # ax4.set_xticks(xbot)
    # ax4.set_xticklabels(xticklabels[NINtemp:])
    
    # first_legend = ax4.legend(handles=bar_patches2, loc='lower right')
    # ax4.add_artist(first_legend)
    # ax4.legend(handles=bar_patches1, loc='upper right')
    
    
    # ax1.set_ylabel("Phase [rad]")
    # ax2.set_ylabel("Transmission")
    
    # # ax4.bar_label(rects4, label_type='center')
    
    # fig.show()
    
    if ('repartition' in args) or ('displayall' in args):
        
        #plt.rcParams.update(rcParamsForBaselines)
        SS = 12     # Small size
        MS = 14     # Medium size
        BS = 16     # Big size
        figsize = (16,12)
        rcParamsForRepartitions = {"font.size":SS,
                "axes.titlesize":SS,
                "axes.labelsize":SS,
                "axes.grid":True,
               
                "xtick.labelsize":SS,
                "ytick.labelsize":SS,
                "legend.fontsize":SS,
                "figure.titlesize":BS,
                # "figure.constrained_layout.use": False,
                "figure.dpi":300,
                "figure.figsize":figsize
                # 'figure.subplot.hspace': 0.05,
                # 'figure.subplot.wspace': 0,
                # 'figure.subplot.left':0.1,
                # 'figure.subplot.right':0.95
                }

        plt.rcParams.update(rcParamsForRepartitions)
        
        if pairwise:
            newfig+=1
            fig=plt.figure(newfig, clear=True)
            fig.suptitle("Photometry repartitions and phaseshift of the beams")
            (ax1,ax3),(ax2,ax4) = fig.subplots(nrows=2, ncols=2,gridspec_kw={'width_ratios': [9,1]})
                        
            NINtemp = NIN//2+1
            xtop = np.arange(NINtemp)  # the label locations
            xbot = np.arange(NINtemp,NIN)
            
            width=0.8
            barwidth=width/8
            
            ax1_firstbar_positions = xtop-width/2+barwidth/2
            ax2_firstbar_positions = xbot-width/2+barwidth/2
            
            bar_patches1=[] ; bar_patches2=[]
            bar_patches1.append(mpatches.Patch(facecolor='gray',edgecolor='black',hatch='///',label="First beam"))
            bar_patches1.append(mpatches.Patch(facecolor='gray',edgecolor='black', label="Second beam"))
            
            linestyles=[]
            linestyles.append(mlines.Line2D([], [], color='black',
                                            linestyle='--',label='Average flux'))    
            linestyles.append(mlines.Line2D([], [], color='black',
                                            linestyle=':',label='Coherent flux'))
                
            for k in range(Nmod):
                
                rects1 = ax1.bar(ax1_firstbar_positions + 2*k*barwidth, photometries[0,:NINtemp,k,0], 
                                 barwidth, hatch='///',color=colors[k],edgecolor='black').patches
                rects2 = ax1.bar(ax1_firstbar_positions + (2*k+1)*barwidth, photometries[0,:NINtemp,k,1], 
                                 barwidth, color=colors[k],edgecolor='black').patches
                rects_moy1 = ax1.bar(ax1_firstbar_positions + (2*k+1/2)*barwidth, np.mean(photometries[0,:NINtemp,k,:],axis=-1), 
                                 barwidth*2, color=colors[k],edgecolor='black',fill=False,linestyle='--').patches
                rects_coh1 = ax1.bar(ax1_firstbar_positions + (2*k+1/2)*barwidth, ModuleCoherentFlux[0,:NINtemp,k]*100, 
                                 barwidth*2, color=colors[k],edgecolor='black',fill=False,linestyle=':').patches
                
                rects4 = ax2.bar(ax2_firstbar_positions + 2*k*barwidth, photometries[0,NINtemp:,k,0],
                                 barwidth, hatch='///',color=colors[k],edgecolor='black').patches
                rects5 = ax2.bar(ax2_firstbar_positions + (2*k+1)*barwidth, photometries[0,NINtemp:,k,1],
                                 barwidth,color=colors[k],edgecolor='black').patches
                rects_moy2 = ax2.bar(ax2_firstbar_positions + (2*k+1/2)*barwidth, np.mean(photometries[0,NINtemp:,k,:],axis=-1), 
                                 barwidth*2, color=colors[k],edgecolor='black',fill=False,linestyle='--').patches
                rects_coh2 = ax2.bar(ax2_firstbar_positions + (2*k+1/2)*barwidth, ModuleCoherentFlux[0,NINtemp:,k]*100, 
                                 barwidth*2, color=colors[k],edgecolor='black',fill=False,linestyle=':').patches
                
                for rect1,rect2,rect_moy in zip(rects1,rects2,rects_moy1):
                    height = np.max([rect1.get_height(),rect2.get_height(),rect_moy.get_height()])
                    # ax1.text(rect_moy.get_x() + rect_moy.get_width() / 2, height+0.1, round(height,1),
                    #         ha='center', va='bottom')
                    
                for rect1,rect2,rect_moy in zip(rects4,rects5,rects_moy2):
                    height = np.max([rect1.get_height(),rect2.get_height(),rect_moy.get_height()])
                    # ax2.text(rect_moy.get_x() + rect_moy.get_width() / 2, height+0.1, round(height,1),
                    #         ha='center', va='bottom')
                    
                bar_patches2.append(mpatches.Patch(color=colors[k],label="ABCD"[k]))
            
            ax1_xmin,ax1_xmax = ax1.get_xlim() ; ax2_xmin,ax2_xmax = ax2.get_xlim()
            ax1_ymin,ax1_ymax = ax1.get_ylim() ; ax2_ymin,ax2_ymax = ax2.get_ylim()
            ax1_normalised_width = 1/(ax1_ymax-ax1_ymin)*width*.7
            ax1_normalised_height = 1/(ax1_ymax-ax1_ymin)*width
            ax2_normalised_width = 1/(ax1_ymax-ax1_ymin)*width*.7
            ax2_normalised_height = 1/(ax1_ymax-ax1_ymin)*width
            ax1_bottomleft = 1/(ax1_xmax-ax1_xmin)*(ax1_firstbar_positions+barwidth/2-ax1_xmin)
            ax2_bottomleft = 1/(ax2_xmax-ax2_xmin)*(ax2_firstbar_positions+barwidth/2-ax2_xmin)
            
            for ib in range(NINtemp):
                subpos=(ax1_bottomleft[ib],0.75,ax1_normalised_width,ax1_normalised_height)#firstbar_pos[0] + 2*k*barwidth,
                label = True if ib==0 else False
                ax=add_subplot_axes(ax1,subpos,polar=True,label=label)
                if label:
                    ax.xaxis.label.set_size(2)
                    ax.yaxis.label.set_size(2)
                ax.set_ylim(0,1)
                for k in range(Nmod):
                    phase = phases[0,ib,k] ; norm = normphasors[0,ib,k]/np.max(normphasors[0,ib,:])
                    ax.arrow(phase, 0, 0, norm, width = 0.05,
                             edgecolor=colors[k],facecolor = colors[k], lw = 2, zorder = 5,length_includes_head=True)
                    
                # ax.set_thetagrids(phases[0,ib,:]*180/np.pi,labels=np.round(phases[0,ib,:]*180/np.pi))
            for ib in range(NIN-NINtemp):
                subpos=(ax2_bottomleft[ib],0.75,ax2_normalised_width,ax2_normalised_height)#firstbar_pos[0] + 2*k*barwidth,
                label = False
                ax=add_subplot_axes(ax2,subpos,polar=True,label=label)
                ax.set_ylim(0,1)
                
                for k in range(Nmod):
                    phase = phases[0,NINtemp+ib,k] ; norm = normphasors[0,NINtemp+ib,k]/np.max(normphasors[0,NINtemp+ib,:])
                    ax.arrow(phase, 0, 0, norm, width = 0.05,
                             edgecolor=colors[k],facecolor = colors[k], lw = 2, zorder = 5,length_includes_head=True)
            
            xticklabels=[]
            for ia in range(NA):
                for iap in range(ia+1,NA):
                    xticklabels.append(f"B{ia+1}{iap+1}")
            
            ax1.set_xticks(xtop)
            ax1.set_xticklabels(xticklabels[:NINtemp])
            
            ax1.set_ylim(0,14)
            ax2.set_ylim(0,14)
            
            ax2.set_xticks(xbot)
            ax2.set_xticklabels(xticklabels[NINtemp:])
            
            # Set legend on ax3 and ax4    
            ax3.axis("off"); ax4.axis("off")
            ax3.legend(handles=bar_patches1+bar_patches2+linestyles, loc='upper left')
            
            
            # ax1.set_ylabel("Phase [rad]")
            ax1.set_ylabel("Transmission \n[%]")
            ax2.set_ylabel("Transmission \n[%]")
            ax1.grid(False)
            ax2.grid(False)
            
            if len(savedir):
                fig.savefig(savedir+f"PICrepartition.{ext}")
            
            fig.show()
    
        plt.rcParams.update(plt.rcParamsDefault)
    
    
# =============================================================================
#     Noise propagation through the matrix
# =============================================================================
    
    if ('noise' in args) or ('displayall' in args):

        newfig+=1
        fig = plt.figure('Noise propagation', clear=True)
        fig.suptitle("Covariance matrices in photon noise regime (1000photons)")
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
    
        ax2.set_yticklabels([])
        # ax2.set_yticks(xticks)
        # ax2.set_yticklabels(xticklabels)
        # ax2.set_yticks(yticks)
        # ax2.set_yticklabels(yticklabels)
    
        ax1.set_xticks(xticks) ; ax1.set_xticklabels(xticklabels)
        ax2.set_xticks(xticks) ; ax2.set_xticklabels(xticklabels)
        ax1.grid(False)
        ax2.grid(False)

        
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
    Get a vector (resp.array) of shape NB=NA² (resp. [NW,NB]).
    Returns it on a non-redundant form of length NIN=NA(NA-1)/2 sorted as follow:
        12,..,1NA,23,..,2NA,34,..,3NA,..,(NA-1)(NA)

    Parameters
    ----------
    vector : FLOAT COMPLEX ARRAY [NB] or [NW,NB]
        Vector of general complex coherences.

    Returns
    -------
    ninvec : FLOAT COMPLEX ARRAY [NIN] or [NW,NIN]
        Complex Vector of non-redundant mutual coherences.
    """

    if vector.ndim==2:
        NW,NB = vector.shape
    else:
        NB = len(vector)
        
    NA = int(np.sqrt(NB))
    NIN = int(NA*(NA-1)/2)
    
    if vector.ndim==2:
        ninvec = np.zeros([NW,NIN])*1j
        for ia in range(NA):
            for iap in range(ia+1,NA):
                k = posk(ia,iap,NA)
                ninvec[:,k] = vector[:,ia*NA+iap]
    else:
        ninvec = np.zeros([NIN])*1j
        for ia in range(NA):
            for iap in range(ia+1,NA):
                k = posk(ia,iap,NA)
                ninvec[k] = vector[ia*NA+iap]
    
    return ninvec


def makeA2P(descr, modulator, verbose=False,reducedmatrix=False):
    """Builds an A2P matrix from a high-level description descr of the FTchip.
       descr (NIN,2) gives for each baseline (order 01,02,..,12,13,...) the amplitude ratio for pups 1 & 2"""
    
    
    descr = np.array(descr) # Make sure it is an array so that the indexation works well
    nb_in=len(descr)
    
    NA=int(round((1+np.sqrt(1+8*nb_in))/2)) # inversion analytique de la ligne suivante
    NIN=NA*(NA-1)//2
    if NIN != nb_in:
        if verbose:
            print('Taille descr bizarre, attendu=',NIN)
        
    NQ = np.shape(modulator)[0] ; NP=NIN*NQ
    if NQ==4:
        alphabet = ['A','B','C','D']
    elif NQ==2:
        alphabet = ['A','C']
    
    conventional_ich = []
    for ia in range(NA):
        for iap in range(ia+1,NA):
            for iq in range(NQ):
                conventional_ich.append(((str(ia+1),str(iap+1)),alphabet[iq]))
    
    ich = conventional_ich

    A2Pgen=np.zeros((NIN,NQ,NA),dtype=complex) #(base_out=[ia,iap],ABCD,ia_in)
    active_ich=[-1]*NIN
    ib=0 ; ibact=0
    for ia in range(NA):
        for iap in range(ia+1,NA):
            ib = posk(ia,iap,NA)
            A2Pgen[ib,:,ia ]=modulator[:,0]*descr[ib,0]
            A2Pgen[ib,:,iap]=modulator[:,1]*descr[ib,1]
            if descr[ib,0]*descr[ib,1]:     # If non null, the baseline is measured
                active_ich[ib]=ibact        # Position of the baseline in the NINmes vector
                ibact+=1
            
    A2P=np.reshape(A2Pgen,(NP,NA))
    lightA2P=np.copy(A2P)
    
    if reducedmatrix:
        inc=0
        for ip in range(NP):
            if (A2P[ip,:] == np.zeros(NA)).all():
                lightA2P = np.delete(lightA2P, ip-inc, axis=0) # Remove the line at position ip because it doesn't receive any flux
                del ich[ip-inc]
                inc+=1

    return lightA2P, ich, active_ich


def MakeV2PfromA2P(Amat):
    """
    

    Parameters
    ----------
    Amat : TYPE
        DESCRIPTION.

    Returns
    -------
    Bmat : TYPE
        DESCRIPTION.
    Bmatgrav : TYPE
        DESCRIPTION.
    Bmatgrav_reduced : TYPE
        DESCRIPTION.

    """
    
    NP,NA = Amat.shape
    NB=NA**2
    Bmat = np.zeros([NP,NB])*1j
    for ip in range(NP):
        for ia in range(NA):
            for iap in range(NA):
                k = ia*NA+iap
                Bmat[ip, k] = Amat[ip,ia]*np.transpose(np.conjugate(Amat[ip,iap]))

    FilledColumns=np.where(np.sum(np.abs(Bmat),axis=0)!=0)[0]
    NBmes = len(FilledColumns)  # count the number of null columns
    NINmes = (NBmes - NA)//2
    Bmatgrav_reduced = np.zeros([NP,NBmes])
    Bmatgrav = np.zeros([NP,NB])
    ib_r=0
    for ia in range(NA):
        Bmatgrav_reduced[:,ia] = np.abs(Bmat[:,ia*(NA+1)])
        Bmatgrav[:,ia] = np.abs(Bmat[:,ia*(NA+1)])
        for iap in range(ia+1,NA):
            k = ia*NA+iap; kp = iap*NA+ia
            ib = posk(ia,iap,NA)
            Bmatgrav[:,NA+ib] = np.real(Bmat[:,k])
            Bmatgrav[:,NA+NINmes+ib] = np.imag(Bmat[:,kp])
            if k in FilledColumns:
                Bmatgrav_reduced[:,NA+ib_r] = np.real(Bmat[:,k])
                Bmatgrav_reduced[:,NA+NINmes+ib_r] = np.imag(Bmat[:,kp])
                
                ib_r+=1

    return Bmat, Bmatgrav, Bmatgrav_reduced


def ReducedVector(vec, active_ich, NA,form=''):
    
    NIN = len(active_ich)
    NINmes = np.sum(np.array(active_ich)>=0)
    NBmes = NA+2*NINmes
    
    if form=='NIN':
        newvec = np.zeros([NINmes]) ; k=0
        for ib in range(NIN):
            if active_ich[ib]>=0:
                newvec[k] = vec[ib]
                k+=1
            
    elif form=='NBcomplex':
        newvec = np.zeros([NBmes]) ; ib_r=0
        for ia in range(NA):
            newvec[ia] = np.abs(vec[ia*(NA+1)])
            for iap in range(ia+1,NA):
                k=ia*NA+iap ; kp=iap*NA+ia
                ib=posk(ia,iap,NA)
                if active_ich[ib]>=0:
                    newvec[NA+ib_r] = np.real(vec[k])
                    newvec[NA+NINmes+ib_r] = np.imag(vec[k])
                    ib_r+=1
        
    elif form=='NBreal':
        newvec = np.zeros([NBmes]) ; ib_r=0
        newvec[:NA] = vec[:NA]
        for ib in range(NIN):
            if active_ich[ib]>=0:
                newvec[NA+ib_r] = vec[NA+ib]
                newvec[NA+NINmes+ib_r] = vec[NA+NIN+ib]
                ib_r+=1
                
    else:
        raise Exception("'form' parameter must be NIN, NBcomplex or NBreal")
    
    return newvec
    
    


def check_nrj(A2P):
    """
    Checks if a (NP,NA) A2P matrix is normalized.
    If it is normalised, it means the PIC it accounts for conserves energy.
    """
    
    A2Pmod2=A2P*np.conj(A2P)
    nrjpup=np.real(np.sum(A2Pmod2,axis=0))   # Somme du carré des éléments de chaque ligne --> vecteur de dimension NA
    if (nrjpup > 1+1e-15).any():             # Check si la somme des carrés de tous les éléments est inférieure à 1.
        T = np.sum(nrjpup)/len(nrjpup)    
        print(f'Pb: A2P is not normalized, transmission is {round(T*100)}%')
        print(f"Detail: {nrjpup}")
        
    elif (nrjpup < 1-1e-15).any():
        T = np.sum(nrjpup)/len(nrjpup)
        print(f"The PIC absorbs {round((1-T)*100)}% of energy")
        print(f"Detail: {nrjpup}")
        
    return

def check_semiunitary(A2P):
    """
    Check if the A2P matrix is pseudo-inversible and if it absorbs, or not, energy.
    """
    
    Rank = np.linalg.rank(A2P)
    
    if not Rank == np.shape(A2P)[1]:
        print(f"The matrix is not semi-unitary")
    
    
    # Not sure about that.
    # u,s,vh = np.linalg.svd(A2P)
    
    # if not (s<=1).all():   # Check if all singular values are lower than 1
    #     print(f"The matrix creates energy.")
        
    # elif not (s==1).all(): # Check if all singular values are equal to 1
    #     print(f"The matrix absorbs energy.")
    
    return
        
def check_cp(gd):
    
    NIN=len(gd)
    NA = int(1/2+np.sqrt(1/4+2*NIN))
    NC = int(binom(NA,3))

    cp=np.zeros(NC)
    for ia in range(NA):
        for iap in range(ia+1,NA):
            for iapp in range(iap+1,NA):
                ib1=posk(ia,iap,NA); ib2=posk(iap,iapp,NA);ib3=posk(ia,iapp,NA)
                ic=poskfai(ia,iap,iapp,NA)
                cp[ic]=gd[ib1]+gd[ib2]-gd[ib3]
    return cp


# def check_cp(gd):
    
#     NIN=len(gd) ; 
#     NA = int(1/2+np.sqrt(1/4+2*NIN))
#     # Number of independant closure phases
#     ND = int((NA-1)*(NA-2)/2)

#     cp=np.zeros(ND)
#     for iap in range(1,NA):
#         for iapp in range(iap+1,NA):
#             ib1=posk(0,iap,NA); ib2=posk(iap,iapp,NA);ib3=posk(0,iapp,NA)
#             ic=poskfai(0,iap,iapp,NA)
#             cp[ic]=gd[ib1]+gd[ib2]-gd[ib3]
#     return cp

def unwrapPhase(pd,gd,wl):
    
    k = np.round((gd-pd)/wl)
    unwrappedPhase = pd + k*wl
    
    return unwrappedPhase

def filterUnwrap(pd,wl):
    NT = len(pd)
    filteredpd = np.copy(pd)
    if pd.ndim == 2:
        NT,NIN = pd.shape
    else:
        NIN=0
    
    if NIN:
        for ib in range(NIN):
            for it in range(NT-1):
                if filteredpd[it+1,ib]-filteredpd[it,ib]>2*wl/3:
                    filteredpd[it+1,ib]-=wl
                elif filteredpd[it+1,ib]-filteredpd[it,ib]<-2*wl/3:
                    filteredpd[it+1,ib]+=wl
                    
    else:
        for it in range(NT-1):
            if filteredpd[it+1]-filteredpd[it]>2*wl/3:
                filteredpd[it+1]-=wl
            elif filteredpd[it+1]-filteredpd[it]<-2*wl/3:
                filteredpd[it+1]+=wl
    
    return filteredpd


def reconstructOpenLoop(pd,gd,wl):
    
    unwrappedSignal = unwrapPhase(pd,gd,wl)
    openloop = filterUnwrap(unwrappedSignal,wl)
    
    return openloop
    

def getPsd(signal, timestamps, cumStd=False,mov_average=10,timebonds=(0,-1),fbonds=()):
    
    NT = len(timestamps) ; dt = np.mean(timestamps[1:]-timestamps[:-1])
    
    if timebonds[1]==-1:
        timerange = range(np.argmin(np.abs(timestamps-timebonds[0])),NT)
    else:
        timerange = range(np.argmin(np.abs(timestamps-timebonds[0])),np.argmin(np.abs(timestamps-timebonds[1])))
    
    timestamps = timestamps[timerange] ; signal = signal[timerange]
    NT = len(timestamps)
    
    frequencySampling1 = np.fft.fftfreq(NT, dt)
    if len(fbonds):
        fmin, fmax = fbonds
    else:
        fmin=0
        fmax=np.max(frequencySampling1)
    
    PresentFrequencies = (frequencySampling1 > fmin) \
        & (frequencySampling1 < fmax)
        
    frequencySampling = frequencySampling1[PresentFrequencies]
    
    psd = 2*np.abs(np.fft.fft(signal,norm="forward",axis=0)[PresentFrequencies])**2
    
    if cumStd:
        cumulativeStd = np.sqrt(np.cumsum(psd,axis=0))
        
    if mov_average:
        psdSmoothed = moving_average(psd, mov_average)
        frequencySamplingSmoothed = moving_average(frequencySampling,mov_average)

    
    
    if cumStd:
        return frequencySampling, psd, frequencySamplingSmoothed, psdSmoothed, cumulativeStd

    return frequencySampling, psd, frequencySamplingSmoothed, psdSmoothed


def estimateT0(frequencySampling,atmospherePsd,f1=0.3,f2=5):
    if atmospherePsd.ndim==2:
        NT,NIN = atmospherePsd.shape
    else:
        NIN=0
        
    NominalRegime = (frequencySampling>f1)*(frequencySampling<f2)
    logFrequencySampling = np.log10(frequencySampling)
    
    coefs = np.polynomial.polynomial.polyfit(logFrequencySampling[NominalRegime], np.nan_to_num(np.log10(np.abs(atmospherePsd[NominalRegime]))), 1)
    
    if NIN:
        fitAtmospherePsd= np.zeros([NT,NIN])
        for ib in range(NIN):
            poly1d_fn = np.poly1d(coefs[::-1,ib])
            fitAtmospherePsd[:,ib] = 10**poly1d_fn(logFrequencySampling)
    else:
        poly1d_fn = np.poly1d(coefs[::-1])
        fitAtmospherePsd = 10**poly1d_fn(logFrequencySampling)
        
    index1Hz = np.argmin(np.abs(frequencySampling-1))
    psdAt1Hz = fitAtmospherePsd[index1Hz]
    # See Buscher et al 1995 - I divide psdAt1Hz by 2 for being compatible with
    # the two-sided PSD definition
    estimatedT0 = (psdAt1Hz/2/2.84e-4/0.55**2)**(-3/5)
    
    return estimatedT0, coefs[1]
    

def model(freq,delay,gain):
    z=np.exp(1J*2*np.pi*freq/(2*np.amax(freq)))
    ftr=1/(1+z**(-delay)*gain*z/(z-1))
    return np.abs(ftr)

def modelleak(freq,delay,gain,leak,constant):
    """
    Parameters
    ----------
    freq,delay,gain,leak,constant : float
        parameters that fit the model
    """
    z_i=np.exp(1J*2*np.pi*freq/(2*np.amax(freq)))
    ftr=1/(1+z_i**(-delay)*gain*z_i/(z_i-leak))
    return 20*np.log10(np.abs(ftr))+constant



def moving_average(x, w):
    """
    Simple homogeneous smooth (moving) average along the first axis.

    Parameters
    ----------
    x : ARRAY
        Array of 1D or more dimensions.
    w : INT
        Length of the smoothing window.

    Returns
    -------
    y : ARRAY
        Array of same dimensions as x in dim>1 and dim=dim(x)-w+1 on first axis.

    """
    
    y = np.apply_along_axis(lambda m: np.convolve(m, np.ones(w), 'valid') / w, axis=0, arr=x)
    
    return y


def addtext(ax, text, loc = 'best', fontsize='small',fancybox=True, 
            framealpha=0.7, handlelength=0, handletextpad=0):
    """
    Add a text in a legend box. Enables to use the very useful "loc" parameter 
    of the built-in legend function.

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    text : TYPE
        DESCRIPTION.
    loc : TYPE, optional
        DESCRIPTION. The default is 'best'.
    fontsize : TYPE, optional
        DESCRIPTION. The default is 'small'.
    fancybox : TYPE, optional
        DESCRIPTION. The default is True.
    framealpha : TYPE, optional
        DESCRIPTION. The default is 0.7.
    handlelength : TYPE, optional
        DESCRIPTION. The default is 0.
    handletextpad : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """
    import matplotlib.patches as mpl_patches
    # create a list with two empty handles (or more if needed)
        
    n = 20
    textList = [text[i:i+n] for i in range(0, len(text), n)]
    Nlines = len(textList)
    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", 
                                     lw=0, alpha=0)] * Nlines

    labels = []
    for texttemp in textList:
        labels.append(texttemp)
    
    # create the legend, supressing the blank space of the empty line symbol and the
    # padding between symbol and label by setting handlelenght and handletextpad
    ax.legend(handles, labels, loc=loc, fontsize='small', 
              fancybox=True, framealpha=0.7, 
              handlelength=0, handletextpad=0)


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


def setaxelim(ax, xdata=[],ydata=[],xmargin=0.1,ymargin=0.1, ylim_min=[0,0], xlim_min=[0,0],**kwargs):
    
    if isinstance(xdata,(float,int)):
        xdata=[xdata]
    if isinstance(ydata,(float,int)):
        ydata=[ydata]
        
    if len(xdata):
        
        if not 'xmin' in kwargs.keys():
            xmin = np.min(xdata) - xmargin * np.abs(np.min(xdata))
        else:
            xmin=kwargs['xmin']
        
        xmax = np.max(xdata) + xmargin*np.abs(np.max(xdata))
        # xmin = (1+xmargin)*np.min(xdata) ; xmax = (1+xmargin)*np.max(xdata)
        # ax.set_xlim([xmin,xmax])
        
        xdown_min, xup_min = xlim_min
        if xup_min !=0:
            xmax = np.max([xmax,xup_min])
        if xdown_min !=0:
            xmin = np.min([xmin,xdown_min])
        
        ax.set_xlim([xmin,xmax])
        
    if len(ydata):
        if not 'ymin' in kwargs.keys():
            ymin = np.min(ydata) - ymargin * np.abs(np.min(ydata))
        else:
            ymin=kwargs['ymin']
        
        ymax = np.max(ydata) + ymargin*np.abs(np.max(ydata))
        
        ydown_min, yup_min = ylim_min
        if yup_min !=0:
            ymax = np.max([ymax,yup_min])
        if ydown_min !=0:
            ymin = np.min([ymin,ydown_min])
        
        ax.set_ylim([ymin,ymax])
    
    