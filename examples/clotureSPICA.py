# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:56:58 2020

@author: cpannetier
"""
# Define the workspace as being the coh_pack main workspace
import os
os.chdir('../')

# import coh_lib.coh_lib as cl
import coh_lib as cl
import coh_tools as ct
import coh_fs_spica as cfs

import numpy as np

lmbda1 = 1.3
lmbda2 = 1.75
R = 22
spectra = cl.generatespectra(lmbda1, lmbda2, 10*R)[1:]
NW = len(spectra)

directory = 'C:/Users/cpannetier/Documents/These/FringeTracking/SPICA-FT/V2PM_SPICA/'
filename = 'MIRCX_ABCD_H_PRISM22_V2PM.fits'
fitsfile = directory+filename

newV2PM, P2VM, spectra , spectraM, ich = cfs.true(fitsfile,spectra)   
newV2PM, newP2VM = cl.coh__GRAV2simu(newV2PM)

V2PM, P2VM, ich = cfs.perfect(1)   
sigma = 1/spectra
sigmaM = 1/spectraM

coh = cl.coh_init(NA=6, spectra=spectra)
coher, pis, ampl = cl.coh_make(ampl=1, dist='coherent', pup=1, debug=True)

MW = len(sigma)
NA = 6
coher3 = np.zeros([MW, 36])*1j
pixels = np.zeros([MW,60])
for iw in range(MW):
    pixels[iw] = np.dot(newV2PM[iw],coher[0,iw])
    coher3[iw]=np.dot(newP2VM[0],pixels[0])
    
# pixels = np.dot(newV2PM,coher[0,0])
# coher3=np.dot(newP2VM[0],pixels[0])

# Calcul Phase delay
coh__ = np.reshape(coher3, [MW, NA, NA]) 
phi_sum = coh__[:,:,0]**(1/NA)               
for ia in range(1,NA):
    phi_sum *= (coh__[:,:,ia])**(1./NA)                 # baseline demodulation: a_k = product(C_ik) over i

# Phase-delay calculation
phasor_meanlmbda = np.mean(phi_sum, axis=0)             # [1xNA] Average of phasors on wavelengths
phiPD = np.angle(phasor_meanlmbda)                      # [1xNA] phi contains the sum of pupil phase k phi_k
phiPD_mean = np.mean(phiPD)                             # [float] Average on pupils
# phiPD -= phiPD_mean                                     # [1xNA] with the mean phase phi_mean --> p_k = phi_k + phi_mean
# import pdb; pdb.set_trace()
    
pisPD = phiPD/(2*np.pi/coh['PDspectra'])         
PD = pisPD