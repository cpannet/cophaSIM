# -*- coding: utf-8 -*-

import unittest
import importlib

import numpy as np

import cophasing.skeleton as sk
import cophasing.coh_tools as ct
from cophasing import config
        
class TestStringMethods(unittest.TestCase):

    def setUp(self):
         pass

    def test_test(self):
        self.assertEqual(2,2)
    
    def test_NB2NIN(self):
        NA=6 ; NB=NA**2 ; NIN=NA*(NA-1)/2
        vector = np.arange(NB)
        vectornin = ct.NB2NIN(vector)
        self.assertTrue(len(vectornin)==NIN)
        self.assertEqual(vectornin[0], vector[1])
                
    def test_addnoise(self):
        config.ron=10 ; config.enf=1.5 ; config.qe=1
        pixels = 100*np.ones(60)
        pixels2 = sk.addnoise(pixels)
        self.assertEqual(pixels2.shape, pixels.shape, 'Images are not same size')
    
    
    
    
        
    def test_fringesensors(self):
        coher = np.ones(36)
        from cophasing.SPICA_FS import SPICAFS_TRUE, SPICAFS_PERFECT
        
        print("\nTry:")
        
        print(" - Oversampling by SPICA-FS?")
        directory = 'C:/Users/cpannetier/Documents/These/FringeTracking/SPICA-FT/V2PM_SPICA/'
        V2PMfilename = 'TESTBENCH_ABCD_H_PRISM75_V2PM.fits'
        FSfitsfile = directory+V2PMfilename
        OW=1
        spectra, spectraM = SPICAFS_TRUE(init=True,fitsfile=FSfitsfile,OW=OW)
        self.assertEqual(len(spectra),len(spectraM)*OW, "The oversampling by SPICA-FS didn't work well")
        
        
        print(" -Estimated photometries and visibilities equal to the input after micro modulation/demodulation ?")
        # Test V2PM & P2VM
        coher = np.ones(36)*1000 + 0j
        Modulation = config.FS['V2PM'] ; Demodulation = config.FS['P2VM']
        pixels = np.real(np.dot(Modulation[0],coher))
        coher2 = np.dot(Demodulation[0],pixels)
        self.assertTrue(np.linalg.norm(coher-coher2)<1e-5,"The demodulation doesn't work well")
        
        
        # Test V2PM & P2VM
        photometries = np.array([0.2,1,1,1,1,1])*1000
        visibilities = np.ones(15)*np.exp(np.pi*1j)
        for ia in range(6):
            for iap in range(6):
                if ia < iap:
                    coher[ia*6+iap] = 2*np.sqrt(photometries[ia]*photometries[iap])*visibilities[ct.posk(ia,iap,6)]
                elif ia > iap:
                    coher[ia*6+iap] = np.transpose(np.conj(2*np.sqrt(photometries[ia]*photometries[iap])*visibilities[ct.posk(ia,iap,6)]))
                else:
                    coher[ia*6+iap] = photometries[ia]

        pixels = np.real(np.dot(Modulation[0],coher))
        coher2 = np.dot(Demodulation[0],pixels)
        self.assertTrue(np.linalg.norm(coher-coher2)<1e-5,f"The demodulation doesn't work well. \n\
Input coher: {coher} \n\
Output coher: {coher2}")

        photometries_chck2 = [coher2[ia*(6+1)] for ia in range(6)]
        visibilities_chck2 = np.zeros(15)*1j
        for ia in range(6):
            for iap in range(6):
                if ia!=iap:
                    visibilities_chck2[ct.posk(ia,iap,6)]  = coher2[ia*6+iap]/(2*np.sqrt(photometries[ia]*photometries[iap]))
                    
        self.assertTrue(np.linalg.norm(photometries-photometries_chck2)<1e-5,f"The estimation of photometries is not correct: \n\
Input:{photometries}\n\
Output:{photometries_chck2}")
        self.assertTrue(np.linalg.norm(visibilities-visibilities_chck2)<1e-5,f"The estimation of coherences is not correct: \n\
Input:{visibilities}\n\
Output:{visibilities_chck2}")


    def test_fs_perfect(self):
        from cophasing import skeleton as sk
        # print(" Fringe Sensor SPICAFS_PERFECT")
        lmbda1 = 1.5
        lmbda2 = 1.7
        MW=10
        OW=10
        # spectra = np.arange(lmbda1, lmbda2, )
        spectra, spectraM = ct.generate_spectra(lmbda1, lmbda2, OW=OW, MW=MW, mode='linear_sig')
        NW = len(spectra)
        
        from cophasing.SPICA_FS import SPICAFS_PERFECT
        SPICAFS_PERFECT(init=True, spectra=spectra, spectraM=spectraM)
        
        datadir = 'C:/Users/cpannetier/Documents/Python_packages/cophasing/cophasing/data/'
        CHARAfile = datadir+'interferometers/CHARAinterferometerH.fits'
        ObservationFile = datadir+'observations/CHARA/Unresolved_mag3.fits'
        datadir2 = "C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/disturbances/NoDisturbances/"
        DisturbanceFile = datadir2+'NoDisturbances.fits'    # Realistic disturbance

        sk.initialize(CHARAfile, ObservationFile, DisturbanceFile,
                      spectra=spectra, spectraM=spectraM,qe=1.)
        
        from cophasing import simu
        importlib.reload(simu)
        
        photometries0 = np.array([0.2,1,0.4,1,1,1])*1000
        photometries = np.repeat(photometries0[np.newaxis,:],NW,0)
        photometriesMacro = np.repeat(photometries0[np.newaxis,:],MW,0)*OW
        
        visibilities = np.ones(15)+0j
        visibilities = np.repeat(visibilities[np.newaxis,:],NW,0)
        coher = np.zeros([NW,36])*1j
        for ia in range(6):
            coher[:,ia*(6+1)] = photometries[:,ia]
            for iap in range(ia+1,6):
                coher[:,ia*6+iap] = np.sqrt(photometries[:,ia]*photometries[:,iap])*visibilities[:,ct.posk(ia,iap,6)]
                coher[:,iap*6+ia] = np.conjugate(coher[:,ia*6+iap])
        
        
        Demodulation = config.FS['MacroP2VM']
        Modulation = config.FS['V2PM']
        
        MW,NB,NP = np.shape(Demodulation)
        MacroImages = np.zeros([MW,NP])
        MicroImages = np.zeros([NW,NP]) ; coher_chck_nw = np.zeros([NW,NB])+0j
        iow=0;imw=0
        for iw in range(config.NW):
            
            Modulation = config.FS['V2PM'][iw,:,:]
            image_iw = np.real(np.dot(Modulation,coher[iw,:]))
            self.assertTrue(np.min(image_iw)>=0)
            MacroImages[imw,:] += image_iw
            MicroImages[iw,:] = image_iw
            
            coher_chck_nw[iw,:] = np.dot(config.FS['P2VM'][iw,:],image_iw)
            iow += 1
            if iow == OW:
                imw+=1
                iow = 0      
                
        coher_chck = np.zeros([MW,NB])+0j
        for imw in range(MW):
            coher_chck[imw,:] = np.dot(Demodulation[imw,:],MacroImages[imw,:])
        
        photometries_chck = np.transpose([coher_chck[:,ia*(6+1)] for ia in range(6)])
        photometries_chck_nw = np.transpose([coher_chck_nw[:,ia*(6+1)] for ia in range(6)])
        # visibilities_chck = np.zeros([MW,15])*1j
        # for ia in range(6):
        #     for iap in range(6):
        #         if ia!=iap:
        #             print(ia*6+iap)
        #             visibilities_chck[:,ct.posk(ia,iap,6)]  = coher_chck[:,ia*6+iap]/(np.sqrt(photometries_chck[:,ia]*photometries_chck[:,iap]))
        # self.assertTrue(np.linalg.norm(photometries-photometries_chck)<1e-5)
        self.assertTrue(np.abs(np.sum(photometries)-np.sum(photometries_chck))<1e-5)
        self.assertTrue(np.linalg.norm(photometriesMacro-photometries_chck)<1e-5,f"{np.linalg.norm(photometriesMacro-photometries_chck)}")
        self.assertTrue(np.linalg.norm(coher) - np.linalg.norm(coher_chck_nw)<1e-5, f"The micro demodulation doesn't conserve energy")
        
        coher2 = SPICAFS_PERFECT(coher)


        photometries_chck2 = np.transpose([coher2[:,ia*(6+1)] for ia in range(6)])
        visibilities_chck2 = np.zeros([MW,15])*1j
        for ia in range(6):
            for iap in range(6):
                if ia!=iap:
                    visibilities_chck2[:,ct.posk(ia,iap,6)]  = coher2[:,ia*6+iap]/(np.sqrt(photometries_chck2[:,ia]*photometries_chck2[:,iap]))
        
        
        self.assertTrue(np.abs(np.sum(coher)-np.sum(coher2))<1e5, f"Energy input: {np.sum(coher)} \n\
Energy output {np.sum(coher2)}")
        self.assertTrue(np.linalg.norm(photometriesMacro-photometries_chck2)<1e-5,f"The estimation of photometries is not correct: \n\
Input:{photometries}\n\
# Output:{photometries_chck2}")
#         self.assertTrue(np.linalg.norm(visibilities-visibilities_chck2)<1e-5,f"The estimation of coherences is not correct: \n\
# Input:{visibilities}\n\
# Output:{visibilities_chck2}")

        
        print("- Test MIRCxFS")
        
        from cophasing.MIRCx_FS import MIRCxFS
        MIRCxFS(init=True, spectra=spectra, spectraM=spectraM)
        
        datadir = 'C:/Users/cpannetier/Documents/Python_packages/cophasing/cophasing/data/'
        CHARAfile = datadir+'interferometers/CHARAinterferometerH.fits'
        ObservationFile = datadir+'observations/CHARA/Unresolved_mag3.fits'
        datadir2 = "C:/Users/cpannetier/Documents/These/FringeTracking/Python/Simulations/data/disturbances/NoDisturbances/"
        DisturbanceFile = datadir2+'NoDisturbances.fits'    # Realistic disturbance

        sk.initialize(CHARAfile, ObservationFile, DisturbanceFile,
                      spectra=spectra, spectraM=spectraM,qe=1.)
        
        from cophasing import simu
        importlib.reload(simu)
        
        photometries0 = np.array([1,1,1,1,1,1])*1000
        photometries = np.repeat(photometries0[np.newaxis,:],NW,0)
        photometriesMacro = np.repeat(photometries0[np.newaxis,:],MW,0)*OW
        
        visibilities = np.ones(15)+0j
        visibilities = np.repeat(visibilities[np.newaxis,:],NW,0)
        coher = np.zeros([NW,36])*1j
        for ia in range(6):
            coher[:,ia*(6+1)] = photometries[:,ia]
            for iap in range(ia+1,6):
                coher[:,ia*6+iap] = np.sqrt(photometries[:,ia]*photometries[:,iap])*visibilities[:,ct.posk(ia,iap,6)]
                coher[:,iap*6+ia] = np.conjugate(coher[:,ia*6+iap])
        
        
        Demodulation = config.FS['MacroP2VM']
        Modulation = config.FS['V2PM']
        
        MW,NB,NP = np.shape(Demodulation)
        MacroImages = np.zeros([MW,NP])
        MicroImages = np.zeros([NW,NP]) ; coher_chck_nw = np.zeros([NW,NB])+0j
        iow=0;imw=0
        for iw in range(config.NW):
            
            Modulation = config.FS['V2PM'][iw,:,:]
            image_iw = np.real(np.dot(Modulation,coher[iw,:]))
            self.assertTrue(np.min(image_iw)>=0)
            MacroImages[imw,:] += image_iw
            MicroImages[iw,:] = image_iw
            
            coher_chck_nw[iw,:] = np.dot(config.FS['P2VM'][iw,:],image_iw)
            iow += 1
            if iow == OW:
                imw+=1
                iow = 0      
                
        coher_chck = np.zeros([MW,NB])+0j
        for imw in range(MW):
            coher_chck[imw,:] = np.dot(Demodulation[imw,:],MacroImages[imw,:])
        
        photometries_chck = np.transpose([coher_chck[:,ia*(6+1)] for ia in range(6)])
        photometries_chck_nw = np.transpose([coher_chck_nw[:,ia*(6+1)] for ia in range(6)])
        
#         self.assertTrue(np.abs(np.sum(photometries)-np.sum(photometries_chck))<1e-5, f"Input Photometries: {photometries[0]} \n\
# Output photometries: {photometries_chck[0]}")
#         self.assertTrue(np.linalg.norm(photometriesMacro-photometries_chck)<1e-5,f"{np.linalg.norm(photometriesMacro-photometries_chck)}")
#         self.assertTrue(np.linalg.norm(coher) - np.linalg.norm(coher_chck_nw)<1e-5, f"The micro demodulation doesn't conserve energy")
        
        coher2 = MIRCxFS(coher)

        photometries_chck2 = np.transpose([coher2[:,ia*(6+1)] for ia in range(6)])
        visibilities_chck2 = np.zeros([MW,15])*1j
        for ia in range(6):
            for iap in range(6):
                if ia!=iap:
                    visibilities_chck2[:,ct.posk(ia,iap,6)]  = coher2[:,ia*6+iap]/(np.sqrt(photometries_chck2[:,ia]*photometries_chck2[:,iap]))
        
        self.assertTrue(np.abs(np.sum(photometries)-np.sum(photometries_chck))<1e-5, f"Input Photometries: {photometries[0]} \n\
# Output photometries: {photometries_chck2[0]}")
        self.assertTrue(np.abs(np.sum(coher)-np.sum(coher2))<1e5, f"Energy input: {np.sum(coher)} \n\
Energy output {np.sum(coher2)}")
        self.assertTrue(np.linalg.norm(photometriesMacro-photometries_chck2)<1e-5,f"The estimation of photometries is not correct: \n\
Input:{photometries}\n\
# Output:{photometries_chck2}")
#         self.assertTrue(np.linalg.norm(visibilities-visibilities_chck2)<1e-5,f"The estimation of coherences is not correct: \n\
# Input:{visibilities}\n\
# Output:{visibilities_chck2}")





if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStringMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)
    