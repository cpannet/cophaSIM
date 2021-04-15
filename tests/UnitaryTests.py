# -*- coding: utf-8 -*-

import unittest

import numpy as np

import coh_lib.skeleton as sk
import coh_lib.coh_tools as ct
from coh_lib import config
        
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
        from coh_lib.SPICA_FS import SPICAFS_TRUE
        directory = 'C:/Users/cpannetier/Documents/These/FringeTracking/SPICA-FT/V2PM_SPICA/'
        V2PMfilename = 'TESTBENCH_ABCD_H_PRISM75_V2PM.fits'
        FSfitsfile = directory+V2PMfilename
        OW=1
        spectra, spectraM = SPICAFS_TRUE(init=True,fitsfile=FSfitsfile,OW=OW)
        self.assertEqual(len(spectra),len(spectraM)*OW, "The oversampling by SPICA-FS didn't work well")
        
        # Test V2PM & P2VM
        coher = np.ones(36)*1000
        Modulation = config.FS['V2PM'] ; Demodulation = config.FS['P2VM']
        pixels = np.real(np.dot(Modulation[0],coher))
        coher2 = np.dot(Demodulation[0],pixels)
        self.assertTrue(np.linalg.norm(coher-coher2)<1e-5,"The demodulation doesn't work well")
        
        

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStringMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)
    