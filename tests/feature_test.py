import unittest
import sys
import numpy as np

sys.path.append('..')

import src.feature as feat
from statsmodels.regression.linear_model import yule_walker 
from numpy.random import uniform,randn
from scipy.signal import lfilter

class FeatureTest(unittest.TestCase):

    def test_tcr_2_oscill(self):
        v=[-1,0.5,1.7,1.5,0.4]
        self.assertEqual(feat.tcr(v,1),2)
        print("test_tcr_2_oscill".upper()+" PASSED")

    def test_tcr_no_oscill(self):
        v=[-1,0.5,1.7,1.5,0.4]
        self.assertFalse(feat.tcr(v,3))
        print("test_tcr_no_oscill".upper()+" PASSED")
    
    #Feature under test
    def test_probAgreement_out_of_range(self):
        n=128
        rr=uniform(2000,333,n)
        rho,sigma=feat.getArModel(rr,order=9)
        nsim=200
        min_entropy=[]
        max_entropy=[]
        
        for i in range(30):
            print("Testing probability agreement - {0}-th simualtion".format(i))
            entropy_distr=feat.simulateArSent(rho,sigma,n,nsim)
            min_entropy.append(min(entropy_distr))
            max_entropy.append(max(entropy_distr))

        min_entropy.sort()
        max_entropy.sort()
        
        #very likley to be out of bound
        sent=min_entropy[0]-2*np.std(min_entropy)
        self.assertFalse(feat.probAgreement(rho,sigma,sent,n,nsim))

        sent=max_entropy[0]+2*np.std(max_entropy)
        self.assertFalse(feat.probAgreement(rho,sigma,sent,n,nsim))

        print("test_probAgreement_out_of_range".upper()+" PASSED")
