import unittest
import sys
import os
import numpy as np

sys.path.append('..')

import src.feature as feat
import src.preproc as pre
import src.records as rec

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
    def test_probAgreement_range(self):
        n=128
        rr=uniform(2000,333,n)
        rho,sigma=feat.getArModel(rr,order=9)
        nsim=200
        min_entropy=[]
        max_entropy=[]
        median_entropy=[]
        
        for i in range(30):
            print("Testing probability agreement - {0}-th simualtion".format(i))
            entropy_distr=feat.simulateArSent(rho,sigma,n,nsim)
            min_entropy.append(min(entropy_distr))
            max_entropy.append(max(entropy_distr))
            median_entropy.append(np.median(entropy_distr))

        min_entropy.sort()
        max_entropy.sort()
        
        #test sample entropy values very likley to be out of bound
        sent=min_entropy[0]-2*np.std(min_entropy)
        self.assertFalse(feat.probAgreement(rho,sigma,sent,n,nsim))

        sent=max_entropy[0]+2*np.std(max_entropy)
        self.assertFalse(feat.probAgreement(rho,sigma,sent,n,nsim))
        
        #test sample entropy values very likely to be around the median
        sent=np.mean(median_entropy)
        tested=feat.probAgreement(rho,sigma,sent,n,nsim)
        self.assertTrue(tested>=0.4)
    
    def test_dump(self):
        r=rec.Record("data/anonymized/record_0.mat")
        dpoints=[]
        
        if os.path.exists("test_dump.pkl"):
            os.remove("test_dump.pkl")

        for e in pre.iterEpochs(r):
            dpoints.append( feat.DataPoint(e) )

        feat.dumpPoints( dpoints, "test_dump.pkl")
        t_dpoints= feat.pointsFromFile("test_dump.pkl")

        self.assertEqual( len( dpoints ), len( t_dpoints ) )
        for i in range( len( dpoints ) ):
            self.assertEqual( dpoints[ i ], t_dpoints[ i ] )

        os.remove("test_dump.pkl")

