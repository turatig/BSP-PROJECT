import unittest
import sys
import math
import random
import numpy as np

sys.path.append('..')
import src.preproc as pre
import src.records as rec
from functools import reduce

class ArtifactsRRTest(unittest.TestCase):
    
    #Feature under test: labeldRr,filterLabeledRr
    def test_all_artifacts_rr(self):

        rr=[60000,60000,60000,100,100,500,1800]
        lab_rr=pre.labeledRr(rr,q1=900,q3=1000,bpml=30,bpmh=180)
        self.assertFalse(len(pre.filterLabeledRr(lab_rr,non_artifact=True)))

    def test_no_artifacts_rr(self):

        rr=[800,900,950,1200,750,1150,1000]
        lab_rr=pre.labeledRr(rr,q1=900,q3=1000,bpml=30,bpmh=180)
        self.assertEqual(len(pre.filterLabeledRr(lab_rr,non_artifact=True)),len(rr))

class EpochsTest(unittest.TestCase):

    def setUp(self):
        self.r=rec.Record("data/anonymized/record_0.mat")

    #Feature under test: rpSplitEpoch
    def test_coherent_split_epoch(self):
        rr=pre.rrSeries(self.r.rPeaksRaw,self.r.fsEdf)
        epochs=pre.rpSplitEpoch(rr,self.r.rPeaksRaw,self.r.tIndexSleep)
        self.assertEqual(len(epochs),len(self.r.sleepStaging))

    def test_fuse_epochs(self):
        epochs=[]

        for e in pre.iterEpochs(self.r,max_iter=5): epochs.append(e)
        
        fused=pre.fuseEpochs(epochs)
        self.assertEqual(fused.dur,30*5)
        self.assertEqual(list(reduce(lambda i,j: np.concatenate([i,j]),[e.rr for e in epochs])),list(fused.rr))
        self.assertEqual(fused.rid,epochs[len(epochs)//2].rid)
    
    def test_iter_fuse_epochs(self):
        fep,ep=[],[]
        fused=random.randint(0,20)
        fused-=fused%2

        for e in pre.iterFusedEpochs(self.r,fused):fep.append(e)
        for e in pre.iterEpochs(self.r):ep.append(e)

        self.assertEqual(len(fep),len(ep)-fused)
        self.assertEqual(fep[0].rid,ep[fused//2].rid)
        self.assertEqual(fep[-1].rid,ep[len(ep)-fused//2-1].rid)

