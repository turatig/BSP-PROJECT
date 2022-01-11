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
    def test_coherent_split(self):
        rr=pre.rrSeries(self.r.rPeaksRaw,self.r.fsEdf)
        epochs=pre.rpSplitEpoch(rr,self.r.rPeaksRaw,self.r.tIndexSleep)
        self.assertEqual(len(epochs),len(self.r.sleepStaging))

    def test_iter_epochs(self):
        epochs=[]
        n=random.randint(0,500)

        for e in pre.iterEpochs(self.r,max_iter=n): epochs.append(e)

        self.assertEqual(len(epochs),n)

        for e in epochs:
            self.assertEqual(e.dur,30)
            self.assertEqual(len(e.vmw),30*e.fsacc)
            self.assertTrue((sum(e.rr)-e.rr[0])/1000<=30)


    def test_fuse_epochs(self):
        epochs=[]
        n=random.randint(2,20)
        n+=1-(n%2)

        for e in pre.iterEpochs(self.r,max_iter=n): epochs.append(e)
        
        epochs[n//2].fuseLeft(epochs[:n//2])
        epochs[n//2].fuseRight(epochs[(n//2)+1:])

        self.assertEqual(epochs[n//2].dur,30*n)
        tested=list(reduce(lambda i,j: np.concatenate([i,j]),[e.rr for e in epochs]))

        for i in range(n):
            self.assertEqual(epochs[n//2].rr[i],tested[i])
    
    def test_iter_fuse_epochs(self):
        fep,ep=[],[]
        fuse=random.randint(0,20)
        fuse-=fuse%2

        for e in pre.iterEpochs(self.r,fuse=fuse):fep.append(e)
        for e in pre.iterEpochs(self.r):ep.append(e)

        self.assertEqual(len(fep),len(ep)-fuse)
        self.assertEqual(fep[0].rid,ep[fuse//2].rid)
        self.assertEqual(fep[-1].rid,ep[len(ep)-fuse//2-1].rid)

    def test_balance_data(self):
        dataset=[]

        for e in pre.iterEpochs(self.r):
            dataset.append(e)
        
        sleep,wake=0,0

        for d in pre.balanceDataset(dataset):
            self.assertTrue(d in dataset)
            if d.label: sleep+=1
            else: wake+=1

        
        self.assertTrue(abs(1-sleep/wake)<=0.1)

