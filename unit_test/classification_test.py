import unittest
import sys
from functools import reduce

sys.path.append('..')
import src.records as rec
import src.feature as feat
import src.preproc as pre
import src.classification as clas

class ClassificationTest(unittest.TestCase):
    def test_svc_dataset(self):
        r=rec.Record("data/anonymized/record_0.mat")
        dpoints=[]

        for e in pre.iterEpochs(r):
            dpoints.append( feat.DataPoint(e) )

        X,y=clas.svcDataset( dpoints,["chest","wrist"] )

        self.assertEqual( len(dpoints), len(X) )

        for i in range(len(dpoints)):
            expected=  dpoints[i].chestRepr() + dpoints[i].wristRepr()
            resulted= X[i]
            self.assertEqual( len(expected), len(resulted) )
            self.assertTrue( all([ expected[j] == resulted[j] for j in range( len(expected) )]) )

    def test_loo(self):
        recs=[]
        epochs=[]

        for r in rec.iterRecords("data/anonymized",verb=True):
            ep_rec=[]
            for e in pre.iterEpochs(r):
                epochs.append(e)
                ep_rec.append(e)

            recs.append(ep_rec)
        
        count=0
        for train_idx,test_idx in clas.leaveOneOutSubj("data/anonymized"):
            expected_test= recs[count]
            expected_train= reduce(lambda i,j: i+j,recs[:count] + recs[count+1:])

            resulted_test=[ epochs[i] for i in test_idx ]
            resulted_train=[ epochs[i] for i in train_idx ]

            self.assertTrue( expected_test==resulted_test )
            self.assertTrue( expected_train==resulted_train )
            count+=1
