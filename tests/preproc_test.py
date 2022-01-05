import unittest
import sys
import math

sys.path.append('..')
import src.preproc as pre
import src.records as rec

class PreprocTest(unittest.TestCase):
    
    #Feature under test: labeldRr,filterLabeledRr
    def test_all_artifacts_rr(self):

        rr=[60000,60000,60000,100,100,500,1800]
        lab_rr=pre.labeledRr(rr,q1=900,q3=1000,bpml=30,bpmh=180)
        self.assertFalse(len(pre.filterLabeledRr(lab_rr,non_artifact=True)))

    def test_no_artifacts_rr(self):

        rr=[800,900,950,1200,750,1150,1000]
        lab_rr=pre.labeledRr(rr,q1=900,q3=1000,bpml=30,bpmh=180)
        self.assertEqual(len(pre.filterLabeledRr(lab_rr,non_artifact=True)),len(rr))
    
    #Feature under test: rpSplitEpoch
    def test_coherent_split_epoch(self):
        for r in rec.iterRecords("data/anonymized/"):
            epoch_idx=pre.rpSplitEpoch(r.rPeaksRaw,r.tIndexSleep)
            self.assertEqual(len(epoch_idx),len(r.sleepStaging))

    
