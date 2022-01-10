import unittest
import sys

sys.path.append('..')
import src.classification as clas
import src.preproc as pre
import src.records as rec
import src.feature as feat

class ClassificationTest(unittest.TestCase):
    def test_balance_data(self):
        dataset=[]

        for r in rec.iterRecords("data/anonymized"):
            print(r)
            for e in pre.iterEpochs(r):
                dataset.append(e)
        
        sleep,wake=0,0

        for d in clas.balanceDataset(dataset):
            self.assertTrue(d in dataset)
            if d.label: sleep+=1
            else: wake+=1

        
        print("Sleep: {0}\nWake: {0}".format(sleep,wake))
        self.assertTrue(abs(1-sleep/wake)<=0.1)

