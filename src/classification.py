import random

#return a balanced dataset (with binary labels) by undersampling the major class
def balanceDataset(dataset):
    counters={0:0,1:0}

    for d in dataset:
        if d.label: counters[1]+=1
        else: counters[0]+=1

    max_dset= 0 if counters[0]>counters[1] else 1
    unbalance= counters[int(not max_dset)]/counters[max_dset]

    for d in dataset:
        if d.label!=max_dset or random.uniform(0,1)<unbalance:
            yield d


