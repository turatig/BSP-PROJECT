import numpy as np
import fathon as fa
from math import pi
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.ar_model import AutoReg 
from scipy.signal import freqz
from pyentrp.entropy import sample_entropy
from functools import reduce

#representation of an epoch as a point in feature space
class DataPoint():

    def __init__(self,meanrr,stdrr,rmssdrr,lfhf,sent,scalexp):
        self.meanrr=meanrr
        self.stdrr=stdrr
        self.rmssdrr=rmssdrr
        self.lfhf=lfhf
        self.sent=sent
        self.scalexp=scalexp

    def __str__(self):
        fmt="-"*20+"\n\nEpoch features\n\n"+"-"*20+"\n\n"
        fmt+="Mean RR: {0}\n".format(self.meanrr)
        fmt+="Std RR: {0}\n".format(self.stdrr)
        fmt+="Rmssd RR: {0}\n".format(self.rmssdrr)
        fmt+="Lf/Hf RR: {0}\n".format(self.lfhf)
        fmt+="Sample entropy RR [m=1,r=0.2*std(rr)]: {0}\n".format(self.sent)
        fmt+="Scaling exponent RR: {0}\n".format(self.scalexp)

        return fmt

#root mean of squared successive differences
def rmssd(rr):
    return np.sqrt( 1/(len(rr)-1)*sum([ (rr[i+1]-rr[i])**2 for i in range(len(rr)-1)]) )

#power spectral density estimation through autoregressive model of order 9
def psd(rr,order=9,n=1024):
    
    rho,sigma=yule_walker(rr,order,'mle')
    rho*=-1

    return sigma**2/np.abs(freqz(1,np.concatenate([[1],rho]),n,whole=True)[1])**2

#compute low frequency/ high frequency ratio of rr series psd
def lhRatio(rr,fs):
    spectrum=psd(rr)

    #right limit of hf band
    rl=0.4 if 0.4<fs/2 else fs/2
    freq=[ i/len(spectrum)*fs for i in range(len(spectrum)) ]
    lfs=[ spectrum[i] for i in range(len(spectrum)) if freq[i]>=0.04 and freq[i]<0.15 ]
    hfs=[ spectrum[i] for i in range(len(spectrum)) if freq[i]>=0.15 and freq[i]<rl ]

    tot_a=np.sum(spectrum)
    lf=np.sum(lfs)/tot_a
    hf=np.sum(hfs)/tot_a

    return lf/hf


#detrended fluctuations analysis short scale exponent
def dfaExp(rr,min_scale=4,max_scale=12):
    pydfa=fa.DFA(rr)
    wins=np.array([i for i in range(min_scale,max_scale)])
    n,_=pydfa.computeFlucVec(wins)
    scalexp,intercept=pydfa.fitFlucVec()

    return scalexp

def rrStatistics(e):
    meanrr=np.mean((e.rr-e.mean)/e.std)
    stdrr=np.std(e.rr)
    rmssdrr=rmssd(e.rr)

    return meanrr,stdrr,rmssdrr

#TODO: probability agreement feature
def extractRrFeatures(e):
    
    meanrr,stdrr,rmssdrr=rrStatistics(e)

    #power spectral density feature
    lfhf=lhRatio(e.rr,1/(np.mean(e.rr)/1000))

    #self-similarity features
    sent=sample_entropy(e.rr,1,0.2*np.std(e.rr))
    dfa=dfaExp(e.rr)
    
    return meanrr,stdrr,rmssd,lfhf,sent,dfa

#threshold-crossing rate
def tcr(vm,th=0):
    count=0

    if vm[0]<th: count-=1
    
    for val in vm:
        if count%2 and val>th: 
            count+=1
        if not count%2 and val<th: count+=1

    if vm[0]<th: count+=1
    
    return count 

def extractAccFeatures(vm):
    return tcr(vm,0.0052),np.mean(vm),np.std(vm),max(vm)

def extract(e):
    pass
