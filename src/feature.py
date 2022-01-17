import numpy as np
import fathon as fa
from math import pi
from statsmodels.regression.linear_model import yule_walker
from scipy.signal import freqz,lfilter
from scipy.stats import percentileofscore
#from neurokit2 import entropy_sample
from antropy import sample_entropy
from functools import reduce
from math import inf,isinf


#root mean of squared successive differences
def rmssd(rr):
    return np.sqrt( 1/(len(rr)-1)*sum([ (rr[i+1]-rr[i])**2 for i in range(len(rr)-1)]) )

def rrStatistics(e):
    meanrr=np.mean((e.rr-e.mean)/e.std)
    stdrr=np.std(e.rr)
    rmssdrr=rmssd(e.rr)

    return meanrr,stdrr,rmssdrr

def getArModel(rr,order=9):
    rho,sigma=yule_walker(rr,order,'mle')
    rho*=-1
    rho=rho[::-1]

    return rho,sigma

#power spectral density estimation through autoregressive model of order 9
def arPsd(rho,sigma,n=1024):
    spectrum=sigma**2/np.abs(freqz(1,np.concatenate([[1],rho]),n,whole=True)[1])**2
    #return np.concatenate([ spectrum[:len(spectrum)//2][::-1] , spectrum[len(spectrum)//2:][::-1] ])
    return spectrum

#compute low frequency/ high frequency ratio of rr series psd
def lhRatio(spectrum,fs=1000):

    #standard psd band for hrv analysis
    lf=(0.04,0.15)
    hf=( 0.15, 0.4 if 0.4<fs/2 else fs/2 )

    freq=[ i/len(spectrum)*fs for i in range(len(spectrum)) ]
    lfs=[ spectrum[i] for i in range(len(spectrum)) if freq[i]>=lf[0] and freq[i]<lf[1] ]
    hfs=[ spectrum[i] for i in range(len(spectrum)) if freq[i]>=hf[0] and freq[i]<hf[1] ]

    tot_a=np.sum(spectrum)
    lf=np.sum(lfs)/tot_a
    hf=np.sum(hfs)/tot_a

    return lf/hf

#rho,sigma: ar model params
#generate syntethic sample entropy distribution of rr series by filtering wgn through ar model.
def simulateArSent(rho,sigma,n=1024,simulations=200):
    entropy_distr=[]

    for i in range(simulations):
        realization=lfilter([1],np.concatenate([[1],rho]),sigma*np.random.randn(n))
        #entropy_distr.append( entropy_sample(realization,dimension=1,tolerance=0.2*np.std(realization))[0] )
        entropy_distr.append( sample_entropy(realization,1) )

    return entropy_distr

#compute a probability agreement coefficient for the sample entropy according to synthetic data generated from model
def probAgreement(rho,sigma,sent,n,simulations=200):

    entropy_distr=simulateArSent(rho,sigma,n,simulations)

    #infinte values of sample entropy must be treated separately to avoid nan values
    if isinf(sent):
        return ( len([e for e in entropy_distr if e==sent])/len(entropy_distr) ) * 0.5

    entropy_distr=[ e for e in entropy_distr if not isinf(e) ]

    if sent<np.quantile(entropy_distr,0.05) or sent>np.quantile(entropy_distr,0.8):
        return 0
    elif sent>np.median(entropy_distr):
        return 1-percentileofscore(entropy_distr,sent)/100
    else:
        return percentileofscore(entropy_distr,sent)/100

#detrended fluctuations analysis short scale exponent
def dfaExp(rr,min_scale=4,max_scale=12):
    pydfa=fa.DFA(rr)
    wins=np.array([i for i in range(min_scale,max_scale)])
    n,_=pydfa.computeFlucVec(wins)
    scalexp,intercept=pydfa.fitFlucVec()

    return scalexp

#threshold-crossing rate of signal vm
def tcr(vm,th=0):
    count=0

    if vm[0]<th: count-=1
    
    for val in vm:
        if count%2 and val>th: count+=1
        if not count%2 and val<th: count+=1

    if vm[0]<th: count+=1
    
    return count 


#representation of an epoch as a point in feature space
class DataPoint():

    def __init__(self,e):
        self.extractRrFeatures(e)
        self.extractAccFeatures(e)
        self.label=e.label

    def extractRrFeatures(self,e):
        
        self.meanrr,self.stdrr,self.rmssdrr=rrStatistics(e)
    
        #power spectral density feature
        rho,sigma=getArModel(e.rr)
        spectrum=arPsd(rho,sigma)

        self.lfhf=lhRatio(spectrum,1/(np.mean(e.rr)/1000))
    
        #self-similarity features
        #self.sent=entropy_sample(e.rr,dimension=1,tolerance=0.2*np.std(e.rr))[0]
        self.sent=sample_entropy(e.rr,1)
        self.scalexp=dfaExp(e.rr)
        self.pa=probAgreement(rho,sigma,self.sent,n=len(e.rr))
    

    def extractAccFeatures(self,e):
        self.vmc_tcr=tcr(e.vmc,0.0052)
        self.vmc_mean=np.mean(e.vmc)
        self.vmc_std=np.std(e.vmc)
        self.vmc_max=max(e.vmc)

        self.vmw_tcr=tcr(e.vmw,0.0052)
        self.vmw_mean=np.mean(e.vmw)
        self.vmw_std=np.std(e.vmw)
        self.vmw_max=max(e.vmw)

    def rrRepr(self):
        return [self.meanrr,self.stdrr,self.rmssdrr,self.sent,self.pa,self.scalexp]

    def chestRepr(self):
        return [self.vmc_tcr,self.vmc_mean,self.vmc_std,self.vmc_max]

    def wristRepr(self):
        return [self.vmw_tcr,self.vmw_mean,self.vmw_std,self.vmw_max]

    def __str__(self):
        div="-"*20+"\n"

        fmt="\n\nRR FEATURES\n\n"+div
        fmt+="Mean: {0} \n".format(self.meanrr)+div
        fmt+="Std: {0}\n".format(self.stdrr)+div
        fmt+="Rmssd: {0}\n".format(self.rmssdrr)+div
        fmt+="Lf/Hf: {0}\n".format(self.lfhf)+div
        fmt+="Sample entropy [m=1,r=0.2*std(rr)]: {0}\n".format(self.sent)+div
        fmt+="Probability agreement: {0}\n".format(self.pa)+div
        fmt+="Scaling exponent: {0}\n".format(self.scalexp)+div

        fmt+="\n\nACCELEROMETER CHEST FEATURES\n\n"+div
        fmt+="Zero-crossing rate: {0}\n".format(self.vmc_tcr)+div
        fmt+="Mean: {0}\n".format(self.vmc_mean)+div
        fmt+="Std: {0}\n".format(self.vmc_std)+div
        fmt+="Max: {0}\n".format(self.vmc_max)+div

        fmt+="\n\nACCELEROMETER WRIST FEATURES\n\n"+div
        fmt+="Zero-crossing rate: {0}\n".format(self.vmw_tcr)+div
        fmt+="Mean: {0}\n".format(self.vmw_mean)+div
        fmt+="Std: {0}\n".format(self.vmw_std)+div
        fmt+="Max: {0}\n".format(self.vmw_max)+div

        return fmt

#Return a list of datapoints in features space representation from an iterable of epochs
def extractFeatures(epochs,verb=False):
        datapoints=[]
        bar=100
        ecount=0

        for e in epochs:
            datapoints.append( DataPoint(e) )
            p=int(ecount*bar/len(epochs))
    
            if verb and int((ecount-1)*bar/len(epochs))<p:
                print("Feature extraction:["+"-"*p+" "*(bar-p)+"]",end=" " )
                print("{0}/{1}".format(ecount,len(epochs)))
            
            ecount+=1

        return datapoints
