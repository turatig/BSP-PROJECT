import numpy as np
from scipy.signal import butter,filtfilt
import random

WAKE=0
SLEEP=1

#preprocessed epoch.
#q1: first quartile of rr series
#q3: third quartile of rr series
#vm(w/c): vector magnitude (wrist/chest)
#rid/rfilename: id of the epoch and filename
class Epoch():
    def __init__(self,dur,fsecg,fsacc,rr,mean,std,q1,q3,vmw,vmc,label,rid,rfilename):
        self.dur=dur
        self.fsecg=fsecg
        self.fsacc=fsacc
        self.rr=np.array(rr)
        self.mean=mean
        self.std=std
        self.q1=q1
        self.q3=q3
        self.vmw=np.array(vmw)
        self.vmc=np.array(vmc)
        self.label=label
        self.rid=rid
        self.rfilename=rfilename
    
    def fuse(self,ep1,ep2,rr_only=True):

        self.rr=np.concatenate([ep1.rr,ep2.rr])
        if not rr_only:
            self.vmc=np.concatenate([ep1.vmc,ep2.vmc])
            self.vmw=np.concatenate([ep1.vmw,ep2.vmw])

        self.dur=ep1.dur+ep2.dur

    def fuseLeft(self,epochs,rr_only=True):
        if epochs:
            for e in epochs[::-1]:
                self.fuse(e,self)
    
    def fuseRight(self,epochs,rr_only=True):
        if epochs:
            for e in epochs:
                self.fuse(self,e)
                
    def copy(self):
        return Epoch(self.dur,self.fsecg,self.fsacc,self.rr,self.mean,\
                self.std,self.q1,self.q3,self.vmw,self.vmc,self.label,self.rid,self.rfilename)

    def __str__(self):
        return "Epoch {0} from record {1}".format(self.rid,self.rfilename)

#compute rr series in ms from r peaks postion list
def rrSeries(rPeaks,fs,start=0):
    return [ (rPeaks[i]-rPeaks[i-1])*1000/fs if i>0 else (rPeaks[i]-start)*1000/fs for i in range(len(rPeaks)) ]

#return array of tuples (rr,True/False) for non_artifact/artifact in rr series given quartile 1 and 3,bpm low and high
def labeledRr(rrSeries,q1,q3,bpml=30,bpmh=180):

    f1=lambda rr: 60000/rr>=bpml and 60000/rr<=bpmh
    f2=lambda rr: rr>=q1-3*(q3-q1) and rr<=q3+3*(q3-q1)

    return [ (rr,f1(rr) and f2(rr)) for rr in rrSeries ]

#filter artifacts/non-artifacts from a labeled rr series vector
def filterLabeledRr(rr,non_artifact=True):
    return [rr[0] for rr in rr if rr[1]==non_artifact]

#compute vector magnitude for triaxial accelerometers signal and filter it with bandpass in 0.25Hz-3Hz
def filterVm(acc,fs):
    vm=np.sqrt(np.sum(np.power(acc,2),axis=0))
    b,a=butter(3,[0.25,3],btype='bp',output='ba',fs=fs)
    
    return filtfilt(b,a,vm)

#return list of rr epochs.
#t_idx: time index sleep list (start of every 30 sec epoch scored)
def rpSplitEpoch(rr,rp,t_idx):
    last,count=0,0
    epochs=list()
     
    for i in range(len(t_idx)-1):
        while count<len(rp) and rp[count]>=t_idx[i] and rp[count]<t_idx[i+1]: count+=1

        epochs.append(rr[last:count])
        last=count
    
    epochs.append( rr[last:len(rp)] )

    return epochs

def fuseEpochs(epochs):
    fuse=len(epochs)
    cep=epochs[fuse//2].copy()

    if fuse>1:
        cep.fuseLeft(epochs[:fuse//2 ])
        cep.fuseRight(epochs[ (fuse//2)+1:])
                
    return cep


#iter preprocessed epochs 30 seconds long of the given record
#fuse: number of subsequent epochs to fuse (half left,half right -- will be converted to an even number). 
def iterEpochs(record,max_iter=None,fuse=0,verb=False):
    
    dur=30
    fuse-=fuse%2
    #buffer for epochs fusion
    buf=[]

    rr=rrSeries(record.rPeaksRaw,record.fsEdf)
    q1,q3=np.quantile(rr,0.25),np.quantile(rr,0.75)
    labeled_rr=labeledRr(rr,q1,q3)
    
    filtrr=filterLabeledRr(labeled_rr,non_artifact=True)
    m,std=np.mean(filtrr),np.std(filtrr)
    
    vmw=filterVm(record.accWrist,record.fsAcc)
    vmc=filterVm(record.accChest,record.fsAcc)


    if verb:
        print("Mean RR: {0}".format(m))
        print("*"*20+" SIGNAL PREPROCESSING COMPLETED "+"*"*20)
        print("Number of epoches labeled: {0}".format(len(record.sleepStaging)))
    
    
    #for every epoch
    acc_step = dur*record.fsAcc
    count,yielded=0,0

    for e_rr in rpSplitEpoch(labeled_rr,record.rPeaksRaw,record.tIndexSleep):

        na=filterLabeledRr(e_rr,non_artifact=True)
        a=filterLabeledRr(e_rr,non_artifact=False)


        #if epoch has an acceptable sleep staging score and has less of 50% artifacts in rr series yield it
        if record.sleepStaging[count]<6 and len(e_rr) and len(a)/len(e_rr)<0.5:
            
            ep=Epoch(dur,record.fsEdf,record.fsAcc,na, m,std,q1,q3,\
                        vmw[ acc_step*count : acc_step *(count+1) ],vmc[ acc_step*count : acc_step *(count+1) ],\
                                int(record.sleepStaging[count]>1),count,record.filename)

            buf.append(ep)

            if len(buf)>=fuse+1:
                
                yield fuseEpochs(buf)
                                
                buf=buf[1:]
                yielded+=1
                if not max_iter is None and yielded>=max_iter: break

        count+=1
        if verb:
            print("Yielded {0} of {1} epochs".format(yielded,count))



#return a balanced dataset (with binary labels) by undersampling the major class
def balanceDataset(dataset):
    #label counters
    counters={0:0,1:0}
    dset=[]

    for d in dataset:
        if d.label: counters[1]+=1
        else: counters[0]+=1

    max_dset= 0 if counters[0]>counters[1] else 1
    unbalance= counters[int(not max_dset)]/counters[max_dset]

    for d in dataset:
        if d.label!=max_dset or random.uniform(0,1)<unbalance:
            dset.append(d)

    return dset

