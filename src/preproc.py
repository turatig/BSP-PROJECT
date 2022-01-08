import numpy as np
from scipy.signal import butter,filtfilt

#preprocessed epoch.
#q1: first quartile of rr series
#q3: third quartile of rr series
#vm(w/c): vector magnitude (wrist/chest)
#label: 0(wake)/1(sleep). N.B: Nrem1 collapse on wake
class Epoch():
    def __init__(self,dur,fsecg,fsacc,rr,mean,std,q1,q3,vmw,vmc,label):
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

#iter preprocessed epochs dur seconds long of the given record
def iterEpochs(record,dur=30,max_iter=None,verb=False):

    
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
    count,acc_samp,yielded=0,0,0
    for e_rr in rpSplitEpoch(labeled_rr,record.rPeaksRaw,record.tIndexSleep):

        na=filterLabeledRr(e_rr,non_artifact=True)
        a=filterLabeledRr(e_rr,non_artifact=False)


        #if epoch has an acceptable sleep staging score and has less of 50% artifacts in rr series yield it
        if record.sleepStaging[count]<6 and len(e_rr) and len(a)/len(e_rr)<0.5:

            yield Epoch(dur,record.fsEdf,record.fsAcc,\
                    na, m,std,q1,q3,\
                        vmw[acc_samp:acc_samp+acc_step],vmc[acc_samp:acc_samp+acc_step],int(record.sleepStaging[count]>2))
            yielded+=1
        
        count+=1
        acc_samp+=acc_step

    if verb:
        print("Yielded {0} of {1} epochs".format(yielded,count))

