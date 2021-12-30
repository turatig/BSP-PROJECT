import records as rec
import sys
import preproc as pre
import  plot as plot
import feature as feat
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from random import randint
from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import freqz

#plot preprocessed epochs of records in a dataset selecting with probability p 
def inspectDatasetEpochs(data_dir,p=0.05,max_per_rec=10,lab="wake"):
    rec_count=0
    _sleep,_wake,_totl=0,0,0
    lab=0 if lab=="wake" else 1
    
    for r in rec.iterRecords(data_dir):
        print("Analyzing record number {0}".format(rec_count))
        print(r.fmtTime(fmt="hh"))

        plotted=0
        epoch=0
        sleep,wake,totl=0,0,0
        
        """rr=pre.rrSeries(r.rPeaks,r.fsAcc)[:1024]
        p=feat.psd(rr,order=150)

        fs=1/(np.mean(rr)/1000)
        fig=plot.plotPSD(p,fs,"PSD of the 1024 samples of the entire series",rr)
        fig.show()
        input()"""
        for e in pre.iterEpochs(r,verb=False):


            if not e.label:wake+=1
            else:sleep+=1
            totl+=1
            
            if lab==e.label and plotted<max_per_rec and randint(0,100)/100<p:
                plotted+=1
                label="sleep" if e.label else "wake"
                fig=plot.plotEpoch(e,"Record {0} Epoch {1}\nLabel: {2}".format(rec_count,epoch,label))
                fig.show()
                fig2=plot.plotPSD(feat.psd(e.rr),1/(np.mean(e.rr)/1000),\
                        "Record {0} Epoch {1}\nLabel: {2}".format(rec_count,epoch,label))
                fig2.show()

                fig3,(ax1,ax2)=plt.subplots(2)
                plot.plotVMTime(ax1,e.vmw,"Vector magnitude wrist",e.fsacc)
                plot.plotVMTime(ax2,e.vmc,"Vector magnitude chest",e.fsacc)
                
                print(e.vmw.shape)
                fig3.show()
                datap=feat.extract(e)
                #print(datap)
                input()
            try: 
                assert not np.isnan(np.sum(feat.psd(e.rr)))
            except AssertionError as e:
                print("Record {0} epoch {1} has nan".format(count,epoch))

            epoch+=1

        _sleep+=sleep
        _wake+=wake
        _totl+=totl

        print("Wake(%): {0}".format(wake*100/totl))
        print("Sleep(%): {0}".format(sleep*100/totl))

        rec_count+=1

    print("-"*20+'\n'+"-"*20)
    print("Total wake(%): {0}".format(_wake*100/_totl))
    print("Total sleep(%): {0}".format(_sleep*100/_totl))
    print("-"*20+'\n'+"-"*20)



if __name__=="__main__":
    inspectDatasetEpochs(sys.argv[1],float(sys.argv[2]),int(sys.argv[3]),sys.argv[4])

