import src.records as rec
import src.preproc as pre
import src.plot as plot
import src.feature as feat
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch,periodogram
from random import randint
from scipy.signal import freqz

def plotData(e,rec_count,epoch):    
    label="sleep" if e.label else "wake"
    fig=plot.plotEpoch(e,"Record {0} Epoch {1}\nLabel: {2}".format(rec_count,e.rid,label))
    fig.show()
    
    rho,sigma=feat.getArModel(e.rr,order=9)
    psd=feat.arPsd(rho,sigma,n=1024)

    if len(e.rr)>32: per=welch(e.rr,nperseg=len(e.rr)//8,nfft=1024,return_onesided=False)[1]
    else: per=periodogram(e.rr,nfft=1024,return_onesided=False)[1]


    fig2=plot.plotPSD(psd,"Record {0} Epoch {1}\nLabel: {2}".format(rec_count,e.rid,label),\
            1/(np.mean(e.rr)/1000),per=per)
    fig2.show()

    fig3,(ax1,ax2)=plt.subplots(2)
    plot.plotVMTime(ax1,"Vector magnitude wrist",e.vmw,e.fsacc)
    plot.plotVMTime(ax2,"Vector magnitude chest",e.vmc,e.fsacc)
    
    fig3.show()

#plot preprocessed epochs of records in a dataset selecting with probability p 
def inspectDatasetEpochs(data_dir,p=0.05,max_per_rec=10,lab="wake",fuse=0):
    rec_count=0
    _sleep,_wake,_totl=0,0,0
    lab=0 if lab=="wake" else 1
    count=0
 
    for r in rec.iterRecords(data_dir):
        
        print("Analyzing record number {0}".format(rec_count))
        print(r.fmtTime(fmt="hh"))

        plotted=0
        epoch=0
        sleep,wake,totl=0,0,0
        
        
        for e in pre.iterEpochs(r,fuse=fuse,verb=False):


            if not e.label:wake+=1
            else:sleep+=1
            totl+=1
            
            if lab==e.label and plotted<max_per_rec and randint(0,100)/100<p:
                plotted+=1
                plotData(e,rec_count,epoch)
                print(e)
                print(feat.DataPoint(e))
                input()

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

    data_dir="data/anonymized"

    #Random sampling probability 
    rs_prob=1
    max_per_rec=10
    label="sleep"
    fuse=0

    i=1
    
    if len(sys.argv)>1 and sys.argv[1]=="--help":    
        print("Usage: python analysis.py [--option value]... in any order and number")
        print("\t\t--data_dir directory: directory containing .mat records.")
        print("\t\t--rs_prob n (integer s.t. 0<=n<=1: random sampling probability."+\
                "\n\t\t\tExample: --rs_prob 0.2 means that (randomly) 2 epochs out 10 will be inspected.")
        print("\t\t--max_per_rec n: maximum number of epochs per record to be inspected.")
        print("\t\t--label (wake/sleep): select the label of epochs to be inspected.")
        print("\t\t--fuse n (integer s.t. 0<=n): select number of subsequent epochs to join for hrv features computation.")
        exit()
    else:
        while i<len(sys.argv):
            if sys.argv[i]=="--data_dir": data_dir=sys.argv[i+1]
            elif sys.argv[i]=="--rs_prob": rs_prob=float(sys.argv[i+1])
            elif sys.argv[i]=="--max_per_rec": max_per_rec=int(sys.argv[i+1])
            elif sys.argv[i]=="--label": label=sys.argv[i+1]
            elif sys.argv[i]=="--fuse": fuse=int(sys.argv[i+1])

            i+=2

    inspectDatasetEpochs(data_dir,rs_prob,max_per_rec,label,fuse)


