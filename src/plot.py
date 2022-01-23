import matplotlib.pyplot as plt
from scipy.fft import fft
import numpy as np
from seaborn import heatmap
from scipy.signal import periodogram

def plotRR(ax,title,e,bpm_lim=False):
    ax.set_title(title)
    ax.set_xlabel("Idx")
    ax.set_ylabel("R-R interval(ms)")
    x=[i for i in range(len(e.rr))]
    ax.plot(x,e.rr,color="blue",label="rr series")
    ax.plot(x,[e.mean for i in x],color="green",linestyle='dashed',label="mean")
    ax.plot(x,[e.q1-3*(e.q3-e.q1) for i in x],color="red",linestyle="dotted",label="q1-3*(q3-q1)")
    ax.plot(x,[e.q3+3*(e.q3-e.q1) for i in x],color="red",linestyle="dotted",label="q3+3*(q3-q1)")
    if bpm_lim:
        ax.plot(x,[60000/30 for i in x],color="orange",linestyle="-.",label="30 bpm")
        ax.plot(x,[60000/180 for i in x],color="orange",linestyle="dashdot",label="180 bpm")
    ax.legend()

def plotVM(ax,title,vm,fs):
    ax.set_title(title)
    sig=fft(vm)
    freq=[(i/len(vm))*fs for i in range(len(vm))]

    ax.set_xlabel("Frequncy(Hz)")
    ax.set_ylabel("Magnitude")
    ax.semilogy(freq[1:len(vm)//2],np.abs(sig[1:len(vm)//2]),color="blue")
    ax.vlines([0.25,3],min(np.abs(sig)),max(np.abs(sig)),colors=["orange","orange"],linestyles=["dotted","dotted"])

#plot vector magnitude of triaxial accelerometer highlighting zero-crossing threshold
def plotVMTime(ax,vm,title,fs,zc=None):
    ax.set_title(title)
    ax.set_ylabel("Power")
    ax.set_xlabel("Time(s)")

    t=[i/fs for i in range(len(vm))]
    ax.plot(t,vm,color="blue",label="vector magnitude")
    if not zc is None:
        ax.plot(t,[zc for i in range(len(vm))],color="red",linestyle="dashed",label="zero-crossing line")
    
    ax.legend()

def plotEpoch(e,stitle):
    fig,(ax1,ax2,ax3)=plt.subplots(3)
    
    title="RR series"
    plotRR(ax1,title,e)
    title="AVM - wrist"
    plotVM(ax2,title,e.vmw,e.fsacc)
    title="AVM - chest"
    plotVM(ax3,title,e.vmc,e.fsacc)
    fig.suptitle(stitle)

    return fig

#per: if periodogram is provided then will be plotted against parametric PSD estimation
def plotPSD(psd,fs,stitle,per=None):
    fig,ax=plt.subplots()

    freq=[i/len(psd)*fs for i in range(len(psd))]
    ax.set_title("Power spectral density - AR order 9")
    ax.set_xlabel("Frequency(Hz)")
    ax.set_ylabel("Magnitude")
    ax.semilogy(freq,psd,color="blue")

    if not per is None: 
        ax.semilogy(freq,per,color="green")

    rl=0.4 if 0.4<fs/2 else fs/2
    ax.vlines([0.04,0.15,rl],min(psd),max(psd),colors=["orange"]*3,linestyles=["dotted"]*3)
    fig.suptitle(stitle)

    return fig

def plotGsCvResults(ax,scores,fusion_ax,title="10-fold Cross validation results"):
    ax.set_title(title)
    ax.set_xlabel("Fusion hyperparameter")
    ax.set_ylabel("Score")

    ax.plot(fusion_ax,[ score['acc'] for score in scores ],color="green",label="accuracy")
    ax.plot(fusion_ax,[ score['cohen'] for score in scores ],color="blue",label="cohen's k-score")
    ax.plot(fusion_ax,[ score['sp'] for score in scores ],color="red",label="specificty")
    ax.plot(fusion_ax,[ score['se'] for score in scores ],color="orange",label="sensitivity")
    
    ax.legend()

def plotCovariance(ax,cov_mat,features):
    heatmap(cov_mat,ax=ax,xticklabels=features,yticklabels=features,annot=True)
