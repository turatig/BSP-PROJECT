import src.records as rec
import src.preproc as pre
import src.feature as feat
import src.classification as clas
import src.plot as plot
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
import time

#grid search cv on fusion hyperparameter -- i.e: number of subsequent of rr epochs to join before extracting features
def searchBestFusion(data_dir,start=2,stop=14,max_rec=None,max_ep=None,verb=False):

    scores=clas.neighbourEpochSelection(data_dir,start=start,stop=stop,max_rec=max_rec,max_ep=max_ep,verb=verb)
    #only even number of epochs are fused with one -- half on the left, half on the right of the labeled epoch
    fused_epochs=[ i for i in range(start,stop+1) if not i%2 ]
    scores_list=[ (fused_epochs[i],scores[i]) for i in range(len(scores)) ]
    best_score=max(scores_list,key=lambda el: el[1]["sp"])
    
    if verb:
        print("Best hyperparam found: {0}".format(best_score[0]))
        print("Score: {0}".format(best_score[1]))

        fig,ax=plt.subplots()
        plot.plotGsCvResults(ax,"10-fold cross-validation results \n\
                fusion hyper: number of subsequent rr epochs",scores,fused_epochs)
        plt.show()

    return best_score

#estimate feature covariance through the following process:
#-extract 30 balancedDataset (3000 epochs of sleep and 3000 epochs of wake each)
#-compute the covariance matrix
#-visualize the matrix of the median covariance of each couple of features
def covarianceAnalysis(dpoints,n_subsets=30):
    cors=[] 
    for i in range(n_subsets):
        X,y=clas.svcDataset( pre.balanceDataset( dpoints ),['hrv'] )
        cors.append( spearmanr( X, axis=0 )[0] )

    #compute median covariance among features
    mcor=np.median(cors,axis=0)
    labels=feat.DataPoint.rrSemantics()
    
    fig,ax=plt.subplots()
    plot.plotSpearmanCor( ax,mcor,labels )
    plt.show()

#filen: name of the output file
def dumpDataset(data_dir,filen,fuse=0,max_rec=None,max_ep=None,verb=False):
    epochs=[]
    for r in rec.iterRecords( data_dir,max_iter=max_rec,verb=verb ):
        for e in pre.iterEpochs(r,max_iter=max_ep,fuse=fuse):
            epochs.append( e )

    dpoints=feat.extractFeatures(epochs,verb=verb)
    feat.dumpPoints(dpoints,filen)

    return dpoints

    

if __name__=="__main__":
    #searchBestFusion("data/anonymized",start=2,stop=6,max_rec=2,max_ep=200,verb=True)
    dpoints=feat.pointsFromFile("dset_fuse14.pkl")
    scores=clas.looScores( dpoints , loo_it= clas.leaveOneOutSubj("data/anonymized",fuse=14,verb=True),verb=True)
    fig,ax=plt.subplots()
    plot.plotCvScores(ax,"Leave-one-out subject evaluation",scores)
    fig.show()
    plt.show()
    #covarianceAnalysis( dpoints )
    #covarianceAnalysis( feat.getDatapoints("data/anonymized",fuse=14 ,verb=True) )
    #dumpDataset("data/anonymized","dset_fuse14.pkl",fuse=14,verb=True)

    

    

