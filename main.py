import src.records as rec
import src.preproc as pre
import src.feature as feat
import src.classification as clas
import src.plot as plot
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
import time
import sys

#grid search cv on fusion hyperparameter -- i.e: number of subsequent of rr epochs to join before extracting features
#tol,max_iter: svc classifier convergence paramters
def searchBestFusion(data_dir,start=2,stop=14,max_rec=None,max_ep=None,tol=1e-3,max_iter=-1,verb=False):

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
#-compute the spearman correlation matrix of each subset
#-compute the matrix of median correlation
def corrAnalysis(dpoints,n_subsets=30):
    cors,cors_chest,cors_wrist=[],[],[] 
    for i in range(n_subsets):
        X,y=clas.svcDataset( pre.balanceDataset( dpoints ),['hrv'] )
        cors.append( spearmanr( X, axis=0 )[0] )
        X,y=clas.svcDataset( pre.balanceDataset( dpoints ),['chest'] )
        cors_chest.append( spearmanr( X, axis=0 )[0] )
        X,y=clas.svcDataset( pre.balanceDataset( dpoints ),['wrist'] )
        cors_wrist.append( spearmanr( X, axis=0 )[0] )
        

    #compute median covariance among features
    mcor=np.median(cors,axis=0)
    labels=feat.DataPoint.rrSemantics()
    
    fig,ax=plt.subplots()
    plot.plotSpearmanCor( ax,"Hrv features correlation",mcor,labels )

    mcor=np.median(cors_chest,axis=0)
    labels=feat.DataPoint.accSemantics()
    
    fig,ax=plt.subplots()
    plot.plotSpearmanCor( ax,"Chest acceleration features correlation",mcor,labels )

    mcor=np.median(cors_wrist,axis=0)
    fig,ax=plt.subplots()
    plot.plotSpearmanCor( ax,"Wrist acceleration features correlation",mcor,labels )
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
    if len(sys.argv)>1 and sys.argv[1]=="--loo":
        if len(sys.argv)>2:
            dpoints=feat.pointsFromFile(sys.argv[2])
        else:
            dpoints=feat.pointsFromFile("dset_fuse14.pkl")

        scores=clas.looScores( dpoints , loo_it= clas.leaveOneOutSubj("data/anonymized",fuse=14),tol=1e-2,verb=True)
        fig,ax=plt.subplots()
        plot.plotCvScores(ax,"Leave-one-out subject evaluation",scores)
        plt.show()
    
    elif len(sys.argv)>1 and sys.argv[1]=="--grid-search":
        searchBestFusion("data/anonymized",verb=True)

    elif len(sys.argv)>1 and sys.argv[1]=="--corr":
        if len(sys.argv)>2:
            dpoints=feat.pointsFromFile(sys.argv[2])
        else:
            dpoints=feat.pointsFromFile("dset_fuse14.pkl")
        corrAnalysis( dpoints )

    elif len(sys.argv)>1 and sys.argv[1]=="--dump":
        if len(sys.argv)>2:
             filen="dset_fuse"+sys.argv[2]+".pkl"
             dpoints=feat.getDatapoints("data/anonymized",fuse=int(sys.argv[2]),verb=True)
        else:
            filen="dset_fuse14.pkl"
            dpoints=feat.getDatapoints("data/anonymized",fuse=14,verb=True)
        feat.dumpPoints( dpoints,filen )
    else:
        print("Usage: python main.py --option [value]")
        print("\t\t--loo [feat_dump_file]: leave-one-out subject evaluation of the classifier.")
        print("\t\t--grid-search: grid-search cross-validation."+\
                "\n\t\tSelect the best number of subsequent epochs to join for hrv features computation.")
        print("\t\t--corr [feat_dump_file]: plot estimate of features correlation.")
        print("\t\t--dump [n]: dump datapoints in data/anonymized on file with pickle."+\
                "\n\t\t n: number of subsequent epochs to join for hrv feature computation.\n\t\tOutput file: dset_fuse[n].pkl")
