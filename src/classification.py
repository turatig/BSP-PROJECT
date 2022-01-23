import src.records as rec
import src.preproc as pre
import src.feature as feat
import src.plot as plot
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,cohen_kappa_score,confusion_matrix
from sklearn.model_selection import cross_validate


def svcDataset(datapoints,feature_set=["hrv"]):
    feature_type={
            "hrv": lambda d: d.rrRepr(),
            "chest": lambda d: d.chestRepr(),
            "wrist": lambda d: d.wristRepr()
        }
    
    for f_type in feature_set:
        if f_type not in feature_type.keys(): raise Exception("Unknown feature type")

    X,y=[],[]

    for d in datapoints:
        features=[]

        for f_type in feature_set:
            features+=feature_type[f_type](d)

        X.append(np.array(features))
        y.append(np.array(d.label))

    return np.array(X),np.array(y)

def svcScore(clf,X,y_truth):
    
    y_pred=clf.predict(X)
    c=confusion_matrix(y_truth,y_pred)
    tn,fn=c[0,0],c[1,0]
    tp,fp=c[1,1],c[0,1]

    return {
            "acc": accuracy_score(y_truth,y_pred),
            "cohen": cohen_kappa_score(y_truth,y_pred),
            "sp": tn/(tn+fp),
            "se": tp/(tp+fn)
            }

#perform a grid-search 10-fold cv on the fusion hyperparameter 
#i.e. number of adjacent epochs to be considered to have reliable
def neighbourEpochSelection(data_dir,start=2,stop=14,max_rec=None,max_ep=None,verb=False):

    model=SVC(kernel="linear")
    fuse=start

    scores=[]

    records=[ r for r in rec.iterRecords(data_dir,max_iter=max_rec,verb=verb) ]
    print("\n\n--- Scoring hrv features for the fusion hyperparam selection: range[{0},{1}] ---\n".format(start,stop))
    while fuse<=stop:
        epochs=[]
        print("--- Nearby epochs to be fused: {0} ---\n--- Preprocessing and epochs extraction ---\n".format(fuse))
        for r in records:
            epochs+=[e for e in pre.iterEpochs(r,fuse=fuse,max_iter=max_ep)]
        
        X,y=svcDataset( feat.extractFeatures(pre.balanceDataset(epochs,verb=verb),verb=verb) )
        print("--- Cross-validated scoring ---")
        cv_res=cross_validate( model,X,y,scoring=svcScore,cv=10,verbose=3 if verb else 0 )

        score={
                "acc": np.mean(cv_res['test_acc']),
                "cohen": np.mean(cv_res['test_cohen']),
                "sp": np.mean(cv_res['test_sp']),
                "se": np.mean(cv_res['test_se'])
            }
        
        scores.append(score)
        fuse+=2

    return scores

#iterator yield (train,test) as list of indices implementing LOO strategy on subjects
#each subject is associated with a record
def leaveOneOutSubj(data_dir,fuse=0,max_rec=None,max_ep=None,verb=False):
    lengths=[]

    for r in rec.iterRecords(data_dir,max_iter=max_rec):
        length=0
        for e in pre.iterEpochs(r,fuse=fuse,max_iter=max_ep): length+=1
        lengths.append(length)
        if verb:
            print("Length (in epochs): {0}".format(length))

    tot=np.sum(lengths)
    n=0
    for l in lengths:
        test=np.arange(n,n+l,dtype=int)
        train=np.concatenate([ np.arange(0,n,dtype=int),np.arange(n+l,tot,dtype=int) ])
        
        if verb:
            print("Train indices\n{0}".format(train))
            print("Test indices \n{0}".format(test))

        yield train,test
        n+=l


