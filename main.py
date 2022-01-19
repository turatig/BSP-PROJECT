import src.records as rec
import src.preproc as pre
import src.feature as feat
import src.classification as clas
import src.plot as plot
import matplotlib.pyplot as plt

if __name__=="__main__":
    scores=clas.neighbourEpochSelection("data/anonymized",verb=True)
    #only even number of epochs are fused with one -- half on the left, half on the right of the labeled epoch
    fused_epochs=[ i for i in range(2,15) if not i%2 ]
    scores_list=[ (fused_epochs[i],scores[i]) for i in range(len(scores)) ]
    best_score=max(scores_list,key=lambda el: el[1]["sp"])

    print("Best hyperparam found: {0}".format(best_score[0]))
    print("Score: {0}".format(best_score[1]))

    fig,ax=plt.subplots()
    plot.plotGsCvResults(ax,scores,fused_epochs)
    plt.show()


