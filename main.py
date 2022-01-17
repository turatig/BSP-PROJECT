import src.records as rec
import src.preproc as pre
import src.feature as feat
import src.classification as clas
import src.plot as plot
import matplotlib.pyplot as plt

if __name__=="__main__":
    scores=clas.neighbourEpochSelection("data/anonymized",verb=True)
    #only even number of epochs are fused with one -- half on the left, half on the right of the labeled epoch
    fused_epochs=[ i for i in range(2,5) if not i%2 ]
    fig,ax=plt.subplots()

    plot.plotGsCvResults(ax,scores,fused_epochs)
    plt.show()


