-requirements.txt: list of all the python libraries that the application rely on.

-data/: directory containing the available dataset (data/anonymized).

-unit_test/: directory containing unittest code used to verify part of the implementation.
From the root directory of the project run: python -m unittest unit_test/module_name.py to run a specific test suite.

-src/: directory containing the implementation code.
Files:
    -records.py: read and manage records read from .mat file
    -preproc.py: preprocess data and manage labeled epochs extraction
    -feature.py: extract features from epochs, dump/read feature's space representation on/from file etc...
    -classification.py: hyperparameter search and performance evaluation of the classifier
    -plot.py: utilities to visualize data and results

-*.pkl: pickle dump of a dataset as a set of points in feature's space representation. fusen: means that n subsequent
epochs were joined to a central at the time the dataset was dumped.

-analysis.py: driver code to inspect dataset (used for initial debug and preliminary data analysis) -- useless for classification
purpouses.
Usage: python analysis.py [--option value]. Run python analysis.py --help for details.

-main.py: main driver code for classification.
Usage: python main.py --option [value]. Run python main.py without options for details.
Disclaimer:

    -for the number of subsequent epochs to join for hrv features computation a simplified version of the paper strategy was adopted.
    Only even number of epochs were considered and they were joined to the central one. 
    (half on the left,half on the right -- enlarging window)

    -convergence parameter `tolerance` in SVC is set to 1e-2 (an order of magnitude bigger than default) for the leave-one-out 
    evaluation to trade-off a faster execution with a worse accuracy while software is tested. 
    During the presentation results with proper values of convergence will be presented.

    -loo and correlation analysis work both on the feature's space dump of the entire dataset to avoid the computation of
    features on the same set every time. A file dset_fuse14.pkl is provided but different dataset could be dumped with
    python main.py --dump n where n indicates the number of epochs to join for hrv features computation.


    
