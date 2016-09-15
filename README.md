Deep Belief Network for QSAR (DBN-Kyoto)
========================================

A drug discovery workload using Deep Belief Network (DBN) model.
This algorithm is used to predict if the chemical structure of a compound
can interact with the protein sequence. Deep learning neural network can
resolve the bottleneck of exponential increase in the calculation time
and memory consumption encountered in SVM algorithm.

Refer to "DBN-Kyoto/run.sh" for running the workload.

This script was used as the workload to optimize 
Theano to Intel MIC architecture at https://github.com/intel/theano.
And the older version is also used as 
the benckmark script in https://github.com/pcs-theano/Benchmarks.
The results were reported in the paper 
_Prediction of Compound-protein Interactions Based on Deep Learningin_, Mol Inform. 2016 Aug 12.

Dependency
----------

- Theano (pip install theano)

Files
-----

- dbn.py  
Template script for parameter search.

- cpi.npz  
Sample data including 250,000 of compound protein interactions.  
Download from https://my.syncplicity.com/share/vvks9oqxas1xneg/cpi
