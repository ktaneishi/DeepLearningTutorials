DBN-Kyoto (Deep Belief Network for Drug Discovery)
==================================================

This is a drug discovery workload using Deep Belief Network (DBN) model.
This algorithm is used to predict if the chemical structure of a compound
can interact with the protein sequence.
Deep Learning resolve the bottleneck of exponential increase in the calculation time
and memory consumption encountered in SVM algorithm.

These scripts require the optimized [Theano by Intel](https://github.com/intel/theano).
The results were reported in the paper _Prediction of Compound-protein Interactions Based on Deep Learning_, Mol Inform. 2016 Aug 12.

Dependency
----------

- Theano (pip install theano)

Files
-----

- dbn.py  
Definition of Deep Belief Network.

- main.py
Main script.

- benchmark.py
Benchmark script.

- cpi.npz  
Sample data including about 250,000 of compound protein interactions.  
Download from https://my.syncplicity.com/share/vvks9oqxas1xneg/cpi

Usage
-----

    $ python benchmark.py
