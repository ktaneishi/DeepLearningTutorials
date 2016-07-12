# Benchmarks
This repository contains representative benchmark used in Deep Learning field.
LSTM-Sentiment Analysis:
    The LSTM model is used to perform sentiment analysis on movie reviews from 
    the Large Movie Review Dataset, sometimes known as the IMDB dataset. 

    Refer to "LSTM-Sentiment_Analysis/run.sh" for running the workload.


DBN-Kyoto:
    A drug discovery workload using DBN model developed by Kyoto University. 
    This algorithm is used to predict if the chemical structure of a compound
    can interact with the protein sequence. Deep learning neural network can
    resolve the bottleneck of exponential increase in the calculation time
    and memory consumption encountered in SVM algorithm.

    Refer to "DBN-Kyoto/run.sh" for running the workload.

AlexNet-CPU:
    AlexNet was the first work that popularized Convolutional Networks in 
    Computer. Please refer to https://github.com/uoguelph-mlrg/theano_alexnet
    for the original alexnet for theano. Since the original version only
    supports GPU, we update the codes to support CPU, and so called AlexNet-CPU.
    
    Refer to "Alexnet_CPU/run.sh" for running the workload.
