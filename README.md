# Music_Genre_Classification

Implemented Genre classification using the GTZAN dataset. Built a deep CNN model with
3 2D convolutional layers, 1 flatten, 1 dense layer in Keras. 
I optimized the model by incorporating batch normalization, max pooling and Adam(speeding up gradient descent).
I used sparse categorical cross entropy loss function and accuracy metric with (20,20,60) percent (test,validation,training) set
respectively with scikit-learn.

Used librosa, numpy, scipy for preprocessing(MFCC feature vectors were fed into the network for training). Developed deeper insights into
audio signal processing ideas (window size, hop size, number of FFT, block processing). Achieved 84 percent accuracy with the network.
Used json formats(read/write) for storing compressed data for training

How can I improve?
1) Data Augmentation : Using pitch shifting, time stretching and other methods to increase the amount of data
2) Complex architecture : Using bidirectional LSTM layers or adding more number of hidden layers, training for 
   more number of epochs, better optimization techniques
3) Signal processing, Trying different features for training and/or combining different features(chroma, spectrograms etc)


