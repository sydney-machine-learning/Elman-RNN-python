# Elman-RNN-Python


Elman-RNN in Python

Description:

    Runs perfectly and output plots of training error and test error over the number of epochs.

    Applied RNN , network with a single hidden layer. The input layer and output layer neurons are decided by looking at the dataset or by the data loader function. 

    Various heuristics can be changed or modified as per requirement of the dataset such as -
    Epoch
    learning rate
    weight decay
    classification yes=1 or no =0
    neurons in hidden layer
    fraction or partition for train and test, (currently set to 0.8)

To run
    write the input file name in main function.
    
    Format the input data to look like
    line 1 contains the number of time steps of the sample
    then each element of the sample, values of the sequence x,
    then corresponding y value on the next line.

    --- sample input for classification problem
    
    10
    0
    1
    1
    0
    0
    0
    1
    1
    0
    1
    0 1

    14
    0
    1
    1
    1
    0
    0
    1
    0
    0
    0
    0
    0
    1
    1
    0 1

    ---- sample input for simple one dimensional 3 length input sequence and output

    3
    0.313276
    0.342539
    0.369255
    0.392381

    3
    0.369255
    0.392381
    0.411436
    0.426266

    Output -- 
    
    typical output in terminal contains
    Epoch number, train rmse, test rmse, time elapsed since the beginning,
    % of correct classifications for train and test (only if classification == 1 )

    Ex -- 
    297  .Train RMSE :  0.5436137743601149 Test RMSE:  0.6253677893731929 Time: 369.80654406547546
    45.5  Train  24.0  Test

Made by - 
    Ashray Aman with a great deal of support and reference from Rohitash Chandra and his github.
    https://github.com/aman17ashray
    https://github.com/rohitash-chandra
    