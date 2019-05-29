# Rohitash Chandra & Ashray Aman, 2019 c.rohitash@gmail.conm

#!/usr/bin/python

#Recurrent Neural Network in Python (used for teaching and learning purpose)
#also works great for multi dimensional inputs and output (such as used in classification problems)

# Sigmoid units used in hidden and output layer.
# this is best for teaching and learning as only numpy package is used in some places for easier calculations and none of the other
# python machine leanring packages has been used Forward and Backward Pass.

# corresponding C++ implementation is given here : 
# https://github.com/rohitash-chandra/VanillaElmanRNN

# some relevant resources for implementation of feedforward NN is mentioned below
# numpy implymentation with momemtum is given here: https://github.com/rohitash-chandra/VanillaFNN-Python
# corresponding C++ implementation which is generalised to any number of hidden layers is given here:
# https://github.com/rohitash-chandra/feedforward-neural-network



import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
# to seed
# np.random.seed(1)


weightdecay = 0.01
Epoch = 100 # default value if not changed in main

class Network:

    def __init__(self,LearnRate, Topo, train_x,train_y,test_x,test_y):
        self.Top  = Topo  # NN topology [input, hidden, output]
        self.Train_x = train_x
        self.Train_y = train_y
        self.Test_x = test_x
        self.Test_y = test_y
        self.learn_rate  = LearnRate
        #initialize weights ( W1 W2 ) and bias ( b1 b2 ) of the network
        self.W1 = np.random.randn(self.Top[0]  , self.Top[1])
        self.B1 = np.random.randn(self.Top[1])      # bias first layer
        self.W2 = np.random.randn(self.Top[1] , self.Top[2])
        self.B2 = np.random.randn(self.Top[2])    # bias second layer
        self.StateW = np.random.randn(self.Top[1] , self.Top[1]) # for feedback
        self.StateOut = np.ones(self.Top[1])
        self.hid_out = np.zeros(self.Top[1]) # output of first hidden layer
        self.hid_delta = np.zeros(self.Top[1]) # gradient of first hidden layer
        self.out = np.zeros(self.Top[2]) #  output last (output) layer
        self.out_delta = np.zeros(self.Top[2]) #  gradient of  output layer
        self.pred_class=0
        self.InlayerOutL0=[]    # to store output from the first layer which is the input itself
        self.ErL1 = [] # to store gradients or error for layer 1 which is hidden layer
        self.OutputSlideL1=[] # to store output corresponding to each ith element(x[0..i]) in the input sequence
        self.OutputSlideL2=[] # to store output from layer 2 output layer for x[0..i]
        self.MaxEpoch = Epoch # edit this as an argument to the class function


    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))


    def ForwardPass(self, sample,slide):
        sample_time = sample[slide]
        layer = 0 
        weightsum = 0.0
        StateWeightSum = 0.0
        forwardout=0.0
        # if self.InlayerOutL0 == []:
        #     print('blah')
        #     quit()
        #     self.InlayerOutL0 = np.zeros((len(sample)+1,self.Top[0]))
        for row in range(0,self.Top[0]):
            self.InlayerOutL0[slide+1][row] = sample_time[row]

        for y in range(0, self.Top[1]):
            for x in range(0, self.Top[0]):
                #print(sample_time)
                #print()
                weightsum += self.InlayerOutL0[slide+1][x] * self.W1[x,y]
            #weightsum = 0

            for x in range(0,self.Top[1]):
                StateWeightSum += self.OutputSlideL1[slide][x] * self.StateW[x,y]
            #print(weightsum,StateWeightSum,self.B1[y],y)
            #print(self.B1)
            #self.StateOut[y] = (weightsum + StateWeightSum) - self.B1[y]
            forwardout = (weightsum + StateWeightSum) - self.B1[y]
            # if self.OutputSlideL1 == []:
            #     print('blah2')
            #     quit()
            #     self.OutputSlideL1 = np.zeros((len(sample)+1,self.Top[1]))
            self.OutputSlideL1[slide+1][y] = self.sigmoid(forwardout)            
            weightsum=0
            StateWeightSum=0


        layer = 1 #   hidden layer to output
        weightsum = 0.0
        #print(self.out,end=' ')
        for y in range(0, self.Top[layer+1]):
            for x in range(0, self.Top[layer]):
                weightsum  +=   self.OutputSlideL1[slide+1][x] * self.W2[x,y]
                forwardout = (weightsum - self.B2[y])
            self.OutputSlideL2[slide+1][y] = self.sigmoid(forwardout) 
            weightsum = 0.0
            StateWeightSum=0.0
        self.pred_class = max(self.out)#added new for classification
        #print(self.out, 'is out')


        # for regression type
        for i in range(0,self.Top[2]):
            self.out[i] = self.OutputSlideL2[slide+1][i]
        #for i in range(0,self.Top[2]):
        #   self.out[i] = 1 if self.OutputSlideL2[slide+1][i] > 0.5 else 0 


    #just one going forward for all the input data to find fx
    def evaluate_proposal(self,x):
        fx = []
        for i,sample in enumerate(x):
            self.StateOut = np.ones(self.Top[1])
            self.ErL1 = np.zeros((len(sample)+1,self.Top[1])) # need to modify for multiple layers
            self.OutputSlideL1 = np.zeros((len(sample)+1,self.Top[1]))
            for z in range(0,self.Top[1]):
                self.OutputSlideL1[0][z] = self.StateOut[z]
            self.InlayerOutL0 = np.zeros((len(sample)+1,self.Top[0]))
            self.OutputSlideL2 = np.zeros((len(sample)+1,self.Top[2]))
            
            for slide in range(0,len(sample)):
                self.ForwardPass(sample,slide)
            temp=np.copy(self.out)
            #print(self.out)
            fx.append(temp)
        #print(fx[0:5])
        return np.array(fx)


    def BackwardPass(self, sample,learnrate, desired,slide): 
    # compute gradients for each layer (output and hidden layer)
        temp = 0.0
        sum = 0.0
        Endslide = len(sample)
        for output in range(0,self.Top[2]):
            self.out_delta[output] = 1 * (desired[output]-self.out[output])

        for x in range(0,self.Top[1]):
            temp=0.0
            for y in range(0,self.Top[2]):
                temp += self.out_delta[y] * self.W2[x][y]
            self.ErL1[Endslide][x] = self.OutputSlideL1[Endslide][x]*(1-self.OutputSlideL1[Endslide][x])*temp
            temp=0.0

        for x in range(0,self.Top[1]):
            for y in range(0,self.Top[1]):
                sum+=self.ErL1[slide][y] * self.StateW[x][y]
            self.ErL1[slide-1][x] = self.OutputSlideL1[slide-1][x]*(1-self.OutputSlideL1[slide-1][x]) * sum
            sum=0.0
        sum=0.0


        '''
        for weight update
        '''
        # below is weight update for W1

        tmp=0.0
        for x in range(0,self.Top[0]):
            for y in range(0,self.Top[1]):
                tmp = (self.learn_rate * self.ErL1[slide][y] * self.InlayerOutL0[slide][x]) # weight change
                self.W1[x][y] += tmp - (tmp * weightdecay)
        


        # for W2
        seeda=1.0
        tmpoo=0.0
        for x in range(0,self.Top[1]):
            for y in range(0,self.Top[2]):
                tmpoo = ((seeda * self.learn_rate * self.out_delta[y] * self.OutputSlideL1[Endslide][x]))
                self.W2[x][y] += tmpoo - (tmpoo * weightdecay)
        seeda=0.0

        # for hidden weights or state weights
        tmp2=0.0
        for x in range(0,self.Top[1]):  # for inner layer
            for y in range(0,self.Top[1]): # for outer layer
                tmp2 = ((self.learn_rate * self.ErL1[slide][y] * self.OutputSlideL1[slide-1][x]))
                self.StateW[x][y] += tmp2 - (tmp2 * weightdecay)

        # now updating the bias
        topbias = 0.0
        seed=1.0
        for y in range(0,self.Top[2]):
            topbias = ((seed * -1 * self.learn_rate * self.out_delta[y]))
            self.B2[y] += topbias - (topbias * weightdecay)
            topbias=0.0
        
        topbias=0.0
        seed=0
        tmp1=0.0
        for y in range(0,self.Top[1]):
            tmp1 = (-1 * self.learn_rate * self.ErL1[slide][y])
            self.B1[y] += tmp1 - (tmp1 * weightdecay)


    # to train the weights
    def BPTT(self): 
        # assuming topo as [input,hidden,out]
        for i,sample in enumerate(self.Train_x):
            # each sample is a training data
            self.StateOut = np.ones(self.Top[1])
            self.ErL1 = np.zeros((len(sample)+1,self.Top[1])) # need to modify for multiple layers
            self.OutputSlideL1 = np.zeros((len(sample)+1,self.Top[1]))
            for x in range(0,self.Top[1]):
                self.OutputSlideL1[0][x] = self.StateOut[x]
            self.InlayerOutL0 = np.zeros((len(sample)+1,self.Top[0]))
            self.OutputSlideL2 = np.zeros((len(sample)+1,self.Top[2]))
            for slide in range(0,len(sample)):
                self.ForwardPass(sample,slide)
            for slide in range(len(sample),0,-1):
                self.BackwardPass(sample,self.learn_rate,self.Train_y[i-1],slide)


def data_loader(filename):
    f=open(filename,'r')
    x=[[[]]]
    count=0
    y=[[]]
    while(True):
        count+=1
        #print(count)
        text = f.readline()
        #print(text)
        if(text==''):
           break
        if(len(text.split()) == 0):
            #print(text)
            text=f.readline()
        if(text==''):
           break
        #print(text)
        t=int(text)
        a=[[]]
        ya=[]
        for i in range(0,t):
            temp=f.readline().split(' ')
            b=[]
            for j in range(0,len(temp)):
                b.append(float(temp[j]))
            a.append(b)
        del a[0]
        x.append(a)
        temp=f.readline().split(' ')
        #print(temp)
        for j in range(0,len(temp)):
            if temp[j] != "\n":
                ya.append(float(temp[j]))
        y.append(ya)
    del x[0]
    del y[0]
    return x,y

def print_data(x,y):
    # assuming x is 3 dimensional and y is 2 dimensional
    for i in range(0,len(x)):
        for j in range(0,len(x[i])):
            print(x[i][j])
        print(y[i])
        print(' ')

def loadersunspot(fname):
    f = open(fname,'r')
    x=[[]]
    count=0
    y=[]
    while(True):
        count+=1
        #print(count)
        text = f.readline()
        #print(text)
        if(text==''):
           break
        if(len(text.split()) == 0):
            #print(text)
            text=f.readline()
        if(text==''):
           break
        #print(text)
        a=[]
        for i in range(0,len(text.split(' '))-1):
            #print(text.split(' ')[i].strip())
            temp = float(text.split(' ')[i].strip())
            a.append([temp])
        y.append([float(text.split(' ')[-1].strip())])
        if a[0] == []:
            del a[0]
        x.append(a)
        #print(count)
    if (x[0]) == [] or x[0] == [[]] :
        del x[0]
    if y[0] == [] or y[0] == [[]]:
        del y[0]
    return x,y



def main():


        #for mackey
        fname = "train_embed.txt"
        x,y = data_loader(fname)
        #print_data(x,y)
        train_x= x[:int(len(x)*0.8)]
        test_x=x[int(len(x)*0.8):]
        train_y= y[:int(len(y)*0.8)]
        test_y=y[int(len(y)*0.8):]
        Input = len(train_x[0][0])
        Output = len(train_y[0])


        # # for sunspot
        # fname = "trainsunspot.txt"
        # x,y = loadersunspot(fname)
        # Input=1
        # Output = 1
        # print(x[10],y[10])
        # train_x= x[:int(len(x)*0.8)]
        # test_x=x[int(len(x)*0.8):]
        # train_y= y[:int(len(y)*0.8)]
        # test_y=y[int(len(y)*0.8):]

        Hidden = 5
        


        #print('printed data. now we use RNN for training ...')
        Topo = [Input, Hidden, Output]
        print(Topo)
        Epoch=300
        MaxRun = Epoch # number of experimental runs

        learnRate = 0.01
        start_time=time.time()
        rnn_net = Network(learnRate,Topo,train_x,train_y,test_x,test_y)        
        trainErr=[] # storing the error for plotting
        testErr=[]
        for run in range(0,MaxRun):
            print('---------------------------')
            print(run, ' .', end = '')
            rnn_net.BPTT()
            trainfx = rnn_net.evaluate_proposal(train_x)
            testfx = rnn_net.evaluate_proposal(test_x)
            #print(test_x[0:5],test_y[0:5])
            err_trainmse = np.sqrt(((np.array(trainfx) - np.array(train_y)) ** 2).mean())
            err_testmse = np.sqrt(((np.array(testfx) - np.array(test_y)) ** 2).mean())
            print('Train RMSE : ', err_trainmse, end= ' ')
            print('Test RMSE: ',err_testmse,end='')
            print(' Time:',time.time()-start_time)
            trainErr.append(err_trainmse)
            testErr.append(err_testmse)
        
        # to plot the error
        plt.figure()
        plt.plot(np.array(trainErr), label = 'train')
        plt.plot(np.array(testErr), label = 'test')
        plt.legend()
        #plt.ylabel('error on train and test')
        plt.savefig('error_out.png')


if __name__ == "__main__": main()
