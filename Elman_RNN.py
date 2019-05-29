# Rohitash Chandra, 2017 c.rohitash@gmail.conm

#!/usr/bin/python

#Feedforward Neural Network in Python (Classification Problem used for teaching and learning purpose)


#Sigmoid units used in hidden and output layer. gradient descent and stocastic gradient descent functions implemented
# this is best for teaching and learning as numpy arrays are not used   in Forward and Backward Pass.
#numpy implymentation with momemtum is given here: https://github.com/rohitash-chandra/VanillaFNN-Python
#corresponding C++ implementation which is generalised to any number of hidden layers is given here:
# https://github.com/rohitash-chandra/feedforward-neural-network

#problems: https://en.wikipedia.org/wiki/XOR_gate, (4 Bit Parity) https://en.wikipedia.org/wiki/Parity_bit
#wine classification: https://archive.ics.uci.edu/ml/datasets/wine

# infor: https://en.wikipedia.org/wiki/Feedforward_neural_network
#http://media.wiley.com/product_data/excerpt/19/04713491/0471349119.pdf


'''

Elman rnn without mcmc, just plain vanilla network with a single hidden layer 
with multiple hidden neurons declared in main function. BPTT and backprop done along with forward prop. hope it works
Hopfully would also work when each sample or each training sequence is a 1D vector. forward prop also works with multidimensional input sequences. not sure about bptt and backprop
'''
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
np.seterr(all='warn')
#np.random.seed(1)


MinimumError = 0.00001
trainsize=299
testsize=99
weightdecay = 0
Epoch = 1000

class Network:

#    def __init__(self, Topo, train_x,train_y,test_x,test_y, MaxTime, LearnRate):
    def __init__(self,LearnRate, Topo, train_x,train_y,test_x,test_y):
        self.Top  = Topo  # NN topology [input, hidden, output]
        #self.Max = MaxTime # max epocs or training time
        self.Train_x = train_x
        self.Train_y = train_y
        self.Test_x = test_x
        self.Test_y = test_y
        #self.NumSamples = Train.shape[0]

        self.learn_rate  = LearnRate # will be updated later with BP call


        #self.minPerf = MinPer
        #initialize weights ( W1 W2 ) and bias ( b1 b2 ) of the network
        #np.random.seed()
        self.W1 = np.random.randn(self.Top[0]  , self.Top[1])
        self.B1 = np.random.randn(self.Top[1])      # bias first layer
        #print(self.B1,' is B1')
        self.BestB1 = self.B1
        self.BestW1 = self.W1
        self.W2 = np.random.randn(self.Top[1] , self.Top[2])
        self.B2 = np.random.randn(self.Top[2])    # bias second layer
        self.BestB2 = self.B2
        self.BestW2 = self.W2
        self.StateW = np.random.randn(self.Top[1] , self.Top[1]) # for feedback
        self.BestStateW = self.StateW
        self.StateOut = np.ones(self.Top[1])
        self.hid_out = np.zeros(self.Top[1]) # output of first hidden layer
        self.hid_delta = np.zeros(self.Top[1]) # gradient of first hidden layer
        self.out = np.zeros(self.Top[2]) #  output last (output) layer
        self.out_delta = np.zeros(self.Top[2]) #  gradient of  output layer
        self.pred_class=0
        self.InlayerOutL0=[]
        self.ErL1 = []
        self.OutputSlideL1=[]
        self.OutputSlideL2=[]
        self.MaxEpoch = Epoch # edit this as an argument to the class function


    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    # def sampleEr(self,actualout):
    #     error = np.subtract(self.out, actualout)
    #     sqerror= np.sum(np.square(error))/self.Top[2]
    #     return sqerror

    def ForwardPass(self, sample,slide):
        sample_time = sample[slide]
        # layersize is topology
        layer = 0 # input to hidden layer
        weightsum = 0.0
        StateWeightSum = 0.0
        end = len(self.Top)-1
        forwardout=0.0

        '''
        just some if else so that if forward prop is called somewhere outside of BPTT
        then this can initialise some important things (important for bptt)
        '''
        if self.InlayerOutL0 == []:
            print('blah')
            quit()
            self.InlayerOutL0 = np.zeros((len(sample)+1,self.Top[0]))
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
            if self.OutputSlideL1 == []:
                print('blah2')
                quit()
                self.OutputSlideL1 = np.zeros((len(sample)+1,self.Top[1]))
            self.OutputSlideL1[slide+1][y] = self.sigmoid(forwardout)
            
            #self.hid_out[y] = self.sigmoid(forwardout)
            weightsum=0
            StateWeightSum=0
            #self.StateOut = np.ones(self.Top[1])


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
        #print(self.out, '          ',end=' ')

        # # for 0 or 1 output from each neuron
        # for out in range(0,self.Top[2]):
        #     self.out[out] = 1 if self.out[out] >= 0.5 else 0
        # print(self.out, end = '')


        #print(self.out, end = '')
        #print(' is self.out')


    #just one going forward for all the input data to find fx
    def evaluate_proposal(self,x):  # BP with SGD (Stocastic BP)
        #self.decode(w)  # method to decode w into W1, W2, B1, B2, StateW.
        #print(self.W1.)
#        size = x.shape[0]
        #print(self.W1)
        #print(self.W2)
        #print(self.StateW)
        Input = x # temp hold input
        #Desired = np.zeros((1, self.Top[2]))
        fx = []
#
        #print(len(x),len(x[0]),' is pat')
        for i,sample in enumerate(x):
#            Input[:] = data[pat, 0:self.Top[0]]
#            Desired[:] = data[pat, self.Top[0]:]
            #self.StateOut = np.random.randn(self.Top[1]) # verify this once with Rohit - Ashray
            self.StateOut = np.ones(self.Top[1])
            self.ErL1 = np.zeros((len(sample)+1,self.Top[1])) # need to modify for multiple layers
            self.OutputSlideL1 = np.zeros((len(sample)+1,self.Top[1]))
            for z in range(0,self.Top[1]):
                self.OutputSlideL1[0][z] = self.StateOut[z]
            self.InlayerOutL0 = np.zeros((len(sample)+1,self.Top[0]))
            self.OutputSlideL2 = np.zeros((len(sample)+1,self.Top[2]))
            
            for slide in range(0,len(sample)):
                self.ForwardPass(sample,slide)
                #print(Input[pat][slide],end=' ')
            temp=np.copy(self.out)
            #print(self.out)
            fx.append(temp)
            #print(fx)
           #print(self.out, ' ', pat)
            #self.StateOut = np.ones(self.Top[1])
        # desired is of no use here i guess so we'll proceed only with input x
        #print(fx, ' is fx')
        #print(fx[0:5])
        return np.array(fx)


    def BackwardPass(self, sample,learnrate, desired,slide): # still remaining to be edited
                # compute gradients for each layer (output and hidden layer)
        sample_time = sample[slide-1]
        layer = 2 #output layer
        temp = 0.0
        sum = 0.0
        Endslide = len(sample)
        for output in range(0,self.Top[2]):
            self.out_delta[output] = 1 * (desired[output]-self.out[output])
        #self.out_delta = desired - self.out # numpy arr vectorised
        #for x in range(0,self.Top[2]):
        layer=1
        for x in range(0,self.Top[1]):
            temp=0.0
            for y in range(0,self.Top[2]):
                temp += self.out_delta[y] * self.W2[x][y]
            #print(self.Er, Endslide,x)
            #print(self.OutputSlide[Endslide][x], temp)
            # if math.isnan(temp):
            #     print(self.out_delta, self.W2[x][y])
            #     quit()
            self.ErL1[Endslide][x] = self.OutputSlideL1[Endslide][x]*(1-self.OutputSlideL1[Endslide][x])*temp
            # if math.isnan(self.ErL1[Endslide][x]):
            #     print('ashray',self.OutputSlideL1[Endslide][x],sample_time,temp)
            #     quit()
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
        
        seeda=1.0
        tmpoo=0.0
        for x in range(0,self.Top[1]):
            for y in range(0,self.Top[2]):
                tmpoo = ((seeda * self.learn_rate * self.out_delta[y] * self.OutputSlideL1[Endslide][x]))
                self.W2[x][y] += tmpoo - (tmpoo * weightdecay)
        seeda=0.0


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
            #print(tmp1, weightdecay,self.B1[y])
            self.B1[y] += tmp1 - (tmp1 * weightdecay)







    # to train the weights
    def BPTT(self): # BP with Vanilla or SGD (Stocastic BP)
        # assuming topo as [input,hidden,out]
        #Input = np.zeros((1, self.Top[0])) # temp hold input
        #Desired = np.zeros((1, self.Top[2]))
        #Er = []#np.zeros((1, self.Max))
        #epoch = -1
        #bestmse = 100
        #bestTrain = 0
        # while  epoch < self.MaxEpoch:
        #     epoch+=1
        #     print(epoch)
        for i,sample in enumerate(self.Train_x):
            # each sample is a training data
            #sse=0
            self.StateOut = np.ones(self.Top[1])
            self.ErL1 = np.zeros((len(sample)+1,self.Top[1])) # need to modify for multiple layers
            self.OutputSlideL1 = np.zeros((len(sample)+1,self.Top[1]))
            for x in range(0,self.Top[1]):
                self.OutputSlideL1[0][x] = self.StateOut[x]
            self.InlayerOutL0 = np.zeros((len(sample)+1,self.Top[0]))
            self.OutputSlideL2 = np.zeros((len(sample)+1,self.Top[2]))
            for slide in range(0,len(sample)):
                self.ForwardPass(sample,slide)
            #print(self.OutputSlide)
            #quit()
            #print(self.out, end = '')
            for slide in range(len(sample),0,-1):
                self.BackwardPass(sample,self.learn_rate,self.Train_y[i-1],slide)
        #return (epoch)

    def decode(self,w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]
        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))
        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]].reshape(self.Top[1])
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]].reshape(self.Top[2])
        w_state = w[w_layer1size + w_layer2size + self.Top[1] + self.Top[2]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]+self.Top[1]*self.Top[1]]
        self.StateW = np.reshape(w_state,(self.Top[1],self.Top[1]))
        #print(self.B1,' after decode')
        #print(self.W1.shape,self.W2.shape,self.StateW.shape,self.B1.shape,self.B2.shape, ' is shape')


    def encode(self):
        w1 = self.W1.ravel()
        w1=w1.reshape(1,w1.shape[0])
        w2 = self.W2.ravel()
        w2=w2.reshape(1,w2.shape[0])
        w3 = self.StateW.ravel()
        w3=w3.reshape(1,w3.shape[0])
        w=np.concatenate([w1.T,w2.T,self.B1.T,self.B2.T,w3.T])
        w=w.reshape(-1)
        return w


# def normalisedata(data, inputsize, outsize): # normalise the data between [0,1]. This is important for most problems.
#     traindt = data[:,np.array(range(0,inputsize))]
#     dt = np.amax(traindt, axis=0)
#     tds = abs(traindt/dt)
#     return np.concatenate(( tds[:,range(0,inputsize)], data[:,range(inputsize,inputsize+outsize)]), axis=1)

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


        problem = 1 # [1,2,3] choose your problem (Iris classfication or 4-bit parity or XOR gate)
        
        #for mackey
        fname = "train_embed.txt"
        x,y = data_loader(fname)
        #print_data(x,y)
        num_samples = len(x)
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


        if problem == 1:
            #TrDat  = np.loadtxt("train.csv", delimiter=',') #  Iris classification problem (UCI dataset)
            #TesDat  = np.loadtxt("test.csv", delimiter=',') #
            Hidden = 5
            TrSamples =  110
            TestSize = 40
            #learnRate = 0.5
            #TrainData  = normalisedata(TrDat, Input, Output)
            #TestData  = normalisedata(TesDat, Input, Output)
            #MaxTime = 500
            #MinCriteria = 95 #stop when learn 95 percent



        #print(TrainData)
        print('printed data. now we use RNN for training ...')
        Topo = [Input, Hidden, Output]
        print(Topo)
        Epoch=500
        MaxRun = Epoch # number of experimental runs



##        trainTolerance = 0.2 # [eg 0.15 would be seen as 0] [ 0.81 would be seen as 1]
#        testTolerance = 0.49



#        trainPerf = np.zeros(MaxRun)
#        testPerf =  np.zeros(MaxRun)

#        trainMSE =  np.zeros(MaxRun)
#        testMSE =  np.zeros(MaxRun)
#        Epochs =  np.zeros(MaxRun)
#        Time =  np.zeros(MaxRun)#\

#        stocastic = 1 # 0 if vanilla (batch mode)
        learnRate = 0.01
        start_time=time.time()
        rnn_net = Network(learnRate,Topo,train_x,train_y,test_x,test_y)        
        # for run in range(0, MaxRun  ):
        #         print(run, 'is the experimental run')         
        #         #(erEp,  trainMSE[run] , trainPerf[run] , Epochs[run]) = fnnSGD.BP_GD( stocastic, trainTolerance )
        #         rnn_net.BPTT()
        #         Time[run]  =time.time()-start_time
        #         (testMSE[run], testPerf[run]) = fnnSGD.TestNetwork(TestData, testTolerance)

        trainErr=[]
        for run in range(0,MaxRun):
            print(run, ' is the run.', end = '')
            #print(rnn_net.B1)
            #print(rnn_net.B1.shape)
            rnn_net.BPTT()
            trainfx = rnn_net.evaluate_proposal(train_x)
            testfx = rnn_net.evaluate_proposal(test_x)
            #print(train_x[0:5],trainfx[0:5],train_y[0:5])
            err_trainmse = np.sqrt(((np.array(trainfx) - np.array(train_y)) ** 2).mean())
            err_testmse = np.sqrt(((np.array(testfx) - np.array(test_y)) ** 2).mean())
            print(err_trainmse , ' is train error',end= ' ')
            print(err_testmse,' is test error')
            trainErr.append(err_trainmse)
        # print (trainPerf, 'train perf % for n exp')
        # print (testPerf,  'test  perf % for n exp')
        # print (trainMSE,  'train mean squared error for n exp')
        # print (testMSE,  'test mean squared error for n exp')

        # print ('mean and std for training perf %')
        # print(np.mean(trainPerf), np.std(trainPerf))

        # print ('mean and std for test perf %')
        # print(np.mean(testPerf), np.std(testPerf))

        # print ('mean and std for time in seconds')
        # print(np.mean(Time), np.std(Time))



        plt.figure()
        plt.plot(np.array(trainErr))
        plt.ylabel('error')
        plt.savefig('error_out.png')


if __name__ == "__main__": main()
