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





import matplotlib.pyplot as plt
import numpy as np
import random
import time

MinimumError = 0.00001
trainsize=299
testsize=99
trainfile = "train_embed.txt"
testfile = "test_embed.txt"

#class Layers:
#    Weights
#    ContextWeight
#    WeightChange
#
#    TransitionProb
#
#    RadialOutput
#    OutputLayer
#    Bias
#    BiasChange
#    Error
#
#    Mean
#    StanDev
#
#    MeanChange
#    StanDevChange
#




class Network:

    def __init__(self, Topo, train_x,train_y,test_x,test_y, MaxTime, LearnRate, MinPer):
        self.Top  = Topo  # NN topology [input, hidden, output]
        self.Max = MaxTime # max epocs or training time
        self.Train_x = train_x
        self.Train_y = train_y
        self.Test_x = test_x
        self.Test_y = test_y
        self.NumSamples = Train.shape[0]

        self.learn_rate  = LearnRate # will be updated later with BP call


        self.minPerf = MinPer
                                        #initialize weights ( W1 W2 ) and bias ( b1 b2 ) of the network
        np.random.seed()
        self.W1 = np.random.randn(self.Top[0]  , self.Top[1])
        self.B1 = np.random.randn(self.Top[1])      # bias first layer
        self.BestB1 = self.B1
        self.BestW1 = self.W1
        self.W2 = np.random.randn(self.Top[1] , self.Top[2])
        self.B2 = np.random.randn(self.Top[2])    # bias second layer
        self.BestB2 = self.B2
        self.BestW2 = self.W2
        self.StateW = np.random.randn(self.Top[1],self.Top[1]) # for feedback
        self.BestStateW = self.StateW
        self.StateOut = np.zeros(self.top[1])
        self.hid_out = np.zeros(self.Top[1]) # output of first hidden layer
        self.hid_delta = np.zeros(self.Top[1]) # gradient of first hidden layer
        self.out = np.zeros(self.Top[2]) #  output last (output) layer
        self.out_delta = np.zeros(self.Top[2]) #  gradient of  output layer


    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sampleEr(self,actualout):
        error = np.subtract(self.out, actualout)
        sqerror= np.sum(np.square(error))/self.Top[2]
        return sqerror

    def ForwardPass(self, sample_time):
        # layersize is topology
        layer = 0 # input to hidden layer
        weightsum = 0
        StateWeightSum=0
        end = len(self.Top)-1
        for y in range(0, self.Top[layer+1]):
            for x in range(0, self.Top[layer]):
                weightsum  +=   sample_time[x] * self.W1[x,y]
            weightsum = 0
            for x in range(0,self.Top[layer+1]):
                StateWeightSum += self.StateOut[x]*self.StateW[x,y]
            self.StateOut[y] = weightsum + StateWeightSum - self.B1[y]
            self.hid_out[y] = self.sigmoid(self.StateOut[y])
            weightsum=0
            StateWeightSum=0


        layer = 1 #   hidden layer to output
        weightsum = 0
        for y in range(0, self.Top[layer+1]):
            for x in range(0, self.Top[layer]):
                weightsum  +=   self.hid_out[x] * self.W2[x,y]
            self.out[y] = self.sigmoid(weightsum - self.B2[y])
            weightsum = 0

#'''
#	for (y = 0; y < Layersize[layer + 1]; y++) {
#		for (x = 0; x < Layersize[layer]; x++) {
#			WeightedSum += (nLayer[layer].Outputlayer[slide + 1][x]
#					* nLayer[layer].Weights[x][y]);
#		}
#		for (x = 0; x < Layersize[layer + 1]; x++) {
#			ContextWeightSum += (nLayer[1].Outputlayer[slide][x]
#					* nLayer[1].ContextWeight[x][y]); // adjust this line when use two hidden layers.
#			//
#		}
#
#		ForwardOutput = (WeightedSum + ContextWeightSum)
#				- nLayer[layer + 1].Bias[y];
#		nLayer[layer + 1].Outputlayer[slide + 1][y] = SigmoidS(ForwardOutput);
#		// cout<<ForwardOutput<<endl;
#		//getchar();
#		WeightedSum = 0;
#		ContextWeightSum = 0;
#	}
#'''
    def BackwardPass(self, input, desired ):
                # compute gradients for each layer (output and hidden layer)

        layer = 2 #output layer
        for x in range(0, self.Top[layer]):
            self.out_delta[x] =  (desired[x] - self.out[x])*(self.out[x]*(1-self.out[x]))

        layer = 1 # hidden layer
        temp = 0
        for x in range(0, self.Top[layer]):
            for y in range(0, self.Top[layer+1]):
                temp += ( self.out_delta[y] * self.W2[x,y]);
                self.hid_delta[x] =  (self.hid_out[x] * (1 - self.hid_out[x])) * temp
                temp = 0

                # update weights and bias
        layer = 1 # hidden to output

        for x in range(0, self.Top[layer]):
            for y in range(0, self.Top[layer+1]):
                    self.W2[x,y] += self.learn_rate * self.out_delta[y] * self.hid_out[x]
            #print self.W2
        for y in range(0, self.Top[layer+1]):
                self.B2[y] += -1 * self.learn_rate * self.out_delta[y]

        layer = 0 # Input to Hidden
        for x in range(0, self.Top[layer]):
                for y in range(0, self.Top[layer+1]):
                    self.W1[x,y] += self.learn_rate * self.hid_delta[y] * input[x]

        for y in range(0, self.Top[layer+1]):
                self.B1[y] += -1 * self.learn_rate * self.hid_delta[y]

    def TestNetwork(self, Data,  erTolerance):

        clasPerf = 0
        sse = 0
        self.W1 = self.BestW1
        self.W2 = self.BestW2 #load best knowledge
        self.B1 = self.BestB1
        self.B2 = self.BestB2 #load best knowledge

        testSize = Data.shape[0]

        for s in range(0, testSize):

                Input  =   Data[s,0:self.Top[0]]
                Desired =  Data[s,self.Top[0]:]

                self.ForwardPass(Input )
                sse = sse+ self.sampleEr(Desired)


                if(np.isclose(self.out, Desired, atol=erTolerance).any()):
                   clasPerf =  clasPerf +1

        return ( sse/testSize, float(clasPerf)/testSize * 100 )


    def saveKnowledge(self):
        self.BestW1 = self.W1
        self.BestW2 = self.W2
        self.BestB1 = self.B1
        self.BestB2 = self.B2

    def BPTT(self, stocastic, trainTolerance,learningRate): # BP with Vanilla or SGD (Stocastic BP)

        Input = np.zeros((1, self.Top[0])) # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        Er = []#np.zeros((1, self.Max))
        epoch = -1
        bestmse = 100
        bestTrain = 0
        while  epoch < self.Max and bestTrain < self.minPerf :
            epoch+=1
            for samples in self.TrainData:
                # each sample is a training data
                sse=0
                for slide in range(0,len(samples)):
                    self.ForwardPass(samples[slide],slide)
                for slide in range(len(samples),0,-1):
                    self.BackwardPass(samples,)
                    '''
                    to edit
                    '''
#            sse = 0
#            for s in range(0, self.NumSamples):
#
#                if(stocastic):
#                   pat = random.randint(0, self.NumSamples-1)
#                else:
#                   pat = s
#
#                Input   =  self.TrainData[pat,0:self.Top[0]]
#                Desired  = self.TrainData[pat,self.Top[0]:]
#
#
#
#                self.ForwardPass(Input )
#                self.BackwardPass(Input , Desired)
#                sse = sse+ self.sampleEr(Desired)
#
#            mse = np.sqrt(sse/self.NumSamples*self.Top[2])
#
#            if mse < bestmse:
#               bestmse = mse
#               self.saveKnowledge()
#               (x,bestTrain) = self.TestNetwork(self.TrainData,  trainTolerance)
#
#
#            Er = np.append(Er, mse)
#
#
#            epoch=epoch+1

        return (Er,bestmse, bestTrain, epoch)



def normalisedata(data, inputsize, outsize): # normalise the data between [0,1]. This is important for most problems.
    traindt = data[:,np.array(range(0,inputsize))]
    dt = np.amax(traindt, axis=0)
    tds = abs(traindt/dt)
    return np.concatenate(( tds[:,range(0,inputsize)], data[:,range(inputsize,inputsize+outsize)]), axis=1)

def data_loader(filename):
    f=open("data_1.txt",'r')
    x=[[[]]]
    y=[[]]
    while(True):
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

def main():


        problem = 1 # [1,2,3] choose your problem (Iris classfication or 4-bit parity or XOR gate)
        fname = "data_1.txt"
        x,y = data_loader(fname)
        num_samples = len(x)
        train_x= x[:int(len(x)*0.8)]
        test_x=x[int(len(x)*0.8):]
        train_y= y[:int(len(y)*0.8)]
        test_y=y[int(len(y)*0.8):]
        if problem == 1:
            #TrDat  = np.loadtxt("train.csv", delimiter=',') #  Iris classification problem (UCI dataset)
            #TesDat  = np.loadtxt("test.csv", delimiter=',') #
            Hidden = 4
            Input = 4
            Output = 2
            TrSamples =  110
            TestSize = 40
            learnRate = 0.1
            #TrainData  = normalisedata(TrDat, Input, Output)
            #TestData  = normalisedata(TesDat, Input, Output)
            MaxTime = 500
            MinCriteria = 95 #stop when learn 95 percent



#
#        if problem == 2:
#        TrainData = np.loadtxt("4bit.csv", delimiter=',') #  4-bit parity problem
#           TestData = np.loadtxt("4bit.csv", delimiter=',') #
#         Hidden = 4
#           Input = 4
#           Output = 1
#           TrSamples =  16
#           TestSize = 16
#           learnRate = 0.9
#           MaxTime = 3000
#           MinCriteria = 95 #stop when learn 95 percent
#
#        if problem == 3:
#        TrainData = np.loadtxt("xor.csv", delimiter=',') #  4-bit parity problem
#           TestData = np.loadtxt("xor.csv", delimiter=',') #
#           Hidden = 3
#           Input = 2
#           Output = 1
#           TrSamples =  4
#           TestSize = 4
#           learnRate = 0.9
#           MaxTime = 500
#           MinCriteria = 100 #stop when learn 95 percent

        print(TrainData)
        print('printed data. now we use FNN for training ...')




        Topo = [Input, Hidden, Output]
        MaxRun = 10 # number of experimental runs



        trainTolerance = 0.2 # [eg 0.15 would be seen as 0] [ 0.81 would be seen as 1]
        testTolerance = 0.49



        trainPerf = np.zeros(MaxRun)
        testPerf =  np.zeros(MaxRun)

        trainMSE =  np.zeros(MaxRun)
        testMSE =  np.zeros(MaxRun)
        Epochs =  np.zeros(MaxRun)
        Time =  np.zeros(MaxRun)

        stocastic = 1 # 0 if vanilla (batch mode)

        for run in range(0, MaxRun  ):
                 print(run, 'is the experimental run')
                 fnnSGD = Network(Topo, train_x,train_y,test_x,test_y, MaxTime, learnRate, MinCriteria)
                 start_time=time.time()
                 (erEp,  trainMSE[run] , trainPerf[run] , Epochs[run]) = fnnSGD.BP_GD( stocastic, trainTolerance )

                 Time[run]  =time.time()-start_time
                 (testMSE[run], testPerf[run]) = fnnSGD.TestNetwork(TestData, testTolerance)

        print (trainPerf, 'train perf % for n exp')
        print (testPerf,  'test  perf % for n exp')
        print (trainMSE,  'train mean squared error for n exp')
        print (testMSE,  'test mean squared error for n exp')

        print ('mean and std for training perf %')
        print(np.mean(trainPerf), np.std(trainPerf))

        print ('mean and std for test perf %')
        print(np.mean(testPerf), np.std(testPerf))

        print ('mean and std for time in seconds')
        print(np.mean(Time), np.std(Time))



        plt.figure()
        plt.plot(erEp )
        plt.ylabel('error')
        plt.savefig(str(problem)+'out.png')


if __name__ == "__main__": main()
