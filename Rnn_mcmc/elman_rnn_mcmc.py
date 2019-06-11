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
Elman RNN with mcmc single core

this is a single core mcmc applied on weights of RNN. (without langevin)

an issue - need to find some better heuristics values as it still converges to a local minimum.
found this when running on mackey dataset
'''



import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
#np.random.seed(1)
import numpy as np
import rnn_mcmc_plots as mcmcplt

mplt = mcmcplt.Mcmcplot()

MinimumError = 0.00001
trainsize=299
testsize=99
weightdecay = 0.01
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
    def evaluate_proposal(self,x,w):
        self.decode(w)
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


    def sampleEr(self,actualout):
        error = np.subtract(self.out, actualout)
        sqerror= np.sum(np.square(error))/self.Top[2]
        return sqerror


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

def shuffledata(x,y):
    a=[]
    for i in range(0,len(x)):
        a.append(i)
    random.shuffle(a)
    x1 = []
    y1=[]
    for item in a:
        x1.append(x[item])
        y1.append(y[item])
    return x1,y1


class MCMC:
    def __init__(self, samples, learnrate, train_x, train_y,test_x,test_y, topology):
        self.samples = samples  # max epocs
        self.topology = topology  # NN topology [input, hidden, output]
        self.train_x = train_x#
        self.test_x = test_x
        self.train_y=train_y
        self.test_y=test_y
        self.learnrate = learnrate
        # ----------------

    def rmse(self, predictions, targets):
        predictions = np.array(predictions)
        targets=np.array(targets)
        return np.sqrt(((predictions - targets) ** 2).mean())

    def likelihood_func(self, neuralnet, x,y, w, tausq):
        #y = data[:, self.topology[0]]
        y=y
        fx = neuralnet.evaluate_proposal(x, w)
        rmse = self.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(np.array(y) - np.array(fx)) / tausq
        return [np.sum(loss), fx, rmse]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def sampler(self):

        # ------------------- initialize MCMC
        testsize = len(self.test_x)   #self.testdata.shape[0]
        trainsize = len(self.train_x)
        samples = self.samples

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        netw = self.topology  # [input, hidden, output]
        y_test = self.test_y  #self.testdata[:, netw[0]]
        y_train = self.train_y #self.traindata[:, netw[0]]
        #print(len(y_train))
        #print(len(y_test))

        # here
        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2] + (netw[1] * netw[1]) # num of weights and bias

        pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
        pos_tau = np.ones((samples, 1))

        # original -->    fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
        #print('shape: ',np.array(y_train).shape[1])
        fxtrain_samples = np.ones((samples, trainsize,int(np.array(y_train).shape[1])))  # fx of train data over all samples
        # original --> fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples || probably for 1 dimensional data
        fxtest_samples = np.ones((samples, testsize,np.array(self.test_y).shape[1]))  # fx of test data over all samples
        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)

        w = np.random.randn(w_size)
        w_proposal = np.random.randn(w_size)

        step_w = 0.02  # defines how much variation you need in changes to w
        step_eta = 0.01
        # --------------------- Declare FNN and initialize

        neuralnet = Network(self.learnrate,self.topology, self.train_x,self.train_y,self.test_x,self.test_y)
        print ('evaluate Initial w')
        #print(w,np.array(self.train_x).shape)
        pred_train = neuralnet.evaluate_proposal(self.train_x, w)
        pred_test = neuralnet.evaluate_proposal(self.test_x, w)

        eta = np.log(np.var(np.array(pred_train) - np.array(y_train)))
        tau_pro = np.exp(eta)
        err_nn = np.sum(np.square(np.array(pred_train) - np.array(y_train)))/(len(pred_train)) #added by ashray mean square sum
        print('err_nn is: ',err_nn)
        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0
        #print(pred_train)
        prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients
        [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.train_x,self.train_y, w, tau_pro)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.test_x,self.test_y, w, tau_pro)

        print(likelihood,' is likelihood of train')
        #print(pred_train)
        #print(pred_train, ' is pred_train')
        naccept = 0
        print ('begin sampling using mcmc random walk')
        plt.plot(x_train, y_train)
        plt.plot(x_train, pred_train)
        plt.title("Plot of Data vs Initial Fx")
        plt.savefig('mcmcresults/begin.png')
        plt.clf()

        plt.plot(x_train, y_train)

        for i in range(samples - 1):
            #print(i)

            w_proposal = w + np.random.normal(0, step_w, w_size)

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.train_x,self.train_y, w_proposal,
                                                                                tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.test_x,self.test_y, w_proposal,
                                                                            tau_pro)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
                                               tau_pro)  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            diff_priorliklihood = prior_prop - prior_likelihood

            #mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))
            mh_prob = min(0, (diff_likelihood + diff_priorliklihood))
            mh_prob = math.exp(mh_prob)
            u = random.uniform(0, 1)

            if u < mh_prob:
                # Update position
                #print(i, ' is the accepted sample')
                naccept += 1
                likelihood = likelihood_proposal
                prior_likelihood = prior_prop
                w = w_proposal
                eta = eta_pro
                # if i % 100 == 0:
                #     #print ( likelihood, prior_likelihood, rmsetrain, rmsetest, w, 'accepted')
                #     print ('Sample:',i, 'RMSE train:', rmsetrain, 'RMSE test:',rmsetest)

                pos_w[i + 1,] = w_proposal
                pos_tau[i + 1,] = tau_pro
                fxtrain_samples[i + 1,] = pred_train
                fxtest_samples[i + 1,] = pred_test
                rmse_train[i + 1,] = rmsetrain
                rmse_test[i + 1,] = rmsetest

                plt.plot(x_train, pred_train)


            else:
                pos_w[i + 1,] = pos_w[i,]
                pos_tau[i + 1,] = pos_tau[i,]
                fxtrain_samples[i + 1,] = fxtrain_samples[i,]
                fxtest_samples[i + 1,] = fxtest_samples[i,]
                rmse_train[i + 1,] = rmse_train[i,]
                rmse_test[i + 1,] = rmse_test[i,]

                # print i, 'rejected and retained'

            if i % 100 == 0:
                #print ( likelihood, prior_likelihood, rmsetrain, rmsetest, w, 'accepted')
                print ('Sample:',i, 'RMSE train:', rmsetrain, 'RMSE test:',rmsetest)

        print (naccept, ' num accepted')
        print ((naccept*100) / (samples * 1.0), '% was accepted')
        accept_ratio = naccept / (samples * 1.0) * 100

        plt.title("Plot of Accepted Proposals")
        plt.savefig('mcmcresults/proposals.png')
        plt.savefig('mcmcresults/proposals.svg', format='svg', dpi=600)
        plt.clf()

        return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test, accept_ratio)





def main():
        outres = open('resultspriors.txt', 'w')
        learnRate = 0.1


        #for mackey
        fname = "trainsunspot.txt"
        x,y = loadersunspot(fname)
        #print_data(x,y)
        x,y = shuffledata(x,y)
        train_x= x[:int(len(x)*0.8)]
        test_x=x[int(len(x)*0.8):]
        train_y= y[:int(len(y)*0.8)]
        test_y=y[int(len(y)*0.8):]
        Input = len(train_x[0][0])
        Output = len(train_y[0])



        #print(traindata)
        Hidden = 5
        topology = [Input, Hidden, Output]
        numSamples = 80000  # need to decide yourself

        mcmc = MCMC(numSamples,learnRate,train_x,train_y,test_x,test_y, topology)  # declare class

        [pos_w, pos_tau, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_ratio] = mcmc.sampler()
        print ('sucessfully sampled')

        burnin = 0.1 * numSamples  # use post burn in samples

        pos_w = pos_w[int(burnin):, ]
        pos_tau = pos_tau[int(burnin):, ]


        ''' 
        to plots the histograms of weight destribution
        '''

        mplt.initialiseweights(len(pos_w),len(pos_w[0]))
        for i in range(len(pos_w)):
            mplt.addweightdata(i,pos_w[i])
        mplt.saveplots()

        fx_mu = fx_test.mean(axis=0)
        fx_high = np.percentile(fx_test, 95, axis=0)
        fx_low = np.percentile(fx_test, 5, axis=0)

        fx_mu_tr = fx_train.mean(axis=0)
        fx_high_tr = np.percentile(fx_train, 95, axis=0)
        fx_low_tr = np.percentile(fx_train, 5, axis=0)

        rmse_tr = np.mean(rmse_train[int(burnin):])
        rmsetr_std = np.std(rmse_train[int(burnin):])
        rmse_tes = np.mean(rmse_test[int(burnin):])
        rmsetest_std = np.std(rmse_test[int(burnin):])
        print (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std)
        np.savetxt(outres, (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio), fmt='%1.5f')

        #ytestdata = testdata[:, input]
        #ytraindata = traindata[:, input]
        ytestdata = test_y
        ytraindata = train_y

        # converting everything to np arrays
        x_test = np.array(x_test)
        fx_low = np.array(fx_low)
        fx_high = np.array(fx_high)
        fx_mu = np.array(fx_mu)
        ytestdata = np.array(ytestdata)
        x_train= np.array(x_train)
        ytraindata= np.array(ytraindata)
        fx_mu_tr = np.array(fx_mu_tr)
        fx_low_tr = np.array(fx_low_tr)
        fx_high_tr = np.array(fx_high_tr)



        plt.plot(x_test, ytestdata, label='actual')
        plt.plot(x_test, fx_mu, label='pred. (mean)')
        plt.plot(x_test, fx_low, label='pred.(5th percen.)')
        plt.plot(x_test, fx_high, label='pred.(95th percen.)')
        #print(np.array(x_test).shape,np.array(fx_low).shape,np.array(fx_high).shape)
        #print(fx_low[:,0],fx_high,x_test)
        plt.fill_between(x_test, fx_low[:,0], fx_high[:,0], facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Plot of Test Data vs MCMC Uncertainty ")
        plt.savefig('mcmcresults/mcmcrestest.png')
        plt.savefig('mcmcresults/mcmcrestest.svg', format='svg', dpi=600)
        plt.clf()
        # -----------------------------------------
        plt.plot(x_train, ytraindata, label='actual')
        plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
        plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
        plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
        plt.fill_between(x_train, fx_low_tr[:,0], fx_high_tr[:,0], facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Plot of Train Data vs MCMC Uncertainty ")
        plt.savefig('mcmcresults/mcmcrestrain.png')
        plt.savefig('mcmcresults/mcmcrestrain.svg', format='svg', dpi=600)
        plt.clf()

        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)

        ax.boxplot(pos_w)

        ax.set_xlabel('[W1] [B1] [W2] [B2]')
        ax.set_ylabel('Posterior')

        plt.legend(loc='upper right')

        plt.title("Boxplot of Posterior W (weights and biases)")
        plt.savefig('mcmcresults/w_pos.png')
        plt.savefig('mcmcresults/w_pos.svg', format='svg', dpi=600)

        plt.clf()




if __name__ == "__main__": main()
