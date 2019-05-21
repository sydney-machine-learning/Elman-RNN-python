#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:31:56 2019

@author: ashrey
"""
import numpy as np
import random

np.random.seed(0)
i1=np.random.randint(1,100)
i2=np.random.randint(i1,150)
x1=[]
for i in range(i1,i2):
    x1.append(i)
y1=i2+1


limit1=10
limit2=14

x=[[[]]]
y=[[]]
for i in range(0,limit1):
    i1= np.random.randint(0,limit1)
    t=[[]]
    tempy=[]
    i2=np.random.randint(i1,limit2)
    for j in range(i1,i2):
        tempx=[]
        for k in range(0,limit2):
            tempx.append(0)
        tempx[j]=1
        t.append(tempx)
    tempy=[]
    for k in range(0,limit2):
        tempy.append(0)
    tempy[i2]=1
    del t[0]
    x.append(t)
    y.append(tempy)

del x[0]
del y[0]
train_x= x[:int(limit2*0.8)]
test_x=x[int(limit2*0.8):]
train_y= y[:int(limit2*0.8)]
test_y=y[int(limit2*0.8):]

# =============================================================================
#
# 3 dimensional x and 2 dimensional y
# an element of train_x is a list of list. basically numbers ranging from a specific integer to another integer
# in one hot vector format and corresponding y contains the next integer value expected.
# for ex - train_x[0] contains integers from 5 to 15 in one hot vector format with max size limit2 corresponding y contains 16. the next integer in one hot.
#
# =============================================================================
print("Dataset created")

'''
inputting the new file

'''


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