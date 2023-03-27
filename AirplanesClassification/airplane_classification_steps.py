#perceptron algorithm
#https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as spo
from sklearn import svm

def line_f(a,b,x):
    return a*x+b

X_bomber_mass = [1.0,1.5, 2, 2.1,3.1]
X_bomber_speed = [ 1.1, 1.2, 1.3, 1.4, 1.5 ]
X_fighter_mass = [0.1,0.15, 0.2 , 0.5, 1.6 ]
X_fighter_speed = [0.3, 0.4, 0.5, 0.6, 0.7]
x=list(zip(X_bomber_mass,X_bomber_speed))
x1=list(zip(X_fighter_mass,X_fighter_speed))
x.extend(x1)
lb=min(min(X_bomber_mass,X_fighter_mass))
ub=max(max(X_bomber_mass,X_fighter_mass))

y=[1,1,1,1,1,-1,-1,-1,-1,-1]

import random

c = list(zip(x, y))

#random.shuffle(c)
x = [e[0] for e in c]
y = [e[1] for e in c]

def partial_plot(i,w0,b0):
    pl.figure()
    xw=[lb,ub]
    yw=np.zeros(2)
    yw[0]=(-b0-w0[0]*xw[0])/w0[1]
    yw[1]=(-b0-w0[0]*xw[1])/w0[1]
    #plot of the separation line
    pl.plot(xw,yw)
    pl.scatter(X_bomber_mass[0:i],X_bomber_speed[0:i],facecolor='blue')
    pl.scatter(X_fighter_mass[0:i],X_fighter_speed[0:i],facecolor='orange')
    pl.title("Perceptron Algorithm, iteration " + str(i))
    pl.xlabel('mass')
    pl.ylabel('speed')
    pl.show()
   
#part1: data classification using perceptron algorithm
w=np.zeros(2)
b=0
x=np.array(x)
y=np.array(y)


#perceptron algorithm#
#adjust w and b 
#until all input points are correctly classified
while True:
    for i in range(len(y)): 
        mistake_counter=0
        ytest = np.dot(w, x[i]) + b 
        if ytest * y[i] <= 0: 
            mistake_counter+=1
            w += np.array(x[i])*y[i]
            b += y[i]  
        partial_plot(i,w,b)
    if mistake_counter==0:
           break
  
partial_plot(i,w,b)


print("decision function value for each observation")     
for i in range(len(y)): 
   print('f(',i,') =',w[0]*x[i][0]+w[1]*x[i][1]+b)
    
