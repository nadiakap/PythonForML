#perceptron algorithm
#https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as spo
from sklearn import svm

def line_f(a,b,x):
    return a*x+b

X_bomber_mass = [1.0, 1.5, 2, 2.1,3.1]
X_bomber_speed = [ 1.1, 1.2, 1.3, 1.4, 1.5]
X_fighter_mass = [0.1,0.15, 0.2 , 0.5, 1.6 ]
X_fighter_speed = [0.3, 0.4, 0.5, 0.6, 0.7]
x=list(zip(X_bomber_mass,X_bomber_speed))
x1=list(zip(X_fighter_mass,X_fighter_speed))
x.extend(x1)
lb=min(min(X_bomber_mass,X_fighter_mass))
ub=max(max(X_bomber_mass,X_fighter_mass))

y=[1,1,1,1,1,-1,-1,-1,-1,-1]

#part1: data classification using perceptron algorithm
w=np.zeros(2)
b=0
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
    if mistake_counter==0:
           break
'''
print("decision function value for each observation")     
for i in range(len(y)): 
   print('f(',i,') =',w[0]*x[i][0]+w[1]*x[i][1]+b)
'''    
pl.figure()
xw=[lb,ub]
yw=np.zeros(2)
yw[0]=(-b-w[0]*xw[0])/w[1]
yw[1]=(-b-w[0]*xw[1]/w[1])
#plot of the separation line
pl.plot(xw,yw)
pl.scatter(X_bomber_mass,X_bomber_speed,facecolor='blue')
pl.scatter(X_fighter_mass,X_fighter_speed,facecolor='orange')
pl.title("Perceptron Algorithm")
pl.xlabel('mass')
pl.ylabel('speed')
pl.show()

#part 2: linprog classification
c = np.zeros(2)
bounds = [(-10,10),(-10,10)]
b_ub = -np.array([-1.1, -1.2, -1.3, -1.4, -1.5,0.3, 0.4, 0.5, 0.6, 0.7])
A_ub = -np.array([[-1.0,-1],[-1.5,-1],[-2,-1],[-2.1,-1],[-3.1,-1],[0.3,1],[0.15,1],[0.2,1],[0.5,1],[1.6,1]])
res=spo.linprog(c,A_ub,b_ub,bounds=bounds)

pl.figure()
pl.scatter(X_bomber_mass,X_bomber_speed)
pl.scatter(X_fighter_mass,X_fighter_speed)
x_line = np.array([0,3])
y_line = line_f(res.x[0],res.x[1],x_line)
pl.plot(x_line,y_line)
pl.title("linear programming solution")
pl.xlabel('mass')
pl.ylabel('speed')
pl.show()


#part 3: support vector machine classification
y = np.array(y)
x=np.array(x)

# "hinge" is the standard SVM loss
clf = svm.LinearSVC(C=10, loss="hinge", random_state=42).fit(x, y)
# obtain the support vectors through the decision function
decision_function = clf.decision_function(x)
# we can also calculate the decision function manually
# decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
support_vector_indices = np.where((2 * y - 1) * decision_function <= 1)[0]
support_vectors = x[support_vector_indices]

pl.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=pl.cm.Paired)
ax = pl.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
pl.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
            linestyles=['--', '-', '--'])
pl.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='k')
pl.title("Support Vector Machine, C=10")
pl.tight_layout()
pl.xlabel('mass')
pl.ylabel('speed')
pl.show()
