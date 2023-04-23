from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)
gnb = GaussianNB()
gnbfitted = gnb.fit(X_train, y_train)
y_pred = gnbfitted.predict(X_test)
print("*****training NB with subset of data**")
print("Number of mislabeled points out of a total %d points : %d"
          % (X_test.shape[0], (y_test != y_pred).sum()))
print('****number of training samples observed in each class**')
print(gnbfitted.class_count_)
print('****unconditional probability of each .**')
print(gnbfitted.class_prior_)
print('****variance of each feature per class s**')
print(gnbfitted.var_)
print('****mean of each feature per class **')
print(gnbfitted.theta_)
'''
gnbwholefitted = gnb.fit(X, y)
print("*****training NB with the whole dataset**")

print('****number of training samples observed in each class**')
print(gnbwholefitted.class_count_)
print('****unconditional probability of each class.**')
print(gnbwholefitted.class_prior_)
print('****variance of each feature per class**')
print(gnbwholefitted.sigma_)
print('****mean of each feature per class **')
print(gnbwholefitted.theta_)
'''
#MLE for estimating mean and variance of the first feature
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
f = np.ndarray.flatten(X[0:49,0:1])
loc11, scale11 = norm.fit(f)
ax.hist(f, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()
f = np.ndarray.flatten(X[50:99,0:1])
loc21, scale21 = norm.fit(f)
f = np.ndarray.flatten(X[100:150,0:1])
loc31, scale31 = norm.fit(f)
print('mean for the first feature:', loc11,loc21,loc31)
print('variance for the first feature:', scale11**2,scale21**2,scale31**2)
#MLE for estimating mean and variance of the second feature
f = np.ndarray.flatten(X[0:49,1:2])
loc12, scale12 = norm.fit(f)
f = np.ndarray.flatten(X[50:99,1:2])
loc22, scale22 = norm.fit(f)
f = np.ndarray.flatten(X[100:150,1:2])
loc32, scale32 = norm.fit(f)
print('mean for the second feature:', loc12,loc22,loc32)
print('variance for the second feature:', scale12**2,scale22**2,scale32**2)

#MLE for estimating mean and variance of the third feature
f = np.ndarray.flatten(X[0:49,2:3])
loc13, scale13 = norm.fit(f)
f = np.ndarray.flatten(X[50:99,2:3])
loc23, scale23 = norm.fit(f)
f = np.ndarray.flatten(X[100:150,2:3])
loc33, scale33 = norm.fit(f)
#MLE for estimating mean and variance of the forth feature
f = np.ndarray.flatten(X[0:49,3:4])
loc14, scale14 = norm.fit(f)
f = np.ndarray.flatten(X[50:99,3:4])
loc24, scale24 = norm.fit(f)
f = np.ndarray.flatten(X[100:150,3:4])
loc34, scale34 = norm.fit(f)

#Gaussian Mixture Model - EM algorithm
n_classes = len(np.unique(y_train))


GMestimator = GaussianMixture(n_components=n_classes,
              covariance_type='full', max_iter=20, random_state=0)
GMfitted = GMestimator.fit(X_train)
print('matrix of mean values for each component of GM for each class:', GMfitted.means_)
print('covariance matrixs of GM:', GMfitted.covariances_)

#Bayesian Gaussian Mixture Model - Bayesian inference algorithm
BGMestimator = BayesianGaussianMixture(n_components=n_classes,
              covariance_type='diag', max_iter=60, random_state=0)
BGMfitted = BGMestimator.fit(X_train)
print('matrix of mean values for each component of BGM for each class:', BGMfitted.means_)
print('covariance matrixs of GM:', BGMfitted.covariances_)

