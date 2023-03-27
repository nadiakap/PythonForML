from numpy import genfromtxt
import numpy as np
import scipy.linalg as lna
from sklearn import decomposition as dc
import matplotlib.pyplot as pl
data_file = 'food_pca.csv'
my_data = genfromtxt(data_file, delimiter = ',', skip_header=1)
#n - number of features
#m - number of food items
#first column contains food names
n = 4
m = 24
X0=my_data[0:m,1:n+1]
#data normalization
X = (X0 - X0.min(0)) / X0.ptp(0)

cor_fat_protein = np.corrcoef(X[:,0],X[:,1])[0,1]
cor_fat_vitC = np.corrcoef(X[:,0],X[:,3])[0,1]
cor_fat_fiber = np.corrcoef(X[:,0],X[:,2])[0,1]
cor_fiber_vitC = np.corrcoef(X[:,2],X[:,3])[0,1]
cor_fiber_protein = np.corrcoef(X[:,2],X[:,1])[0,1]
cor_vitC_protein = np.corrcoef(X[:,3],X[:,1])[0,1]

print('cor_fat_protein = ',cor_fat_protein)
print('cor_fat_vitC = ',cor_fat_vitC)
print('cor_fat_fiber = ',cor_fat_fiber)
print('cor_fiber_vitC = ',cor_fiber_vitC)
print('cor_fiber_protein = ',cor_fiber_protein)
print('cor_vitC_protein = ',cor_vitC_protein)
pca = dc.PCA(n_components=2)
res = pca.fit(X)
cmp=res.components_
newfeatures = pca.transform(X)
expl_var=res.explained_variance_ratio_
cv = np.cov(X,rowvar=False)
eigenvals,eigenvecs = lna.eig(cv)
r = np.matmul(X,eigenvecs)

food_labels = ['oranges', 'eggs', 'peppers', 'tomato','broccoli','carrots']
pl.figure()
pl.title('Depiction of several food items in fat/protein space')
xxx=X[17:23,0]
yyy=X[17:23,1]
pl.scatter(xxx,yyy,facecolor='orange')
pl.ylabel('protein')
pl.xlabel('fat')

for i, txt1 in enumerate(food_labels):
    pl.annotate(txt1, (xxx[i], yyy[i]))
pl.show()

pl.figure()
pl.title('Depiction of several food items in the space of two largest principal components')
xxx=r[17:23,0]
yyy=r[17:23,1]
pl.scatter(xxx,yyy,facecolor='blue')
for j, txt in enumerate(food_labels):
    pl.annotate(txt, (xxx[j], yyy[j]))
pl.ylabel('pc2')
pl.xlabel('pc1')
pl.show()
'''
pl.figure()
pl.title('Depiction of several food items in fiber/vitC space')
xxx=X[17:23,2]
yyy=X[17:23,3]
pl.scatter(xxx,yyy,facecolor='orange')
pl.ylabel('vitC')
pl.xlabel('fiber')

for i, txt1 in enumerate(food_labels):
    pl.annotate(txt1, (xxx[i], yyy[i]))
pl.show()
'''
