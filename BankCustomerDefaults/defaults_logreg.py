#logistic regression example from this website:
#https://medium.com/@polanitzer/logistic-regression-in-python-predict-the-probability-of-default-of-an-individual-8a0091da3775
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc('font', size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)

data=pd.read_csv('bank.csv')
data = data.dropna()
print(data.shape)
print(list(data.columns))

sns.countplot(x='y', data=data, palette='hls')
plt.show()

count_no_default = len(data[data['y']==0])
count_default = len(data[data['y']==1])
pct_of_no_default = count_no_default/(count_no_default+count_default)
print('\033[1m percentage of no default is', pct_of_no_default*100)
pct_of_default = count_default/(count_no_default+count_default)
print('\033[1m percentage of default', pct_of_default*100)

#one-hot encoding for categorical variables
cat_vars=['education']
for var in cat_vars:
 cat_list='var'+'_'+var
 cat_list = pd.get_dummies(data[var], prefix=var)
 data1=data.join(cat_list)
 data=data1
cat_vars=['education']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

#final data columns
data_final=data[to_keep]
data_final.drop(['loan_applicant_id'], axis=1, inplace=True)
#Over-sampling using SMOTE (Synthetic Minority Oversampling Technique)
#see this tutorial for unbalance data handling
#https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
columns = X_train.columns
os_data_X,os_data_y = os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y = pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print('\033[1m length of oversampled data is ',len(os_data_X))
print('\033[1m Number of no default in oversampled data',len(os_data_y[os_data_y['y']==0]))
print('\033[1m Number of default',len(os_data_y[os_data_y['y']==1]))
print('\033[1m Proportion of no default data in oversampled data is ',len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print('\033[1m Proportion of default data in oversampled data is ',len(os_data_y[os_data_y['y']==1])/len(os_data_X))

#feature selection using RFE - recursive feature elimination
#tutorial on computing feature importance
#https://machinelearningmastery.com/calculate-feature-importance-with-python/
data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, n_features_to_select=9, step=3)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
print(os_data_X.columns)
data_X1 = pd.DataFrame({
 'Feature': os_data_X.columns,
 'Importance': rfe.ranking_},)
data_X1.sort_values(by=['Importance'])
print(data_X1)
cols=[]
for i in range (0, len(data_X1['Importance'])):
 if data_X1['Importance'][i] == 1:
  cols.append(data_X1['Feature'][i])
print(cols)
print(len(cols))

#statistical analysis
X=os_data_X[cols]
y=os_data_y['y']
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
pvalue = pd.DataFrame(result.pvalues,columns=['p_value'],)
pvs=[]
for i in range (0, len(pvalue['p_value'])):
 if pvalue['p_value'][i] < 0.05:
   pvs.append(pvalue.index[i])
if 'const' in pvs:
 pvs.remove('const')
else:
 pvs 
X=os_data_X[pvs]
y=os_data_y['y']
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary()) 
#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
logreg = LogisticRegression(max_iter=600,)
logreg.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(accuracy_score(y_test, y_pred)))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
