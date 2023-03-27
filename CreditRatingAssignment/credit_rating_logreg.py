#credit rating assignment
#https://www.mathworks.com/help/stats/credit-rating-by-bagging-decision-trees.html
import pandas as pd
from sklearn.linear_model import LogisticRegression

data=pd.read_csv('CreditRating1.csv',header=0)

def isDefault(x):
    if x == "'CCC'":
        return 1
    return 0 

#create new column 'Default': it is 1 for companies with rating 'CCC' and 0 for other firms
data['Default'] = data['Rating'].apply(lambda x: isDefault(x))

data.drop('Industry',axis=1,inplace = True)
data.drop('Rating',axis=1,inplace = True)
y=(data['Default']).to_numpy()
data.drop('Default',axis=1,inplace = True)

X = data.to_numpy()
print(y)
print(X)

clf = LogisticRegression(random_state=0).fit(X, y)
res = clf.predict(X[150:151,:])
res_prob = clf.predict_proba(X[150:151,:])
