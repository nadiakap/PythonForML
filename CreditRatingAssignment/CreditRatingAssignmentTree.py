#credit rating assignment
#https://www.mathworks.com/help/stats/credit-rating-by-bagging-decision-trees.html
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


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
cr_feature_names = data.columns.to_list()


X = data.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, stratify = y,random_state=42)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
res_prob = clf.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))

clf_tree = tree.DecisionTreeClassifier(random_state=0, max_depth=2)
clf_tree = clf_tree.fit(X_train, y_train)
y_tree_pred = clf_tree.predict(X_test)
print(confusion_matrix(y_test, y_tree_pred))

tree.plot_tree(clf_tree)

r = tree.export_text(clf_tree, feature_names = cr_feature_names)
print(r)
