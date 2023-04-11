import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import GridSearchCV


data=pd.read_csv('C:/Users\dmitr\Desktop\AI_2020\Telecom-Churn-Data-Analysis-master\Telecom-Churn-Data-Analysis-master\Telecom Churn.csv',header=0)
'''
def isDefault(x):
    if x == "'CCC'":
        return 1
    return 0 

#create new column 'Default': it is 1 for companies with rating 'CCC' and 0 for other firms
data['Default'] = data['Rating'].apply(lambda x: isDefault(x))

data.drop('Industry',axis=1,inplace = True)
data.drop('Rating',axis=1,inplace = True)
'''
y=(data['churn']).astype('int')
data.drop('phone number',axis=1,inplace = True)

data.drop('churn',axis=1,inplace = True)


#dummy1 = pd.get_dummies(data[['state','international plan','voice mail plan']], drop_first=True)
#data = pd.concat([data, dummy1], axis=1)
data.drop(['state','voice mail plan'],axis=1,inplace = True)
data['international plan']=\
  data['international plan'].map({'yes':1,'no':0})

cr_feature_names = data.columns.to_list()

#X = data.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3,stratify=y, random_state=42)
'''
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
res_prob = clf.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
'''
clf_tree = tree.DecisionTreeClassifier(random_state=17)
clf_tree = clf_tree.fit(X_train, y_train)
cvs=cross_val_score(clf_tree,X_train,y_train)
mn = np.mean(cvs)
y_tree_pred = clf_tree.predict(X_test)
print(confusion_matrix(y_test, y_tree_pred))

tree.plot_tree(clf_tree)

r = tree.export_text(clf_tree, feature_names = cr_feature_names)
print(r)

cls_knn = KNeighborsClassifier()
cvs_knn=cross_val_score(cls_knn,X_train,y_train)
mn_knn = np.mean(cvs_knn)

tree_params = {'max_depth': np.arange(1,11), 'max_features': (0.5,0.7,0.9)}
#grid search can be done in parallel if there are several processors available. specify n_jobs parametr
tree_grid = GridSearchCV(clf_tree,tree_params,cv=5)
tree_grid = tree_grid.fit(X_train,y_train)
#tree parameters are optimized
print(tree_grid.best_estimator_, tree_grid.best_score_)
