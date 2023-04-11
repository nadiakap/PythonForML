#telecom churn data analysis
#for example, see this tutorial
#https://towardsdatascience.com/end-to-end-machine-learning-project-telco-customer-churn-90744a8df97d

import pandas as pd

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz


data=pd.read_csv('C:/Users\dmitr\Desktop\AI_2020\Telecom-Churn-Data-Analysis-master\Telecom-Churn-Data-Analysis-master\Telecom Churn.csv',header=0)

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


X_train, X_valid, y_train, y_valid = train_test_split(data, y, test_size=0.3,stratify=y, random_state=42)

clf_tree = tree.DecisionTreeClassifier(random_state=17)
clf_tree = clf_tree.fit(X_train, y_train)
cvs=cross_val_score(clf_tree,X_train,y_train)
mn = np.mean(cvs)
y_tree_pred = clf_tree.predict(X_valid)
print(confusion_matrix(y_valid, y_tree_pred))



cls_knn = KNeighborsClassifier()
cvs_knn=cross_val_score(cls_knn,X_train,y_train)
mn_knn = np.mean(cvs_knn)

tree_params = {'max_depth': np.arange(1,11), 'max_features': (0.5,0.7,0.9)}
#grid search can be done in parallel if there are several processors available. specify n_jobs parametr
tree_grid = GridSearchCV(clf_tree,tree_params,cv=5)
tree_grid = tree_grid.fit(X_train,y_train)
#tree parameters are optimized
print(tree_grid.best_estimator_, tree_grid.best_score_)

knn_params = {'n_neighbors': (1,2,3,4,5,10,15,50,60,70,80,90,100)}
#grid search can be done in parallel if there are several processors available. specify n_jobs parametr
knn_grid = GridSearchCV(cls_knn,knn_params,cv=5)
knn_grid = knn_grid.fit(X_train,y_train)
#tree parameters are optimized
print(knn_grid.best_estimator_, knn_grid.best_score_)

y_predict = tree_grid.predict(X_valid)
score = tree_grid.score(X_valid,y_valid)
#accuracy score - proportion of correct answers
acc_sc = accuracy_score(y_valid,y_predict)
#how good is the answer? we need to compare accuracy_score 
#with proportiona of bad clients in hold out sample )
bc = 1-np.mean(y)#proportion of bad client in hold out sample
#if proportion of bad clinets in hold out sample
# is less than accuracy score then we have good results
print('good results:',bc<acc_sc)
#to see the tree:
export_graphviz(tree_grid.best_estimator_,out_file='telecom_tree.dot',feature_names=cr_feature_names, filled=True)
#if dot utility is installed then use the following 
#command to convert to .png file
# !dot -Tpngtelecom_tree.dot -o telecom_tree.png


cl_tree = tree.DecisionTreeClassifier(random_state=17, max_depth=3)
cl_tree = cl_tree.fit(X_train, y_train)

tree.plot_tree(cl_tree)

r = tree.export_text(cl_tree, feature_names = cr_feature_names)
print(r)
