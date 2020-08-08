# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 12:42:58 2020

@author: ThinkPad
"""



from IPython.display import Image

from sklearn import __version__ as sklearn_version
 
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import math
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

class Classification:
    
    def __init__(self, method, data):
        self.method = method
        #np.random.shuffle(data)
        self.data = data
    
    def _randomforest(self, train_data):
        train_x = train_data[:, 0:-1]
        train_y = train_data[:, -1]

        forest = RandomForestClassifier(criterion='entropy', n_estimators=180, 
                                        class_weight = 'balanced',
                                        random_state=1, n_jobs=2,
                                        min_samples_split=13,
                                        oob_score = True
                                        )
    
        forest.fit(train_x, train_y)
        return forest
    
    def _adaboost(self, train_data):
        train_x = train_data[:, 0:-1]
        train_y = train_data[:, -1]       
        
        estimatorCart = DecisionTreeClassifier(max_depth=1)
        ada = AdaBoostClassifier(base_estimator=estimatorCart,
                                    n_estimators=200,
                                    )

        ada.fit(train_x, train_y)
        return ada
    
    def _knn(self, train_data):
        train_x = train_data[:, 0:-1]
        train_y = train_data[:, -1]
        
        smo = SMOTE()
        
        new_train_x, new_train_y = smo.fit_sample(train_x, train_y)
        
        
        knn = KNeighborsClassifier(n_neighbors=3, weights = 'distance', p=2, metric='minkowski')

        knn.fit(train_x, train_y)
        return knn
    
    
    def _xgb(self, train_data):
        train_x = train_data[:, 0:-1]
        train_y = train_data[:, -1]
        
        model = XGBClassifier(learning_rate=0.1,
                      n_estimators=10,           
                      max_depth=4,               
                      min_child_weight = 1,      
                      gamma=0.,                  
                      subsample=1,               
                      colsample_btree=1,         
                      scale_pos_weight=1,        
                      random_state=27,           
                      slient = 0
                      )
        model.fit(X_train, y_train)

        return model
    
    def find_minimum_n_components(self):
        print("Exploring explained variance ratio for dataset ...")
        candidate_components = range(2, int(self.data.shape[1]/2+1), 2)
        explained_ratios = []
        #t = time()
        for c in candidate_components:
            pca = PCA(n_components=c)
            X_pca = pca.fit_transform(self.data[:,0:-1])
            explained_ratios.append(np.sum(pca.explained_variance_ratio_))
            print('candidate_components:{} is done'.format(c))
            
        n_components = candidate_components[np.argmax(explained_ratios)]
        print('accuracy is :{}'.format(np.max(explained_ratios)))
        return n_components  
        
    def classify(self, test_data):
        X_train = self.data[:,0:-1]
        y_train = self.data[:,-1]#

        X_test = test_data[:, 0:-1]
        y_test = test_data[:,-1]

        #print('Class labels:', np.unique(y_train))
        #n_components = self.find_minimum_n_components()
       # print('best n_component is {}'.format(n_components))
        
        
# =============================================================================
#         pca = PCA(n_components=n_components).fit(X_train)
#         new_X_train = pca.transform(X_train)
#         new_X_test = pca.transform(X_test)
# =============================================================================
        
        train_dataset = np.hstack((X_train, y_train.reshape(-1,1)))
        print("start model fitting")
        if(self.method == 'Randomforest'):
            model = self._randomforest(train_dataset)
        if(self.method == 'knn'):           
            model = self._knn(train_dataset)
        if(self.method == 'adaboost'):
            model = self._adaboost(train_dataset)
        print("end model fitting")
        if(self.method == 'xgboost'):
            model = self._xgb(train_dataset)
        
            
        y_pred = model.predict(X_test)  
        con_mat = confusion_matrix(y_test, y_pred, np.unique(y_test))

        print(con_mat)
    
    
                 
            
        
    

