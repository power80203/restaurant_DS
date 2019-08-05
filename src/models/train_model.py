import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib 
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import confusion_matrix
import xgboost
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN
from sklearn.cluster import KMeans
import collections

sys.path.append(os.path.abspath("."))
import config
from src.features import fe
from keras.utils import to_categorical

# read data
df_main_ori = fe.mergred_store_and_user()

print("original data", df_main_ori.shape)

# select

# smoker	drink_level	dress_preference	ambience	
# transport	marital_status	hijos	birth_year	interest	
# personality	religion	activity	color	weight	budget	height
# 	Upayment	latitude_y	longitude_y	store_the_geom_meter	
#     store_address	store_city	store_state	store_country	fax	zip	store_alcohol	
#     store_smoking_area	store_dress_code	store_accessibility	store_price	url	
#     store_Rambience	franchise	store_area	store_other_services	cuisin	fri_hours	
#     mon_hours	sat_hours	sun_hours	thu_hours	tue_hours	wed_hours	park	
#     payment	distance_between_user_and_store

# ******************** lack operating time

modeling_list = ['smoker', 'drink_level', 'budget',
                 'transport', 'marital_status', 'hijos', 'interest', 'personality', 
                 'religion', 'activity', 
                 'store_alcohol', #'store_smoking_area',
                 'store_dress_code', 'store_accessibility', 'store_price', 'store_Rambience','franchise',
                 'store_area', 'store_other_services', 'park', 
                ]

target = 'rating'

# add numeric vars
modeling_numeric_list = ['birth_year', 'payment_methods', 'number_of_store_cuisin','cuisine_match', 'num_of_Upayment']

# full list
full_list = modeling_list.copy()
full_list.append(target)

for i in modeling_numeric_list:
    full_list.append(i)

df_main = df_main_ori[full_list]

df_main['number_of_store_cuisin'] = df_main['number_of_store_cuisin'].fillna(0)

print("data after selection", df_main.shape)

df_main = df_main.dropna()

print("data after dropping", df_main.shape)

# from fancyimpute import KNN    
# # Use 5 nearest rows which have a feature to fill in each row's missing features
# knnOutput = KNN(k=5).fit_transform(df_main)

# print(knnOutput.shape)

# one hot encoding

for i in modeling_list:
    tmp_oneHotEncoding = pd.get_dummies(df_main[i],prefix = i)
    df_main = df_main.drop(i, axis = 1)
    df_main = pd.concat([df_main,tmp_oneHotEncoding], axis = 1)
    print("data after one hot %s"%i, df_main.shape)

df_main['rating'] = df_main['rating'].apply( lambda x : 0 if x < 2 else 1)

# start to modeling

y = df_main['rating'].values

df_main = df_main.drop('rating', axis = 1)

x = df_main.values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


def ann(X_train, X_test, y_train, y_test):
    ##################################### DL
        
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    dacay = 1.5
    input_number = X_train.shape[1]
    while True:
        if input_number < 5:
            classifier.add(Dense(output_dim = 2, activation = 'sigmoid'))
            break
        else:
            classifier.add(Dense(output_dim = int(input_number / dacay), activation = 'relu', 
                                input_dim = input_number))
            # classifier.add(Dropout(0.1))

        input_number = int(input_number / dacay)

    # classifier.add(Dropout(0.5))
    # # Adding the second hidden layer
    # classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu'))
    # classifier.add(Dropout(0.5))
    # # Adding the output layer
    # classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'softmax'))
    # earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

    # Compiling the ANN
    SGD = keras.optimizers.SGD(lr = 0.1, momentum=0.0, decay=0.0, nesterov=False)
    classifier.compile(optimizer = 'RMSprop', loss = 'binary_crossentropy', 
                    metrics = ['accuracy'])
    #binary_crossentropy
    #categorical_crossentropy



    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, validation_data = (X_test, y_test), 
                   batch_size = 4, nb_epoch = 100, verbose = 0)

    y_pred = classifier.predict(X_test)
    
    

    # y_pred = np.argmax(y_pred, axis =1 )
    y_pred = y_pred >= 0.5
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    # Making the Confusion Matrix
    print("acc by ANN model", accuracy_score(y_test, y_pred))

    return y_pred


    ##################################### DL

def xgBoost(X_train, X_test, y_train, y_test):
    ##################################### XGBOOST

    cl = xgboost.XGBClassifier(max_depth= 11, learning_rate= 0.1, n_estimators= 1000, 
                                                verbosity = 1, 
                                                #    objective='multi:softmax', num_class = 3,
                                                objective = 'binary:logistic',
                                                booster='gbtree', n_jobs=1, nthread= 2, gamma= 0.1, 
                                                min_child_weight= 5, max_delta_step=0,
                                                colsample_bylevel=1, tree_method= 'exact',
                                                colsample_bynode=1, reg_alpha=0, reg_lambda= 0.8,
                                                base_score=0.5, random_state= 1)

    # cl = KNeighborsClassifier(n_neighbors= 6)

    cl.fit(X_train, y_train)

    y_predictive_data = cl.predict(X_test)

    # print(y_predictive_data)

    cm = confusion_matrix(y_test, y_predictive_data)

    print(cm)
    print("acc by xgboost model", accuracy_score(y_test, y_predictive_data))


    param_test1 = {
    'max_depth':range(3,15,2),
    'min_child_weight':range(1,6,2)
    }

    param_test2 = {
    'max_depth':[10, 11,  12],
    'min_child_weight': [4, 5,  6]
    }

    param_test3 = {
    'gamma':[i/10.0 for i in range(0,5)]
    }

    param_test = {'scale_pos_weight =':  [0.1, 1,  5], 
                    'max_depth': [4,10,20], 
                    'n_estimators ':[10,20, 50],
                    'colsample_bytree' : [0.1, 0.5, 1],
                    'learning_rate':[1, 0.05,  0.1, ],
                    'n_estimators' : [1,5,10]
                    }

    gsearch = GridSearchCV(
        # estimator = xgboost.XGBClassifier( 
        #     learning_rate =0.01, n_estimators= 10, max_depth=4, verbosity = 0,
        #     min_child_weight= 1, gamma=0, subsample= 1, colsample_bytree= 0.8, 
        #     reg_alpha = 0, num_class = 3,objective= 'multi:softmax', nthread = 4, 
        #     scale_pos_weight= 0.1, ), 
            estimator = cl, 
            param_grid = param_test , 
            n_jobs= 3, 
            scoring='accuracy',
            cv= 10,
            verbose = 2
            
        )

    gsearch_in = 0
    if gsearch_in:
            
        gsearch.fit(X_train, y_train)

        # print(gsearch.cv_results_)
        print("")
        print("best para:", gsearch.best_params_)
        print("best score:",gsearch.best_score_)
    
    return y_predictive_data


def svm_c(X_train, X_test, y_train, y_test):

    model = svm.SVC(kernel='rbf',  C= 10)

    train_sizes, train_scores, test_scores = learning_curve(estimator = model,
                                                            X =  X_train,
                                                            y = y_train,
                                                            train_sizes= np.linspace(0.1, 1.0, 10),
                                                            cv = 10,
                                                            n_jobs = 1)
    train_mean = np.mean(train_scores, axis = 1)
    train_std = np.std(train_scores, axis = 1)
    test_mean = np.mean(test_scores, axis = 1)
    test_std = np.std(test_scores, axis = 1)

    plt.plot(train_sizes, train_mean, color = 'blue', marker = 'o', markersize = 5, label = 'training acc')
    
    plt.fill_between(train_sizes, 
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha = 0.15, color = 'blue')

    plt.plot(train_sizes, test_mean, color = 'green', marker = 's', markersize = 5, label = 'validation acc')

    plt.fill_between(train_sizes, 
                        test_mean + test_std,
                        test_mean - test_std,
                        alpha = 0.15, color = 'green')

    plt.grid()
    plt.xlabel('number of training samples')
    plt.ylabel('acc')
    plt.legend(loc = 'lower right')
    plt.ylim([0.5 , 1])
    plt.show()

    model.fit(X_train, y_train)

    y_predictive_data = model.predict(X_test)

    # print(y_predictive_data)

    cm = confusion_matrix(y_test, y_predictive_data)

    print('svm model')
    print(cm)
    print("acc by svm model", accuracy_score(y_test, y_predictive_data))

def knn(X_train, X_test, y_train, y_test):

    cl = KNeighborsClassifier(n_neighbors= 7)

    cl.fit(X_train, y_train)

    y_predictive_data = cl.predict(X_test)


    cm = confusion_matrix(y_test, y_predictive_data)
    
    print(cm)
    print("acc by knn model", accuracy_score(y_test, y_predictive_data))

    return y_predictive_data
    # test
    """
    for number in range(1,25):

        cl = KNeighborsClassifier(n_neighbors= number)

        cl.fit(X_train, y_train)

        y_predictive_data = cl.predict(X_test)


        cm = confusion_matrix(y_test, y_predictive_data)

        print(cm)
        print("acc by knn %s model"%number, accuracy_score(y_test, y_predictive_data))
    """

def emsemble( list1 , list2, list3, y_test):
    """

    ann and xgboost has best result currently

    """

    result = list()

    for x, y in zip(list1, list2):
        x = np.argmax(x)
        if x == 1 or  y == 1:
            result.append(1)
        else:
            result.append(0)

    # result2 = list()
    # for x, y in zip(list3, result):
    #         if x == 1 and y == 1:
    #             result2.append(1)
    #         else:
    #             result2.append(0)

    print("acc by emsemble model", accuracy_score(result, y_test))



if __name__ == "__main__":
    knn_pred = knn(X_train, X_test, y_train, y_test)
    ann_pred = ann(X_train, X_test, y_train, y_test)
    xgBoost_pred = xgBoost(X_train, X_test, y_train, y_test)
    # emsemble(ann_pred, xgBoost_pred, knn_pred,y_test)
    svm_c(X_train, X_test, y_train, y_test)

    