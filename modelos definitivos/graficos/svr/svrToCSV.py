###
# A
# Autor: Jose Benjumeda Rubio
# 202102 24-3
# Voy a probar una svr con los parametros que me ha dicho Angela que uso Maria.
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import json
import time

data2016 = pd.read_csv('datos_sotavento/data_stv_2016.csv')
data2017 = pd.read_csv('datos_sotavento/data_stv_2017.csv')
data2018 = pd.read_csv('datos_sotavento/data_stv_2018.csv')

target2016 = pd.read_csv('datos_sotavento/target_stv_2016.csv',names=['prediction date','total_power'])
target2017 = pd.read_csv('datos_sotavento/target_stv_2017.csv',names=['prediction date','total_power'])
target2018 = pd.read_csv('datos_sotavento/target_stv_2018.csv',names=['prediction date','total_power'])


sotavento = [43.354377,-7.881213] #(latitud,longitud)
PM = 17560

# 43.25 <= latitud <= 43.5
# -8 <= longitud <= -7.75

max_latitud = 43.5
min_latitud = 43.25
max_longitud = -7.75
min_longitud = -8

# key = 10u_(latitud, longitud)
# return (latitud,longitud)
def coord(key):
   return list(map(float, (key.split("_")[1].replace('(','').replace(')','').split(','))))


def new_dataframe(datayear):
    keysyear= list(datayear.keys())
    data = {}
    for key in keysyear:
        if key.find('_') != -1:
            coordinate = coord(key)
            if coordinate[0] >= min_latitud and coordinate[0] <= max_latitud:
                if coordinate[1] >= min_longitud and coordinate[1] <= max_longitud:
                    data[key] = datayear[key]
        else:
            data[key] = datayear[key]

    return pd.DataFrame(data)

data2016 = new_dataframe(data2016)
data2017 = new_dataframe(data2017)
data2018 = new_dataframe(data2018)

# CORRECT PREDICTION DATE
def correct_prediction_date():
    target2016['prediction date'] = data2016['prediction date']
    target2017['prediction date'] = data2017['prediction date']
    target2018['prediction date'] = data2018['prediction date']

def np_dataset(X_train,Y_train,X_test,Y_test):
    return np.array(X_train), np.array(Y_train),np.array(X_test),np.array(Y_test)


correct_prediction_date()

X_train = pd.concat([data2016, data2017])
Y_train = pd.concat([target2016, target2017])
X_test = data2018
Y_test = target2018

Y_train = Y_train['total_power']
Y_test = Y_test['total_power']
X_train = X_train.drop(['prediction date'],axis=1)
X_test = X_test.drop(['prediction date'],axis=1)

X_train,Y_train,X_test,Y_test = np_dataset(X_train,Y_train,X_test,Y_test)
#print_shape_dataset(X_train,Y_train,X_test,Y_test)

scaler = StandardScaler(with_mean=False).fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


svr = SVR(C=10000.0, cache_size=200, coef0=0.0, degree=3, epsilon=53.46025114271836,
    gamma=0.003472222222222222, kernel='rbf', max_iter=-1, shrinking=True,
    tol=0.001, verbose=False)


svr.fit(X_train, Y_train)
Y_test_predict = svr.predict(X_test)

pd.DataFrame(Y_test_predict).to_csv('svr_Y_test_predict.csv', index=False)
