###
# A
# Autor: Jose Benjumeda Rubio
# 202102 24-3
# Este programa carga los datos a dataframe y luego hace una ridge regression.
# Divide en conjunto de test (20%) y entrenamiento (80%) y luego predice,
# obteniendo el error cuadratico medio y el r2.
# El modelo se crea con distintos valores para el hiperparámetro de regularizacion
# para ver los disintos rendimientos segun el valor.
#

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

data2016 = pd.read_csv('datos_sotavento/data_stv_2016.csv')
data2017 = pd.read_csv('datos_sotavento/data_stv_2017.csv')
data2018 = pd.read_csv('datos_sotavento/data_stv_2018.csv')

target2016 = pd.read_csv('datos_sotavento/target_stv_2016.csv',names=['prediction date','total_power'])
target2017 = pd.read_csv('datos_sotavento/target_stv_2017.csv',names=['prediction date','total_power'])
target2018 = pd.read_csv('datos_sotavento/target_stv_2018.csv',names=['prediction date','total_power'])


lat_max = 43.5
lat_min = 43.25
len_max = -7.75
len_min = -8

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
            if coordinate[0] >= lat_min and coordinate[0] <= lat_max:
                if coordinate[1] >= len_min and coordinate[1] <= len_max:
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

parameters = {
    'C': [10.**exp for exp in range (3, 6)],
    'epsilon': list(Y_train.std() * np.array([2.**exp for exp in range(-6,-3)])),
    'gamma': list(np.array([4.**exp for exp in range(-2, 2)])/X_test.shape[1])}

grid = GridSearchCV(SVR(), param_grid=parameters, scoring='neg_mean_absolute_error')
grid.fit(X_train, Y_train)
print(grid.best_score_)
print(grid.cv_results_)
print(grid.best_estimator_)
Y_test_predict = grid.predict(X_test)

print("Mean absolute error:")
print(mean_absolute_error(Y_test, Y_test_predict)/17560)
print("R^2:")
print(r2_score(Y_test_predict, Y_test))
