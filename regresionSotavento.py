from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.metrics import r2_score

data2016WithAllColumns = pd.read_csv('datos_sotavento/data_stv_2016.csv')

data2016 = data2016WithAllColumns[['10u_(43.5, -8.0)', '10u_(43.5, -7.875)', '10u_(43.5, -7.75)',
'10u_(43.375, -8.0)', '10u_(43.375, -7.875)', '10u_(43.375, -7.75)',
'10u_(43.25, -8.0)', '10u_(43.25, -7.875)', '10u_(43.25, -7.75)',
'10v_(43.5, -8.0)', '10v_(43.5, -7.875)', '10v_(43.5, -7.75)',
'10v_(43.375, -8.0)', '10v_(43.375, -7.875)', '10v_(43.375, -7.75)',
'10v_(43.25, -8.0)', '10v_(43.25, -7.875)', '10v_(43.25, -7.75)',
'100u_(43.5, -8.0)', '100u_(43.5, -7.875)', '100u_(43.5, -7.75)',
'100u_(43.375, -8.0)', '100u_(43.375, -7.875)', '100u_(43.375, -7.75)',
'100u_(43.25, -8.0)', '100u_(43.25, -7.875)', '100u_(43.25, -7.75)',
'100v_(43.5, -8.0)', '100v_(43.5, -7.875)', '100v_(43.5, -7.75)',
'100v_(43.375, -8.0)', '100v_(43.375, -7.875)', '100v_(43.375, -7.75)',
'100v_(43.25, -8.0)', '100v_(43.25, -7.875)', '100v_(43.25, -7.75)',
'2t_(43.5, -8.0)', '2t_(43.5, -7.875)', '2t_(43.5, -7.75)',
'2t_(43.375, -8.0)', '2t_(43.375, -7.875)', '2t_(43.375, -7.75)',
'2t_(43.25, -8.0)', '2t_(43.25, -7.875)', '2t_(43.25, -7.75)',
'sp_(43.5, -8.0)', 'sp_(43.5, -7.875)', 'sp_(43.5, -7.75)',
'sp_(43.375, -8.0)', 'sp_(43.375, -7.875)', 'sp_(43.375, -7.75)',
'sp_(43.25, -8.0)', 'sp_(43.25, -7.875)', 'sp_(43.25, -7.75)',
'vel10_(43.5, -8.0)', 'vel10_(43.5, -7.875)', 'vel10_(43.5, -7.75)',
'vel10_(43.375, -8.0)', 'vel10_(43.375, -7.875)', 'vel10_(43.375, -7.75)',
'vel10_(43.25, -8.0)', 'vel10_(43.25, -7.875)', 'vel10_(43.25, -7.75)',
'vel100_(43.5, -8.0)', 'vel100_(43.5, -7.875)', 'vel100_(43.5, -7.75)',
'vel100_(43.375, -8.0)', 'vel100_(43.375, -7.875)', 'vel100_(43.375, -7.75)',
'vel100_(43.25, -8.0)', 'vel100_(43.25, -7.875)', 'vel100_(43.25, -7.75)']]

target2016 = pd.read_csv('datos_sotavento/target_stv_2016.csv', names=['date','obtainedPower'])

data2017WithAllColumns = pd.read_csv('datos_sotavento/data_stv_2017.csv')

data2017 = data2017WithAllColumns[['10u_(43.5, -8.0)', '10u_(43.5, -7.875)', '10u_(43.5, -7.75)',
'10u_(43.375, -8.0)', '10u_(43.375, -7.875)', '10u_(43.375, -7.75)',
'10u_(43.25, -8.0)', '10u_(43.25, -7.875)', '10u_(43.25, -7.75)',
'10v_(43.5, -8.0)', '10v_(43.5, -7.875)', '10v_(43.5, -7.75)',
'10v_(43.375, -8.0)', '10v_(43.375, -7.875)', '10v_(43.375, -7.75)',
'10v_(43.25, -8.0)', '10v_(43.25, -7.875)', '10v_(43.25, -7.75)',
'100u_(43.5, -8.0)', '100u_(43.5, -7.875)', '100u_(43.5, -7.75)',
'100u_(43.375, -8.0)', '100u_(43.375, -7.875)', '100u_(43.375, -7.75)',
'100u_(43.25, -8.0)', '100u_(43.25, -7.875)', '100u_(43.25, -7.75)',
'100v_(43.5, -8.0)', '100v_(43.5, -7.875)', '100v_(43.5, -7.75)',
'100v_(43.375, -8.0)', '100v_(43.375, -7.875)', '100v_(43.375, -7.75)',
'100v_(43.25, -8.0)', '100v_(43.25, -7.875)', '100v_(43.25, -7.75)',
'2t_(43.5, -8.0)', '2t_(43.5, -7.875)', '2t_(43.5, -7.75)',
'2t_(43.375, -8.0)', '2t_(43.375, -7.875)', '2t_(43.375, -7.75)',
'2t_(43.25, -8.0)', '2t_(43.25, -7.875)', '2t_(43.25, -7.75)',
'sp_(43.5, -8.0)', 'sp_(43.5, -7.875)', 'sp_(43.5, -7.75)',
'sp_(43.375, -8.0)', 'sp_(43.375, -7.875)', 'sp_(43.375, -7.75)',
'sp_(43.25, -8.0)', 'sp_(43.25, -7.875)', 'sp_(43.25, -7.75)',
'vel10_(43.5, -8.0)', 'vel10_(43.5, -7.875)', 'vel10_(43.5, -7.75)',
'vel10_(43.375, -8.0)', 'vel10_(43.375, -7.875)', 'vel10_(43.375, -7.75)',
'vel10_(43.25, -8.0)', 'vel10_(43.25, -7.875)', 'vel10_(43.25, -7.75)',
'vel100_(43.5, -8.0)', 'vel100_(43.5, -7.875)', 'vel100_(43.5, -7.75)',
'vel100_(43.375, -8.0)', 'vel100_(43.375, -7.875)', 'vel100_(43.375, -7.75)',
'vel100_(43.25, -8.0)', 'vel100_(43.25, -7.875)', 'vel100_(43.25, -7.75)']]

target2017 = pd.read_csv('datos_sotavento/target_stv_2017.csv', names=['date','obtainedPower'])

data2018WithAllColumns = pd.read_csv('datos_sotavento/data_stv_2018.csv')

data2018 = data2018WithAllColumns[['10u_(43.5, -8.0)', '10u_(43.5, -7.875)', '10u_(43.5, -7.75)',
'10u_(43.375, -8.0)', '10u_(43.375, -7.875)', '10u_(43.375, -7.75)',
'10u_(43.25, -8.0)', '10u_(43.25, -7.875)', '10u_(43.25, -7.75)',
'10v_(43.5, -8.0)', '10v_(43.5, -7.875)', '10v_(43.5, -7.75)',
'10v_(43.375, -8.0)', '10v_(43.375, -7.875)', '10v_(43.375, -7.75)',
'10v_(43.25, -8.0)', '10v_(43.25, -7.875)', '10v_(43.25, -7.75)',
'100u_(43.5, -8.0)', '100u_(43.5, -7.875)', '100u_(43.5, -7.75)',
'100u_(43.375, -8.0)', '100u_(43.375, -7.875)', '100u_(43.375, -7.75)',
'100u_(43.25, -8.0)', '100u_(43.25, -7.875)', '100u_(43.25, -7.75)',
'100v_(43.5, -8.0)', '100v_(43.5, -7.875)', '100v_(43.5, -7.75)',
'100v_(43.375, -8.0)', '100v_(43.375, -7.875)', '100v_(43.375, -7.75)',
'100v_(43.25, -8.0)', '100v_(43.25, -7.875)', '100v_(43.25, -7.75)',
'2t_(43.5, -8.0)', '2t_(43.5, -7.875)', '2t_(43.5, -7.75)',
'2t_(43.375, -8.0)', '2t_(43.375, -7.875)', '2t_(43.375, -7.75)',
'2t_(43.25, -8.0)', '2t_(43.25, -7.875)', '2t_(43.25, -7.75)',
'sp_(43.5, -8.0)', 'sp_(43.5, -7.875)', 'sp_(43.5, -7.75)',
'sp_(43.375, -8.0)', 'sp_(43.375, -7.875)', 'sp_(43.375, -7.75)',
'sp_(43.25, -8.0)', 'sp_(43.25, -7.875)', 'sp_(43.25, -7.75)',
'vel10_(43.5, -8.0)', 'vel10_(43.5, -7.875)', 'vel10_(43.5, -7.75)',
'vel10_(43.375, -8.0)', 'vel10_(43.375, -7.875)', 'vel10_(43.375, -7.75)',
'vel10_(43.25, -8.0)', 'vel10_(43.25, -7.875)', 'vel10_(43.25, -7.75)',
'vel100_(43.5, -8.0)', 'vel100_(43.5, -7.875)', 'vel100_(43.5, -7.75)',
'vel100_(43.375, -8.0)', 'vel100_(43.375, -7.875)', 'vel100_(43.375, -7.75)',
'vel100_(43.25, -8.0)', 'vel100_(43.25, -7.875)', 'vel100_(43.25, -7.75)']]

target2018 = pd.read_csv('datos_sotavento/target_stv_2018.csv', names=['date','obtainedPower'])

data3Years = pd.concat([data2016, data2017, data2018])
target3Years = pd.concat([target2016, target2017, target2018])


lin_model = LinearRegression()

X_train, X_test, Y_train, Y_test = train_test_split(data3Years, target3Years['obtainedPower'], test_size = 0.2, random_state=4)
lin_model.fit(X_train, Y_train)
Y_test_predict = lin_model.predict(X_test)


print("Mean squared error:")
print(np.sqrt(mean_squared_error(Y_test, Y_test_predict)))
print("R^2:")
print(r2_score(Y_test_predict, Y_test))


# for j in range(100):
#         lin_model = LinearRegression()
#         X_train, X_test, Y_train, Y_test = train_test_split(dataThreeYears, targetThreeYears, test_size = 0.2, random_state=j)
#         lin_model.fit(X_train, Y_train)
#         Y_test_predict = lin_model.predict(X_test)
#         rmseForEachSplitSeed.append(np.sqrt(mean_squared_error(Y_test, Y_test_predict)))
#         r2ForEachSplitSeed.append(r2_score(Y_test_predict, Y_test))

# print("Mean squared error:")
# print(sum(rmseForEachSplitSeed)/len(rmseForEachSplitSeed))
# print("R^2:")
# print(sum(r2ForEachSplitSeed)/len(r2ForEachSplitSeed))
