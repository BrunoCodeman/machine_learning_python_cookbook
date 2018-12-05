import numpy as np
from sklearn import preprocessing

data = np.array([[3, -1.5,  2, -5.4], [0,  4,  -0.3, 2.1], [1,  3.3, -1.9, -4.3]])
# https://scikit-learn.org/stable/modules/preprocessing.html
data_standardized = preprocessing.scale(data)
print(f"scaled data = {data_standardized}")
#centraliza a média em 0 para que não haja features enviesadas
mean = data_standardized.mean(axis=0)
std_deviation = data_standardized.std(axis=0)
print(f"\nMean = {mean}")
print(f"Std deviation = {std_deviation}")

#Escalar os valores de um datapoint pode ajudar a torná-los úteis, pois são aleatórios.
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print(f"\n Min max scaled data = {data_scaled}")

