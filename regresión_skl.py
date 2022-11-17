import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

!gdown '1YB1vIna6NmitE36EmqO1ZwfE3ViQopeb' -O dataset.csv

datos = pd.read_csv("dataset.csv")
x = np.array(datos["x"])
y = np.array(datos["y"])

media = np.mean(x)
#desviación estandar
sigma = np.std(x)
#estanderización
x = (x-media)/sigma
plt.title("Visualización de datos")
plt.plot(x,y, 'o', color='cyan', mec='black')

x = x.reshape(-1, 1)
modelo = linear_model.LinearRegression()
modelo.fit(x, y)

h = modelo.predict(x)
plt.plot(x, h, 'g')
print('a0 = ', modelo.intercept_, 'a1 = ', modelo.coef_[0])
