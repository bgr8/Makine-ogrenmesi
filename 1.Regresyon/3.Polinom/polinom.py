# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:37:16 2019

@author: Buğra
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial_regression.csv", sep = ';')

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("Araba Max Hız")
plt.xlabel("Araba Fiyatı")
plt.show()

# Linear Reg = y = b0 + b1*x
# Multi Reg = y = b0 + b1*x1 + b2*x2

# %% Linear Reg
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

# %% predict - hız tahmini
y_head = lr.predict(x)

plt.plot(x,y_head,color="red", label="Linear")
plt.show()

print("10 milyonTL lik araba hızı tahmini: ",lr.predict([[10000]]))

# %%
# Poly Reg. y = b0 + b1*x + b2*x^2 + b3*x^3 + .... + bn*x^n

from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree = 4)

x_poly = polyreg.fit_transform(x)

# %% fit
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

# %% 
y_head2 = lin_reg2.predict(x_poly)

plt.plot(x,y_head2,color="black",label="poly")
plt.legend()
plt.show()

























