# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:12:49 2019

@author: Buğra
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("multiple_linear_regression_dataset.csv", sep = ';')

# iloc ile : tüm satırları al 0 ve 2 sütunları al
x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)

# %%
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0: ", multiple_linear_regression.intercept_)
print("b1,b2", multiple_linear_regression.coef_)

multiple_linear_regression.predict(np.array([[10,35],[5,35]]))