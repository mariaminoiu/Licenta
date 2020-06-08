# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:58:58 2020

@author: maria
"""
# Pandas is used for data manipulation
import pandas as pd

# Read in data as pandas dataframe and display it
df = pd.read_csv('DF.csv')
print(df)

print('The shape of our features is:', df.shape)

# Descriptive statistics for each column
dsts=df.describe()
print(df.describe())


import matplotlib.pyplot as plt
from sklearn import linear_model

# Data preparation


X = df['HTEC'].values.reshape(-1,1)
y = df['PIB'].values

# Train

ols = linear_model.LinearRegression()
model = ols.fit(X, y)
response = model.predict(X)

# Evaluate

r2 = model.score(X, y)

# Plot 

plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(X, response, color='k', label='Dreapta de regresie')
ax.scatter(X, y, edgecolor='k', facecolor='grey', alpha=0.7, label='Valori')
ax.set_ylabel('Produsul intern brut (milioane RON)', fontsize=14)
ax.set_xlabel('Angajari in domeniul High Tech (persoane)', fontsize=14)
ax.text(0.8, 0.1, 'Realizat de Minoiu Maria', fontsize=13, ha='right', va='center',
         transform=ax.transAxes, color='grey', alpha=0.5)
ax.legend(facecolor='white', fontsize=11)
ax.set_title('$R^2= %.2f$' % r2, fontsize=18)

fig.tight_layout()


X = df['CH_CD'].values.reshape(-1,1)
y = df['PIB'].values

# Train

ols = linear_model.LinearRegression()
model = ols.fit(X, y)
response = model.predict(X)

# Evaluate 

r2 = model.score(X, y)

# Plot 

plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(X, response, color='k', label='Dreapta de regresie')
ax.scatter(X, y, edgecolor='k', facecolor='grey', alpha=0.7, label='Valori')
ax.set_ylabel('Produsul intern brut (milioane RON)', fontsize=14)
ax.set_xlabel('Cheltuieli cu cercetarea si dezvoltarea (milioane RON)', fontsize=14)
ax.text(0.8, 0.1, 'Realizat de Minoiu Maria', fontsize=13, ha='center', va='center',
         transform=ax.transAxes, color='grey', alpha=0.5)
ax.legend(facecolor='white', fontsize=11)
ax.set_title('$R^2= %.2f$' % r2, fontsize=18)

fig.tight_layout()



import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

# Data preparation



X = df[['CH_CD', 'HTEC']].values.reshape(-1,2)
Y = df['PIB']

# Prepare model data point for visualization

x = X[:, 0]
y = X[:, 1]
z = Y

x_pred = np.linspace(6, 24, 30)   
y_pred = np.linspace(0, 100, 30)  
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

# Train 

ols = linear_model.LinearRegression()
model = ols.fit(X, Y)
predicted = model.predict(model_viz)

# Evaluate 

r2 = model.score(X, Y)

# Plot 

plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('Cheltuieli cercetare si dezvoltare', fontsize=8)
    ax.set_ylabel('Angajari in High Tech', fontsize=8)
    ax.set_zlabel('Produsul intern brut', fontsize=8)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')


ax1.view_init(elev=28, azim=120)
ax2.view_init(elev=4, azim=114)
ax3.view_init(elev=60, azim=165)

fig.suptitle('$R^2 = %.2f$' % r2, fontsize=20)

fig.tight_layout()

#for ii in np.arange(0, 360, 1):
   #ax.view_init(elev=32, azim=ii)
   #fig.savefig('gif_image%d.png' % ii) # for making a gif



from sklearn import linear_model



features = ['CH_CD', 'CFE', 'HTEC']
target = 'PIB'

X = df[features].values.reshape(-1, len(features))
y = df[target].values

ols = linear_model.LinearRegression()
model = ols.fit(X, y)
print(model.coef_)
print(model.intercept_)

print(model.score(X, y))
x_pred = np.array([20, 30000, 9000])
x_pred = x_pred.reshape(-1, len(features))
print(model.predict(x_pred))



