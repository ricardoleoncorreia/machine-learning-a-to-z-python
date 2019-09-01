# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

# Fitting SVR to the dataset (Gaussian Kernel)
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
X_pred = sc_X.transform([[6.5]])
y_pred = regressor.predict(X_pred)
y_pred = sc_y.inverse_transform(y_pred)

# Revert feature scaling for training set
X_plot = sc_X.inverse_transform(X)
y_plot = sc_y.inverse_transform(y)
y_plot_pred = sc_y.inverse_transform(regressor.predict(X))

# Visualising the SVR results
plt.scatter(X_plot, y_plot, color = 'red')
plt.plot(X_plot, y_plot_pred, color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Revert feature scaling for test data
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
X_grid_plot = sc_X.inverse_transform(X_grid)
y_grid_plot = sc_y.inverse_transform(regressor.predict(X_grid))

# Visualising the SVR results (for higher resolution and smoother curve)
plt.scatter(X_plot, y_plot, color = 'red')
plt.plot(X_grid_plot, y_grid_plot, color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()