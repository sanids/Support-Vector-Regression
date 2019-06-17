# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values #must do this to convert to 2-D array

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling (MUST APPLY TO SVR AS IT DOES NOT DO AUTOMATICALLY)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y) 

# Fitting the SVR Model to the dataset

from sklearn.svm import SVR #importing SVR
regressor = SVR(kernel = 'rbf') #rbf is the gaussian kernel
regressor.fit(X, y)




# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]])))) #only transforming because data scaling is already fitted
#transforming 6.5 to a 2-D array
#inverse transforming so we can get the unscaled result

# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red') #make sure to inverse transform to convert back to unscaled
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue') #make sure to inverse transform to convert back to unscaled
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

