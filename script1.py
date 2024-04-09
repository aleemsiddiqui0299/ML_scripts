import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
n_samples, n_features = X.shape


print(f"The dataset consists of {n_samples} samples and {n_features} features")
# print(X.describe())
# print(X, y)
housing = fetch_california_housing()
print(housing.keys())
# print(housing.DESCR)
print(housing.target)
print(housing.feature_names)
print(housing.target_names)


#Preparing dataset with some analysis
dataset = pd.DataFrame(housing.data,columns=housing.feature_names)
dataset['Price'] = housing.target
print(dataset.head())
print(dataset.info())


## Summarizing data statistics
#1. check missing values
print(dataset.describe())
print(dataset.isnull().sum())

### Exploratory Data Analysis
## Correlation
## pearson method by default
print(dataset.corr())

#plot corr matrix
corr = dataset.corr()
corr.style.background_gradient(cmap='BrBG')

#Heatmap
import seaborn
seaborn.heatmap(corr)   

import matplotlib.pyplot as plt

#scatter plot for individual correlation representation
# plt.scatter(dataset['HouseAge'], dataset['Population'])
# plt.xlabel("House Age in years")
# plt.ylabel("Population")

plt.scatter(dataset['AveBedrms'], dataset['AveRooms'])
plt.xlabel("Avg bedroom count")
plt.ylabel("Avg room count")
plt.legend()

import seaborn as sns

sns.regplot(x='AveBedrms',y='AveRooms',data=dataset)
# sns.regplot(x='HouseAge',y='Population', data = dataset)

# Independent and dependent features differentiated

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
print("X head : ",X.head())

# Train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
print("X train : ",X_train.head())
print("X test : ",X_test.head())

# Standardize dataset

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("X train : ",X_train.shape)
print("X test : ",X_test.shape)

# Model training

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)

