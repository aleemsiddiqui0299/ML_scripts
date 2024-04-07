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
