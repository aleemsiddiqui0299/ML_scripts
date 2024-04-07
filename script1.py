import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

# boston_df = load_boston()
print(housing.keys())
