#File created for advanced decision tree pipeline to be used to fetch data from a stream / file 
#Then model creation followed by model training happens and it will save the model in some compressed format


#adding some neccessary imports
import numpy as np
import pandas as pd
from sklearn.datasets import  fetch_openml

from sklearn.model_selection import train_test_split, GridSearchCV
