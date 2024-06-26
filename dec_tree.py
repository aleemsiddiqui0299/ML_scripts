import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
preprocessor = Pipeline([
    ('scaler', StandardScaler())  # Scale features
])

# Define the Decision Tree Classifier
classifier = DecisionTreeClassifier(random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

test_predictions = pipeline.predict(X_test)
acc = accuracy_score(y_test, test_predictions)
print("Accuracy : ", acc)

print("Saving model pipeline")
joblib.dump(pipeline,"decision_tree_model.pkl")
