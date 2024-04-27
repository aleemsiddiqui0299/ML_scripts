import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm

data = pd.read_csv('movies_data.csv')
print(data.describe())

X = data['review'][:1000]
y = data['sentiment'][:1000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=1000)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

print(X_train_features.shape, X_test_features.shape)
clf = RandomForestClassifier(n_estimators=100,random_state=42)
with tqdm(total=len(X_train)) as pbar:
    print("Training the model...")
    for _ in range(clf.n_estimators):
        clf.fit(X_train_features, y_train)
        pbar.update(1)
print("Model Training completed.")

