import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm


def load_data():
    data = pd.read_csv('movies_data.csv')
    print(data.describe())
    return data

def split_data(data):
    X = data['review'][:1000]
    y = data['sentiment'][:1000]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test 


def extract_features(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    print(X_train_features.shape, X_test_features.shape)
    return vectorizer, X_train_features, X_test_features

def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100,random_state=42)
    with tqdm(total=clf.n_estimators) as pbar:
        print("Training the model...")
        for _ in range(clf.n_estimators):
            clf.fit(X_train, y_train)
            pbar.update(1)
    print("Model Training completed.")

def make_predictions(model, X_test):
    pass

def evaluate_model(y_true, y_pred):
    pass

def save_model(model, filename):
    pass

def load_model(filename):
    model = joblib.load(filename)
    return model

def retrain_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

