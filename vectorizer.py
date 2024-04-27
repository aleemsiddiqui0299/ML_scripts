from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]


vectorizer = TfidfVectorizer()
tfidf_mat= vectorizer.fit_transform(documents)
print(tfidf_mat.shape)

feature_names = vectorizer.get_feature_names_out()
print(tfidf_mat.toarray())
print(feature_names)