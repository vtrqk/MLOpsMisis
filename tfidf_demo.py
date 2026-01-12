from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "мне очень понравилось",
    "мне совсем не понравилось",
    "это было прекрасно",
]

vec = TfidfVectorizer(ngram_range=(1, 2))
X = vec.fit_transform(docs)

print("Тип X:", type(X))
print("Форма X:", X.shape)
print("\nПризнаки (feature names):")
print(list(vec.get_feature_names_out()))

print("\nМатрица X (в плотном виде, для понимания):")
print(X.toarray())

print("\nРазреженная запись первой строки (только ненулевые координаты):")
row0 = X[0]
idx = row0.nonzero()[1]
for i, v in zip(idx, row0.data):
    print(i, vec.get_feature_names_out()[i], float(v))
