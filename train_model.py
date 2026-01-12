from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

texts = [
    "я люблю этот фильм",
    "это было прекрасно",
    "мне очень понравилось",
    "всё просто супер",
    "это лучший день",
    "это отвратительно",
    "я ненавижу это",
    "ужасный опыт",
    "мне совсем не понравилось",
    "это худшее, что я видел",
    "сегодня обычный день",
    "нормально",
    "так себе",
    "ничего особенного",
    "просто ок",
]

labels = [
    "positive","positive","positive","positive","positive",
    "negative","negative","negative","negative","negative",
    "neutral","neutral","neutral","neutral","neutral",
]

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000)),
])

print("Training model...")
model.fit(texts, labels)

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "sentiment_model.pkl"
joblib.dump(model, model_path)
print(f"Saved to {model_path}")
