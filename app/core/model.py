from pathlib import Path
from typing import Dict, Tuple

import joblib


class SentimentModel:
    def __init__(self, model_path: str | Path = "models/sentiment_model.pkl"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Run: python train_model.py"
            )
        self._model = joblib.load(self.model_path)
        self.labels: list[str] = list(self._model.classes_)

    def predict(self, text: str) -> Tuple[str, Dict[str, float]]:
        proba = self._model.predict_proba([text])[0]
        label_idx = int(proba.argmax())
        label = self.labels[label_idx]
        probs = {lbl: float(p) for lbl, p in zip(self.labels, proba)}
        return label, probs
