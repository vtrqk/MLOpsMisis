from app.core.model import SentimentModel


def test_model_predict():
    model = SentimentModel()
    label, probs = model.predict("мне очень понравилось")

    assert label in model.labels
    assert set(probs.keys()) == set(model.labels)
    assert abs(sum(probs.values()) - 1.0) < 1e-3
