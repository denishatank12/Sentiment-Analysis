"""Unit tests for the three-class SVM contribution."""

import pytest

from src.three_class_svm import ThreeClassSentimentSVM


TEXTS = ["great service", "excellent product", "terrible service", "awful product", "ordinary service", "average product"]
LABELS = ["positive", "positive", "negative", "negative", "neutral", "neutral"]


def test_fit_and_predict_returns_valid_labels() -> None:
    model = ThreeClassSentimentSVM().fit(TEXTS, LABELS)
    predictions = model.predict(["excellent service", "ordinary product", "terrible product"])
    assert len(predictions) == 3
    assert set(predictions).issubset({"negative", "neutral", "positive"})


def test_score_is_between_zero_and_one() -> None:
    model = ThreeClassSentimentSVM().fit(TEXTS, LABELS)
    assert 0.0 <= model.score(TEXTS, LABELS) <= 1.0


def test_rejects_invalid_label() -> None:
    with pytest.raises(ValueError, match="Unsupported labels"):
        ThreeClassSentimentSVM().fit(TEXTS, [*LABELS[:-1], "mixed"])


def test_requires_all_three_classes() -> None:
    with pytest.raises(ValueError, match="negative, neutral, and positive"):
        ThreeClassSentimentSVM().fit(TEXTS[:4], LABELS[:4])


def test_requires_fit_before_prediction() -> None:
    with pytest.raises(RuntimeError, match="Call fit"):
        ThreeClassSentimentSVM().predict(["hello"])
