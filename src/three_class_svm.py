"""Three-class TF-IDF + linear SVM sentiment classifier.

This small, dependency-light example extends the repository's binary sentiment
examples with a neutral class.  It is designed for educational use: callers
provide text and labels, and the returned model can classify new text as
``positive``, ``negative``, or ``neutral``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


VALID_LABELS = frozenset({"negative", "neutral", "positive"})


@dataclass
class ThreeClassSentimentSVM:
    """Train and use a linear SVM for three-class text sentiment."""

    max_features: int = 10_000
    ngram_range: tuple[int, int] = (1, 2)
    random_state: int = 42

    def __post_init__(self) -> None:
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        max_features=self.max_features,
                        ngram_range=self.ngram_range,
                    ),
                ),
                ("svm", LinearSVC(random_state=self.random_state)),
            ]
        )
        self._is_fitted = False

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "ThreeClassSentimentSVM":
        """Fit the model after validating text and sentiment labels."""
        if len(texts) != len(labels) or not texts:
            raise ValueError("texts and labels must be non-empty and have equal length")
        invalid = set(labels) - VALID_LABELS
        if invalid:
            raise ValueError(f"Unsupported labels: {sorted(invalid)}")
        if len(set(labels)) < 3:
            raise ValueError("Training data must contain negative, neutral, and positive labels")
        self.pipeline.fit(texts, labels)
        self._is_fitted = True
        return self

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Return one sentiment label for each input text."""
        if not self._is_fitted:
            raise RuntimeError("Call fit before predict")
        if not texts:
            raise ValueError("texts must not be empty")
        return [str(label) for label in self.pipeline.predict(texts)]

    def score(self, texts: Sequence[str], labels: Sequence[str]) -> float:
        """Return mean accuracy on a labeled evaluation set."""
        if len(texts) != len(labels) or not texts:
            raise ValueError("texts and labels must be non-empty and have equal length")
        return float(self.pipeline.score(texts, labels))
