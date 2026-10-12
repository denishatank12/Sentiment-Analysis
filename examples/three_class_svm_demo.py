"""Run a small, reproducible three-class SVM sentiment demonstration."""

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.three_class_svm import ThreeClassSentimentSVM


POSITIVE = ["excellent service", "great product", "wonderful support", "amazing value", "best purchase", "friendly staff"]
NEGATIVE = ["terrible service", "bad product", "awful support", "poor value", "worst purchase", "rude staff"]
NEUTRAL = ["ordinary service", "average product", "routine support", "reasonable value", "expected purchase", "factual staff"]
TEXTS = POSITIVE * 3 + NEGATIVE * 3 + NEUTRAL * 3
LABELS = (["positive"] * (len(POSITIVE) * 3) +
          ["negative"] * (len(NEGATIVE) * 3) +
          ["neutral"] * (len(NEUTRAL) * 3))


def main() -> None:
    """Train the model and print a classification report."""
    x_train, x_test, y_train, y_test = train_test_split(
        TEXTS, LABELS, test_size=0.25, random_state=42, stratify=LABELS
    )
    model = ThreeClassSentimentSVM().fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions, labels=["negative", "neutral", "positive"], zero_division=0))
    for text, prediction in zip(
        ["The experience was okay, but not memorable."],
        model.predict(["The experience was okay, but not memorable."]),
    ):
        print(f"{prediction:>8}: {text}")


if __name__ == "__main__":
    main()
