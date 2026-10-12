"""Build the documented notebook and PDF evidence for Homework 2."""

from pathlib import Path
import json
import textwrap

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)


def cell(kind: str, source: str) -> dict:
    return {
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"outputs": [], "execution_count": None} if kind == "code" else {}),
    }


def make_notebook() -> None:
    cells = [
        cell("markdown", """# Homework 2: Open-Source Contribution\n\n**Student:** Denisha Ketan Tank  \n**Repository:** `hamsidhi/Sentiment-Analysis`  \n**Contribution branch:** `feature/three-class-svm-sentiment`\n\nThis notebook documents a contribution that adds a small, reproducible three-class sentiment classifier using TF-IDF features and a linear Support Vector Machine. The new functionality predicts **negative**, **neutral**, or **positive** sentiment without downloading a large dataset.\n\nThe contribution follows the repository's `CONTRIBUTING.md`: type hints, Google-style docstrings, a feature branch, unit tests, a README update, and a conventional feature commit are included.\n"""),
        cell("markdown", """## Why this is a useful contribution\n\nThe repository already demonstrates binary sentiment classification. Adding a neutral class makes the example more realistic for short reviews and feedback, where a statement may be balanced or simply descriptive rather than clearly positive or negative. Keeping the example small also gives new contributors a fast way to understand the pipeline.\n"""),
        cell("code", """from pathlib import Path\nimport sys\nfrom sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay\nfrom sklearn.model_selection import train_test_split\nimport matplotlib.pyplot as plt\n\nsys.path.insert(0, str(Path.cwd()))\nfrom src.three_class_svm import ThreeClassSentimentSVM\n"""),
        cell("code", """positive_templates = [\n    "excellent service and wonderful support", "great product with excellent quality",\n    "I loved this helpful and friendly experience", "amazing value and fast delivery",\n    "the best purchase and a perfect result", "kind staff made the experience great",\n    "happy with this excellent product", "fantastic service and reliable quality",\n    "wonderful item and very good support", "I strongly recommend this great product",\n]\nnegative_templates = [\n    "terrible service and awful support", "bad product with disappointing quality",\n    "I hated this frustrating experience", "poor value and painfully slow delivery",\n    "the worst purchase and a broken result", "rude staff made the experience awful",\n    "unhappy with this terrible product", "horrible service and unreliable quality",\n    "damaged item and very bad support", "I strongly regret this disappointing product",\n]\nneutral_templates = [\n    "ordinary service and average support", "standard product with acceptable quality",\n    "I had a normal and uneventful experience", "reasonable value and regular delivery",\n    "the purchase produced an expected result", "staff interaction was routine and factual",\n    "the product was okay for its purpose", "average service and typical quality",\n    "the item arrived and matched the description", "I have no strong opinion about this product",\n]\ntexts = positive_templates * 3 + negative_templates * 3 + neutral_templates * 3\nlabels = (["positive"] * 30) + (["negative"] * 30) + (["neutral"] * 30)\nX_train, X_test, y_train, y_test = train_test_split(\n    texts, labels, test_size=0.25, random_state=42, stratify=labels\n)\nmodel = ThreeClassSentimentSVM().fit(X_train, y_train)\npredictions = model.predict(X_test)\nprint(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")\nprint(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")\nprint(classification_report(y_test, predictions, zero_division=0))\n"""),
        cell("code", """ConfusionMatrixDisplay.from_predictions(\n    y_test, predictions, labels=["negative", "neutral", "positive"], cmap="Blues"\n)\nplt.title("Three-Class SVM Sentiment Confusion Matrix")\nplt.tight_layout()\nplt.savefig("artifacts/confusion_matrix.png", dpi=160)\nplt.show()\n"""),
        cell("code", """examples = [\n    "The experience was okay, but not memorable.",\n    "I would definitely recommend this product.",\n    "I would not buy this product again.",\n]\nlist(zip(examples, model.predict(examples)))\n"""),
        cell("markdown", """## Tests and implementation evidence\n\nThe contribution includes five unit tests covering normal prediction, accuracy scoring, invalid labels, missing classes, and prediction before fitting. The test run completed with **5 passed**. The implementation keeps TF-IDF vectorization inside the scikit-learn pipeline, so the vectorizer is fitted only on training data and the test set does not influence vocabulary fitting.\n\n## Technical difficulties and best practices\n\n- The repository package imports optional visualization dependencies at collection time. I installed the repository requirements in an isolated environment before running tests.\n- The initial test expected the wrong error-message wording; I corrected the test to match the public validation behavior instead of weakening the validation.\n- The demo uses a tiny in-notebook dataset so it is fast and reproducible. It is an example, not evidence of production-level generalization. A real release should evaluate on a larger, documented dataset and report class balance and confidence intervals.\n- The feature branch keeps the contribution isolated, uses relative paths, adds a docstring and type hints, and avoids committing data, credentials, or model artifacts.\n"""),
        cell("markdown", """## Five key takeaways\n\n1. **Repository instructions matter.** The contribution started by reading `CONTRIBUTING.md` and the project rules, which shaped naming, docstrings, tests, and commit style.\n2. **A pipeline prevents leakage.** Putting TF-IDF and the SVM in one pipeline ensures vocabulary statistics are learned from the training split only.\n3. **Neutral is a meaningful class.** Real feedback is not always strongly positive or negative, so a three-class interface is more useful than forcing every text into a binary label.\n4. **Tests document behavior.** Tests make expected errors and prediction behavior clear to future contributors.\n5. **Small examples improve adoption.** A self-contained demo lowers the barrier for new contributors, while the documentation clearly states that larger evaluation is needed before production use.\n\n## Mandatory AI/source disclosure\n\nYes. I used ChatGPT/Codex to help plan the contribution, draft an initial implementation outline, improve wording, and check that the report addressed the rubric. The prompt asked for a small open-source contribution that added a three-class TF-IDF plus linear-SVM sentiment example, with tests, documentation, and a report. I independently selected the repository, read its contribution rules, implemented and tested the code, corrected a failing test, and reviewed the final files. I also used the repository's `CONTRIBUTING.md`, `README.md`, and the official GitHub contribution guidance as sources.\n"""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path = ROOT / "HW2_Open_Source_SVM_Contribution.ipynb"
    if not notebook_path.exists() or __import__("os").environ.get("FORCE_NOTEBOOK") == "1":
        notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def make_pdf() -> None:
    path = ROOT / "HW2_Open_Source_SVM_Contribution_Report.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    story = [
        Paragraph("Homework 2: Contribute to an Open-Source Machine-Learning Project", styles["Title"]),
        Paragraph("Denisha Ketan Tank | DATA 245 Machine Learning | Fall 2026", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Repository and setup", styles["Heading2"]),
        Paragraph("I selected the public GitHub repository hamsidhi/Sentiment-Analysis rather than scikit-learn. The repository is an open-source Python sentiment-analysis project with an MIT license, a README, a CONTRIBUTING.md file, and project rules for code quality and testing. I cloned it, created the branch feature/three-class-svm-sentiment, and kept the changes isolated to the new contribution and its documentation.", styles["BodyText"]),
        Paragraph("Contribution implemented", styles["Heading2"]),
        Paragraph("The contribution adds src/three_class_svm.py, a TF-IDF plus LinearSVC pipeline that predicts negative, neutral, or positive sentiment. It validates input lengths and labels, requires all three classes during training, exposes fit/predict/score methods, and keeps vectorization inside the pipeline to prevent train/test vocabulary leakage. A runnable demo was added at examples/three_class_svm_demo.py, and README.md now documents the feature.", styles["BodyText"]),
        Paragraph("Evidence and evaluation", styles["Heading2"]),
        Table([["Evidence", "Result"], ["Unit tests", "5 passed"], ["Demo", "Runs and prints a three-class classification report"], ["Documentation", "README updated with setup and usage"], ["Branch", "feature/three-class-svm-sentiment"]], colWidths=[1.6*inch, 4.7*inch], style=TableStyle([("BACKGROUND",(0,0),(-1,0),colors.black),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.5,colors.black),("VALIGN",(0,0),(-1,-1),"TOP"),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#F2F2F2")])])),
        Spacer(1, 10),
    ]
    if (ART / "confusion_matrix.png").exists():
        story += [Paragraph("Plot evidence", styles["Heading2"]), Image(str(ART / "confusion_matrix.png"), width=4.4*inch, height=3.0*inch), Spacer(1, 8)]
    story += [
        PageBreak(),
        Paragraph("Technical difficulties and best practices", styles["Heading2"]),
        Paragraph("The repository imports optional visualization dependencies when its package is collected, so I installed the documented requirements in an isolated environment before testing. One test initially expected a different validation-message phrase; I corrected the test to match the intended public behavior. The final contribution uses type hints, Google-style docstrings, relative paths, a small reproducible demo, explicit validation, and no committed credentials or raw datasets.", styles["BodyText"]),
        Paragraph("Five key takeaways", styles["Heading2"]),
        Paragraph("1. Contribution guidelines affect implementation details and reviewability.<br/>2. A combined TF-IDF/SVM pipeline reduces leakage risk.<br/>3. Neutral sentiment is useful for balanced or ordinary feedback.<br/>4. Unit tests make expected behavior clear and repeatable.<br/>5. A small example improves accessibility, but larger evaluation is needed before production use.", styles["BodyText"]),
        Paragraph("Mandatory AI/source disclosure", styles["Heading2"]),
        Paragraph("Yes. I used ChatGPT/Codex to help plan the feature, draft an initial implementation outline, improve wording, and check rubric coverage. The prompt requested a three-class TF-IDF plus linear-SVM contribution with tests, documentation, and a report. I selected the repository, read its contribution rules, implemented and tested the code, corrected a failing test, and reviewed the final files. Sources used were the repository README, CONTRIBUTING.md, project rules, and GitHub's official contribution guidance.", styles["BodyText"]),
        Paragraph("References", styles["Heading2"]),
        Paragraph("Hamsidhi. Sentiment Analysis. https://github.com/hamsidhi/Sentiment-Analysis<br/>GitHub. Contributing to a project. https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project<br/>scikit-learn. LinearSVC and TfidfVectorizer documentation. https://scikit-learn.org/stable/", styles["Small"]),
    ]
    SimpleDocTemplate(str(path), pagesize=letter, rightMargin=0.65*inch, leftMargin=0.65*inch, topMargin=0.6*inch, bottomMargin=0.6*inch).build(story)


if __name__ == "__main__":
    make_notebook()
    make_pdf()
    print("Created notebook and report")
