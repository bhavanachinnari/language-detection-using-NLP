# Language Detection using NLP 🌐

A machine learning model that detects the language of a given text across **17 languages**, achieving **98.45% accuracy** using a Multinomial Naive Bayes classifier with Bag-of-Words text representation.

---

## Features

- **17 Language Support** — Detects Arabic, Danish, Dutch, English, French, German, Greek, Hindi, Italian, Kannada, Malayalam, Portugese, Russian, Spanish, Swedish, Tamil, and Turkish
- **High Accuracy** — Achieves **98.45% test accuracy** on 10,337 text samples
- **Fast Inference** — Lightweight Naive Bayes model with CountVectorizer for real-time predictions
- **Confusion Matrix Visualization** — Heatmap visualization of classification performance

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Model | Multinomial Naive Bayes (scikit-learn) |
| Text Vectorization | CountVectorizer (Bag-of-Words) |
| Label Encoding | LabelEncoder (scikit-learn) |
| Visualization | Seaborn, Matplotlib |
| Language | Python (Jupyter Notebook) |

---

## Dataset

- **10,337 text samples** across 17 languages
- Each sample is a paragraph of text labeled with its language
- Languages range from European (French, German, Spanish) to South Asian (Hindi, Kannada, Malayalam, Tamil)

| Language | Samples |
|---|---|
| English | 1385 |
| French | 1014 |
| Spanish | 819 |
| Portugese | 739 |
| Italian | 698 |
| Russian | 692 |
| Swedish | 676 |
| Malayalam | 594 |
| Dutch | 546 |
| Arabic | 536 |
| Turkish | 474 |
| German | 470 |
| Tamil | 469 |
| Danish | 428 |
| Kannada | 369 |
| Greek | 365 |
| Hindi | 63 |

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Install Dependencies

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

### Run the Notebook

```bash
git clone https://github.com/bhavanachinnari/language-detection-using-NLP.git
cd language-detection-using-NLP
jupyter notebook
```

---

## How It Works

1. **Preprocessing** — Text is cleaned by removing symbols, numbers, and special characters, then converted to lowercase
2. **Vectorization** — Text is converted to a Bag-of-Words representation using `CountVectorizer` (vocabulary size: 34,937 features)
3. **Label Encoding** — Language labels are encoded numerically using `LabelEncoder`
4. **Train/Test Split** — 80/20 split (8,269 train / 2,068 test samples)
5. **Training** — `MultinomialNB` model trained on the vectorized text
6. **Prediction** — New text is vectorized and classified into one of 17 languages

---

## Results

| Metric | Score |
|---|---|
| **Test Accuracy** | **98.45%** |
| Macro Avg Precision | 0.99 |
| Macro Avg Recall | 0.98 |
| Macro Avg F1 | 0.99 |

---

## Example Usage

```python
def predict(text):
    x = cv.transform([text]).toarray()
    lang = model.predict(x)
    lang = le.inverse_transform(lang)
    print("The language is:", lang[0])

predict("Hi, how are you?")        # → English
predict("Salut comment ça va?")    # → French
predict("مرحبا كيف حالك؟")        # → Arabic
predict("नमस्ते, आप कैसे हैं?")   # → Hindi
predict("Γεια πως εισαι?")        # → Greek
```

---

## Dependencies

```
pandas
numpy
scikit-learn
seaborn
matplotlib
```
