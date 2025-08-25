
# 📝 Sentiment Analyzer: Disaster Tweets Classification

This repository contains a Jupyter Notebook that builds a **sentiment analyzer** to classify tweets as **disaster-related** or **non-disaster**.  
The dataset is sourced from Kaggle's ["Natural Language Processing with Disaster Tweets"](https://www.kaggle.com/c/nlp-getting-started/data) competition.

---

## 📂 Project Structure
```
.
├── TrainAndTestSentimentAnalyzer.ipynb  # Main notebook for training and testing
├── train.csv                            # Training dataset (from Kaggle)
├── test.csv                             # Test dataset (from Kaggle)
└── README.md                            # Project documentation
```

---

## 🚀 Features
- **Exploratory Data Analysis (EDA):**
  - Visualizes class distribution, word counts, character counts, and unique word counts.
- **Text Preprocessing:**
  - Tokenization, stemming, lemmatization
  - Stopword removal, punctuation cleaning
- **Feature Engineering:**
  - Bag-of-Words and TF-IDF vectorization
  - Optionally supports Word2Vec embeddings
- **Model Training:**
  - Logistic Regression
  - Naive Bayes (MultinomialNB)
  - SGDClassifier
- **Model Evaluation:**
  - Accuracy, F1-score, ROC-AUC
  - Confusion Matrix visualization

---

## 🛠️ Installation

Clone this repository and install required dependencies:

```bash
git clone https://github.com/your-username/sentiment-analyzer.git
cd sentiment-analyzer
pip install -r requirements.txt
```

---

## 📦 Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk
- gensim

Install all dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk gensim
```

---

## 📊 Dataset
Download the dataset from Kaggle:  
[NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/data)

Files:
- `train.csv` — Training data
- `test.csv` — Test data

---

## ▶️ Usage

Run the notebook:

```bash
jupyter notebook TrainAndTestSentimentAnalyzer.ipynb
```

Follow these steps inside the notebook:
1. Load the dataset.
2. Perform text cleaning & preprocessing.
3. Train chosen ML models.
4. Evaluate performance metrics.

---

## 🔮 Future Improvements
- Experiment with deep learning models (LSTM, BERT).
- Hyperparameter tuning with GridSearchCV.
- Deploy as a web app (Flask/FastAPI).

---

## 📜 License
This project is licensed under the MIT License.

---


