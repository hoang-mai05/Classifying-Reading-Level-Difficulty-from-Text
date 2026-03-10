# Reading-Level Difficulty Classifier

Project A: Classifying Reading Level Difficulty from Text

This project implements an automated system to assess the reading difficulty of English text passages. By leveraging machine learning pipelines, we aim to distinguish between lower-level educational materials and advanced literature to help students find appropriate reading resources!

---

## Project Overview

The goal is to build a **binary classifier** that categorizes texts into two coarse difficulty levels based on the UK National Curriculum:
  * **Class 0: Lower-level** (Ages 7-14, UK Key Stage 2-3).
  * **Class 1: Upper-level** (Ages 14-18, UK Key Stage 4-5).
The project followed a two-stage development cycle:
  1. **Problem 1 (Restricted)**: Logistic Regression using a Bag-of-Words (BoW) representation.
  2. **Problem 2 (Open Challenge)**: An open-ended exploration allowing advanced features like LLM embeddings and non-linear models.

---

## Dataset: UK Key Stage Readability

The dataset is a remixed version of the UK Key Stage Readability for English Texts prepared by J. Bird (2024).
**Training Set:** 5,557 input/output pairs.
**Test Set:** 1,197 input instances (unlabeled).
**Zero Overlap:** There is no overlap in authors or titles between the training and test sets to ensure generalization to new writers.

We assigned Lexile Scores of 400 - 1000 as Lower-Level passages, and 1001 - 1201+ as Upper-Level passages

---

## Model Progression

### Problem 1: Bag-of-Words Baseline
1. **Features:** Unigram counts processed through a CountVectorizer and a custom ReadingLevelTransformer.
2. **Techniques:** Purely Lower Cased preprocessing, Additionally split training data to incorporate a testable measurement to prevent overfitting
3. **Classifier:** Logistic Regression.
4. **Best CV AUROC:** ~0.7411.


### Problem 2: DeBERTa & Random Forest Implementation

1. **Semantic Features:** Pretrained DeBERTa-v3-small embeddings to capture contextual nuances.
2. **Structural Features:** Numerical metrics including average sentence length, Flesch Reading Ease, and Dale-Chall Index.
3. **Classifier:** Random Forest with hyperparameter tuning via GridSearchCV.
4. **Strategy:** Pre-calculation and caching of embeddings as .npy files to accelerate training.
5. **Best CV AUROC:** ~0.7800.

### Problem 2: Word2Vec & Logistic Regression Implementation

1. **Features:** Unigram counts processed through a CountVectorizer and a custom ReadingLevelTransformer.
2. **Techniques:** Weighted TF-IDF Word2Vec means, PCA dimensionality reduction (20 components), and OOV (Out-of-Vocabulary) rate calculation.
3. **Classifier:** Logistic Regression.
4. **Best CV AUROC:** ~0.7729.

---

## Evaluation & Best Practices

**Performance Metric:** Area Under the ROC Curve (AUROC).
**Cross-Validation:** $5$-fold GroupKFold split by author to prevent data leakage and ensure the model learns readability, not author style.
**Hyperparameter Search:** Systematic search over regularization strengths ($C$) for Logistic Regression and max_depth / n_estimators for Random Forest.

---

## Setup & Installation

To run the notebooks, ensure you have the following dependencies installed:

```pip install torch transformers sentencepiece protobuf tiktoken gensim scikit-learn pandas numpy matplotlib```

  1. **Baseline Run:** Open Word2Vec.ipynb and ensure the GoogleNews-vectors-negative300.bin file is in the word2vec/ directory.

  2. **Advanced Run:** Open randomForest.ipynb. The notebook will automatically download the DeBERTa model from the Hugging Face Hub.

---

## References

* Bird, J. J. (2024). _What Differentiates Educational Literature? A Multimodal Fusion Approach of Transformers and Computational Linguistics. arXiv preprint 2411.17593._

* Tufts CS 135 Course Materials.
