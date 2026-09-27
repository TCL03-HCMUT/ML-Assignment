# GoEmotions Machine Learning Project Requirements

This document outlines the detailed requirements, objectives, and task breakdown for our Machine Learning course assignment using the **GoEmotions** dataset.

## 🎯 Project Objective
The goal is to build, evaluate, and compare complete machine learning pipelines for text classification. We must implement a **Traditional Machine Learning Pipeline** (mandatory) and are highly encouraged to implement a **Deep Learning Pipeline** for bonus points.

## 📊 The Dataset: GoEmotions
- **Type**: Text Data.
- **Task**: Multi-label emotion classification (28 classes including neutral).
- **Key Characteristics**: The dataset is highly imbalanced and allows multiple labels per text. We must handle this via appropriate metrics (like Micro/Macro F1-score) and multi-label classifiers (e.g., `OneVsRestClassifier` or `BCEWithLogitsLoss`).

---

## 🛠️ Task Breakdown & Work Distribution

To ensure everyone contributes equally and effectively, the work is divided into four main roles:

### 1. Data & Foundation Engineer
**Responsibilities:**
- Download the dataset via a public link (e.g., HuggingFace `datasets`).
- Conduct Exploratory Data Analysis (EDA): statistics on text lengths, word frequencies, and class distributions.
- Implement a configurable text preprocessing pipeline (lowercasing, punctuation removal, stop-word removal, tokenization).
- **Output:** `modules/data_loader.py` & `modules/preprocessing.py`.

### 2. Traditional ML Engineer (Mandatory Pipeline)
**Responsibilities:**
- Implement traditional feature extraction: Bag-of-Words (BoW), TF-IDF, and n-grams.
- Train and evaluate traditional classifiers (e.g., Logistic Regression, SVM, Random Forest).
- Implement configurable hyperparameter tuning (GridSearch / RandomSearch).
- **Output:** `modules/traditional_pipeline.py`.

### 3. Modern Feature Engineer (Static Embeddings)
**Responsibilities:**
- Implement deep learning embeddings (e.g., Word2Vec, GloVe, or extracting frozen features from BERT/RoBERTa).
- Save extracted feature representations as `.npy` or `.h5` files in the `features/` directory.
- Train downstream models (traditional classifiers or simple MLPs) on these static embeddings.
- **Output:** `modules/modern_features.py`.

### 4. Deep Learning & Integration Engineer (Bonus & Delivery)
**Responsibilities:**
- Implement an End-to-End Deep Learning pipeline (fine-tuning BERT or DistilBERT).
- Assemble all modules into the main `Google Colab Notebook`. Ensure `Runtime -> Run all` executes without errors.
- Conduct a final evaluation comparing all three approaches (Traditional, Modern Features, E2E Deep Learning).
- **Output:** `modules/dl_pipeline.py`, `modules/evaluation.py`, and `notebooks/main_pipeline.ipynb`.

---

## 📦 Deliverables & Submission Requirements

1. **Google Colab Notebook (`notebooks/main_pipeline.ipynb`)**:
   - Must execute successfully from start to finish via `Runtime -> Run all`.
   - Data must be downloaded automatically via a public link.
   - Must NOT rely on mounting personal cloud drives (like Google Drive).
   - Code must clone this GitHub repository and install it as a package (`pip install -e .`).

2. **PDF Report (`reports/final_report.pdf`)**:
   - Must include EDA findings, pipeline design, experiment results, and performance analysis.
   - Must contain a detailed task distribution table outlining individual contribution percentages.

3. **Extracted Feature Files (`features/`)**:
   - Embeddings must be saved in `.npy` or `.h5` formats.

4. **Directory Structure (Zip File)**:
   - The final submission zip must cleanly contain: `notebooks/`, `modules/`, `reports/`, and `features/`.

5. **GitHub Repository (Bonus)**:
   - Must include a `README.md` with course info, instructor details, group members, and execution guides.

---

## 🏆 Evaluation Criteria

- **40%**: Traditional Pipeline Completion (EDA, preprocessing, feature extraction, training, evaluation).
- **25%**: Quality of Analysis and Experiments (comparing configs, explaining results/limitations).
- **20%**: Report Quality (structure, figures, tables, work distribution table).
- **10%**: Completeness of Deliverables (Colab runs successfully, correct directory structure, saved feature formats).
- **5%**: Team Collaboration Spirit (Verifiable evidence of teamwork: photos, meeting minutes in `reports/evidence/`).

**Bonus Points:**
- **+5%**: Deep Learning Pipeline implementation and comparison.
- **+5%**: GitHub Repository publication with a well-organized structure and detailed README.
