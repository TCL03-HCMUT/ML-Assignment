"""
Modules for GoEmotions ML Pipeline.

Architecture:
    ┌─────────────────────┐     ┌─────────────────────┐
    │  TraditionalFeature │     │   ModernFeature      │
    │  Extractor          │     │   Extractor          │
    │  (BoW, TF-IDF, …)  │     │   (W2V, GloVe, BERT) │
    └────────┬────────────┘     └────────┬────────────┘
             │  .npy / .h5               │  .npy / .h5
             │  (N, D) array             │  (N, D) array
             └──────────┬────────────────┘
                        ▼
               ┌────────────────┐
               │   Classifier   │  (shared: LogReg, SVM, RF, MLP …)
               └────────────────┘

Submodules:
    config              – Centralized dataclass configs for every pipeline step.
    data_loader         – Download, load, and explore the GoEmotions dataset.
    preprocessing       – Configurable text cleaning and normalization.
    feature_io          – Shared save/load for .npy / .h5 feature arrays.
    traditional_pipeline – BoW / TF-IDF / n-gram feature extraction.
    modern_features     – Word2Vec / GloVe / frozen-BERT embedding extraction.
    classifier          – Shared multi-label classifier for any feature array.
    dl_pipeline         – End-to-end Transformer fine-tuning (Bonus).
    evaluation          – Metrics computation and pipeline comparison plots.
"""
