
# Credit Card Dataset Sampling and Model Performance Analysis

## 1. Introduction

This project explores how different sampling techniques can help balance a highly imbalanced credit card fraud dataset and how these techniques affect the performance of various machine learning models.

## 2. Dataset Description

The dataset `Creditcard_data.csv` contains credit card transaction records with a binary class label:

- **Class 0:** Legitimate transactions (763 samples)
- **Class 1:** Fraudulent transactions (9 samples)

This severe imbalance motivates the use of sampling techniques before model training.

## 3. Sampling Process

- **Target Population:** All credit card transactions in the dataset.
- **Sampling Frame:** The complete dataset file (`Creditcard_data.csv`).
- **Sample Size:** 9 samples from Class 0 and 9 samples from Class 1 (per technique).

## 4. Sampling Techniques Used

Five sampling techniques were applied to create balanced datasets:

- **Sampling1 – Simple Random Sampling:** Randomly selects equal numbers of samples from each class.
- **Sampling2 – Systematic Sampling:** Selects samples at fixed intervals after a random starting point.
- **Sampling3 – Stratified Sampling:** Ensures equal representation from each class while preserving class structure.
- **Sampling4 – Cluster Sampling:** Divides data into partitions (clusters) and selects clusters to form a balanced sample.
- **Sampling5 – Bootstrap Sampling:** Uses sampling with replacement to generate balanced class distributions.

Each technique results in a balanced dataset of 18 samples.

## 5. Machine Learning Models

Five classification models were evaluated:

- **M1:** Logistic Regression
- **M2:** Decision Tree
- **M3:** Random Forest
- **M4:** Support Vector Machine
- **M5:** K-Nearest Neighbors

## 6. Experimental Setup

- **Train–test split:** 70% training, 30% testing
- **Stratified split** to maintain class balance
- **Fixed random seed** for reproducibility
- **Evaluation metric:** Accuracy
- **Total experiments:** 25 (5 sampling techniques × 5 models)

## 7. Results

| Model | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|-------|-----------|-----------|-----------|-----------|-----------|
| M1 (Logistic Regression) | 0.6667 | 0.5000 | 0.6667 | 0.5000 | 0.6667 |
| M2 (Decision Tree)       | 0.5000 | 0.5000 | 0.5000 | 0.3333 | 0.8333 |
| M3 (Random Forest)       | 0.5000 | 0.6667 | 0.5000 | 0.5000 | 0.6667 |
| M4 (SVM)                 | 0.5000 | 0.3333 | 0.5000 | 0.6667 | 0.6667 |
| M5 (KNN)                 | 0.5000 | 0.3333 | 0.5000 | 0.5000 | 0.6667 |

## 8. Observations

- Different sampling techniques impact models differently.
- Bootstrap Sampling consistently improves performance for tree-based and distance-based models.
- Due to the small dataset size after balancing, accuracy values change in discrete steps.
- Some techniques yield similar results because the minority class contains very few samples.

## 9. Best Performing Combinations

**Best Sampling Technique per Model**

- M1: Sampling1 (0.6667)
- M2: Sampling5 (0.8333)
- M3: Sampling2 (0.6667)
- M4: Sampling4 (0.6667)
- M5: Sampling5 (0.6667)

**Best Model Overall**

- Decision Tree (M2) using Bootstrap Sampling (Sampling5)
- Accuracy: 0.8333

## How to Run

```bash
python sampling_analysis.py
```

**Run `sampling_analysis.py` to reproduce results.**
