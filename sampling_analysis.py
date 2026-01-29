import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample

 # Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

 # Load the credit card dataset
DATA_PATH = "Creditcard_data.csv"
df = pd.read_csv(DATA_PATH)

 # Identify the target column for classification
target_col = "Class" if "Class" in df.columns else df.columns[-1]

print("Class distribution before sampling:")
print(df[target_col].value_counts())

 # Helper function to print class distribution after sampling
def print_class_counts(data, label):
    print(f"\nClass distribution after {label}:")
    print(data[target_col].value_counts())

min_class_size = df[target_col].value_counts().min()

 # Sampling technique implementations

    # Simple Random Sampling: randomly select equal samples from each class
def simple_random_sampling(df):
    samples = []
    for cls in df[target_col].unique():
        cls_df = df[df[target_col] == cls]
        samples.append(cls_df.sample(min_class_size, random_state=RANDOM_STATE))
    return pd.concat(samples).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Systematic Sampling: select samples at fixed intervals after shuffling
def systematic_sampling(df):
    samples = []
    for cls in df[target_col].unique():
        cls_df = df[df[target_col] == cls].sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        k = len(cls_df) // min_class_size
        start = np.random.randint(0, k)
        idx = np.arange(start, start + k * min_class_size, k)
        samples.append(cls_df.iloc[idx[:min_class_size]])
    return pd.concat(samples).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Stratified Sampling: sample equal numbers from each class, preserving structure
def stratified_sampling(df):
    return (
        df.groupby(target_col, group_keys=False)
        .apply(lambda x: x.sample(min_class_size, random_state=RANDOM_STATE))
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    # Cluster Sampling: split each class into clusters, then select clusters
def cluster_sampling(df):
    samples = []
    n_clusters = 5

    for cls in df[target_col].unique():
        cls_df = df[df[target_col] == cls].sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        clusters = np.array_split(cls_df, n_clusters)
        random.shuffle(clusters)

        selected = pd.concat(clusters)
        samples.append(selected.iloc[:min_class_size])

    return pd.concat(samples).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Bootstrap Sampling: sample with replacement to balance classes
def bootstrap_sampling(df):
    samples = []
    for cls in df[target_col].unique():
        cls_df = df[df[target_col] == cls]
        samples.append(
            resample(
                cls_df,
                replace=True,
                n_samples=min_class_size,
                random_state=RANDOM_STATE
            )
        )
    return pd.concat(samples).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

 # Create balanced datasets using each sampling technique
datasets = {
    "Sampling1": simple_random_sampling(df),
    "Sampling2": systematic_sampling(df),
    "Sampling3": stratified_sampling(df),
    "Sampling4": cluster_sampling(df),
    "Sampling5": bootstrap_sampling(df)
}

for name, data in datasets.items():
    print_class_counts(data, name)

 # Define machine learning models to evaluate
models = {
    "M1": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "M2": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "M3": RandomForestClassifier(random_state=RANDOM_STATE),
    "M4": SVC(random_state=RANDOM_STATE),
    "M5": KNeighborsClassifier()
}

 # Train and evaluate each model on each sampled dataset
results = pd.DataFrame(index=models.keys(), columns=datasets.keys())

for sample_name, sample_df in datasets.items():
    X = sample_df.drop(columns=[target_col])
    y = sample_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y
    )

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        results.loc[model_name, sample_name] = round(acc, 4)

 # Display results and identify best sampling/model combinations
print("\nAccuracy Results Table:")
print(results)

print("\nBest Sampling Technique for Each Model:")
for model in results.index:
    best_sampling = results.loc[model].astype(float).idxmax()
    best_acc = results.loc[model].astype(float).max()
    print(f"{model}: {best_sampling} (Accuracy: {best_acc})")

best_model = results.max(axis=1).astype(float).idxmax()
best_model_acc = results.max(axis=1).astype(float).max()
best_sampling_for_best_model = results.loc[best_model].astype(float).idxmax()

print(
    f"\nBest Model Overall: {best_model} "
    f"(Accuracy: {best_model_acc}) using {best_sampling_for_best_model}"
)
