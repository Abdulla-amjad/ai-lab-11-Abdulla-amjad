"""
AI_stats_lab.py

Lab: Unsupervised Learning and K-Means Clustering

Topics:
- Unsupervised learning with unlabeled data
- Iris dataset without labels
- Feature standardization
- K-Means clustering
- K-Means objective function
- Elbow method for choosing K
- Underfitting and overfitting in clustering
- Distance-based outlier detection
- Visualization of unlabeled data, clusters, centroids, and elbow curve

Instructions:
- Implement all functions.
- Do NOT change function names.
- Do NOT print inside functions.
- Return exactly the required formats.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


# ============================================================
# Question 1: Unlabeled Data and K-Means Clustering
# ============================================================

def load_iris_unlabeled(feature_indices=(0, 1)):
    iris = load_iris()

    X = iris.data[:, feature_indices]
    feature_names = [iris.feature_names[i] for i in feature_indices]

    return {
        "X": X,
        "feature_names": feature_names
    }


def standardize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std = np.where(std == 0, 1, std)

    X_scaled = (X - mean) / std

    return {
        "X_scaled": X_scaled,
        "mean": mean,
        "std": std
    }


def fit_kmeans(X, K, random_state=0, n_init=10):
    model = KMeans(
        n_clusters=K,
        random_state=random_state,
        n_init=n_init
    )

    model.fit(X)

    return {
        "centroids": model.cluster_centers_,
        "labels": model.labels_,
        "objective": model.inertia_,
        "n_iter": model.n_iter_
    }


def compute_kmeans_objective(X, centroids, labels):
    assigned_centroids = centroids[labels]

    objective = np.sum(
        np.sum((X - assigned_centroids) ** 2, axis=1)
    )

    return objective


# ============================================================
# Question 2: Choosing K, Underfitting/Overfitting, and Outliers
# ============================================================

def evaluate_k_values(X, k_values, random_state=0, n_init=10):
    objectives = []

    for k in k_values:
        result = fit_kmeans(
            X,
            k,
            random_state=random_state,
            n_init=n_init
        )
        objectives.append(result["objective"])

    relative_improvements = [0.0]

    for i in range(1, len(objectives)):
        improvement = (
            (objectives[i - 1] - objectives[i])
            / objectives[i - 1]
        )
        relative_improvements.append(improvement)

    return {
        "k_values": list(k_values),
        "objectives": objectives,
        "relative_improvements": relative_improvements
    }


def choose_elbow_k(k_values, objectives):
    if len(k_values) < 3:
        return k_values[0]

    x = np.array(k_values, dtype=float)
    y = np.array(objectives, dtype=float)

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    line = p2 - p1
    line_norm = np.linalg.norm(line)

    if line_norm == 0:
        return k_values[0]

    distances = []

    for xi, yi in zip(x, y):
        point = np.array([xi, yi])

        distance = (
            abs(np.cross(line, point - p1))
            / line_norm
        )

        distances.append(distance)

    return k_values[int(np.argmax(distances))]


def cluster_size_summary(labels, K):
    return {
        k: int(np.sum(np.asarray(labels) == k))
        for k in range(K)
    }


def identify_outliers_by_distance(X, centroids, labels, top_n=5):
    assigned_centroids = centroids[labels]

    distances = np.sum(
        (X - assigned_centroids) ** 2,
        axis=1
    )

    indices = np.argsort(distances)[::-1][:top_n]

    return {
        "indices": indices,
        "distances": distances[indices]
    }


def diagnose_clustering_fit(K, elbow_k):
    if K < elbow_k:
        return "underfitting"

    if K == elbow_k:
        return "good_fit"

    return "overfitting"


# ============================================================
# Question 3: Visualization
# ============================================================

def plot_unlabeled_data(X, feature_names=None, title="Unlabeled Data"):
    fig, ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1])

    if feature_names is not None and len(feature_names) >= 2:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])

    ax.set_title(title)

    return fig, ax


def plot_kmeans_clusters(X, labels, centroids, feature_names=None, title="K-Means Clusters"):
    fig, ax = plt.subplots()

    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels
    )

    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=200
    )

    if feature_names is not None and len(feature_names) >= 2:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])

    ax.set_title(title)

    return fig, ax


def plot_elbow_curve(k_values, objectives, title="Elbow Method"):
    fig, ax = plt.subplots()

    ax.plot(k_values, objectives, marker="o")

    ax.set_xlabel("K")
    ax.set_ylabel("Objective Value")
    ax.set_title(title)

    return fig, ax


if __name__ == "__main__":
    print("Implement all required functions.")
