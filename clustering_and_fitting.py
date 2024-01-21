# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:08:21 2024

@author: lisar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import errors as err

# to set environment variable for parallel processing
os.environ["OMP_NUM_THREADS"] = "2"


def read_data(url, indicator):
    """
    Read data from a CSV file based on the given URL and indicator.

    Parameters:
    - url (str): URL of the CSV file.
    - indicator (str): Name of the indicator.

    Returns:
    - DataFrame: Processed data.
    """
    df = pd.read_csv(url, skiprows=4)
    df = df.loc[df['Indicator Name'] == indicator]
    return df


def plot_silhouette_scores(x):
    """
    Plot silhouette scores for different cluster numbers.

    Parameters:
    - x (array): Data for clustering.

    Returns:
    - None: Displays the silhouette score plot.
    """
    silhouette_scores = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=30, n_init=10, random_state=0)
        labels = kmeans.fit_predict(x)
        silhouette_scores.append(silhouette_score(x, labels))

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-', color='b')
    plt.title("Silhouette Scores for Optimal Cluster Selection", fontsize=16)
    plt.xlabel("Number of Clusters", fontsize=14)
    plt.ylabel("Silhouette Score", fontsize=14)
    plt.xticks(range(2, 11))
    plt.grid(True)
    plt.savefig('silhouette_scores.png')
    plt.show()


def normalize_and_cluster(x, n_clusters):
    """
    Normalize input data and perform KMeans clustering.

    Parameters:
    - x (array): Data for clustering.
    - n_clusters (int): Number of clusters.

    Returns:
    - array: Cluster labels.
    - array: Normalized data.
    - array: Cluster centers (back-scaled).
    """
    # To Normalize input data
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x)

    # To perform KMeans clustering on normalized data
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=30, n_init=10, random_state=42)
    labels = kmeans.fit_predict(x_normalized)

    # Back-scale cluster centers
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    return labels, x_normalized, centroids


def visualize_kmeans_clusters(x, labels, centroids, colors):
    """
    Visualize KMeans clustering results with custom labels for clusters, centroid as a diamond shape, and connecting lines.

    Parameters:
    - x (array): Data for clustering.
    - labels (array): Cluster labels.
    - centroids (array): Cluster centroids.
    - colors (list): Colors for different clusters.

    Returns:
    - None: Displays the KMeans clustering plot.
    """
    plt.figure(figsize=(12, 8))

    # Plotting points and centroids with connecting lines
    for i, label in enumerate(set(labels)):
        cluster_points = x[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        cluster_size = 30 + 10 * i 
        cluster_color = colors[i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=cluster_size, label=f'Cluster {label + 1}', c=cluster_color, alpha=0.7)
        centroid_size = 10
        plt.scatter(centroid[0], centroid[1], s=centroid_size, color='black', marker='D')  # Diamond shape marker
        plt.plot([point[0] for point in cluster_points], [point[1] for point in cluster_points], linestyle='--', color=colors[i], alpha=0.5)

    plt.title("KMeans Clustering of Countries based on Population Growth", fontsize=16)
    plt.xlabel(f"Population Growth in {Year1}", fontsize=14)
    plt.ylabel(f"Population Growth in {Year2}", fontsize=14)

    #  to create custom legend
    cluster_legend = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i + 1}', markerfacecolor=colors[i], markersize=10) for i in range(len(set(labels)))]
    centroid_legend = plt.Line2D([0], [0], marker='D', color='w', label='Centroid', markerfacecolor='black', markersize=10)  # Diamond shape marker
    plt.legend(handles=[*cluster_legend, centroid_legend], loc='lower left')

    plt.savefig('kmeans_clusters.png')
    plt.show()




def extract_and_fit_model(df_year, target_country):
    """
    Extract data for a specific country, fit a model, and plot the results.

    Parameters:
    - df_year (DataFrame): Time-series data.
    - target_country (str): Name of the target country.

    Returns:
    - None: Displays the fitted model plot.
    """
    # Ensure 'Year' is a column
    df_year.reset_index(inplace=True)

    df_fitting = df_year[['Year', target_country]].apply(pd.to_numeric, errors='coerce')
    m = df_fitting.dropna().values
    x_d, y_d = m[:, 0], m[:, 1]

    # Fitting the model
    def model(x, a, b, c, d):
        x = x - 1990
        return a * x**3 + b * x**2 + c * x + d

    popt, covar = opt.curve_fit(model, x_d, y_d)
    a, b, c, d = popt

    # Error propagation
    error = err.error_prop(x_d, model, popt, covar)

    # Scatter plot of data points
    plt.scatter(x_d, y_d, label='Data', color='blue', marker='o', alpha=0.7)

    # plotting Fitted model
    x_line = np.arange(min(x_d), max(x_d) + 1, 1)
    y_line = model(x_line, a, b, c, d)
    plt.plot(x_line, y_line, '--', color='red', label='Fitted Model')

    # Error region
    plt.fill_between(x_line, y_line - error, y_line + error, alpha=0.7, color='green', label='Error Region')

    # Adding labels and legend
    plt.title(f"Curve Fit for {target_country} Population Growth", fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Population Growth", fontsize=14)
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent title cut-off
    plt.savefig(f'fitted_model_{target_country}.png')
    plt.show()


# URL and parameters
url = 'API_SP.POP.GROW_DS2_en_csv_v2_6298705.csv'
indicator = 'Population growth (annual %)'
Year1 = '1990'
Year2 = '2022'

# to read data using the new function
df = read_data(url, indicator)

df_cluster = df.loc[df.index, ['Country Name', Year1, Year2]].dropna()

# to extract data for clustering
x = df_cluster[[Year1, Year2]].values

# Plotting silhouette scores
plot_silhouette_scores(x)

# Normalize input data, perform KMeans clustering, and get back-scaled centroids
labels, x_normalized, centroids = normalize_and_cluster(x, n_clusters=3)

# Add cluster labels to the DataFrame
df_cluster['label'] = labels

# Visualize KMeans clustering results with country names
colors = ['blue', 'red', 'green']
visualize_kmeans_clusters(x_normalized, labels, centroids, colors)

# Extract and fit model for a specific country ( United Kingdom)
df_year = df.T
df_year = df_year.rename(columns=df_year.iloc[0])
df_year = df_year.drop(index=df_year.index[0], axis=0)
df_year['Year'] = df_year.index
extract_and_fit_model(df_year, 'United Kingdom')
