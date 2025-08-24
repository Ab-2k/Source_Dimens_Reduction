from sklearn.cluster import KMeans
import numpy as np

# K-Means for dimensionality reduction
n_clusters = 50
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(df_scaled.T)  # Transpose to treat features as data points
selected_features_indices = [np.random.choice(np.where(kmeans.labels_ == i)[0]) for i in range(n_clusters)]
selected_features = df_scaled[:, selected_features_indices]

# Split the data with reduced features
X_train_reduced, X_test_reduced = train_test_split(selected_features, test_size=0.2, random_state=42)