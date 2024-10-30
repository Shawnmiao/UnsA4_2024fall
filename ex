#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the Olivetti faces dataset
faces_data = fetch_olivetti_faces()
features, labels = faces_data.data, faces_data.target

# Split dataset into training, validation, and test sets with stratification
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.4, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Standardize the features for Principal Component Analysis (PCA)
standardizer = StandardScaler()
X_train_std = standardizer.fit_transform(X_train)
X_val_std = standardizer.transform(X_val)
X_test_std = standardizer.transform(X_test)

# Apply PCA to retain 99% of the variance
pca_model = PCA(0.99, random_state=42)
X_train_pca = pca_model.fit_transform(X_train_std)
X_val_pca = pca_model.transform(X_val_std)
X_test_pca = pca_model.transform(X_test_std)

# Determine the best covariance type for Gaussian Mixture Model (GMM) using AIC
best_aic_score, optimal_gmm, optimal_cov_type = np.inf, None, ''
for cov_type in ['full', 'tied', 'diag', 'spherical']:
    gmm_model = GaussianMixture(n_components=20, covariance_type=cov_type, reg_covar=1e-6, random_state=42)
    gmm_model.fit(X_train_pca)
    aic_score = gmm_model.aic(X_val_pca)
    if aic_score < best_aic_score:
        best_aic_score, optimal_gmm, optimal_cov_type = aic_score, gmm_model, cov_type

print(f"Optimal covariance type: {optimal_cov_type} with AIC score: {best_aic_score}")

# Determine minimum clusters using BIC and plot results
bic_scores = []
for cluster_count in range(1, 21):  # Limited to 1-20 clusters for stability
    gmm_model = GaussianMixture(n_components=cluster_count, covariance_type=optimal_cov_type, reg_covar=1e-6, random_state=42)
    gmm_model.fit(X_train_pca)
    bic_scores.append(gmm_model.bic(X_val_pca))

plt.plot(range(1, 21), bic_scores, label='BIC')
plt.xlabel("Number of Clusters")
plt.ylabel("BIC Score")
plt.legend()
plt.show()

# Train final GMM with optimal cluster count based on BIC
optimal_cluster_count = np.argmin(bic_scores) + 1
final_gmm_model = GaussianMixture(n_components=optimal_cluster_count, covariance_type=optimal_cov_type, reg_covar=1e-6, random_state=42)
final_gmm_model.fit(X_train_pca)

# Output hard clustering assignments and soft clustering probabilities
hard_clusters = final_gmm_model.predict(X_test_pca)
soft_cluster_probabilities = final_gmm_model.predict_proba(X_test_pca)

print("Hard Clustering Assignments:", hard_clusters)
print("Soft Clustering Probabilities:\n", soft_cluster_probabilities)

# Generate new faces using the final GMM model
generated_faces_samples, _ = final_gmm_model.sample(10)
generated_faces_images = standardizer.inverse_transform(pca_model.inverse_transform(generated_faces_samples))

# Display generated faces
fig, axs = plt.subplots(1, 10, figsize=(15, 2))
for i, face in enumerate(generated_faces_images):
    axs[i].imshow(face.reshape(64, 64), cmap='gray')
    axs[i].axis('off')
plt.show()

# Modify images to create anomalies
anomalous_images = X_test.copy()
anomalous_images[0] = np.flip(anomalous_images[0].reshape(64, 64), axis=1).flatten()  # Horizontal flip
anomalous_images[1] = np.roll(anomalous_images[1], 20)  # Pixel roll
anomalous_images[2] *= 0.5  # Darken image

# Compare scores between normal and anomalous images
normal_image_scores = final_gmm_model.score_samples(X_test_pca)
anomalous_image_scores = final_gmm_model.score_samples(pca_model.transform(standardizer.transform(anomalous_images)))

print("Normal Image Scores:", normal_image_scores)
print("Anomalous Image Scores:", anomalous_image_scores)


# In[ ]:




