Approach:

Baseline Model: Constructed an initial model using all 561 features from various smartphone sensors.
Dimensionality Reduction: Applied k-means clustering to group similar features, reducing the feature set to 50 by selecting representative features from each cluster.
Normalized the data and used Gaussian Naive Bayes for classification.
Results:

Baseline Model (All Features): Achieved an accuracy of 73.15% with a training time of 0.15 seconds.
Reduced Model (K-Means): Enhanced accuracy to 78.59% with a significantly reduced training time of 0.01 seconds.
Efficiency: Reduced the number of features from 561 to 50, improving model interpretability and computational efficiency.
