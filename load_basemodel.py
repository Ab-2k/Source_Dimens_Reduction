import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import time

# Function to load data (assume it's already defined)

# Load dataset
df, y = load_data()

# Encode class labels
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y.values.ravel())

# Scale the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Split the data into training and testing sets (all features)
X_train_full, X_test_full, y_train, y_test = train_test_split(df_scaled, encoded_y, test_size=0.2, random_state=42)


import time
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Baseline model with all features using Naive Bayes
start_time = time.time()
classifier_pipeline_full = Pipeline([
    ('classifier', GaussianNB())
])
classifier_pipeline_full.fit(X_train_full, y_train)
y_pred_full = classifier_pipeline_full.predict(X_test_full)
end_time = time.time()
full_features_time = end_time - start_time
accuracy_full = accuracy_score(y_test, y_pred_full)
