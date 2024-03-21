import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer

# Define the path to the CrisisLexT26 dataset directory
dataset_dir = 'CrisisLexT26-v1.0/CrisisLexT26/'

# Initialize an empty list to store individual DataFrames
dfs = []

# Function to process each file
def process_file(subdir_path, filename):
    try:
        # Read the CSV file without specifying the delimiter
        df = pd.read_csv(os.path.join(subdir_path, filename))

        # Remove leading spaces from column names
        df.columns = df.columns.str.strip()

        # Check if the required columns exist
        if 'Tweet Text' in df.columns and 'Informativeness' in df.columns:
            return df
        else:
            print("    Required columns not found in this file.")
            return None
    except Exception as e:
        print(f"    Error processing file: {e}")
        return None

# Iterate over subdirectories
for subdir in os.listdir(dataset_dir):
    subdir_path = os.path.join(dataset_dir, subdir)

    # Check if the subdirectory is indeed a directory
    if os.path.isdir(subdir_path):
        print(f"Processing subdirectory: {subdir}")

        # Iterate over files in the subdirectory
        processed_dfs = Parallel(n_jobs=-1)(delayed(process_file)(subdir_path, filename) for filename in os.listdir(subdir_path) if filename.endswith("_labeled.csv"))
        processed_dfs = [df for df in processed_dfs if df is not None]
        dfs.extend(processed_dfs)

# Concatenate all DataFrames into a single DataFrame
if dfs:
    dataset = pd.concat(dfs, ignore_index=True)
    print("line 45")
    print(dataset.head())
else:
    print("No CSV files with required columns found.")

# Shuffle and sample a subset of the dataset
dataset_subset = dataset.sample(frac=0.1, random_state=42)

# Split the subset dataset into training and testing sets
X_subset_train, X_subset_test, y_subset_train, y_subset_test = train_test_split(dataset_subset['Tweet Text'], dataset_subset['Informativeness'], test_size=0.2, random_state=42)

# Extract features from the text data using CountVectorizer with optimized parameters
vectorizer = CountVectorizer(min_df=5, max_df=0.7)
X_subset_train_vectorized = vectorizer.fit_transform(X_subset_train)
X_subset_test_vectorized = vectorizer.transform(X_subset_test)

# Handle missing values in y_subset_train
imputer = SimpleImputer(strategy='most_frequent')
y_subset_train_imputed = imputer.fit_transform(y_subset_train.values.reshape(-1, 1))

# Convert y_subset_train_imputed to a NumPy array
y_subset_train_imputed = np.array(y_subset_train_imputed)

# Convert the string labels to numeric labels
label_encoder = LabelEncoder()
y_subset_train_encoded = label_encoder.fit_transform(y_subset_train_imputed.flatten())

# Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_subset_train_vectorized, y_subset_train_encoded)

# Train the Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_subset_train_vectorized, y_subset_train_encoded)

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_subset_train_vectorized, y_subset_train_encoded)

# Make predictions on the testing data
y_pred_nb = nb_classifier.predict(X_subset_test_vectorized)
y_pred_svm = svm_classifier.predict(X_subset_test_vectorized)
y_pred_rf = rf_classifier.predict(X_subset_test_vectorized)

# Convert y_subset_test to numeric labels using the same label encoder
y_subset_test_encoded = label_encoder.transform(y_subset_test)

# Print unique labels in y_subset_test_encoded and y_pred_nb
print("Unique labels in y_subset_test_encoded:", np.unique(y_subset_test_encoded))
print("Unique labels in y_pred_nb:", np.unique(y_pred_nb))

# Evaluate the performance of the classifiers
print("Naive Bayes Classifier:")
print(classification_report(y_subset_test_encoded, y_pred_nb))

print("Support Vector Machine (SVM) Classifier:")
print(classification_report(y_subset_test_encoded, y_pred_svm))

print("Random Forest Classifier:")
print(classification_report(y_subset_test_encoded, y_pred_rf))
