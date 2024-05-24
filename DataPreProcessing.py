import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

# Load the dataset
anime_df = pd.read_csv('anime.csv')  # replace 'path_to_anime.csv' with the actual path to your dataset
ratings_df = pd.read_csv('rating.csv')  # replace 'path_to_ratings.csv' with the actual path to your dataset

### Step 1: Data Preprocessing

# Handle missing values in anime dataset
anime_df.dropna(inplace=True)

# Handle missing values in ratings dataset (e.g., removing rows with missing user_id or anime_id)
ratings_df.dropna(subset=['user_id', 'anime_id'], inplace=True)

# Replace 'Unknown' values with NaN in anime_df
anime_df.replace('Unknown', pd.NA, inplace=True)

# Drop rows with missing values in anime_df after replacing 'Unknown'
anime_df.dropna(inplace=True)

# Drop duplicate entries in ratings_df
ratings_df.drop_duplicates(inplace=True)

# Check for and handle duplicate entries in the ratings DataFrame index
duplicate_index = ratings_df.duplicated(subset=['user_id', 'anime_id'], keep=False)
if duplicate_index.any():
    print("Duplicate entries detected. Aggregating ratings...")
    ratings_df = ratings_df.groupby(['user_id', 'anime_id']).mean().reset_index()

# Save cleaned DataFrames to new CSV files
anime_df.to_csv('cleaned_anime.csv', index=False)
ratings_df.to_csv('cleaned_ratings.csv', index=False)

### Step 2: Collaborative Filtering Model Development

# Create a pivot table for user-item interactions
user_item_matrix = ratings_df.pivot(index='user_id', columns='anime_id', values='rating').fillna(0)

# Split data into training and testing sets
train_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)

# Compute similarity between users using k-nearest neighbors algorithm
k = 10  # Number of nearest neighbors
knn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
knn.fit(train_data)

# Function to predict ratings
def predict_ratings(user_id, anime_id):
    if user_id in train_data.index:
        user_index = train_data.index.get_loc(user_id)
        anime_index = np.where(train_data.columns == anime_id)[0][0]
        distances, indices = knn.kneighbors(train_data.iloc[user_index].values.reshape(1, -1))
        neighbor_ratings = train_data.iloc[indices[0], anime_index]
        neighbor_similarity = 1 - distances[0]
        prediction = np.dot(neighbor_ratings, neighbor_similarity) / np.sum(neighbor_similarity)
        return prediction
    else:
        # If user_id is not present in the training data, return a default value or handle the case accordingly
        return 0  # For simplicity, return 0 as the default rating

### Step 3: Model Evaluation

# Evaluate the model
test_data_matrix = test_data.values
predictions = np.zeros(test_data_matrix.shape)

for i in range(test_data_matrix.shape[0]):
    for j in range(test_data_matrix.shape[1]):
        if test_data_matrix[i, j] != 0:
            user_id = test_data.index[i]
            anime_id = test_data.columns[j]
            predictions[i, j] = predict_ratings(user_id, anime_id)

# Calculate RMSE
rmse = sqrt(mean_squared_error(test_data_matrix[test_data_matrix.nonzero()], predictions[test_data_matrix.nonzero()]))
print(f'User-based CF RMSE: {rmse}')

### Step 4: Anime Similarity Computation

# Fill any missing values in the 'genre' column with an empty string
anime_df['genre'] = anime_df['genre'].fillna('')

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Apply TF-IDF Vectorizer to the 'genre' column
tfidf_matrix = tfidf_vectorizer.fit_transform(anime_df['genre'])

# Save the TF-IDF matrix to a CSV file
pd.DataFrame(tfidf_matrix.toarray()).to_csv('tfidf_matrix.csv', index=False)

# Compute cosine similarity between all anime
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save the cosine similarity matrix to a CSV file
pd.DataFrame(cosine_sim).to_csv('cosine_similarity_matrix.csv', index=False)
