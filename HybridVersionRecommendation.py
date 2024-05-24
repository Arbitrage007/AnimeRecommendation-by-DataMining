import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse import csr_matrix

# Load the dataset
anime_df = pd.read_csv('anime.csv')  # replace with your actual dataset path
ratings_df = pd.read_csv('rating.csv')  # replace with your actual dataset path

### Step 1: Data Preprocessing

# Handle missing values in anime dataset
anime_df.dropna(inplace=True)

# Handle missing values in ratings dataset
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

# Ensure the user_id 1 is in the sampled dataset
if 1 in ratings_df['user_id'].unique():
    sampled_ratings_df = ratings_df.sample(frac=0.001, random_state=42)
    if 1 not in sampled_ratings_df['user_id'].unique():
        sampled_ratings_df = pd.concat([sampled_ratings_df, ratings_df[ratings_df['user_id'] == 1]])
else:
    sampled_ratings_df = ratings_df.sample(frac=0.001, random_state=42)

### Step 2: Collaborative Filtering Model Development

# Create a pivot table for user-item interactions
user_item_matrix = sampled_ratings_df.pivot(index='user_id', columns='anime_id', values='rating').fillna(0)
user_item_sparse_matrix = csr_matrix(user_item_matrix.values)

# Compute similarity between users using Pearson correlation
user_similarity = 1 - pairwise_distances(user_item_matrix, metric='correlation')
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Apply k-Nearest Neighbors algorithm
k = 10  # Number of nearest neighbors
knn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
knn.fit(user_item_sparse_matrix)

# Function to predict ratings with regularization
def predict_ratings(user_id, anime_id, regularization=0.02):
    if user_id in user_item_matrix.index:
        user_bias = user_item_matrix.mean(axis=1)[user_id]
        if anime_id in user_item_matrix.columns:
            item_bias = user_item_matrix.mean(axis=0)[anime_id]
        else:
            item_bias = 0  # If anime_id not found, set item_bias to 0

        global_mean = user_item_matrix.values.mean()
        base_prediction = global_mean + user_bias + item_bias

        user_index = user_item_matrix.index.get_loc(user_id)
        if anime_id in user_item_matrix.columns:
            anime_index = user_item_matrix.columns.get_loc(anime_id)
        else:
            return base_prediction  # Return base prediction if anime_id not found

        distances, indices = knn.kneighbors(user_item_sparse_matrix[user_index].reshape(1, -1))
        neighbor_ratings = user_item_matrix.iloc[indices[0], anime_index]
        neighbor_similarity = 1 - distances[0]

        weighted_sum = np.dot(neighbor_ratings - item_bias, neighbor_similarity)
        normalization_factor = np.sum(neighbor_similarity) + regularization

        if normalization_factor == 0:
            return base_prediction
        return base_prediction + weighted_sum / normalization_factor
    else:
        global_mean = user_item_matrix.values.mean()  # Return global mean if user not found
        return global_mean

### Step 3: Model Evaluation
# Create a test set from the user_item_matrix
test_set_fraction = 0.1
test_data_matrix = user_item_matrix.sample(frac=test_set_fraction, random_state=42).values
predictions = np.zeros(test_data_matrix.shape)

for i in range(test_data_matrix.shape[0]):
    for j in range(test_data_matrix.shape[1]):
        if test_data_matrix[i, j] != 0:
            user_id = user_item_matrix.index[i]
            anime_id = user_item_matrix.columns[j]
            predictions[i, j] = predict_ratings(user_id, anime_id)

# Calculate RMSE
rmse = sqrt(mean_squared_error(test_data_matrix[test_data_matrix.nonzero()], predictions[test_data_matrix.nonzero()]))
print(f'User-based CF RMSE: {rmse}')

### Step 4: Content-Based Filtering

# Fill any missing values in the 'genre' column with an empty string
anime_df['genre'] = anime_df['genre'].fillna('')
anime_df['type'] = anime_df['type'].fillna('')
anime_df['episodes'] = anime_df['episodes'].fillna(0)
anime_df['rating'] = anime_df['rating'].fillna(0)

# Combine features for a richer feature matrix
anime_df['combined_features'] = anime_df['genre'] + ' ' + anime_df['type'] + ' ' + anime_df['episodes'].astype(str) + ' ' + anime_df['rating'].astype(str)

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Apply TF-IDF Vectorizer to the combined features
tfidf_matrix = tfidf_vectorizer.fit_transform(anime_df['combined_features'])

# Compute cosine similarity between all anime
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a function to recommend anime based on content
def get_recommendations(anime_title, cosine_sim=cosine_sim):
    idx = anime_df[anime_df['name'] == anime_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    anime_indices = [i[0] for i in sim_scores]
    return anime_df['name'].iloc[anime_indices]

# Test the recommendation function
recommended_anime = get_recommendations('Naruto')
print("Anime recommendations for 'Naruto':")
print(recommended_anime)

### Step 5: Hybrid Recommendations

def hybrid_recommendations(user_id, anime_title, user_weight=0.5, content_weight=0.5):
    # Check if the user_id exists in the user_item_matrix
    if user_id not in user_item_matrix.index:
        print(f'User ID {user_id} not found in the dataset.')
        return None

    # Check if the anime_title exists in the anime_df
    if anime_title not in anime_df['name'].values:
        print(f'Anime title "{anime_title}" not found in the dataset.')
        return None

    anime_id = anime_df[anime_df['name'] == anime_title].index[0]
    user_based_pred = predict_ratings(user_id, anime_id)  # Collaborative filtering prediction
    content_based_recs = get_recommendations(anime_title)  # Content-based filtering
    
    # Convert anime names to indices
    anime_indices = anime_df.index[anime_df['name'].isin(content_based_recs)].tolist()
    
    # Normalize content-based scores to be in the same scale as collaborative predictions
    normalized_recs = (anime_df.loc[anime_indices, 'rating'] - anime_df['rating'].min()) / (anime_df['rating'].max() - anime_df['rating'].min())
    
    # Combine user-based prediction and content-based scores with given weights
    hybrid_scores = user_weight * user_based_pred + content_weight * normalized_recs
    
    # Get anime names corresponding to indices
    anime_names = anime_df.loc[anime_indices, 'name'].tolist()
    
    # Calculate percentages
    total_score = sum(hybrid_scores)
    percentages = [score / total_score * 100 for score in hybrid_scores]
    
    # Create a dictionary with anime names and their corresponding recommendation percentages
    hybrid_recommendations = dict(zip(anime_names, percentages))
    
    return hybrid_recommendations

# Example hybrid recommendation
hybrid_recs = hybrid_recommendations(1, 'Naruto')
if hybrid_recs is not None:
    print("Hybrid Recommendations:")
    for anime, percentage in hybrid_recs.items():
        print(f"{anime}: {percentage:.2f}%")

