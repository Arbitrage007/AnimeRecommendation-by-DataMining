import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
anime_df = pd.read_csv('cleaned_anime.csv')

# Handle missing values and duplicates
anime_df.dropna(inplace=True)
anime_df.drop_duplicates(inplace=True)

# Fill any missing values in the 'genre' column with an empty string
anime_df['genre'] = anime_df['genre'].fillna('')

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Apply TF-IDF Vectorizer to the 'genre' column
tfidf_matrix = tfidf_vectorizer.fit_transform(anime_df['genre'])

# Compute cosine similarity between all anime
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a function to recommend anime
def get_recommendations(anime_title, cosine_sim=cosine_sim):
    # Get the index of the anime that matches the title
    idx = anime_df[anime_df['name'] == anime_title].index[0]
    
    # Get the pairwise similarity scores of all anime with that anime
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the anime based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar anime
    sim_scores = sim_scores[1:11]
    
    # Get the anime indices
    anime_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar anime
    return anime_df['name'].iloc[anime_indices]

# Test the recommendation function
recommended_anime = get_recommendations('Gintama')
print("Anime recommendations':")
print(recommended_anime)



