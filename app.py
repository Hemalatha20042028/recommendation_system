# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_data():
    if not os.path.exists('animes.pkl') or not os.path.exists('similarity.pkl'):
        # Load and preprocess data
        anime = pd.read_csv("anime.csv")
        
        # Handle missing values
        anime['genre'].fillna('Unknown', inplace=True)
        anime['type'].fillna('Unknown', inplace=True)
        anime['rating'].fillna(anime['rating'].median(), inplace=True)
        
        # Create tags column
        anime['tags'] = anime['genre'] + " ," + anime['type'] + "," + anime['episodes'].astype(str)
        
        # Create new dataframe
        new_df = anime[['anime_id', 'name', 'tags']]
        
        # Vectorize tags
        cv = CountVectorizer(max_features=2000, stop_words='english')
        vectors = cv.fit_transform(new_df['tags']).toarray()
        
        # Compute similarity matrix
        similarity = cosine_similarity(vectors)
        
        # Save to files
        pickle.dump(new_df, open('animes.pkl', 'wb'))
        pickle.dump(similarity, open('similarity.pkl', 'wb'))
        print("Created and saved data files")
    else:
        # Load precomputed data
        new_df = pickle.load(open('animes.pkl', 'rb'))
        similarity = pickle.load(open('similarity.pkl', 'rb'))
        print("Loaded precomputed data files")
    
    return new_df, similarity

# Load data
new_df, similarity = load_data()
anime_names = sorted(new_df['name'].tolist())

# UI Elements
st.title("🎬 Anime Recommendation System")
st.markdown("Discover new anime similar to your favorites!")

# Search input with typeahead
selected_anime = st.selectbox(
    "Search for an anime:",
    options=anime_names,
    index=None,
    placeholder="Start typing..."
)

# Recommendation logic
if selected_anime:
    try:
        # Find the anime index
        anime_index = new_df[new_df['name'] == selected_anime].index[0]
        
        # Get similarity scores
        distances = similarity[anime_index]
        
        # Get top 6 recommendations
        anime_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        st.success(f"Top recommendations for **{selected_anime}**:")
        
        # Display recommendations in columns
        cols = st.columns(5)
        for idx, (i, _) in enumerate(anime_list):
            with cols[idx]:
                st.subheader(new_df.iloc[i]['name'])
                st.caption(f"Similarity score: {similarity[anime_index][i]:.2f}")
                
    except IndexError:
        st.error(f"❌ '{selected_anime}' not found in database. Please try another title.")

# Add some info/instructions
st.divider()
st.markdown("### How it works:")
st.markdown("""
1. The system analyzes anime genres, types, and episode counts
2. Uses cosine similarity to find shows with similar characteristics
3. Recommends the top 5 most similar anime titles
""")

# Optional: Show raw data
if st.checkbox("Show raw data"):
    st.subheader("Anime Dataset")
    st.dataframe(new_df)