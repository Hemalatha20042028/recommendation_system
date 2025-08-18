# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import requests
from time import sleep

# Set page config - MODIFIED to add theme="auto"
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎬",
    layout="wide",
    theme="auto"  # Added this line for theme support
)

# NEW THEME DETECTION CODE - Add this right after page config
def get_theme():
    """Detect current theme using native Streamlit methods"""
    try:
        # For Streamlit >= 1.16
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx and hasattr(ctx, "theme"):
            return ctx.theme.config.get("base", "light")
    except:
        pass
    return "light"

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

def get_anime_image(anime_name):
    """Fetch anime image URL from Jikan API"""
    try:
        url = f"https://api.jikan.moe/v4/anime?q={anime_name}&limit=1"
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()
        
        if data['data']:
            return data['data'][0]['images']['jpg']['image_url']
        return None
    except Exception as e:
        st.warning(f"Couldn't fetch image for {anime_name}: {str(e)}")
        return None

# Load data
new_df, similarity = load_data()
anime_names = sorted(new_df['name'].tolist())

# UI Elements - MODIFIED to add theme toggle
st.title("🎬 Anime Recommendation System")
st.markdown("Discover new anime similar to your favorites!")

# NEW CODE - Add theme toggle to sidebar
with st.sidebar:
    current_theme = get_theme()
    dark_mode = st.toggle("🌙 Dark Mode", 
                         value=(current_theme == "dark"),
                         key="dark_mode_toggle")

# Apply custom CSS based on theme - NEW CODE
if dark_mode:
    st.markdown("""
    <style>
        /* Dark background */
        .stApp {
            background-color: #0E1117;
        }
        /* Text color */
        .st-bw, .st-cm, h1, h2, h3, h4, h5, h6, p {
            color: white !important;
        }
        /* Cards and containers */
        .st-bb, .st-at {
            background-color: #1E1E1E !important;
            border-color: #333 !important;
        }
        /* Input fields */
        .st-bq, .st-cf {
            background-color: #2D2D2D !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

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
        
        # Get top 5 recommendations (excluding the anime itself)
        anime_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        st.success(f"Top recommendations for **{selected_anime}**:")
        
        # Display recommendations in columns
        cols = st.columns(5)
        for idx, (i, score) in enumerate(anime_list):
            with cols[idx]:
                anime_name = new_df.iloc[i]['name']
                st.subheader(anime_name)
                st.caption(f"Similarity score: {score:.2f}")
                
                # Get and display anime image
                image_url = get_anime_image(anime_name)
                if image_url:
                    st.image(image_url, caption=anime_name, use_column_width=True)
                    sleep(1)  # Respect API rate limits (3 requests/second)
                else:
                    st.warning("Image not available")
                
    except IndexError:
        st.error(f"❌ '{selected_anime}' not found in database. Please try another title.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add some info/instructions
st.divider()
st.markdown("### How it works:")
st.markdown("""
1. The system analyzes anime genres, types, and episode counts
2. Uses cosine similarity to find shows with similar characteristics
3. Recommends the top 5 most similar anime titles
4. Fetches images from MyAnimeList's API
""")

# Optional: Show raw data
if st.checkbox("Show raw data"):
    st.subheader("Anime Dataset")
    st.dataframe(new_df)