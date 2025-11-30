import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("PROJECT_5", "merged_books.csv")  # update with your file path
    df['Primary_Genre'] = df['Ranks and Genre'].str.extract(r'in (.*?)\s*\(')
    df['Combined_Features'] = (
        df['Description'].fillna('') + " " +
        df['Primary_Genre'].fillna('') + " " +
        df['Author'].fillna('')
    )
    return df

df = load_data()

# TF-IDF Vectorizer
@st.cache_resource
def build_model(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    feature_matrix = vectorizer.fit_transform(df['Combined_Features'])
    cosine_sim = cosine_similarity(feature_matrix)
    return vectorizer, cosine_sim

vectorizer, cosine_sim = build_model(df)


# -------------------------
# Helper Function
# -------------------------
def recommend_books(book_title, n=5):
    if book_title not in df['Book Name'].values:
        return []
    idx = df[df['Book Name'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    recommended_books = df.iloc[[i[0] for i in sim_scores]][['Book Name', 'Author', 'Rating']]
    return recommended_books


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="📚 Book Recommendation System", layout="wide")

st.title("📚 Book Recommendation System")
st.markdown("Built with **Streamlit + NLP + Clustering**")

# Sidebar - user input
st.sidebar.header("🔍 User Preferences")
option = st.sidebar.radio("Choose Recommendation Mode:", ["By Genre", "By Book"])

if option == "By Genre":
    genre_list = df['Primary_Genre'].dropna().unique().tolist()
    genre_choice = st.sidebar.selectbox("Select a Genre:", sorted(genre_list))
    top_books = df[df['Primary_Genre'] == genre_choice].sort_values(by="Rating", ascending=False).head(5)
    st.subheader(f"✨ Top 5 Books in {genre_choice}")
    st.table(top_books[['Book Name', 'Author', 'Rating']])

elif option == "By Book":
    book_list = df['Book Name'].dropna().unique().tolist()
    book_choice = st.sidebar.selectbox("Select a Book:", sorted(book_list))
    recommendations = recommend_books(book_choice, n=5)
    st.subheader(f"📖 Because you liked *{book_choice}*")
    st.table(recommendations)


# -------------------------
# EDA Section
# -------------------------
st.header("📊 Exploratory Data Analysis (EDA)")

# 1. Popular Genres
st.subheader("Top 10 Most Popular Genres")
top_genres = df['Primary_Genre'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x=top_genres.values, y=top_genres.index, ax=ax)
st.pyplot(fig)

# 2. Rating Distribution
st.subheader("Distribution of Ratings")
fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df['Rating'], bins=20, kde=True, ax=ax)
st.pyplot(fig)

# 3. Ratings vs Number of Reviews
st.subheader("Ratings vs Number of Reviews")
fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(x='Number of Reviews', y='Rating', data=df, alpha=0.6, ax=ax)
ax.set_xscale("log")
st.pyplot(fig)

st.markdown("✅ End of Analysis. Use the sidebar to explore recommendations!")

