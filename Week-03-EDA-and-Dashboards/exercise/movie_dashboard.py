import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set up the page
st.set_page_config(page_title="MovieLens Analysis", page_icon="🎬", layout="wide")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('/workspaces/ds-fall-2025-wed-S.B/Week-03-EDA-and-Dashboards/data/movie_ratings.csv')
    # Convert genres from string to list
    df['genres'] = df['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    return df

df = load_data()

# Title and introduction
st.title("🎬 MovieLens Data Analysis Dashboard")
st.markdown("""
This dashboard analyzes movie ratings from the MovieLens 200k dataset to uncover insights about:
- Genre popularity and satisfaction
- Rating trends over time
- Top-rated movies
- And more!
""")

# Sidebar for filters
st.sidebar.header("Filters")
min_ratings = st.sidebar.slider("Minimum ratings for analysis", 1, 200, 50)
selected_genres = st.sidebar.multiselect(
    "Select genres for analysis", 
    options=sorted(set([genre for sublist in df['genres'] for genre in sublist])),
    default=["Action", "Comedy", "Drama", "Sci-Fi"]
)

# Tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Genre Breakdown", 
    "Genre Satisfaction", 
    "Rating Trends", 
    "Top Movies", 
    "Age vs Ratings"
])

# 1. Genre breakdown
with tab1:
    st.header("Genre Breakdown")
    
    # Count genres (each movie can be in multiple genres)
    all_genres = [genre for sublist in df['genres'] for genre in sublist]
    genre_counts = Counter(all_genres)
    
    # Create dataframe for visualization
    genre_df = pd.DataFrame.from_dict(genre_counts, orient='index').reset_index()
    genre_df.columns = ['Genre', 'Count']
    genre_df = genre_df.sort_values('Count', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=genre_df, x='Count', y='Genre', palette='viridis', ax=ax)
    ax.set_title("Number of Ratings by Genre")
    st.pyplot(fig)
    
    st.markdown(f"""
    **Insights:**
    - The dataset contains {len(genre_df)} different genres
    - {genre_df.iloc[0]['Genre']} has the most ratings ({genre_df.iloc[0]['Count']})
    - {genre_df.iloc[-1]['Genre']} has the fewest ratings ({genre_df.iloc[-1]['Count']})
    """)

# 2. Genre satisfaction (highest ratings)
with tab2:
    st.header("Genre Satisfaction")
    
    # Explode the genres to have one row per genre per movie
    exploded_df = df.explode('genres')
    
    # Calculate mean rating per genre with minimum ratings threshold
    genre_ratings = exploded_df.groupby('genres').agg(
        mean_rating=('rating', 'mean'),
        count_ratings=('rating', 'count')
    ).reset_index()
    
    # Filter by minimum ratings
    genre_ratings = genre_ratings[genre_ratings['count_ratings'] >= min_ratings]
    genre_ratings = genre_ratings.sort_values('mean_rating', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=genre_ratings, x='mean_rating', y='genres', palette='coolwarm', ax=ax)
    ax.set_title(f"Average Rating by Genre (min {min_ratings} ratings)")
    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Genre")
    st.pyplot(fig)
    
    st.markdown(f"""
    **Insights:**
    - {genre_ratings.iloc[0]['genres']} has the highest average rating ({genre_ratings.iloc[0]['mean_rating']:.2f})
    - {genre_ratings.iloc[-1]['genres']} has the lowest average rating ({genre_ratings.iloc[-1]['mean_rating']:.2f})
    - The overall average rating across all genres is {df['rating'].mean():.2f}
    """)

# 3. Rating trends by release year
with tab3:
    st.header("Rating Trends by Release Year")
    
    # Calculate mean rating by year with minimum ratings threshold
    yearly_ratings = df.groupby('year').agg(
        mean_rating=('rating', 'mean'),
        count_ratings=('rating', 'count')
    ).reset_index()
    
    # Filter years with sufficient ratings
    yearly_ratings = yearly_ratings[yearly_ratings['count_ratings'] >= min_ratings]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(yearly_ratings['year'], yearly_ratings['mean_rating'], marker='o', linewidth=2)
    ax.set_title(f"Average Rating by Movie Release Year (min {min_ratings} ratings)")
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Average Rating")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Highlight top years
    top_years = yearly_ratings.nlargest(5, 'mean_rating')
    st.markdown("**Top 5 years with highest average ratings:**")
    for i, row in top_years.iterrows():
        st.markdown(f"- {int(row['year'])}: {row['mean_rating']:.2f} ({row['count_ratings']} ratings)")

# 4. Top movies with minimum ratings
with tab4:
    st.header("Top Rated Movies")
    
    # Calculate movie statistics
    movie_stats = df.groupby(['title', 'year']).agg(
        mean_rating=('rating', 'mean'),
        count_ratings=('rating', 'count')
    ).reset_index()
    
    # Filter by minimum ratings
    min_50 = movie_stats[movie_stats['count_ratings'] >= 50].nlargest(5, 'mean_rating')
    min_150 = movie_stats[movie_stats['count_ratings'] >= 150].nlargest(5, 'mean_rating')
    
    # Display tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 5 Movies (≥50 ratings)")
        st.dataframe(min_50[['title', 'year', 'mean_rating', 'count_ratings']].reset_index(drop=True))
    
    with col2:
        st.subheader("Top 5 Movies (≥150 ratings)")
        st.dataframe(min_150[['title', 'year', 'mean_rating', 'count_ratings']].reset_index(drop=True))
    
    st.markdown("""
    **Insights:**
    - Movies with more ratings tend to have more reliable average ratings
    - The highest rated movies often have strong cult followings or critical acclaim
    """)

# 5. Age vs ratings for selected genres
with tab5:
    st.header("Age vs Ratings by Genre")
    
    # Create age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 45, 55, 100], 
                            labels=['<18', '18-25', '26-35', '36-45', '46-55', '55+'])
    
    # Filter for selected genres
    exploded_df = df.explode('genres')
    filtered_df = exploded_df[exploded_df['genres'].isin(selected_genres)]
    
    # Calculate mean rating by age group and genre
    age_genre_ratings = filtered_df.groupby(['age_group', 'genres']).agg(
        mean_rating=('rating', 'mean'),
        count_ratings=('rating', 'count')
    ).reset_index()
    
    # Filter by minimum ratings
    age_genre_ratings = age_genre_ratings[age_genre_ratings['count_ratings'] >= min_ratings]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(data=age_genre_ratings, x='age_group', y='mean_rating', 
                 hue='genres', marker='o', ax=ax)
    ax.set_title(f"Average Rating by Age Group and Genre (min {min_ratings} ratings)")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Average Rating")
    ax.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
    
    st.markdown(f"""
    **Insights:**
    - Different age groups have varying preferences for genres
    - Some genres may be consistently rated across age groups while others show significant variation
    """)

# Extra credit: Ratings volume vs mean rating per genre
st.header("Extra: Ratings Volume vs Mean Rating")
# Calculate genre statistics
genre_stats = exploded_df.groupby('genres').agg(
    mean_rating=('rating', 'mean'),
    count_ratings=('rating', 'count')
).reset_index()

# Filter by minimum ratings
genre_stats = genre_stats[genre_stats['count_ratings'] >= min_ratings]

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(genre_stats['count_ratings'], genre_stats['mean_rating'], alpha=0.6)

# Add labels for some points
for i, row in genre_stats.iterrows():
    if row['count_ratings'] > 10000 or row['mean_rating'] > 4.0:
        ax.annotate(row['genres'], (row['count_ratings'], row['mean_rating']), 
                   xytext=(5, 5), textcoords='offset points')

ax.set_xlabel("Number of Ratings")
ax.set_ylabel("Average Rating")
ax.set_title(f"Ratings Volume vs Mean Rating by Genre (min {min_ratings} ratings)")
ax.grid(True, alpha=0.3)

# Calculate correlation
correlation = np.corrcoef(genre_stats['count_ratings'], genre_stats['mean_rating'])[0, 1]
ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes, 
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

st.pyplot(fig)

st.markdown(f"""
**Insights:**
- The correlation between number of ratings and average rating is {correlation:.3f}
- {'' if correlation > 0 else 'Not '}able to conclude that popular genres receive higher ratings
- Some niche genres with fewer ratings can still achieve high average ratings
""")

# Footer
st.markdown("---")
st.markdown("MovieLens 200k Dataset Analysis | Created with Streamlit")