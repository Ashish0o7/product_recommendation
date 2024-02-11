import pickle
import streamlit as st
import requests
import pandas as pd

def fetch_poster(id):
    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=305730f2bd944fa5d1dc3dce9b546fcc&&language=eb-US'.format(id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances=similarity[movie_index]
    movies_list= sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]

    recommended_movies = []
    recommended_movies_posters=[]
    for i in movies_list:
        movie_id=movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))
    return recommended_movies,recommended_movies_posters


st.header('Movie Recommender System')
movies_dict = pickle.load(open('movie_dict.pkl','rb'))
movies=pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl','rb'))

selected_movie_name=st.selectbox('', movies['title'].values)
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #B6BBBB;
            color: #ffffff;
            font-weight: bold;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        .stButton>button:hover {
            background-color:#B6BBBB;
        }
    </style>
    """,
    unsafe_allow_html=True
)
if st.button('Recommend', key='recommend_button'):
    names, posters = recommend(selected_movie_name)

    num_recommendations = min(10, len(names))  

    col1, col2, col3, col4, col5 = st.columns(5)
    poster_style = "margin-bottom: 10px;"
    for i in range(0, num_recommendations, 5):
        with col1:
            if i < num_recommendations:
                st.image(posters[i])
                st.text(names[i])

        with col2:
            if i + 1 < num_recommendations:
                st.image(posters[i + 1])
                st.text(names[i + 1])

        with col3:
            if i + 2 < num_recommendations:
                st.image(posters[i + 2])
                st.text(names[i + 2])

        with col4:
            if i + 3 < num_recommendations:
                st.image(posters[i + 3])
                st.text(names[i + 3])

        with col5:
            if i + 4 < num_recommendations:
                st.image(posters[i + 4])
                st.text(names[i + 4])
