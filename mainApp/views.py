from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout
from django.conf import settings
import tensorflow
from django.shortcuts import render
import pandas as pd
import pickle
import requests
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
def fetch_poster(movie_title):
    api_key = "bdef3705"  # You can replace this with your actual API key
    response = requests.get(f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}")
    data = response.json()
    if 'Poster' in data:
        return data['Poster']
    else:
        return None

def recommend(movie, movies, similarity):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

def movie_recommender(request):
    # Load movie data from pickle files
    movie_dict_path = os.path.join(settings.BASE_DIR, 'mainApp', 'movie_dict.pkl')
    similarity_path = os.path.join(settings.BASE_DIR, 'mainApp', 'similarity.pkl')

    with open(movie_dict_path, 'rb') as f:
        movies_dict = pickle.load(f)
    movies = pd.DataFrame(movies_dict)['title'].tolist()
    movies2 = pd.DataFrame(movies_dict)
    if request.method == 'POST':
        selected_movie = request.POST.get('selected_movie')

        with open(similarity_path, 'rb') as f:
            similarity = pickle.load(f)

        # Check if the selected movie exists in the dictionary
        if selected_movie not in movies2['title'].values:
            return render(request, 'error.html', {'error_message': 'Selected movie not found'})

        # Perform recommendation
        recommended_movies = recommend(selected_movie, movies2, similarity)
        posters = [fetch_poster(movie) for movie in recommended_movies]

        poster_url = fetch_poster(selected_movie)
        recommended_data = list(zip(recommended_movies, posters))
        return render(request, 'movie_recommender.html', {'selected_movie': selected_movie,'recommended_data': recommended_data, 'recommended_movies': recommended_movies, 'poster_url': poster_url, 'movies': movies,'posters': posters})
    else:
        return render(request, 'movie_recommender.html', {'movies': movies})
def login(request):
    return render(request, 'home.html')


@login_required
def home(request):
    return render(request,'home.html')


def logout_view(request):
    logout(request)
    return redirect('home')



from django.core.files.storage import FileSystemStorage
from .models import UploadedImage
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import pandas as pd
import os
import pickle


# Load the embeddings and filenames

embd = os.path.join(settings.BASE_DIR, 'mainApp', 'embeddings.pkl')
with open(embd, 'rb') as f:
    feature_list=pickle.load(f)
fln= os.path.join(settings.BASE_DIR, 'mainApp', 'filenames.pkl')


with open(fln, 'rb') as f:
    filenames=pickle.load(f)
# Define ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to extract features from an image
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function to recommend similar images
def recommend2(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Load the styles data from CSV
styles_data = None
styles_csv_path = os.path.join(settings.BASE_DIR, 'mainApp', 'styles.csv')
if os.path.exists(styles_csv_path):
    try:
        styles_data = pd.read_csv(styles_csv_path, usecols=['id', 'gender', 'masterCategory', 'subCategory', 'articleType',
                                                     'baseColour', 'season', 'year', 'usage', 'productDisplayName'])
    except pd.errors.ParserError:
        # Handle the error gracefully in your application
        pass
else:
    # Handle the case where the file does not exist
    pass
# Function to extract ID from filename
def extract_id_from_filename(filename):
    # Get just the filename without the directory path
    filename = os.path.basename(filename)
    # Split the filename by directory separator '\\'
    parts = filename.split('\\')
    # Take the last part which should be the filename without the directory
    filename = parts[-1]
    # Split the filename by '_'
    parts = filename.split('_')
    # The ID is usually the last part of the filename
    id_part = parts[-1]
    # Remove the file extension
    id_part = os.path.splitext(id_part)[0]
    # Check if the id_part can be converted to an integer
    if id_part.isdigit():
        return int(id_part)
    else:
        print(id_part)
        return None
# Main view
def fashion_recommender(request):
    if request.method == 'POST' and request.FILES.get('uploaded_file'):
        uploaded_file = request.FILES['uploaded_file']
        if uploaded_file:
            fs = FileSystemStorage(location='mainApp/static/uploads/')
            filename = fs.save(uploaded_file.name, uploaded_file)
            uploaded_file_url = fs.url(filename)

            img_path = os.path.join(fs.location, filename)
            print(img_path)

            try:
                features = feature_extraction(img_path, model)
                indices = recommend2(features, feature_list)

                recommended_images = [filenames[index] for index in indices[0]]
                recommended_details = []

                for img_file in recommended_images:
                    id = extract_id_from_filename(img_file)
                    csv_entry = styles_data[styles_data['id'] == int(id)] if styles_data is not None else None
                    recommended_details.append({
                        'image_path': img_file,
                        'id': id,
                        'product_display_name': csv_entry['productDisplayName'].values[0] if csv_entry is not None and not csv_entry.empty else None,
                        'gender': csv_entry['gender'].values[0] if csv_entry is not None and not csv_entry.empty else None,
                        'master_category': csv_entry['masterCategory'].values[0] if csv_entry is not None and not csv_entry.empty else None,
                        'sub_category': csv_entry['subCategory'].values[0] if csv_entry is not None and not csv_entry.empty else None,
                        'article_type': csv_entry['articleType'].values[0] if csv_entry is not None and not csv_entry.empty else None,
                        'base_colour': csv_entry['baseColour'].values[0] if csv_entry is not None and not csv_entry.empty else None,
                        'season': csv_entry['season'].values[0] if csv_entry is not None and not csv_entry.empty else None,
                        'year': csv_entry['year'].values[0] if csv_entry is not None and not csv_entry.empty else None,
                        'usage': csv_entry['usage'].values[0] if csv_entry is not None and not csv_entry.empty else None,
                    })

                return render(request, 'fashion_recommender.html', {
                    'uploaded_file_url': uploaded_file_url,
                    'recommended_details': recommended_details
                })
            except Exception as e:
                # Handle any exceptions that might occur during feature extraction or recommendation
                print(f"Error: {e}")
                return render(request, 'error.html', {'error_message': 'An error occurred during recommendation'})
        else:
            return render(request, 'error.html', {'error_message': 'No file provided'})
    return render(request, 'fashion_recommender.html')
