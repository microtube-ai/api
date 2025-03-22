from flask import Flask, request, jsonify
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

import os
import re
import glob
import numpy as np
import pandas as pd
import ffmpeg

import requests

import whisper
import yt_dlp
import googleapiclient.discovery

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Gensim packages
from gensim import corpora
from gensim import models
from gensim.parsing import strip_tags, strip_numeric, strip_multiple_whitespaces
from gensim.parsing import stem_text, strip_punctuation, remove_stopwords, preprocess_string


import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_query():
    API_KEY = 'AIzaSyAdMTNN6ERopF-Z1FijQgOrQvQIR_atzVs'
    BASE_URL = 'https://www.googleapis.com/youtube/v3/'

    user_input = request.json.get('query', '')
    query = user_input
    
    max_results = 15
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)


    search_request = youtube.search().list(
        q=query,
        part="snippet",
        type='video',
        videoCategoryId=27,
        videoCaption="closedCaption",
        relevanceLanguage='en',
        maxResults=max_results
    )

    search_response = search_request.execute()

    video_data = []
    for item in search_response['items']:
        video_id = item['id']['videoId']
        video_title = item['snippet']['title']
        video_thumbnail = (item['snippet']['thumbnails']['high']['url'])
        video_data.append({'video_id': video_id, 'title': video_title, 'thumbnail':video_thumbnail})

    video_request = youtube.videos().list(
        part='contentDetails',
        id=','.join([video['video_id'] for video in video_data])
    )

    video_response = video_request.execute()

    def parse_duration(duration_str):
        duration_elements = duration_str[2:].split('H', 1)
        hours = int(duration_elements[0]) if len(duration_elements) > 1 else 0

        duration_elements = duration_elements[-1].split('M', 1)
        minutes = int(duration_elements[0]) if len(duration_elements) > 1 else 0

        duration_elements = duration_elements[-1].split('S', 1)
        seconds = int(duration_elements[0]) if len(duration_elements) > 1 else 0

        return hours * 3600 + minutes * 60 + seconds

    for video, video_info in zip(video_data, video_response['items']):
        video['link'] = f'https://www.youtube.com/watch?v={video_info["id"]}'
        video['duration'] = parse_duration(video_info['contentDetails']['duration'])

    filtered_results = [video for video in video_data if video['duration'] < 720]

    df = pd.DataFrame(filtered_results, columns=['title', 'link','thumbnail', 'duration'])

    for index, video_link in enumerate(df['link']):
        try:
            print('Downloading Video:',index,'-',video_link)
            download_youtube_video(video_link, index)
        except Exception as e:
            print(f"Error downloading video at index {index}: {e}")
            df = df.drop(index, inplace=False)
    
    model = whisper.load_model("base")

    video_dir_paths = []
    for vid_dir in os.scandir('./videos'):
        if vid_dir.is_file() and vid_dir.name.endswith('.mp4'):
            video_dir_paths.append(vid_dir.path)

    video_dir_paths = sorted(video_dir_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    list_text = []

    for path in video_dir_paths:
        print(path)
        result = model.transcribe(path)

        # Adding Language Restriction - Only English Language Videos
        if result['language'] != 'en':
            index = int(path.split('/')[-1].split('.')[0])
            df = df.drop([index])
        elif result['text'] == '' or len(set(result['text'].split())) <= 25:
            index = int(path.split('/')[-1].split('.')[0])
            df = df.drop([index])
        else:
            list_text.append(result['text'])

    df['original_text'] = list_text
    df = df.reset_index(drop=True)

    # Removing saved videos
    files = glob.glob('videos/*')
    for f in files:
        os.remove(f)

    # Restricting no. of videos
    no_of_videos_restriction = 10

    if df.shape[0] > no_of_videos_restriction:
        list_text = list_text[:no_of_videos_restriction]
        df = df.iloc[:no_of_videos_restriction]

    stop_words = set(stopwords.words('english'))

    wnl = WordNetLemmatizer()

    def custom_tokenizer(doc):
        tokens = word_tokenize(doc)
        filtered_tokens = [wnl.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha()]
        return filtered_tokens

    search_terms = query

    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, stop_words=None)
    doc_vectors = vectorizer.fit_transform([search_terms] + list_text)

    # Calculate similarity
    cosine_similarities = linear_kernel(doc_vectors[0:1], doc_vectors).flatten()
    document_scores = [item.item() for item in cosine_similarities[1:]]

    df['doc_sim'] = pd.Series(document_scores)

    transform_to_lower = lambda s: s.lower()

    remove_single_char = lambda s: re.sub(r'\s+\w{1}\s+', '', s)

    # Adding Lemmatization
    def lemmatize_text(text):
        lemmatizer = WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    CLEAN_FILTERS = [strip_tags,
                    strip_numeric,
                    strip_punctuation,
                    strip_multiple_whitespaces,
                    transform_to_lower,
                    remove_stopwords,
                    remove_single_char,
                    lemmatize_text]

    def cleaning_pipe(document):
        processed_text = preprocess_string(document, CLEAN_FILTERS)
        return processed_text

    list_text_clean = []

    for i in list_text:
        list_text_clean.append(cleaning_pipe(i))

    list_text_clean.append(search_terms.split(" "))

    dictionary = corpora.Dictionary()
    BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in list_text_clean]

    tfidf = models.TfidfModel(BoW_corpus, smartirs='ntc')
    word_vector = tfidf[BoW_corpus]

    # Calculate similarity
    cosine_similarities = linear_kernel(doc_vectors[-1], doc_vectors).flatten()
    document_scores = [item.item() for item in cosine_similarities[0:-1]]

    df['doc_sim_gensim'] = pd.Series(document_scores)

    result = df.to_dict(orient='records')  # Convert DataFrame to list of dictionaries
    return jsonify(result)
    # return query


def download_youtube_video(url, index_num):
    ydl_opts = {
        'format': 'best',
         'outtmpl': f'./videos/{index_num}.%(ext)s',
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")


@app.route('/api', methods=['POST'])
def api_query():
    user_input = request.json.get('query', '')
    return f'Response --> {user_input}'

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
