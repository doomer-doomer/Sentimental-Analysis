from flask import Flask, request, render_template
import joblib
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Initialize the Flask app
app = Flask(__name__)

import nltk
nltk.download('stopwords')

# Load the trained sentiment analysis model and TF-IDF vectorizer
model = joblib.load('sentiment_model_400k.pkl')
tfidf = joblib.load('tfidf_vectorizer_400k.pkl')

# Define the cleanText function (you can customize it as per your previous definition)
def cleanText(raw_text, remove_stopwords=False, stemming=False, split_text=False):
    # Remove HTML
    text = BeautifulSoup(raw_text, 'lxml').get_text()

    # Remove non-alphabet characters
    letters_only = re.sub("[^a-zA-Z]", " ", text)

    # Convert to lower case and split into words
    words = letters_only.lower().split()

    # Optionally remove stopwords
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]

    # Optionally perform stemming
    if stemming:
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(w) for w in words]

    # Return list of words or joined text
    return " ".join(words) if not split_text else words

# Define the prediction function
def predict_review(text):
    clean_input = cleanText(text, remove_stopwords=True, stemming=True)
    vectorized_input = tfidf.transform([clean_input])
    prediction = model.predict(vectorized_input)
    return 'Positive' if prediction == 1 else 'Negative'

# Define a route to handle requests to the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the user input from the form
        user_review = request.form['review']

        # Predict sentiment of the input review
        sentiment = predict_review(user_review)

        # Return the result to the user
        return render_template('index.html', review=user_review, result=sentiment)

    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
