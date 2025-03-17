
from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re

# Download required NLTK data
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

# Load the trained model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

# Preprocessing function
def preprocess_text(text):
    if not text.strip():  # Handle empty input
        return ""

    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove special characters & numbers
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming

    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    result = None

    if request.method == "POST":
        news_text = request.form.get("news", "").strip()  # Get input & handle empty string
        
        if news_text:
            processed_text = preprocess_text(news_text)
            text_vector = vectorizer.transform([processed_text])  # Convert text to vector
            prediction = model.predict(text_vector)[0]  # Predict (0 = Fake, 1 = Real)
            result = "Real News ✅" if prediction == 1 else "Fake News ❌"
        else:
            result = "Please enter a news article!"

    return render_template("index.html", result=result, news_text=news_text if request.method == "POST" else "")

if __name__ == "__main__":
    app.run(debug=True)
