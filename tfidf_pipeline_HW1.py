import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Generate a corpus of 10 sentences about AI in English
corpus = [
    "Artificial Intelligence is the future.",
    "AI is changing the world.",
    "AI is a branch of computer science.",
]

# 2. Define the preprocessing pipeline
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = text.lower().split()
    tokens = [token.strip(string.punctuation) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

# 3. Apply preprocessing to the corpus
processed_corpus = [preprocess(doc) for doc in corpus]

# 4. TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_corpus)

# 5. Print feature names and TF-IDF array
print(vectorizer.get_feature_names_out())
print(X.toarray()) 