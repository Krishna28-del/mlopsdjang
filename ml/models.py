import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import os
# Load dataset
# Using an IMDb movie review dataset or creating a simple one for this example
data = pd.DataFrame({
    'text': ["I loved the movie", "The movie was terrible", "Best film ever", "Worst film", "It was okay","This was an amazing experience!","I'm thrilled with the outcome!.","I hated this movie."," Very disappointed.","Terrible acting a weak script and awful special effects.", "This movie was a complete disaster."],
    'label': [1, 0, 1, 0, 1 , 1 , 1 , 0 , 0 , 0, 0]
})

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Create a pipeline to vectorize the text and apply Naive Bayes
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'sentiment_model.pkl')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'sentiment_model.pkl')
sentiment_model = joblib.load(model_path)

def predict_sentiment(text):
    # Use the loaded model to predict sentiment
    prediction = sentiment_model.predict([text])[0]
    return "Positive" if prediction == 1 else "Negative"
