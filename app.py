from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models and scalers
# These models and scalers are used for air quality index prediction and sentiment analysis

# Load the trained model and scaler for air quality prediction
# The model was trained to predict air quality based on various features
air_quality_model = pickle.load(open('models/air_quality_model.pkl', 'rb'))

# The scaler standardizes the input data before prediction to match the model's training conditions
air_quality_scaler = pickle.load(open('helpers/scaler.pkl', 'rb'))

# Load the TensorFlow model for sentiment analysis
# This model predicts the sentiment (positive/negative/neutral) of text data
sentiment_analyzer = tf.keras.models.load_model('models/sentiment_analysis_model.h5')

# The vectorizer converts input text into numerical form for the sentiment model to process
# Using traditional file handling for loading due to issues with default pickle loading
with open('helpers/vectorizer.pkl', 'rb') as file:
    sentiment_vectorizer = pickle.load(file)


# Default route: renders the home page
@app.route('/')
def home():
    return render_template('home.html')


# Route for air quality prediction app
@app.route('/air_quality_prediction', methods=['GET', 'POST'])
def app1():
    return render_template('air_quality.html')


# API route for predicting air quality based on user-provided features
@app.route('/air_quality_prediction/predict', methods=['POST'])
def predict():
    # Accepts JSON data via POST, extracts the features, and makes a prediction
    data = request.json
    
    # Convert the input data into a numpy array and scale it
    input_data = np.array([data['features']])
    scaled_data = air_quality_scaler.transform(input_data)
    
    # Predict air quality using the pre-loaded model
    prediction = air_quality_model.predict(scaled_data)
    
    # Return the prediction result as a JSON response
    return jsonify({'prediction': prediction.tolist()})


# Route for sentiment analysis app
@app.route('/sentiment_analysis', methods=['GET', 'POST'])
def app2():
    sentiment_result = None
    
    if request.method == 'POST':
        # Get the text headline from the form
        headline = request.form['headline']

        sentiment_result = predict_sentiment(headline)
    
    return render_template('sentiment_analysis.html', result=sentiment_result)


def predict_sentiment(headline):

    """
    Helper function to predict sentiment for a given Text Headline

    Input: Technical News Headline: string

    Return: Predicted sentiment as either positive, negative or neutral
    """

    # Vectorize the input text to match the model's expected input format
    X = sentiment_vectorizer.transform([headline]).toarray()
    
    # Use the sentiment model to predict the sentiment
    prediction = sentiment_analyzer.predict(X)
    
    # Map the prediction to a label (neutral, positive, or negative)
    sentiment_labels = ["neutral", "positive", "negative"]
    predicted_sentiment = sentiment_labels[np.argmax(prediction)]
    
    # Return the predicted sentiment
    return predicted_sentiment


# Entry point of the Flask app
if __name__ == '__main__':
    app.run(debug=True)
