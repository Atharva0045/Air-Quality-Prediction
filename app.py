from flask import Flask, request, jsonify, render_template
from joblib import load
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = load('models/air_quality_model.joblib')
scaler = pickle.load(open('helpers/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # request form data given by the user
    data = request.json

    # modify and scale the data
    input_data = np.array([data['features']])
    scaled_data = scaler.transform(input_data)

    # predict the index (Relative Humidity)
    prediction = model.predict(scaled_data)

    # return the prediction for display
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
