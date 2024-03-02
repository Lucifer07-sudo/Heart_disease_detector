from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained machine learning model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the request
    data = request.json

    # Preprocess input data if necessary
    # For example, convert data types, scale features, etc.

    # Make prediction using the model
    prediction = model.predict([list(data.values())])

    # Format the prediction result
    prediction_text ="Patient has heart disease" if prediction == 1 else "Patient has no heart disease"

    # Return the prediction result as JSON
    return jsonify({'prediction_text': prediction_text})

if __name__ == '__main__':
    app.run(debug=True)
