from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load the model
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Load the location and column information
with open('columns.json', 'r') as f:
    columns = json.load(f)
    locations = columns['data_columns'][3:]

@app.route('/')
def index():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    total_sqft = float(request.form['total_sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    loc_index = locations.index(location)

    input_data = np.zeros(len(locations) + 3)
    input_data[0] = total_sqft
    input_data[1] = bath
    input_data[2] = bhk

    if loc_index >= 0:
        input_data[loc_index + 3] = 1

    prediction = model.predict([input_data])[0]

    return render_template('index.html', locations=locations, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
