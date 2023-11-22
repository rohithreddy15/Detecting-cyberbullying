import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load your trained model using pickle
with open('CBDmodel1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the route for serving the HTML template
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for text classification
@app.route('/classify', methods=['POST'])
def classify():
    text = request.json['text']
    
    # Use your loaded model for prediction
    prediction = model.predict([text])[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

