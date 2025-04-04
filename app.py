from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the pre-trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__)

# Crop dictionary mapping prediction output to crop names
CROP_DICT = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Lentil", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

@app.route('/')
def index():
    """
    Renders the main page with the input form.
    """
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    """
    Handles crop prediction based on user input.
    """
    try:
        # Retrieve form inputs
        inputs = ['Nitrogen', 'Phosporus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        feature_list = [float(request.form[input_name]) for input_name in inputs]

        # Prepare the input features as a NumPy array
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply MinMaxScaler and StandardScaler transformations
        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)

        # Predict the crop using the trained model
        prediction = model.predict(sc_mx_features)
        crop = CROP_DICT.get(prediction[0], "Unknown")
        
        # Generate the result message
        result = (
            f"{crop} is the best crop to be cultivated right there."
            if crop != "Unknown" 
            else "Sorry, we could not determine the best crop to be cultivated with the provided data."
        )

    except Exception as e:
        # Handle exceptions and display an error message
        result = f"An error occurred: {str(e)}"

    # Render the index page with the prediction result
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
