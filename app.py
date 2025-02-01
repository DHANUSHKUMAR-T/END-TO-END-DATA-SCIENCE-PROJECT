from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('C:\\Users\\Dhanush\\OneDrive\\Desktop\\churn_prediction_project\\heart_disease_prediction\\saved_model\\heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = [float(request.form[key]) for key in request.form]
    
    # Reshape data for prediction
    data = np.array(data).reshape(1, -1)
    
    # Predict heart disease (1 = Yes, 0 = No)
    prediction = model.predict(data)
    
    return render_template('result.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
