from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## import lassocv and standard scaler.
lassocv_model = pickle.load(open('C:\Data\dataScience\ALGERIA_PROJECT\models\lassocv.pkl','rb'))
standard_scaler = pickle.load(open('C:\Data\dataScience\ALGERIA_PROJECT\models\scaler.pkl','rb'))



@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form input
    try:
        input_features = [float(request.form[key]) for key in ["Temperature", "RH", "Ws", "Rain", 
                                                               "FFMC", "DMC","ISI", 
                                                               "Classes", "Region"]]
        
        # Convert to NumPy array and reshape for prediction
        input_array = np.array(input_features).reshape(1, -1)
        new_scaled_data = standard_scaler.transform(input_array)
        # Predict FWI
        predicted_fwi = lassocv_model.predict(new_scaled_data)[0]

        return render_template('index.html', prediction=f"Predicted Fire Weather Index (FWI): {predicted_fwi:.2f}")

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ =="__main__":
    app.run(debug=True,host='0.0.0.0')