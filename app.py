import pandas as pd  
from flask import Flask, render_template, request
import numpy as np
import joblib
import base64
import io
from data_processing import (plot_age_distribution, plot_gender_distribution, plot_height_distribution, 
                             plot_weight_distribution, plot_systolic_bp_distribution, plot_diastolic_bp_distribution,
                             plot_cholesterol_distribution, plot_glucose_distribution, plot_smoking_status_distribution,
                             plot_alcohol_consumption_distribution, plot_physical_activity_distribution,
                             plot_cardiovascular_disease_distribution, plot_correlation_matrix)

app = Flask(__name__)

model = joblib.load('models/heart_disease_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            features = {
                "id": request.form['id'],
                "age": request.form['age'],
                "gender": request.form['gender'],
                "height": request.form['height'],
                "weight": request.form['weight'],
                "ap_hi": request.form['ap_hi'],
                "ap_lo": request.form['ap_lo'],
                "cholesterol": request.form['cholesterol'],
                "gluc": request.form['gluc'],
                "smoke": request.form['smoke'],
                "alco": request.form['alco'],
                "active": request.form['active'],
            }

            for key, value in features.items():
                if value.strip() == '':
                    raise ValueError(f"{key.capitalize()} is missing.")
                features[key] = float(value) 

            data_input = pd.DataFrame([features])

            scaled_data = scaler.transform(data_input)
            
            prediction = model.predict(scaled_data)
            
            result = "Positive (Risk of Heart Disease)" if prediction[0] == 1 else "Negative (No Risk)"
            
            return render_template('predict.html', prediction_text=result)

        except ValueError as e:
            error_message = f"Input error: {str(e)}"
            return render_template('predict.html', prediction_text=error_message)
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            return render_template('predict.html', prediction_text=error_message)

    return render_template('predict.html')

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/plot')
def plot():
    plot_type = request.args.get('plot_type')
    img = io.BytesIO()

    try:
        if plot_type == 'age_distribution':
            img = plot_age_distribution()
        elif plot_type == 'gender_distribution':
            img = plot_gender_distribution()
        elif plot_type == 'height_distribution':
            img = plot_height_distribution()
        elif plot_type == 'weight_distribution':
            img = plot_weight_distribution()
        elif plot_type == 'systolic_bp_distribution':
            img = plot_systolic_bp_distribution()
        elif plot_type == 'diastolic_bp_distribution':
            img = plot_diastolic_bp_distribution()
        elif plot_type == 'cholesterol_distribution':
            img = plot_cholesterol_distribution()
        elif plot_type == 'glucose_distribution':
            img = plot_glucose_distribution()
        elif plot_type == 'smoking_status_distribution':
            img = plot_smoking_status_distribution()
        elif plot_type == 'alcohol_consumption_distribution':
            img = plot_alcohol_consumption_distribution()
        elif plot_type == 'physical_activity_distribution':
            img = plot_physical_activity_distribution()
        elif plot_type == 'cardiovascular_disease_distribution':
            img = plot_cardiovascular_disease_distribution()
        elif plot_type == 'correlation_matrix':
            img = plot_correlation_matrix()
        else:
            return "Plot type not supported", 400

        plot_url = base64.b64encode(img.getvalue()).decode()
        return f'<img src="data:image/png;base64,{plot_url}" />'

    except Exception as e:
        return f"Error generating plot: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
