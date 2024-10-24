Heart Disease Prediction Project

Overvie
w
This project aims to develop a machine learning model that predicts the likelihood of heart disease based on various health metrics. The model is built using Python and leverages libraries such as Scikit-learn for machine learning, Pandas for data manipulation, and Matplotlib/Seaborn for data visualization.

Project Structure

heart_disease_prediction/
│
├── app.py                     # Main application file for running the Flask web server
├── data/                      # Directory containing the dataset
│   └── cardio_train.csv       # Training dataset for the model
├── data_processing.py         # Script for preprocessing the dataset
├── model_training.py          # Script for training the machine learning model
├── models/                    # Directory for storing model files
│   ├── heart_disease_model.pkl # Trained model file (generated after running model training)
│   └── scaler.pkl             # Scaler file for feature normalization
├── requirements.txt           # List of dependencies required to run the project
├── static/                    # Directory for static files (CSS, images)
│   └── images/                # Images used in the application
├── templates/                 # Directory for HTML templates
│   ├── index.html             # Homepage
│   ├── predict.html           # Prediction results page
│   ├── visualizations.html     # Visualizations page
│   └── plot.html              # Page for displaying plots
├── .gitignore                  # Files and directories to ignore in Git
└── README.md                  # Project documentation
Getting Started
Prerequisites
Before you begin, ensure you have the following installed:

Python 3.x
pip (Python package installer)
Installation
Clone the repository:

git clone https://github.com/ebbran/heart_disease_prediction.git
cd heart_disease_prediction
Install the required packages:

pip install -r requirements.txt
Running the Model Training
To train the model and generate the required .pkl files, run the following command:

python model_training.py
This will create two files in the models directory:

heart_disease_model.pkl: The trained model file.
scaler.pkl: The scaler used for normalizing input features.
Running the Application
To start the Flask web server, execute:

python app.py
Then open your web browser and navigate to http://127.0.0.1:5000/ to access the application.

Usage
Once the application is running, you can input the necessary health metrics to get predictions about heart disease. The application will utilize the trained model to provide insights based on the data you provide.

Notes
Ensure that you run the model training script before using the application to generate the required model files.
You can modify the data_processing.py script if you want to preprocess the dataset differently.
License
This project is licensed under the MIT License - see the LICENSE file for details.
