import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC 
from sklearn.linear_model import SGDClassifier  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

current_dir = os.path.dirname(__file__)

data_path = os.path.join(current_dir, 'data', 'cardio_train.csv')

data = pd.read_csv(data_path, sep=';')

X = data.drop('cardio', axis=1)  
y = data['cardio']    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'models/scaler.pkl')
print("Scaler saved as scaler.pkl")

models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SGD Support Vector Machine': SGDClassifier()
}

best_accuracy = 0
best_model = None

for name, model in models.items():
    try:
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy * 100:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
        
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\n{name} Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
    except Exception as e:
        print(f"An error occurred while evaluating {name}: {e}")
    
    print('-' * 50)

if best_model:
    joblib.dump(best_model, 'models/heart_disease_model.pkl')
    print(f"Best model saved as heart_disease_model.pkl with accuracy: {best_accuracy * 100:.2f}%") 
