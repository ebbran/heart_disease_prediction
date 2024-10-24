import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

plt.switch_backend('Agg')

current_dir = os.path.dirname(__file__)

data_path = os.path.join(current_dir, 'data', 'cardio_train.csv')

data = pd.read_csv(data_path, sep=';')

print("Missing Values:\n", data.isnull().sum())
print("\nData Summary:\n", data.describe())
print("\nData Information:\n", data.info())

def get_image():
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

def plot_age_distribution():
    plt.figure(figsize=(10, 6))
    sns.histplot(data['age'], kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age (days)')
    plt.ylabel('Frequency')
    return get_image()

def plot_gender_distribution():
    plt.figure(figsize=(10, 6))
    sns.countplot(x='gender', data=data)
    plt.title('Gender Distribution')
    plt.xlabel('Sex (1: Female, 2: Male)')
    plt.ylabel('Count')
    return get_image()

def plot_height_distribution():
    plt.figure(figsize=(10, 6))
    sns.histplot(data['height'], kde=True)
    plt.title('Height Distribution')
    plt.xlabel('Height (cm)')
    plt.ylabel('Frequency')
    return get_image()

def plot_weight_distribution():
    plt.figure(figsize=(10, 6))
    sns.histplot(data['weight'], kde=True)
    plt.title('Weight Distribution')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Frequency')
    return get_image()

def plot_systolic_bp_distribution():
    plt.figure(figsize=(10, 6))
    sns.histplot(data['ap_hi'], kde=True)
    plt.title('Systolic Blood Pressure Distribution')
    plt.xlabel('Systolic Blood Pressure (mmHg)')
    plt.ylabel('Frequency')
    return get_image()

def plot_diastolic_bp_distribution():
    plt.figure(figsize=(10, 6))
    sns.histplot(data['ap_lo'], kde=True)
    plt.title('Diastolic Blood Pressure Distribution')
    plt.xlabel('Diastolic Blood Pressure (mmHg)')
    plt.ylabel('Frequency')
    return get_image()

def plot_cholesterol_distribution():
    plt.figure(figsize=(10, 6))
    sns.countplot(x='cholesterol', data=data)
    plt.title('Cholesterol Level Distribution')
    plt.xlabel('Cholesterol Level (1: Normal, 2: Above Normal, 3: Well Above Normal)')
    plt.ylabel('Count')
    return get_image()

def plot_glucose_distribution():
    plt.figure(figsize=(10, 6))
    sns.countplot(x='gluc', data=data)
    plt.title('Glucose Level Distribution')
    plt.xlabel('Glucose Level (1: Normal, 2: Above Normal, 3: Well Above Normal)')
    plt.ylabel('Count')
    return get_image()

def plot_smoking_status_distribution():
    plt.figure(figsize=(10, 6))
    sns.countplot(x='smoke', data=data)
    plt.title('Smoking Status Distribution')
    plt.xlabel('Smoking Status (0: Non-smoker, 1: Smoker)')
    plt.ylabel('Count')
    return get_image()

def plot_alcohol_consumption_distribution():
    plt.figure(figsize=(10, 6))
    sns.countplot(x='alco', data=data)
    plt.title('Alcohol Consumption Distribution')
    plt.xlabel('Alcohol Consumption (0: Non-drinker, 1: Drinker)')
    plt.ylabel('Count')
    return get_image()

def plot_physical_activity_distribution():
    plt.figure(figsize=(10, 6))
    sns.countplot(x='active', data=data)
    plt.title('Physical Activity Distribution')
    plt.xlabel('Physical Activity (0: No, 1: Yes)')
    plt.ylabel('Count')
    return get_image()

def plot_cardiovascular_disease_distribution():
    plt.figure(figsize=(10, 6))
    sns.countplot(x='cardio', data=data)
    plt.title('Cardiovascular Disease Distribution')
    plt.xlabel('Cardiovascular Disease (0: No, 1: Yes)')
    plt.ylabel('Count')
    return get_image()

def plot_correlation_matrix():
    plt.figure(figsize=(15, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    return get_image()
