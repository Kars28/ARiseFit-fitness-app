import os
from flask import Flask, request, jsonify
import PyPDF2
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__)
CORS(app)

# Ensure the folder for uploaded files exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the Indian diet dataset
try:
    diet_dataset = pd.read_csv('indian_diet_dataset.csv')
except FileNotFoundError:
    print("Dataset not found. Please run indian_diet_dataset.py first.")
    diet_dataset = None

# Default average values for missing parameters
AVERAGE_VALUES = {
    "Fasting Blood Sugar": 90,  # mg/dL
    "Post Prandial Blood Sugar": 120,  # mg/dL
    "Thyroxine": 1.5,  # ng/dL
    "Cholesterol": 180,  # mg/dL (Total)
    "LDL Cholesterol": 90,  # mg/dL
    "HDL Cholesterol": 50  # mg/dL
}

def get_indian_diet_recommendations(blood_data):
    if diet_dataset is None:
        return {
            "error": "Diet dataset not loaded. Please contact administrator."
        }
    
    # Convert blood data to match dataset format
    user_data = {
        'Fasting_Blood_Sugar': float(blood_data.get("Fasting Blood Sugar", AVERAGE_VALUES["Fasting Blood Sugar"])),
        'Post_Prandial_Blood_Sugar': float(blood_data.get("Post Prandial Blood Sugar", AVERAGE_VALUES["Post Prandial Blood Sugar"])),
        'Thyroxine': float(blood_data.get("Thyroxine", AVERAGE_VALUES["Thyroxine"])),
        'Cholesterol': float(blood_data.get("Cholesterol", AVERAGE_VALUES["Cholesterol"])),
        'LDL_Cholesterol': float(blood_data.get("LDL Cholesterol", AVERAGE_VALUES["LDL Cholesterol"])),
        'HDL_Cholesterol': float(blood_data.get("HDL Cholesterol", AVERAGE_VALUES["HDL Cholesterol"]))
    }
    
    # Find the most similar case in the dataset
    features = ['Fasting_Blood_Sugar', 'Post_Prandial_Blood_Sugar', 'Thyroxine', 
                'Cholesterol', 'LDL_Cholesterol', 'HDL_Cholesterol']
    
    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(diet_dataset[features])
    user_scaled = scaler.transform([list(user_data.values())])
    
    # Use KMeans to find the closest cluster
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(scaled_data)
    user_cluster = kmeans.predict(user_scaled)[0]
    
    # Get recommendations from the same cluster
    cluster_indices = np.where(kmeans.labels_ == user_cluster)[0]
    similar_cases = diet_dataset.iloc[cluster_indices]
    
    # Get the most similar case
    distances = np.linalg.norm(scaled_data[cluster_indices] - user_scaled, axis=1)
    most_similar_idx = cluster_indices[np.argmin(distances)]
    
    return diet_dataset.iloc[most_similar_idx]['Diet_Recommendations']

def extract_data_from_report(text, report_type):
    extracted_data = {}
    
    if report_type == 'blood_sugar':
        fasting_key = "Blood Sugar Fasting"
        post_prandial_key = "Glucose - Post Prandial"
        
        if fasting_key in text:
            extracted_data["Fasting Blood Sugar"] = text.split(fasting_key)[1].split()[0]
        if post_prandial_key in text:
            extracted_data["Post Prandial Blood Sugar"] = text.split(post_prandial_key)[1].split()[0]
            
    elif report_type == 'cholesterol':
        cholesterol_key = "Cholesterol"
        ldl_key = "LDL Cholesterol"
        hdl_key = "HDL Cholesterol"
        
        if cholesterol_key in text:
            extracted_data["Cholesterol"] = text.split(cholesterol_key)[1].split()[0]
        if ldl_key in text:
            extracted_data["LDL Cholesterol"] = text.split(ldl_key)[1].split()[0]
        if hdl_key in text:
            extracted_data["HDL Cholesterol"] = text.split(hdl_key)[1].split()[0]
            
    elif report_type == 'thyroxine':
        thyroxine_key = "Thyroxine"
        if thyroxine_key in text:
            extracted_data["Thyroxine"] = text.split(thyroxine_key)[1].split()[0]
    
    return extracted_data

def extract_data_from_file(file_path, report_type):
    with open(file_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    return extract_data_from_report(text, report_type)

@app.route('/analyzereport', methods=['POST'])
def upload_files():
    if not all(key in request.files for key in ['blood_sugar', 'cholesterol', 'thyroxine']):
        return jsonify({"error": "Missing required reports"}), 400

    # Initialize combined data
    combined_data = {
        "Fasting Blood Sugar": None,
        "Post Prandial Blood Sugar": None,
        "Thyroxine": None,
        "Cholesterol": None,
        "LDL Cholesterol": None,
        "HDL Cholesterol": None
    }

    # Process each report
    for report_type, file in request.files.items():
        if file.filename == '':
            continue
            
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Extract data from the report
        extracted_data = extract_data_from_file(file_path, report_type)
        
        # Update combined data with extracted values
        for key, value in extracted_data.items():
            if value is not None:
                combined_data[key] = value

    # Replace any missing values with averages
    for key, value in combined_data.items():
        if value is None:
            combined_data[key] = AVERAGE_VALUES.get(key)

    # Get diet recommendations
    recommendations = get_indian_diet_recommendations(combined_data)
    
    return jsonify({
        "Results": {
            "extracted_data": combined_data,
            "diet_recommendations": recommendations
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
