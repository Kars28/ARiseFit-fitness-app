import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random

# Generate synthetic blood test data
def generate_blood_data(n_samples=2000):
    np.random.seed(42)
    
    # Generate blood parameters with realistic ranges
    data = {
        'Fasting_Blood_Sugar': np.random.normal(90, 15, n_samples),  # mg/dL
        'Post_Prandial_Blood_Sugar': np.random.normal(120, 20, n_samples),  # mg/dL
        'Thyroxine': np.random.normal(1.5, 0.3, n_samples),  # ng/dL
        'Cholesterol': np.random.normal(180, 30, n_samples),  # mg/dL
        'LDL_Cholesterol': np.random.normal(90, 20, n_samples),  # mg/dL
        'HDL_Cholesterol': np.random.normal(50, 10, n_samples),  # mg/dL
    }
    
    return pd.DataFrame(data)

# Indian diet recommendations based on health conditions
def get_diet_recommendations(blood_data):
    recommendations = []
    
    # Common Indian foods and their health benefits
    indian_foods = {
        'diabetes': [
            'Bitter gourd (Karela) curry',
            'Fenugreek (Methi) leaves',
            'Whole grain roti',
            'Moong dal',
            'Curd with flaxseeds',
            'Sprouted salads',
            'Green leafy vegetables',
            'Cinnamon tea',
            'Jamun (Indian blackberry)',
            'Amla (Indian gooseberry)'
        ],
        'cholesterol': [
            'Oats porridge',
            'Green tea',
            'Garlic in meals',
            'Turmeric milk',
            'Flaxseed chutney',
            'Walnuts',
            'Almonds',
            'Olive oil cooking',
            'Green leafy vegetables',
            'Whole grains'
        ],
        'thyroid': [
            'Coconut oil',
            'Seafood',
            'Dairy products',
            'Nuts and seeds',
            'Whole grains',
            'Fresh fruits',
            'Green vegetables',
            'Lentils',
            'Eggs',
            'Berries'
        ],
        'general_health': [
            'Khichdi with vegetables',
            'Daliya (broken wheat)',
            'Sprouts salad',
            'Buttermilk',
            'Fresh fruits',
            'Nuts and seeds',
            'Green vegetables',
            'Whole grain roti',
            'Lentil soup',
            'Herbal teas'
        ]
    }
    
    for _, row in blood_data.iterrows():
        recommendation = {
            'breakfast': [],
            'lunch': [],
            'dinner': [],
            'snacks': []
        }
        
        # Diabetes management
        if row['Fasting_Blood_Sugar'] > 100 or row['Post_Prandial_Blood_Sugar'] > 140:
            recommendation['breakfast'].extend(random.sample(indian_foods['diabetes'], 2))
            recommendation['lunch'].extend(random.sample(indian_foods['diabetes'], 2))
            recommendation['dinner'].extend(random.sample(indian_foods['diabetes'], 2))
            recommendation['snacks'].extend(random.sample(indian_foods['diabetes'], 1))
        
        # Cholesterol management
        if row['Cholesterol'] > 200 or row['LDL_Cholesterol'] > 130:
            recommendation['breakfast'].extend(random.sample(indian_foods['cholesterol'], 2))
            recommendation['lunch'].extend(random.sample(indian_foods['cholesterol'], 2))
            recommendation['dinner'].extend(random.sample(indian_foods['cholesterol'], 2))
            recommendation['snacks'].extend(random.sample(indian_foods['cholesterol'], 1))
        
        # Thyroid management
        if row['Thyroxine'] < 0.9 or row['Thyroxine'] > 2.3:
            recommendation['breakfast'].extend(random.sample(indian_foods['thyroid'], 2))
            recommendation['lunch'].extend(random.sample(indian_foods['thyroid'], 2))
            recommendation['dinner'].extend(random.sample(indian_foods['thyroid'], 2))
            recommendation['snacks'].extend(random.sample(indian_foods['thyroid'], 1))
        
        # If no specific condition, recommend general healthy diet
        if not any([
            row['Fasting_Blood_Sugar'] > 100,
            row['Post_Prandial_Blood_Sugar'] > 140,
            row['Cholesterol'] > 200,
            row['LDL_Cholesterol'] > 130,
            row['Thyroxine'] < 0.9,
            row['Thyroxine'] > 2.3
        ]):
            recommendation['breakfast'].extend(random.sample(indian_foods['general_health'], 2))
            recommendation['lunch'].extend(random.sample(indian_foods['general_health'], 2))
            recommendation['dinner'].extend(random.sample(indian_foods['general_health'], 2))
            recommendation['snacks'].extend(random.sample(indian_foods['general_health'], 1))
        
        recommendations.append(recommendation)
    
    return recommendations

# Generate and save the dataset
def generate_dataset(n_samples=2000):
    blood_data = generate_blood_data(n_samples)
    recommendations = get_diet_recommendations(blood_data)
    
    # Combine blood data with recommendations
    dataset = pd.DataFrame(blood_data)
    dataset['Diet_Recommendations'] = recommendations
    
    # Save the dataset
    dataset.to_csv('indian_diet_dataset.csv', index=False)
    return dataset

if __name__ == '__main__':
    dataset = generate_dataset()
    print("Dataset generated and saved successfully!") 