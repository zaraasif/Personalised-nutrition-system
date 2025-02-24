import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_data(file_path):
  
    data = pd.read_csv(file_path)
    drop_columns = ['index', 'NDB_No', 'Ash_(g)', 'GmWt_Desc1', 'GmWt_Desc2', 'Refuse_Pct', 
                    'Choline_Tot_ (mg)', 'Lycopene_(µg)', 'Lut+Zea_ (µg)']
    data_cleaned = data.drop(columns=drop_columns)
    data_cleaned.fillna(data_cleaned.median(numeric_only=True), inplace=True)
    
    features = data_cleaned.select_dtypes(include=['float64', 'int64']).drop(columns=['Energ_Kcal'])
    target = data_cleaned['Energ_Kcal']

    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)
    
    return data_cleaned, features_normalized, target, scaler
