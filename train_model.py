import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# 1. Cleaned file check karein
file_path = 'data_cleaned.csv'

if not os.path.exists(file_path):
    print(f"Error: '{file_path}' nahi mili! Pehle 'clean_data.py' chalayen.")
else:
    print("1. Data load ho raha hai...")
    df = pd.read_csv(file_path, header=None)

    # X mein landmarks aur y mein labels
    X = df.iloc[:, 1:] 
    y = df.iloc[:, 0]

    print(f"2. Model training shuru... (Total Gestures: {len(y.unique())})")
    
    # Updated Model: 42 signs ke liye zyada trees (300) aur depth set ki hai
    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
    model.fit(X, y)

    # Model save karna
    with open('model.p', 'wb') as f:
        pickle.dump(model, f)

    print("3. Mubarak ho! Naya 'model.p' taiyar hai.")