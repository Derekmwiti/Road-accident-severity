import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore
import joblib # type: ignore

def create_sample_data():
    length = 100  # Ensure all arrays have the same length
    data = {
        'accident_severity': np.random.randint(1, 5, length),
        'speed': np.random.randint(20, 100, length),
        'weather_condition': np.random.choice(['clear', 'snowy', 'sunny', 'drizling'], length),
        'road_condition': np.random.choice(['rocky', 'muddy', 'icy'], length),
        'vehicle_condition': np.random.choice(['good', 'fair', 'poor'], length),
        'time_of_day': np.random.choice(['morning', 'midday', 'evening', 'night'], length)
    }
    return pd.DataFrame(data)

def preprocess_data(df):
    X = df.drop('accident_severity', axis=1)
    y = df['accident_severity']
    return X, y

def build_model():
    numeric_features = ['speed']
    categorical_features = ['weather_condition', 'road_condition', 'vehicle_condition', 'time_of_day']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'\nMean Squared Error: {mse}')
    print(f'R-squared: {r2}')

def save_model(model, filename='road_accident_severity_model.pkl'):
    joblib.dump(model, filename)
    print(f"\nModel saved as '{filename}'")

def predict_severity(model, data):
    predicted_severity = model.predict(data)
    print(f'\nPredicted Accident Severity for hypothetical data: {predicted_severity[0]}')

if __name__ == "__main__":
    # Create a sample dataset
    df = create_sample_data()
    print("Sample DataFrame:")
    print(df.head())
    
    # Preprocess the data
    X, y = preprocess_data(df)
    
    # Build and train the model
    model = build_model()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(model, X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Save the model
    save_model(model)
    
    # Example prediction
    hypothetical_data = pd.DataFrame({
        'speed': [80],
        'weather_condition': ['sunny'],
        'road_condition': ['rocky'],
        'vehicle_condition': ['fair'],
        'time_of_day': ['evening']
    })
    
    predict_severity(model, hypothetical_data)
