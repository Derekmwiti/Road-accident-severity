# Road Accident Severity Analysis

This project demonstrates the use of a linear regression model to analyze and predict road accident severity based on various factors. The model is trained on a sample dataset and can be used to make predictions for hypothetical scenarios.

# Table of Contents

- Installation
- Usage
- Model Details

# Installation

To get started with this project, follow the steps below:

1. Clone the repository:
    
    git clone https://github.com/Derekmwiti/Road-accident-severity.git
    
2. Navigate to the project directory:
   
    cd road-accident-severity-analysis
 
3. Install the required dependencies:
    
    pip install -r requirements.txt
   
# Usage

Run the script to train the model and make predictions:

python3 severity-analysis.py

The script will:
1. Create a sample dataset.
2. Train a linear regression model.
3. Evaluate the model.
4. Save the trained model.
5. Provide an example prediction.

# Model Details

The model predicts road accident severity based on the following features:
- Speed
- Weather condition
- Road condition
- Vehicle condition
- Time of day

The `python3 severity-analysis.py` performs the following steps:
1. Creates a sample dataset.
2. Preprocesses the data.
3. Builds a linear regression model using `sklearn`.
4. Trains and evaluates the model.
5. Saves the model for future use.
6. Provides an example prediction for a hypothetical scenario.

## Example Prediction

The script includes an example prediction for a hypothetical set of independent variables:

hypothetical_data = pd.DataFrame({
    'speed': [70],
    'weather_condition': ['rainy'],
    'road_condition': ['wet'],
    'vehicle_condition': ['fair'],
    'time_of_day': ['evening']
})

predicted_severity = model.predict(hypothetical_data)
print(f'\nPredicted Accident Severity for hypothetical data: {predicted_severity[0]}')
