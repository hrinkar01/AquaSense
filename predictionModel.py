import numpy as np
import pandas as pd
import tensorflow as tf # type: ignore
from sklearn.preprocessing import MinMaxScaler, StandardScaler # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

def classify_water_quality(ph, turbidity, tds, latitude, longitude):
    """
    Classify water quality based on multiple parameters
    
    Args:
    ph (float): pH level
    turbidity (float): Water turbidity
    tds (float): Total Dissolved Solids
    latitude (float): Latitude coordinate
    longitude (float): Longitude coordinate
    
    Returns:
    str: Water quality classification
    """
    # Drinking Water Criteria (based on WHO guidelines)
    def is_drinkable():
        # Strict drinking water conditions
        if (6.5 <= ph <= 8.5 and  # pH range
            turbidity < 5 and      # NTU (Nephelometric Turbidity Units)
            tds < 500):            # ppm (parts per million)
            return True
        return False
    
    # Agriculture Water Criteria
    def is_agricultural():
        # Slightly less strict conditions
        if (6 <= ph <= 8 and      # Broader pH range for crops
            turbidity < 10 and    # Higher turbidity tolerance
            tds < 2000):          # Higher TDS tolerance
            return True
        return False
    
    # Industrial Use Criteria
    def is_industrial():
        # Industrial processes have varying water quality needs
        if (5.5 <= ph <= 9 and    # Very wide pH range
            turbidity < 20 and    # Can tolerate higher turbidity
            tds < 3000):          # Higher TDS tolerance
            return True
        return False
    
    # Classify water quality
    if is_drinkable():
        return "Drinkable"
    elif is_industrial():
        return "Industrial Use"
    elif is_agricultural():
        return "Agricultural Use"
    else:
        return "Unusable (Requires Treatment)"

def prepare_classification_model(data):
    """
    Prepare and train a water quality classification model
    
    Args:
    data (pd.DataFrame): Input water quality data
    
    Returns:
    tuple: Trained model, scalers, and classification mapping
    """
    # Add classification column
    data['category'] = data.apply(
        lambda row: classify_water_quality(
            row['ph'], row['turbidity'], row['tds'], 
            row['latitude'], row['longitude']
        ), 
        axis=1
    )
    
    # Create label encoding
    category_mapping = {
        'Drinkable': 0,
        'Agricultural Use': 1,
        'Industrial Use': 2,
        'Unusable (Requires Treatment)': 3
    }
    data['category_encoded'] = data['category'].map(category_mapping)
    
    # Prepare features and labels
    X = data[['ph', 'turbidity', 'tds', 'latitude', 'longitude']]
    y = data['category_encoded']
    
    # Scale features
    scaler_features = StandardScaler()
    X_scaled = scaler_features.fit_transform(X)
    
    # One-hot encode labels
    y_encoded = to_categorical(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )
    
    # Create classification model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(y_encoded.shape[1], activation='softmax')
    ])
    
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # Train model
    model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    return model, scaler_features, category_mapping

def predict_water_quality_category(model, scaler, data, months_to_predict=3):
    """
    Predict water quality categories for future months
    
    Args:
    model (keras.Model): Trained classification model
    scaler (StandardScaler): Feature scaler
    data (pd.DataFrame): Historical water quality data
    months_to_predict (int): Number of months to predict
    
    Returns:
    pd.DataFrame: Predicted water quality categories
    """
    # Use the last available data for prediction
    last_data = data.iloc[-1]
    
    # Predict for future months
    predictions = []
    for month in range(months_to_predict):
        # Prepare input data
        input_data = np.array([
            [last_data['ph'], last_data['turbidity'], last_data['tds'], 
             last_data['latitude'], last_data['longitude']]
        ])
        
        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Predict category
        pred_prob = model.predict(input_scaled)
        pred_category_encoded = np.argmax(pred_prob, axis=1)[0]
        
        # Reverse mapping
        category_mapping_reverse = {
            0: 'Drinkable',
            1: 'Agricultural Use',
            2: 'Industrial Use',
            3: 'Unusable (Requires Treatment)'
        }
        pred_category = category_mapping_reverse[pred_category_encoded]
        
        # Print category for each month
        print(f"\n{pred_category}")
        
        predictions.append({
            'Category': pred_category,
            'PH': last_data['ph'],
            'Turbidity': last_data['turbidity'],
            'Tds': last_data['tds'],
            'Latitude': last_data['latitude'],
            'Longitude': last_data['longitude']
        })
    
    return pd.DataFrame(predictions)

def main():
    # Load water quality data
    csv_path = 'dummy_data.csv'
    
    # Load data
    water_data = pd.read_csv(csv_path)
    
    # Prepare classification model
    classification_model, scaler, category_mapping = prepare_classification_model(water_data)
    
    # Predict future water quality categories
    print("\n=== Water Quality Predictions ===")
    future_predictions = predict_water_quality_category(
        classification_model, 
        scaler, 
        water_data,
        months_to_predict=1
    )
    
    # Print full predictions dataframe
    print("\nDetailed Predictions:")
    print(future_predictions)
    

    # Save predictions
    future_predictions.to_csv('prediction_result.csv', index=False)

if __name__ == "__main__":
    main()