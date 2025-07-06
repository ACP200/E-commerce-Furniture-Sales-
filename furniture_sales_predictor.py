import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from textblob import TextBlob
import re

# Ensure TextBlob corpora are downloaded
try:
    _ = TextBlob("test").sentiment
except Exception:
    from textblob import download_corpora
    download_corpora()
    print("Downloaded TextBlob corpora.")

def preprocess_data(df):
    # Convert price and originalPrice to numeric, removing $ and , characters
    df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
    df['originalPrice'] = df['originalPrice'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Fill missing values with median using proper assignment
    df.loc[:, 'price'] = df['price'].fillna(df['price'].median())
    df.loc[:, 'originalPrice'] = df['originalPrice'].fillna(df['originalPrice'].median())
    
    # Fill NaNs in text columns with empty strings
    df['productTitle'] = df['productTitle'].fillna("")
    df['tagText'] = df['tagText'].fillna("")
    
    # Create price-related features
    df['price_diff'] = df['originalPrice'] - df['price']
    df['discount_percentage'] = (df['price_diff'] / df['originalPrice']) * 100
    df['price_bin'] = pd.qcut(df['price'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    # Create text-based features
    df['title_length'] = df['productTitle'].str.len()
    df['description_length'] = df['tagText'].str.len()
    
    # Sentiment analysis
    df['title_sentiment'] = df['productTitle'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['description_sentiment'] = df['tagText'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # Keyword features
    keywords = ['wood', 'metal', 'plastic', 'fabric', 'leather', 'modern', 'vintage', 'classic', 
                'contemporary', 'rustic', 'industrial', 'minimalist', 'luxury', 'eco-friendly', 
                'ergonomic', 'adjustable', 'foldable', 'portable', 'storage', 'multifunctional']
    
    for keyword in keywords:
        df[f'has_{keyword}'] = df['productTitle'].str.lower().str.contains(keyword).astype(int)
        df[f'desc_{keyword}'] = df['tagText'].str.lower().str.contains(keyword).astype(int)
    
    # Create interaction features
    df['price_sentiment'] = df['price'] * df['title_sentiment']
    df['price_description_sentiment'] = df['price'] * df['description_sentiment']
    
    # Select features for model
    feature_columns = ['price', 'originalPrice', 'price_diff', 'discount_percentage', 
                      'title_length', 'description_length', 'title_sentiment', 'description_sentiment',
                      'price_sentiment', 'price_description_sentiment']
    
    # Add keyword features
    feature_columns.extend([f'has_{k}' for k in keywords])
    feature_columns.extend([f'desc_{k}' for k in keywords])
    
    # Prepare X and y
    X = df[feature_columns]
    y = df['sold']
    
    # Remove outliers using IQR method
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    mask = (y >= Q1 - 1.5 * IQR) & (y <= Q3 + 1.5 * IQR)
    X = X[mask]
    y = y[mask]
    
    # Apply log transformation to target variable
    y = np.log1p(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_columns 

def train_models(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        results[name] = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    return results, X_test, y_test

def predict_sales(model_results, X_test, y_test, scaler, feature_columns):
    # Example product features
    example_product = pd.DataFrame({
        'price': [199.99],
        'originalPrice': [299.99],
        'title_length': [50],
        'description_length': [200],
        'title_sentiment': [0.5],
        'description_sentiment': [0.3]
    })
    
    # Calculate derived features
    example_product['price_diff'] = example_product['originalPrice'] - example_product['price']
    example_product['discount_percentage'] = (example_product['price_diff'] / example_product['originalPrice']) * 100
    example_product['price_sentiment'] = example_product['price'] * example_product['title_sentiment']
    example_product['price_description_sentiment'] = example_product['price'] * example_product['description_sentiment']
    
    # Add keyword features (all set to 0 for example)
    for keyword in ['wood', 'metal', 'plastic', 'fabric', 'leather', 'modern', 'vintage', 'classic', 
                   'contemporary', 'rustic', 'industrial', 'minimalist', 'luxury', 'eco-friendly', 
                   'ergonomic', 'adjustable', 'foldable', 'portable', 'storage', 'multifunctional']:
        example_product[f'has_{keyword}'] = 0
        example_product[f'desc_{keyword}'] = 0
    
    # Scale features
    example_scaled = scaler.transform(example_product[feature_columns])
    
    # Make predictions
    predictions = {}
    for name, result in model_results.items():
        pred = result['model'].predict(example_scaled)
        # Convert back from log scale
        predictions[name] = np.expm1(pred)[0]
    
    return predictions

def main():
    try:
        # Load the dataset
        df = pd.read_csv('ecommerce_furniture_dataset_2024.csv')
        
        # Preprocess data
        X, y, scaler, feature_columns = preprocess_data(df)
        
        # Train models
        model_results, X_test, y_test = train_models(X, y)
        
        # Print model performance
        print("\nModel Performance:")
        print("-" * 50)
        for name, result in model_results.items(): 
            print(f"\n{name}:")
            print(f"Mean Squared Error: {result['mse']:.2f}")
            print(f"R² Score: {result['r2']:.2f}")
            print(f"Cross-validation R² Score: {result['cv_mean']:.2f} (+/- {result['cv_std']:.2f})")
        
        # Make predictions for example product
        predictions = predict_sales(model_results, X_test, y_test, scaler, feature_columns)
        
        print("\nPredicted Sales for Example Product:")
        print("-" * 50)
        for model_name, pred in predictions.items():
            print(f"{model_name}: {pred:.0f} units")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 