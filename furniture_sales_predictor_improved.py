import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from textblob import TextBlob
import re
import warnings
warnings.filterwarnings('ignore')

# Ensure TextBlob corpora are downloaded
try:
    _ = TextBlob("test").sentiment
except Exception:
    from textblob import download_corpora
    download_corpora()
    print("Downloaded TextBlob corpora.")

def advanced_text_analysis(text):
    """Advanced text analysis with multiple features"""
    if pd.isna(text) or text == "":
        return {
            'length': 0, 'word_count': 0, 'avg_word_length': 0,
            'sentiment': 0, 'subjectivity': 0, 'exclamation_count': 0,
            'question_count': 0, 'uppercase_ratio': 0, 'digit_count': 0,
            'special_char_count': 0
        }
    
    text = str(text)
    blob = TextBlob(text)
    
    return {
        'length': len(text),
        'word_count': len(text.split()),
        'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
        'sentiment': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        'digit_count': sum(1 for c in text if c.isdigit()),
        'special_char_count': len(re.findall(r'[^a-zA-Z0-9\s]', text))
    }

def extract_price_features(df):
    """Extract comprehensive price-related features"""
    # Clean and convert prices
    df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
    df['originalPrice'] = df['originalPrice'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Fill missing values
    df['price'] = df['price'].fillna(df['price'].median())
    df['originalPrice'] = df['originalPrice'].fillna(df['price'])
    
    # Price features
    df['price_diff'] = df['originalPrice'] - df['price']
    df['discount_percentage'] = (df['price_diff'] / df['originalPrice']) * 100
    df['price_ratio'] = df['price'] / df['originalPrice']
    
    # Price bins
    df['price_bin'] = pd.qcut(df['price'], q=10, labels=False, duplicates='drop')
    df['original_price_bin'] = pd.qcut(df['originalPrice'], q=10, labels=False, duplicates='drop')
    
    # Price statistics
    df['price_percentile'] = df['price'].rank(pct=True)
    df['original_price_percentile'] = df['originalPrice'].rank(pct=True)
    
    return df

def extract_keyword_features(df):
    """Extract comprehensive keyword features"""
    # Furniture categories
    categories = {
        'patio': ['patio', 'outdoor', 'garden', 'balcony', 'poolside', 'backyard', 'terrace'],
        'bedroom': ['bedroom', 'bed', 'nightstand', 'dresser', 'wardrobe', 'closet'],
        'living_room': ['living room', 'sofa', 'couch', 'loveseat', 'sectional', 'accent chair'],
        'office': ['office', 'desk', 'computer', 'gaming', 'workstation'],
        'dining': ['dining', 'table', 'chair', 'kitchen'],
        'storage': ['storage', 'drawer', 'cabinet', 'shelf', 'organizer'],
        'modern': ['modern', 'contemporary', 'minimalist', 'scandinavian'],
        'traditional': ['traditional', 'classic', 'vintage', 'antique'],
        'material_wood': ['wood', 'wooden', 'oak', 'walnut', 'mahogany', 'pine'],
        'material_metal': ['metal', 'steel', 'aluminum', 'iron'],
        'material_fabric': ['fabric', 'fabric', 'velvet', 'linen', 'chenille'],
        'material_leather': ['leather', 'faux leather', 'pu leather'],
        'material_plastic': ['plastic', 'acrylic', 'polyester'],
        'material_rattan': ['rattan', 'wicker', 'bamboo'],
        'features': ['adjustable', 'foldable', 'portable', 'ergonomic', 'multifunctional'],
        'quality': ['luxury', 'premium', 'high-quality', 'sturdy', 'durable'],
        'shipping': ['free shipping', 'shipping', 'delivery'],
        'assembly': ['assembly', 'easy assembly', 'no assembly', 'pre-assembled']
    }
    
    for category, keywords in categories.items():
        df[f'has_{category}'] = df['productTitle'].str.lower().str.contains('|'.join(keywords)).astype(int)
        df[f'desc_{category}'] = df['tagText'].str.lower().str.contains('|'.join(keywords)).astype(int)
    
    return df

def extract_advanced_features(df):
    """Extract advanced features including text analysis"""
    # Text analysis for product title
    title_features = df['productTitle'].apply(advanced_text_analysis)
    title_df = pd.DataFrame(title_features.tolist(), index=df.index)
    title_df.columns = [f'title_{col}' for col in title_df.columns]
    
    # Text analysis for tag text
    tag_features = df['tagText'].apply(advanced_text_analysis)
    tag_df = pd.DataFrame(tag_features.tolist(), index=df.index)
    tag_df.columns = [f'tag_{col}' for col in tag_df.columns]
    
    # Combine with original dataframe
    df = pd.concat([df, title_df, tag_df], axis=1)
    
    # Interaction features
    df['price_title_sentiment'] = df['price'] * df['title_sentiment']
    df['price_tag_sentiment'] = df['price'] * df['tag_sentiment']
    df['discount_title_sentiment'] = df['discount_percentage'] * df['title_sentiment']
    
    # Complexity features
    df['title_complexity'] = df['title_word_count'] * df['title_avg_word_length']
    df['tag_complexity'] = df['tag_word_count'] * df['tag_avg_word_length']
    
    return df

def remove_outliers(df, target_col, method='iqr'):
    """Remove outliers using different methods"""
    if method == 'iqr':
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = (df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
        mask = z_scores < 3
    else:
        mask = pd.Series([True] * len(df), index=df.index)
    
    return df[mask]

def preprocess_data_improved(df):
    """Improved data preprocessing with advanced feature engineering"""
    print("Starting advanced data preprocessing...")
    
    # Basic cleaning
    df['productTitle'] = df['productTitle'].fillna("")
    df['tagText'] = df['tagText'].fillna("")
    
    # Extract price features
    df = extract_price_features(df)
    
    # Extract keyword features
    df = extract_keyword_features(df)
    
    # Extract advanced features
    df = extract_advanced_features(df)
    
    # Remove outliers from target variable
    print(f"Original dataset size: {len(df)}")
    df_clean = remove_outliers(df, 'sold', method='iqr')
    print(f"After outlier removal: {len(df_clean)}")
    
    # Select features for modeling
    feature_columns = [
        # Price features
        'price', 'originalPrice', 'price_diff', 'discount_percentage', 'price_ratio',
        'price_bin', 'original_price_bin', 'price_percentile', 'original_price_percentile',
        
        # Text features
        'title_length', 'title_word_count', 'title_avg_word_length', 'title_sentiment', 'title_subjectivity',
        'title_exclamation_count', 'title_question_count', 'title_uppercase_ratio', 'title_digit_count',
        'title_special_char_count', 'title_complexity',
        
        'tag_length', 'tag_word_count', 'tag_avg_word_length', 'tag_sentiment', 'tag_subjectivity',
        'tag_exclamation_count', 'tag_question_count', 'tag_uppercase_ratio', 'tag_digit_count',
        'tag_special_char_count', 'tag_complexity',
        
        # Interaction features
        'price_title_sentiment', 'price_tag_sentiment', 'discount_title_sentiment',
        
        # Keyword features
        'has_patio', 'has_bedroom', 'has_living_room', 'has_office', 'has_dining', 'has_storage',
        'has_modern', 'has_traditional', 'has_material_wood', 'has_material_metal', 'has_material_fabric',
        'has_material_leather', 'has_material_plastic', 'has_material_rattan', 'has_features',
        'has_quality', 'has_shipping', 'has_assembly',
        
        'desc_patio', 'desc_bedroom', 'desc_living_room', 'desc_office', 'desc_dining', 'desc_storage',
        'desc_modern', 'desc_traditional', 'desc_material_wood', 'desc_material_metal', 'desc_material_fabric',
        'desc_material_leather', 'desc_material_plastic', 'desc_material_rattan', 'desc_features',
        'desc_quality', 'desc_shipping', 'desc_assembly'
    ]
    
    # Filter columns that exist in the dataframe
    available_features = [col for col in feature_columns if col in df_clean.columns]
    
    X = df_clean[available_features]
    y = df_clean['sold']
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Apply log transformation to target variable
    y = np.log1p(y)
    
    # Scale features using RobustScaler (more robust to outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Final feature matrix shape: {X_scaled.shape}")
    print(f"Features used: {len(available_features)}")
    
    return X_scaled, y, scaler, available_features

def train_models_improved(X, y):
    """Train multiple models with hyperparameter tuning"""
    print("\nTraining improved models with hyperparameter tuning...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
    
    # Define models with hyperparameter grids
    models_config = {
        'Ridge Regression': {
            'model': Ridge(),
            'params': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
            }
        },
        'Lasso Regression': {
            'model': Lasso(),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'max_iter': [1000, 2000]
            }
        },
        'Elastic Net': {
            'model': ElasticNet(),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'max_iter': [1000, 2000]
            }
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'Extra Trees': {
            'model': ExtraTreesRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        }
    }
    
    results = {}
    
    for name, config in models_config.items():
        print(f"\nTraining {name}...")
        
        # Use RandomizedSearchCV for faster tuning
        search = RandomizedSearchCV(
            config['model'],
            config['params'],
            n_iter=20,  # Number of parameter settings sampled
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        search.fit(X_train, y_train)
        
        # Get best model
        best_model = search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
        
        results[name] = {
            'model': best_model,
            'best_params': search.best_params_,
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred
        }
        
        print(f"Best parameters: {search.best_params_}")
        print(f"R² Score: {r2:.4f}")
        print(f"Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return results, X_test, y_test

def ensemble_predictions(results, X_test, y_test, scaler, feature_columns):
    """Create ensemble predictions"""
    print("\nCreating ensemble predictions...")
    
    # Get predictions from all models
    predictions = {}
    for name, result in results.items():
        predictions[name] = result['y_pred']
    
    # Create ensemble (simple average)
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    
    # Calculate ensemble metrics
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
    print(f"Ensemble Performance:")
    print(f"MSE: {ensemble_mse:.4f}")
    print(f"R² Score: {ensemble_r2:.4f}")
    print(f"MAE: {ensemble_mae:.4f}")
    
    return ensemble_pred, ensemble_r2

def main():
    try:
        print("🚀 Starting Improved Furniture Sales Predictor")
        print("=" * 50)
        
        # Load the dataset
        print("Loading dataset...")
        df = pd.read_csv('ecommerce_furniture_dataset_2024.csv')
        print(f"Dataset loaded: {len(df)} records")
        
        # Preprocess data with advanced features
        X, y, scaler, feature_columns = preprocess_data_improved(df)
        
        # Train models with hyperparameter tuning
        model_results, X_test, y_test = train_models_improved(X, y)
        
        # Print comprehensive model performance
        print("\n" + "=" * 50)
        print("FINAL MODEL PERFORMANCE")
        print("=" * 50)
        
        for name, result in model_results.items():
            print(f"\n{name}:")
            print(f"  Mean Squared Error: {result['mse']:.4f}")
            print(f"  R² Score: {result['r2']:.4f}")
            print(f"  Mean Absolute Error: {result['mae']:.4f}")
            print(f"  Cross-validation R²: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
            print(f"  Best Parameters: {result['best_params']}")
        
        # Create ensemble predictions
        ensemble_pred, ensemble_r2 = ensemble_predictions(model_results, X_test, y_test, scaler, feature_columns)
        
        # Find best model
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['r2'])
        best_r2 = model_results[best_model_name]['r2']
        
        print(f"\n" + "=" * 50)
        print(f"BEST MODEL: {best_model_name}")
        print(f"BEST R² SCORE: {best_r2:.4f}")
        print(f"ENSEMBLE R² SCORE: {ensemble_r2:.4f}")
        print("=" * 50)
        
        if best_r2 > 0.3:
            print("✅ Significant improvement achieved!")
        elif best_r2 > 0.2:
            print("🔄 Moderate improvement achieved - further tuning recommended")
        else:
            print("⚠️  Limited improvement - consider additional feature engineering")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 