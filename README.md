# 🛋️ Furniture Sales Predictor

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An e-commerce machine learning project that predicts furniture sales based on product attributes using **Linear Regression** and **Random Forest** models.

## 📖 Project Overview

The **Furniture Sales Predictor** is a machine learning-powered web application designed to forecast the number of units sold for furniture products on an e-commerce platform. By analyzing features such as product title, price, original price, and shipping information, this project provides actionable insights for e-commerce businesses to optimize pricing and inventory strategies.

The model leverages **Linear Regression** and **Random Forest Regressor** algorithms to deliver accurate predictions, with a focus on interpretability and practical application.

## ✨ Features

- **🔍 Sales Prediction**: Predicts furniture sales based on product attributes like price, discount, and title characteristics.
- **📊 Feature Engineering**: Incorporates features such as:
  - Price and Original Price
  - Discount Percentage
  - Title Length and Word Count
  - Presence of keywords (e.g., patio, outdoor, bedroom, modern, wood)
  - Free Shipping Status and Installation Options
- **📈 Model Evaluation**: Displays performance metrics including Mean Squared Error (MSE) and R² Score.
- **🛠️ Easy-to-Use Pipeline**: Includes data preprocessing, model training, and prediction scripts.
- **🔎 Feature Importance**: Visualizes key factors influencing sales predictions.

## 💻 Prerequisites

Before running the project, ensure you have the following:

- **Python**: Version 3.6 or higher
- **Dataset**: `ecommerce_furniture_dataset_2024.csv` (place in the project root directory)
- **Dependencies**: Listed in `requirements.txt`
- **Hardware**: Standard CPU (GPU optional for faster training)

## 🛠️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Furniture_Sales_Predictor.git
   cd Furniture_Sales_Predictor
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**:
   Ensure the `ecommerce_furniture_dataset_2024.csv` file is in the project root directory.

## 🚀 Usage

Run the main script to train the model and view predictions:

```bash
python furniture_sales_predictor.py
```

The script will:
- Load and preprocess the dataset
- Train a **Random Forest Regressor** model
- Display performance metrics (MSE and R² Score)
- Show feature importance rankings
- Provide an example prediction

### Making Predictions

Use the `predict_sales()` function to forecast sales for new products:

```python
predicted_sales = predict_sales(
    product_title="Modern Patio Furniture Set with Cushions",
    price=299.99,
    original_price=399.99,
    tag_text="Free shipping"
)
print(f"Predicted sales: {predicted_sales:.2f} units")
```

## 🧠 Model Details

- **Algorithm**: Random Forest Regressor
- **Parameters**:
  - Number of trees: 100
  - Random state: 42 (for reproducibility)
- **Preprocessing**:
  - Features scaled using `StandardScaler`
  - Train/Test split: 80/20
- **Performance Metrics**:
  - Mean Squared Error (MSE)
  - R-squared (R²) Score
- **Features**:
  - Price, Original Price, Discount Percentage
  - Title Length, Word Count
  - Keywords: patio, outdoor, bedroom, living room, office, modern, wood, metal, fabric
  - Free Shipping Status, Installation

## 🛠️ Technologies Used

- **Programming Language**: Python (v3.6+)
- **Libraries**:
  - **scikit-learn**: For implementing Linear Regression, Random Forest Regressor, and StandardScaler
  - **pandas**: For data manipulation and preprocessing
  - **numpy**: For numerical computations
  - **matplotlib** and **seaborn** (optional): For visualizing feature importance and model performance
  - Additional dependencies listed in `requirements.txt`

## 📂 Project Structure

```
Furniture_Sales_Predictor/
├── ecommerce_furniture_dataset_2024.csv
├── furniture_sales_predictor.py
├── requirements.txt
└── README.md
```

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request

Please ensure your code follows PEP 8 standards and includes relevant documentation.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**#MachineLearning #Python #Ecommerce #DataScience #RandomForest #LinearRegression #SalesPrediction #OpenSource**
