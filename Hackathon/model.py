import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# 1. Load the dataset (Make sure paimana_dataset.csv is in your directory)
# For infrastructure: features might be target cost, duration variance, sector indicators.
# For diabetes/health: features might be Glucose, BMI, Age, etc.
try:
    df = pd.read_csv("paimana_dataset.csv")
    print("Dataset successfully loaded!")
except FileNotFoundError:
    print("Error: 'paimana_dataset.csv' not found. Creating a synthetic sample mock for testing...")
    # Mock fallback to prevent script failures if dataset path isn't finalized
    np.random.seed(42)
    mock_data = {
        'feature_1': np.random.rand(500) * 100,
        'feature_2': np.random.rand(500) * 10,
        'feature_3': np.random.randint(20, 70, size=500),
        'continuous_risk_score': np.random.rand(500) * 100, # Continuous target
    }
    df = pd.DataFrame(mock_data)
    # Binary classification target derived from risk threshold
    df['high_risk_class'] = (df['continuous_risk_score'] > 50).astype(int)

# 2. Define Features (X) and Target variables (y)
# Adjust these column names based on your precise CSV structure
feature_cols = [col for col in df.columns if col not in ['continuous_risk_score', 'high_risk_class']]
X = df[feature_cols]
y_reg = df['continuous_risk_score']  # For Linear Regression
y_clf = df['high_risk_class']        # For Classification

# 3. Handle data splitting
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# 4. Feature Scaling (Crucial for reliable performance in linear/logistic models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# PART A: LINEAR REGRESSION (Continuous Risk)
# ==========================================
print("\n--- Training Linear Regression Model ---")
reg_model = LinearRegression()
reg_model.fit(X_train_scaled, y_train_reg)

# Predict & Evaluate
y_pred_reg = reg_model.predict(X_test_scaled)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# ==========================================
# PART B: CLASSIFICATION (Binary Risk Profile)
# ==========================================
print("\n--- Training Classification Model (Logistic Regression) ---")
clf_model = LogisticRegression(random_state=42)
clf_model.fit(X_train_scaled, y_train_clf)

# Predict & Evaluate
y_pred_clf = clf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test_clf, y_pred_clf))

# ==========================================
# PART C: BACKEND PREDICTION ENGINE
# ==========================================
def predict_risk(new_data_features):
    """
    Accepts raw raw feature values array/list, scales it, and fires back both predictions.
    """
    data_df = pd.DataFrame([new_data_features], columns=feature_cols)
    scaled_data = scaler.transform(data_df)
    
    predicted_score = reg_model.predict(scaled_data)[0]
    predicted_class = clf_model.predict(scaled_data)[0]
    
    risk_label = "High Risk" if predicted_class == 1 else "Low Risk"
    
    return {
        "continuous_risk_forecast": round(float(predicted_score), 2),
        "binary_risk_classification": risk_label
    }

# Mock backend API execution check
example_features = X.iloc[0].tolist()
print("\n--- Mock Backend API Trigger Test ---")
print(f"Input features: {example_features}")
print("API Response:", predict_risk(example_features))
