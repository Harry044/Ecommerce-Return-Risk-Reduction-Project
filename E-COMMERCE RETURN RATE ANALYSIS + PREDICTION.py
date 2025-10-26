# ===========================================
# E-COMMERCE RETURN RATE ANALYSIS + PREDICTION
# ===========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load Clean Dataset
df = pd.read_csv('Clean_Ecommerce.csv')

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
print(df.head())

# ===========================================
# Step 2: Create Return Label
# ===========================================
# 'IsReturn' column exists — ensure it's boolean
df['IsReturn'] = df['IsReturn'].astype(bool)

# ===========================================
# Step 3: Exploratory Data Analysis (EDA)
# ===========================================

# 3.1 Return rate by Product
return_by_product = df.groupby('Description')['IsReturn'].mean().sort_values(ascending=False)
print("\nTop 10 Products by Return Rate:\n", return_by_product.head(10))

plt.figure(figsize=(12,6))
return_by_product.head(10).plot(kind='bar', color='salmon')
plt.title('Top 10 Products by Return Rate')
plt.ylabel('Return Rate')
plt.show()

# 3.2 Return rate by Country
return_by_country = df.groupby('Country')['IsReturn'].mean().sort_values(ascending=False)
print("\nReturn Rate by Country:\n", return_by_country)

plt.figure(figsize=(10,5))
sns.barplot(x=return_by_country.index, y=return_by_country.values, palette='coolwarm')
plt.xticks(rotation=45)
plt.title('Return Rate by Country')
plt.ylabel('Return Rate')
plt.show()

# ===========================================
# Step 4: Feature Engineering
# ===========================================
# Extract month from InvoiceDate
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['Month'] = df['InvoiceDate'].dt.month

# Customer-level return history
df['PreviousReturns'] = df.groupby('CustomerID')['IsReturn'].cumsum() - df['IsReturn']

# Handle any negative values (first purchase)
df['PreviousReturns'] = df['PreviousReturns'].clip(lower=0)

# Select relevant features
features = ['Quantity', 'UnitPrice', 'PreviousReturns', 'Month', 'Country']
X = df[features]
y = df['IsReturn']

# One-hot encode categorical feature (Country)
X = pd.get_dummies(X, columns=['Country'], drop_first=True)

# ===========================================
# Step 5: Train-Test Split
# ===========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===========================================
# Step 6: Logistic Regression Model
# ===========================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n=== MODEL PERFORMANCE ===")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ===========================================
# Step 7: Predict Return Probability for All Records
# ===========================================
df['ReturnProbability'] = model.predict_proba(X)[:, 1]

# ===========================================
# Step 8: Export High-Risk Products
# ===========================================
threshold = 0.7  # Change if needed
df_high_risk = df[df['ReturnProbability'] > threshold]

df_high_risk.to_csv('High_Risk_Products.csv', index=False)
print(f"\n Exported {len(df_high_risk)} high-risk records to 'High_Risk_Products.csv'")

# ===========================================
# Step 9: Visualization - Return Probability Distribution
# ===========================================
plt.figure(figsize=(10,5))
sns.histplot(df['ReturnProbability'], bins=50, color='skyblue')
plt.title('Distribution of Return Probabilities')
plt.xlabel('Return Probability')
plt.show()

# ===========================================
# Step 10: Save Final Dataset (Optional)
# ===========================================
df.to_csv('Ecommerce_With_Predictions.csv', index=False)
print("\n Final dataset saved as 'Ecommerce_With_Predictions.csv'")
