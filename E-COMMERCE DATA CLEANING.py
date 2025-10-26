# ===========================================
# E-COMMERCE DATA CLEANING SCRIPT
# ===========================================

import pandas as pd

# Step 1: Load Dataset
# (Replace 'Ecommerce.csv' with your actual file name)
df = pd.read_csv('data.csv', encoding='ISO-8859-1')

# Step 2: Quick Overview
print("Initial Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# Step 3: Drop rows with missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Step 4: Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Drop rows where date conversion failed
df = df.dropna(subset=['InvoiceDate'])

# Step 5: Remove Duplicates
df = df.drop_duplicates()

# Step 6: Create 'IsReturn' column before removing negatives
df['IsReturn'] = df['InvoiceNo'].astype(str).str.startswith('C')

# Step 7: Remove only invalid (not negative) rows
# Keep returns for analysis
df = df[df['UnitPrice'] > 0]


# Step 8: Create 'TotalSales' column
df['TotalSales'] = df['Quantity'] * df['UnitPrice']

# Step 9: (Optional) Focus on UK only
# Comment this line if you want all countries
# df = df[df['Country'] == 'United Kingdom']

# Step 10: Reset Index and Save
df.reset_index(drop=True, inplace=True)
df.to_csv('Clean_Ecommerce.csv', index=False)

print("\n DATA CLEANING COMPLETE!")
print("Final Shape:", df.shape)
print("Cleaned file saved as 'Clean_Ecommerce.csv'")
