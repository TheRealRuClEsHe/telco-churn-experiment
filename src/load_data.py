import pandas as pd
import sqlite3

# Load CSV
df = pd.read_csv('Telco-Customer-Churn.csv')

# Clean column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Data wrangling
# 1. Fix totalcharges: it loaded as text, convert to number
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')

# 2. Check what we broke
print(f"Nulls in totalcharges after conversion: {df['totalcharges'].isna().sum()}")

# 3. Drop those rows - should be ~11 rows
df = df.dropna(subset=['totalcharges'])

# 4. Confirm final row count
print(f"Rows after cleaning: {len(df)}")

# Create SQLite database
conn = sqlite3.connect('telco_churn.db')
df.to_sql('customers', conn, if_exists='replace', index=False)

print(f"Loaded {len(df)} rows successfully")
print(f"Columns: {list(df.columns)}")
conn.close()