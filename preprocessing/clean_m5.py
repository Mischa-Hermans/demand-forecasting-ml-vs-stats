import pandas as pd
import os

data_dir = "data/m5"
save_path = os.path.join(data_dir, "m5_clean.csv")

# Load raw M5 files
sales = pd.read_csv(os.path.join(data_dir, "sales_train_validation.csv"))
calendar = pd.read_csv(os.path.join(data_dir, "calendar.csv"))
prices = pd.read_csv(os.path.join(data_dir, "sell_prices.csv"))

# Reshape to long format
date_cols = [c for c in sales.columns if c.startswith("d_")]
sales_long = sales.melt(
    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    value_vars=date_cols,
    var_name='d',
    value_name='sales'
)

# Merge with calendar for date and week
sales_long = sales_long.merge(calendar[['d', 'date', 'wm_yr_wk']], on='d', how='left')
sales_long['date'] = pd.to_datetime(sales_long['date'])

# Join prices (fix key type first)
sales_long['wm_yr_wk'] = sales_long['wm_yr_wk'].astype(str)
prices['wm_yr_wk'] = prices['wm_yr_wk'].astype(str)

merged = sales_long.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

# Add time features
merged['year'] = merged['date'].dt.year
merged['month'] = merged['date'].dt.month

# Save
merged.to_csv(save_path, index=False)
print(f"Cleaned data saved to {save_path}")
