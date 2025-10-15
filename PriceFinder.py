import csv
import math
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

def load_data(file_path):
    columns = [
        'price',            # The price paid for the property
        'date',             # Date of the transfer
        'postcode',         # The postcode of the property
        'property_type',    # D=Detached, S=Semi-Detached, T=Terraced, F=Flats/Maisonettes, O=Other
        'old_new',          # Y = a new build, N = an established residential building
        'duration',         # F = Freehold, L = Leasehold
        'paon',             # Primary Addressable Object Name (e.g., house number or name)
        'saon',             # Secondary Addressable Object Name (e.g., flat number)
        'street',           # Street name
        'locality',         # e.g., a small village  ## A LOT OF NULLS
        'town_city',        # The town or city
        'district',         # The district/local authority area
        'county',           # The county
        'ppd_category',     # Price Paid Data Category Type - A = Standard, B = Additional
        'record_status'     # Record Status - Not useful for this dataset, safe to drop.
    ]
    try:
        print("Counting total rows...")
        with open(file_path, 'r', encoding='utf-8') as f:
            total_rows = sum(1 for line in f)
        print(f"Total rows: {total_rows:,}")

        chunk_size = 100000
        chunks = []

        # --- THE FIX IS HERE ---
        # We tell pandas that fields are enclosed in double quotes (")
        # and use the robust 'python' engine to handle this correctly.
        reader = pd.read_csv(
            file_path,
            header=None,
            names=columns,
            chunksize=chunk_size,
            engine='python',        # Use the more robust engine
            quotechar='"',          # Specify that fields are quoted with "
            quoting=csv.QUOTE_MINIMAL # A standard quoting behavior setting
        )

        # total is the total amount of 100k chunks needed to read the whole file
        for chunk in tqdm(reader, total=math.ceil(total_rows/chunk_size), desc="Loading data"):
            chunks.append(chunk)
            
        df = pd.concat(chunks, ignore_index=True)
        print("\nData loaded successfully.")
        return df
        
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def clean_data(df_raw):
    if(df_raw is None):
        print("No data to clean.")
        return
    print("Starting data cleaning...")
    #copying data
    df = df_raw.copy()
    df.drop(columns=['record_status','locality','saon','paon'], inplace=True) #maybe paon too idk how that is useful
    print("Dropped unnecessary columns.")
    
    print("Checking for null values...")
    null_counts = df.isnull().sum()
    remaining_nulls = null_counts[null_counts > 0]

    if not remaining_nulls.empty:
        print("Columns with remaining nulls:")
        print(remaining_nulls)
        
        # 3. Drop any rows that have ANY null values left
        print("\nDropping rows with any null values...")
        rows_before_drop = df.shape[0]
        df.dropna(inplace=True)
        rows_after_drop = df.shape[0]
        
        print(f"Removed {rows_before_drop - rows_after_drop:,} rows.")
    else:
        print("No remaining null values to drop.")

    print(f"--- Cleaning Complete. Final shape: {df.shape} ---")
    return df

def engineer_features(x_train_raw, y_train, X_val_raw, X_test_raw):
    print("Starting feature engineering...")

    X_train = x_train_raw.copy()
    print("Learning encodings from training data...")

    simple_mappings = ['property_type', 'old_new', 'duration', 'ppd_category']
    hard_mappings = ['postcode', 'street', 'town_city', 'district', 'county']
    mappings = {}
    mappings['target_encoding'] = {}

    for col in hard_mappings:
        print(f"Calculating target encoding for {col}...")
        mean_map = y_train.groupby(X_train[col]).mean()
        std_map = y_train.groupby(X_train[col]).std()
        count_map = X_train[col].value_counts()
        mappings['target_encoding'][col] = {
            'mean': mean_map, 'std': std_map, 'count': count_map
            }

    print("Applying encodings to datasets...")

    datasets = [x_train_raw.copy(), X_val_raw.copy(), X_test_raw.copy()]
    transformed_dfs = []

    for df in datasets:
        df = pd.get_dummies(df, columns=simple_mappings, drop_first=True)
        for col in hard_mappings:
            maps_for_col = mappings['target_encoding'][col]
            df[f'{col}_mean'] = df[col].map(maps_for_col['mean'])
            df[f'{col}_std'] = df[col].map(maps_for_col['std'])
            df[f'{col}_count'] = df[col].map(maps_for_col['count'])
        
        df.drop(columns=hard_mappings, inplace=True)
        transformed_dfs.append(df)

    X_train, X_val, X_test = transformed_dfs[0], transformed_dfs[1], transformed_dfs[2]

    print("final cleanup for feature engineered data...")
    train_cols = X_train.columns
    X_val = X_val.reindex(columns=train_cols, fill_value=0)
    X_test = X_test.reindex(columns=train_cols, fill_value=0)

    X_train.fillna(0, inplace=True)
    X_val.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    print("Feature engineering complete.")
    return X_train, X_val, X_test

def process_data(input, output_cleaned_path, output_covid_path, output_decade_path, output_year_path):
    if(input is None):
        print("No data to clean.")
        return
    # Function to clean the data and save to new files
    print("Starting to process data...")
    df_raw = input.copy()
    df = clean_data(df_raw)
    
    # Extract year, month, day of week
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M', errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df.drop(columns=['date'], inplace=True)  # Drop date col
    df.dropna(subset=['year', 'month', 'day_of_week'], inplace=True)  # Drop rows where date conversion failed
    
    # cleaned full dataset
    print("Saving cleaned full dataset...")
    df.to_csv(output_cleaned_path, index=False)

    # Create a separate after COVID-19 dataset
    df_covid = df[df['year'] >= 2020]
    df_covid.to_csv(output_covid_path, index=False)

    # Create a separate decade dataset
    df_decade = df[df['year'] >= 2015]
    df_decade.to_csv(output_decade_path, index=False)

    df_year = df[df['year'] >= 2024]
    df_year.to_csv(output_year_path, index=False)

if __name__ == "__main__":
    #data_path = "data/pp-complete.csv"
    try:
        #print("Starting to load data...")
        #data = load_data(data_path)
        #print(data.head())
        ### cleaning data // reupload into new data file
        clean_data_path = "data/pp-cleaned.csv"
        covid_data_path = "data/pp-covid.csv"
        decade_data_path = "data/pp-decade.csv"
        year_data_path = "data/pp-year.csv"
        #did not feature engineer because of data leaking into test set
        #process_data(data, clean_data_path, covid_data_path, decade_data_path, year_data_path)
        print("Data processing complete.")
        ### data splitting 70/15/15
        print("Starting to split data...")
        clean_df = load_data(clean_data_path)  #need to update load
        ### model training
        print("Starting to train SVR model...")
        print("Starting to train XGBoost model...")
        print("Starting to train LightGBM model...")
        print("Starting to train Random Forest model...")
        print("Starting to train Linear Regression model...")
        print("Starting to train CatBoost model...")

        ### predictions
        print("Starting to make predictions...")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data file exists at the specified path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")