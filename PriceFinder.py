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
    #copying data
    print("Starting data cleaning...")
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

def engineer_features(df): ### I STOPPED HERE


def process_data(input, output_cleaned_path, output_covid_path, output_decade_path):
    if(input is None):
        print("No data to clean.")
        return
    # Function to clean the data and save to new files
    df_raw = input.copy()
    df_cleaned = clean_data(df_raw)
    df = engineer_features(df_cleaned)
    # cleaned full dataset
    df.to_csv(output_cleaned_path, index=False)

    # Create a separate after COVID-19 dataset
    df_covid = df[df['date'] >= '2020-01-01']
    df_covid.to_csv(output_covid_path, index=False)

    # Create a separate decade dataset
    df_decade = df[df['date'] < '2015-01-01']
    df_decade.to_csv(output_decade_path, index=False)


if __name__ == "__main__":
    data_path = "data/pp-complete.csv"
    try:
        print("Starting to load data...")
        data = load_data(data_path)
        print(data.head())
        ### cleaning data // reupload into new data file
        print("Starting to clean data...")
        process_data(data,"data/pp-cleaned.csv","data/pp-covid.csv","data/pp-decade.csv")
        
        ### feature engineering
        print("Starting to load SVM model...")

        print("Starting to load XGBoost model...")

        print("Starting to load Random Forest model...")

        print("Starting to load Linear Regression model...")


        ### predictions
        print("Starting to make predictions...")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data file exists at the specified path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")