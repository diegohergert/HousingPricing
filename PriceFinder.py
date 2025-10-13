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
    columns = ['price', 'date', 'postcode', 'property_type', 'old/new', 'duration', 'paon',
                    'saon', 'street', 'locality', 'town_city', 'district', 'county', 'PPD_category', 'DELETEROW']
    try:
        print("counting total rows...")
        total_rows = sum(1 for line in open(file_path))  # no headers
        print(f"Total rows: {total_rows}")

        chunk_size = 100000
        chunks = []
        reader = pd.read_csv(file_path, header=None, names=columns, chunksize=chunk_size)
        for chunk in tqdm(reader, total=-(total_rows//-chunk_size), desc="Loading data"):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None



def clean_data(input, output_cleaned_path, output_covid_path, output_decade_path):
    # Function to clean the data and save to new files
    df = input.copy()
    df.drop(columns=['DELETEROW'])
    # Example cleaning steps (to be customized based on actual data)
    df = df.dropna()
    df.to_csv(output_cleaned_path, index=False)

    # Create a separate COVID-19 dataset
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
        clean_data(data, "data/cleaned_data.csv", "data/covid_data.csv", "data/decade_data.csv")
        
        
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