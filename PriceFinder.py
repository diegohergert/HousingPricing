import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def clean_data(input_path, output_cleaned_path, output_covid_path, output_decade_path):
    # Function to clean the data and save to new files
    df = pd.read_csv(input_path)
    df.columns = ['price', 'date', 'postcode', 'property_type', 'old/new', 'duration', 'paon',
                  'saon', 'street', 'locality', 'town_city', 'district', 'county', 'PPD_category', 'DELETEROW']
    df['DELETEROW'] = df['DELETEROW'].astype(str).str.strip().str.lower() == 'true' ##I STOPED HERE
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
    
    ### cleaning data // reupload into new data file
    print("Starting to clean data...")
    data_cleaned = "data/pp-complete-cleaned.csv"
    data_covid = "data/pp-complete-covid.csv"
    data_decade = "data/pp-complete-decade.csv"
    
    
    ### feature engineering
    print("Starting to load SVM model...")

    print("Starting to load XGBoost model...")

    print("Starting to load Random Forest model...")

    print("Starting to load Linear Regression model...")


    ### predictions
    print("Starting to make predictions...")