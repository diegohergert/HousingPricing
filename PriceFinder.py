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
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


#I CHANGED THIS AFTER CLEANING TO USE 
def load_data(file_path):
    try:
        print("Counting total rows...")
        with open(file_path, 'r', encoding='utf-8') as f:
            total_rows = sum(1 for line in f) -1 #for the header
        print(f"Total rows: {total_rows:,}")

        chunk_size = 100000
        chunks = []

        # --- THE FIX IS HERE ---
        # We tell pandas that fields are enclosed in double quotes (")
        # and use the robust 'python' engine to handle this correctly.
        reader = pd.read_csv(
            file_path,
            chunksize=chunk_size,
            engine='python'      # Use the more robust engine
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
    
    for df in [x_train_raw, X_val_raw, X_test_raw]:
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Create cyclical features for day of the week
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        df['postcode_area'] = df['postcode'].str.extract(r'(^[A-Z]{1,2})', expand=False)

    X_train = x_train_raw.copy()
    print("Learning encodings from training data...")

    simple_mappings = ['property_type', 'old_new', 'duration', 'ppd_category']
    hard_mappings = ['town_city', 'district', 'county', 'postcode_area']
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

        #THESE WERE TOO unique
        df.drop(columns=['postcode','street'], inplace=True)
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
        ## cleaning data // reupload into new data file
        clean_data_path = "data/pp-cleaned.csv"
        covid_data_path = "data/pp-covid.csv"
        decade_data_path = "data/pp-decade.csv"
        year_data_path = "data/pp-year.csv"
        #did not feature engineer because of data leaking into test set
        #process_data(data, clean_data_path, covid_data_path, decade_data_path, year_data_path)
        print("Data processing complete.")
        ### data splitting 70/15/15
        print("Starting to split data...")
        #clean_df = load_data(clean_data_path)  #later this is huge
        covid_df = load_data(covid_data_path)  #2nd smallest
        #decade_df = load_data(decade_data_path)  #2nd largest
        year_df = load_data(year_data_path)  #smallest

        datasets = {#"Clean Dataset": clean_df,
                    "Covid Dataset": covid_df, 
                    #"Decade Dataset": decade_df, 
                    "Year Dataset": year_df}
        

        for name, df in datasets.items():
            #splitting data
            if df is None:
                print(f"Dataset {name} is None, skipping...")
                continue
            print(f"Processing dataset {name} with shape {df.shape}...")
            X = df.drop(columns=['price'])
            print(X.head())
            y_actual = df['price']

            X_train, X_temp, y_train_orig, y_temp = train_test_split(X, y_actual, test_size=0.3, random_state=1)
            X_val, X_test, y_val_orig, y_test_orig = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)
            print(f"Dataset {name} split into train ({X_train.shape}), val ({X_val.shape}), test ({X_test.shape})")

            ### feature engineering
            X_train, X_val, X_test = engineer_features(X_train, y_train_orig, X_val, X_test)
            y_train_log = np.log1p(y_train_orig)  # log1p for numerical stability
            y_val_log = np.log1p(y_val_orig)  # log1p for numerical stability
            y_test_log = np.log1p(y_test_orig)  # log1p for numerical stability

            print(X_train.head())

            ### normalization
            print("Starting normalization...")
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
            print("Normalization complete.")

            ### model training
            print("Starting to train SVR model...")

            print("Starting to train XGBoost model...")
            xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=1)
            xgb_model.fit(X_train_scaled, y_train_log)
            print("XGBoost model training complete.")

            print("Starting to train LightGBM model...")
            lgb_model = lgb.LGBMRegressor()
            lgb_model.fit(X_train_scaled, y_train_log)
            print("LightGBM model training complete.")

            """
            print("Starting to train Random Forest model...")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=1)
            rf_model.fit(X_train_scaled, y_train_log)
            print("Random Forest model training complete.")
            """
            
            print("Starting to train Linear Regression model...")
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train_log)
            print("Linear Regression model training complete.")

            print("Starting to train CatBoost model...")
            cat_model = CatBoostRegressor(verbose=0, random_state=1)
            cat_model.fit(X_train_scaled, y_train_log)
            print("CatBoost model training complete.")

            ### predictions
            print("Starting to make predictions...")
            y_test = np.expm1(y_test_log)
            y_val = np.expm1(y_val_log)
            print("Starting to make predictions with SVR model...")

            print("Starting to make predictions with XGBoost model...")
            xgb_preds = xgb_model.predict(X_test_scaled)
            xgb_val_preds = xgb_model.predict(X_val_scaled)
            xgb_preds = np.expm1(xgb_preds)  # Inverse of log1p
            xgb_val_preds = np.expm1(xgb_val_preds)  # Inverse of log1p
            xgb_medae = mean_absolute_error(y_test, xgb_preds)
            xgb_mse = mean_squared_error(y_test, xgb_preds)
            xgb_rmse = np.sqrt(xgb_mse)
            xgb_r2 = r2_score(y_test, xgb_preds)
            xgb_mape = mean_absolute_percentage_error(y_test, xgb_preds)
            print(f"XGBoost Test MAE: {xgb_medae:.2f}, MSE: {xgb_mse:.2f}, RMSE: {xgb_rmse:.2f}, R2: {xgb_r2:.4f}, MAPE: {xgb_mape:.4f}")
            xgb_val_medae = mean_absolute_error(y_val, xgb_val_preds)
            xgb_val_mse = mean_squared_error(y_val, xgb_val_preds)
            xgb_val_rmse = np.sqrt(xgb_val_mse)
            xgb_val_r2 = r2_score(y_val, xgb_val_preds)
            xgb_val_mape = mean_absolute_percentage_error(y_val, xgb_val_preds)
            print(f"XGBoost Val MAE: {xgb_val_medae:.2f}, MSE: {xgb_val_mse:.2f}, RMSE: {xgb_val_rmse:.2f}, R2: {xgb_val_r2:.4f}, MAPE: {xgb_val_mape:.4f}")

            print("Starting to make predictions with LightGBM model...")
            lgb_preds = lgb_model.predict(X_test_scaled)
            lgb_val_preds = lgb_model.predict(X_val_scaled)
            lgb_preds = np.expm1(lgb_preds)  # Inverse of log1p
            lgb_val_preds = np.expm1(lgb_val_preds)  # Inverse of log1p
            lgb_medae = mean_absolute_error(y_test, lgb_preds)
            lgb_mse = mean_squared_error(y_test, lgb_preds)
            lgb_rmse = np.sqrt(lgb_mse)
            lgb_r2 = r2_score(y_test, lgb_preds)
            lgb_mape = mean_absolute_percentage_error(y_test, lgb_preds)
            print(f"LightGBM Test MAE: {lgb_medae:.2f}, MSE: {lgb_mse:.2f}, RMSE: {lgb_rmse:.2f}, R2: {lgb_r2:.4f}, MAPE: {lgb_mape:.4f}")
            lgb_val_medae = mean_absolute_error(y_val, lgb_val_preds)
            lgb_val_mse = mean_squared_error(y_val, lgb_val_preds)
            lgb_val_rmse = np.sqrt(lgb_val_mse)
            lgb_val_r2 = r2_score(y_val, lgb_val_preds)
            lgb_val_mape = mean_absolute_percentage_error(y_val, lgb_val_preds)
            print(f"LightGBM Val MAE: {lgb_val_medae:.2f}, MSE: {lgb_val_mse:.2f}, RMSE: {lgb_val_rmse:.2f}, R2: {lgb_val_r2:.4f}, MAPE: {lgb_val_mape:.4f}")

            """"
            print("Starting to make predictions with Random Forest model...")
            rf_preds = rf_model.predict(X_test_scaled)
            rf_val_preds = rf_model.predict(X_val_scaled)
            rf_preds = np.expm1(rf_preds)  # Inverse of log1p
            rf_val_preds = np.expm1(rf_val_preds)  # Inverse of
            rf_medae = mean_absolute_error(y_test, rf_preds)
            rf_mse = mean_squared_error(y_test, rf_preds)
            rf_rmse = np.sqrt(rf_mse)
            rf_r2 = r2_score(y_test, rf_preds)
            rf_mape = mean_absolute_percentage_error(y_test, rf_preds)
            print(f"Random Forest Test MAE: {rf_medae:.2f}, MSE: {rf_mse:.2f}, RMSE: {rf_rmse:.2f}, R2: {rf_r2:.4f}, MAPE: {rf_mape:.4f}")
            rf_val_medae = mean_absolute_error(y_val, rf_val_preds)
            rf_val_mse = mean_squared_error(y_val, rf_val_preds)
            rf_val_rmse = np.sqrt(rf_val_mse)
            rf_val_r2 = r2_score(y_val, rf_val_preds)
            rf_val_mape = mean_absolute_percentage_error(y_val, rf_val_preds)
            print(f"Random Forest Val MAE: {rf_val_medae:.2f}, MSE: {rf_val_mse:.2f}, RMSE: {rf_val_rmse:.2f}, R2: {rf_val_r2:.4f}, MAPE: {rf_val_mape:.4f}")
            """
            
            print("Starting to make predictions with Linear Regression model...")
            lr_preds = lr_model.predict(X_test_scaled)
            lr_val_preds = lr_model.predict(X_val_scaled)
            lr_preds = np.expm1(lr_preds)  # Inverse of log1p
            lr_val_preds = np.expm1(lr_val_preds)  # Inverse of log1p
            lr_medae = mean_absolute_error(y_test, lr_preds)
            lr_mse = mean_squared_error(y_test, lr_preds)
            lr_rmse = np.sqrt(lr_mse)
            lr_r2 = r2_score(y_test, lr_preds)
            lr_mape = mean_absolute_percentage_error(y_test, lr_preds)
            print(f"Linear Regression Test MAE: {lr_medae:.2f}, MSE: {lr_mse:.2f}, RMSE: {lr_rmse:.2f}, R2: {lr_r2:.4f}, MAPE: {lr_mape:.4f}")
            lr_val_medae = mean_absolute_error(y_val, lr_val_preds)
            lr_val_mse = mean_squared_error(y_val, lr_val_preds)
            lr_val_rmse = np.sqrt(lr_val_mse)
            lr_val_r2 = r2_score(y_val, lr_val_preds)
            lr_val_mape = mean_absolute_percentage_error(y_val, lr_val_preds)
            print(f"Linear Regression Val MAE: {lr_val_medae:.2f}, MSE: {lr_val_mse:.2f}, RMSE: {lr_val_rmse:.2f}, R2: {lr_val_r2:.4f}, MAPE: {lr_val_mape:.4f}")

            print("Starting to make predictions with CatBoost model...")
            cat_preds = cat_model.predict(X_test_scaled)
            cat_val_preds = cat_model.predict(X_val_scaled)
            cat_preds = np.expm1(cat_preds)  # Inverse of log1p
            cat_val_preds = np.expm1(cat_val_preds)  # Inverse of log1p
            cat_medae = mean_absolute_error(y_test, cat_preds)
            cat_mse = mean_squared_error(y_test, cat_preds)
            cat_rmse = np.sqrt(cat_mse)
            cat_r2 = r2_score(y_test, cat_preds)
            cat_mape = mean_absolute_percentage_error(y_test, cat_preds)
            print(f"CatBoost Test MAE: {cat_medae:.2f}, MSE: {cat_mse:.2f}, RMSE: {cat_rmse:.2f}, R2: {cat_r2:.4f}, MAPE: {cat_mape:.4f}")
            cat_val_medae = mean_absolute_error(y_val, cat_val_preds)
            cat_val_mse = mean_squared_error(y_val, cat_val_preds)
            cat_val_rmse = np.sqrt(cat_val_mse)
            cat_val_r2 = r2_score(y_val, cat_val_preds)
            cat_val_mape = mean_absolute_percentage_error(y_val, cat_val_preds)
            print(f"CatBoost Val MAE: {cat_val_medae:.2f}, MSE: {cat_val_mse:.2f}, RMSE: {cat_val_rmse:.2f}, R2: {cat_val_r2:.4f}, MAPE: {cat_val_mape:.4f}")

            print(f"Finished processing dataset {name}.\n")
            
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data file exists at the specified path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")