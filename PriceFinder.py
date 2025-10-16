import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

#models
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from catboost import CatBoostRegressor

#I CHANGED THIS AFTER CLEANING TO USE 
def load_data(file_path):
    try:
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
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
    
    df['MSSubClass'] = df['MSSubClass'].apply(str)
    df['OverallCond'] = df['OverallCond'].astype(str)
    df['YrSold'] = df['YrSold'].astype(str)
    df['MoSold'] = df['MoSold'].astype(str)

    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"One-hot encoding {len(categorical_cols)} categorical columns...")
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print("Feature engineering complete.")
    return df_encoded

if __name__ == "__main__":
    train_data_path = "dataKaggle/train.csv"
    test_data_path = "dataKaggle/test.csv"

    try:
        data = load_data(train_data_path)
        test_data = load_data(test_data_path)
        if data is None or test_data is None:
            print("Data loading failed. Exiting.")
            exit(1)
        print(data.head())
        datasets = {"full": clean_data(data)} #might do more splits
        

        for name, df in datasets.items():
            if df is None:
                print(f"Dataset {name} is None, skipping...")
                continue
            print(f"Processing dataset {name} with shape {df.shape}...")
            
            # Separate features and target
            X = df.drop(columns=['Id', 'SalePrice'])
            y = df['SalePrice']
            
            X_submission = test_data.copy()
            X_submission_ids = X_submission['Id']

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

            print(f"Dataset {name} split into train ({X_train.shape}), val ({X_val.shape}), submission ({X_submission.shape})")

            



            ### feature engineering (until fixed)
            #X_train, X_val, X_test = engineer_features(X_train, y_train_orig, X_val, X_test)
            y_train_log = np.log1p(y_train)  # log1p for numerical stability
            y_val_log = np.log1p(y_val)  # log1p for numerical stability
        
            print(X_train.head())

            ### normalization
            print("Starting normalization...")
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
            #X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
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
            y_val = np.expm1(y_val_log)
            print("Starting to make predictions with SVR model...")

            print("Starting to make predictions with XGBoost model...")
            xgb_val_preds = xgb_model.predict(X_val_scaled)
            xgb_preds = np.expm1(xgb_preds)  # Inverse of log1p
            xgb_val_preds = np.expm1(xgb_val_preds)  # Inverse of log1p

            xgb_val_medae = mean_absolute_error(y_val, xgb_val_preds)
            xgb_val_mse = mean_squared_error(y_val, xgb_val_preds)
            xgb_val_rmse = np.sqrt(xgb_val_mse)
            xgb_val_r2 = r2_score(y_val, xgb_val_preds)
            xgb_val_mape = mean_absolute_percentage_error(y_val, xgb_val_preds)
            print(f"XGBoost Val MAE: {xgb_val_medae:.2f}, MSE: {xgb_val_mse:.2f}, RMSE: {xgb_val_rmse:.2f}, R2: {xgb_val_r2:.4f}, MAPE: {xgb_val_mape:.4f}")

            print("Starting to make predictions with LightGBM model...")
            lgb_val_preds = lgb_model.predict(X_val_scaled)
            lgb_preds = np.expm1(lgb_preds)  # Inverse of log1p
            lgb_val_preds = np.expm1(lgb_val_preds)  # Inverse of log1p

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
            lr_val_preds = lr_model.predict(X_val_scaled)
            lr_preds = np.expm1(lr_preds)  # Inverse of log1p
            lr_val_preds = np.expm1(lr_val_preds)  # Inverse of log1p

            lr_val_medae = mean_absolute_error(y_val, lr_val_preds)
            lr_val_mse = mean_squared_error(y_val, lr_val_preds)
            lr_val_rmse = np.sqrt(lr_val_mse)
            lr_val_r2 = r2_score(y_val, lr_val_preds)
            lr_val_mape = mean_absolute_percentage_error(y_val, lr_val_preds)
            print(f"Linear Regression Val MAE: {lr_val_medae:.2f}, MSE: {lr_val_mse:.2f}, RMSE: {lr_val_rmse:.2f}, R2: {lr_val_r2:.4f}, MAPE: {lr_val_mape:.4f}")

            print("Starting to make predictions with CatBoost model...")
            cat_val_preds = cat_model.predict(X_val_scaled)
            cat_preds = np.expm1(cat_preds)  # Inverse of log1p
            cat_val_preds = np.expm1(cat_val_preds)  # Inverse of log1p
            
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