import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

#models
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from catboost import CatBoostRegressor

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

def engineer_features(train_df, test_df):
    print("Starting feature engineering...")
    
    train_len = train_df.shape[0]
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    
    #fill nans with median
    num_cols = combined_df.select_dtypes(include=np.number).isnull().sum()
    num_cols = num_cols[num_cols > 0].index
    for col in num_cols:
        median_value = train_df[col].median()  #NOT FULL BC DATA LEAKAGE
        combined_df[col].fillna(median_value, inplace=True)
    print(f"Imputed {len(num_cols)} numerical columns with their median.")
    
    #fill nans with mode
    cat_cols = combined_df.select_dtypes(include='object').isnull().sum()
    cat_cols = cat_cols[cat_cols > 0].index
    for col in cat_cols:
        mode_value = train_df[col].mode()[0] #NOT FULL BC DATA LEAKAGE
        combined_df[col].fillna(mode_value, inplace=True)
    print(f"Imputed {len(cat_cols)} categorical columns with their mode.")

    combined_df['MSSubClass'] = combined_df['MSSubClass'].astype(str)
    combined_df['OverallCond'] = combined_df['OverallCond'].astype(str)
    combined_df['YrSold'] = combined_df['YrSold'].astype(str)
    combined_df['MoSold'] = combined_df['MoSold'].astype(str)
    
    #encode categorical variables
    df_encoded = pd.get_dummies(combined_df, columns=combined_df.select_dtypes(include='object').columns, drop_first=True)

    #split back
    train_df_encoded = df_encoded.iloc[:train_len]
    test_df_encoded = df_encoded.iloc[train_len:]

    print("Feature engineering complete.")
    return train_df_encoded, test_df_encoded

def plot_results(model_performance):
    sorted_performance = sorted(model_performance.items(), key=lambda item: item[1])
    models = [item[0] for item in sorted_performance]
    rmse_values = [item[1] for item in sorted_performance]

    plt.figure(figsize=(12, 12))
    bars = plt.barh(models, rmse_values, color='red')
    plt.xlabel('RMSE')
    plt.ylabel('Model')
    plt.title('Model Performance Comparison (RMSE)')
    plt.gca().invert_yaxis()  # Highest performance on top

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 100, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center')
    
    plt.xlim(0, max(rmse_values) * 1.1)
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.show()

def plot_EDA(data, top_n_numerical, categorical_features_to_plot):
    numerical_cols = data.select_dtypes(include=np.number).columns
    correlations = data[numerical_cols].corr()['SalePrice'].sort_values(ascending=False)
    
    # Get the names of the top N most correlated features
    top_numerical_features = correlations[1:top_n_numerical + 1].index
    
    # Create the subplot grid
    n_cols = 3
    n_rows = int(np.ceil(len(top_numerical_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten() # Flatten the grid to easily loop through it

    fig.suptitle('Top Correlated Numerical Features vs. SalePrice', fontsize=18, y=1.03)

    for i, col in enumerate(top_numerical_features):
        sns.scatterplot(data=data, x=col, y='SalePrice', ax=axes[i], alpha=0.5)
        axes[i].set_title(f'{col} (Corr: {correlations[col]:.2f})', fontsize=12, pad=10)

    # Hide any unused subplots
    for i in range(len(top_numerical_features), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(
    left=0.069,
    bottom=0.07,
    right=0.975,
    top=0.922,
    wspace=0.26,
    hspace=0.507
    )
    plt.savefig('numerical_features_vs_saleprice.png')
    plt.show()

    # --- Part 2: Key Categorical Features vs. SalePrice ---
    if categorical_features_to_plot:
        n_cols_cat = 3
        n_rows_cat = int(np.ceil(len(categorical_features_to_plot) / n_cols_cat))
        fig_cat, axes_cat = plt.subplots(n_rows_cat, n_cols_cat, figsize=(n_cols_cat * 6, n_rows_cat * 5))
        axes_cat = axes_cat.flatten()

        fig_cat.suptitle('Key Categorical Features vs. SalePrice', fontsize=18, y=1.03)
        
        for i, col in enumerate(categorical_features_to_plot):
            # Order by median SalePrice for a cleaner look
            order = data.groupby(col)['SalePrice'].median().sort_values().index
            sns.boxplot(data=data, x=col, y='SalePrice', ax=axes_cat[i], order=order)
            axes_cat[i].tick_params(axis='x', rotation=45)
        
        for i in range(len(categorical_features_to_plot), len(axes_cat)):
            axes_cat[i].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.subplots_adjust(
        left=0.069,
        bottom=0.07,
        right=0.975,
        top=0.922,
        wspace=0.26,
        hspace=0.507
        )
        plt.savefig('categorical_features_vs_saleprice.png')
        plt.show()

def plot_actual_predicted_best(y_true, y_pred, model_name):
    plt.figure(figsize=(12, 12))
    plt.scatter(y_true, y_pred, alpha=0.5, color='green')
    perfect_line_max = max(y_true.max(), y_pred.max())
    perfect_line_min = min(y_true.min(), y_pred.min())
    plt.plot([perfect_line_min, perfect_line_max], [perfect_line_min, perfect_line_max], color='red', linestyle='--', lw=2, label='Perfect Prediction')
    plt.title(f'Actual vs Predicted SalePrice - {model_name}')
    plt.xlabel('Actual SalePrice')
    plt.ylabel('Predicted SalePrice')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'actual_vs_predicted_{model_name}.png')
    plt.show()  

def plot_top_models_predictions(top_models_list, all_trained_models, X_val_data, y_val_data):
    plt.figure(figsize=(12, 12))

    colors = plt.cm.viridis(np.linspace(0, 1, len(top_models_list)))
    
    for i, (model_name, rmse) in enumerate(top_models_list):
        # Get the trained model from the dictionary
        model = all_trained_models[model_name]
        
        # Make predictions
        preds_log = model.predict(X_val_data)
        preds = np.expm1(preds_log)
        
        # Plot the predictions for this model
        plt.scatter(y_val_data, preds, alpha=0.5, color=colors[i], 
                    label=f'{model_name} (RMSE: ${rmse:,.2f})')

    # Add the 45-degree "perfect prediction" line
    max_val = max(y_val_data.max(), preds.max())
    min_val = min(y_val_data.min(), preds.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='Perfect Prediction')

    plt.title('Actual vs. Predicted Prices (Top Models)', fontsize=15)
    plt.xlabel('Actual Prices', fontsize=12)
    plt.ylabel('Predicted Prices', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_data_path = "dataKaggle/train.csv"
    test_data_path = "dataKaggle/test.csv"

    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)
    if train_data is None or test_data is None:
        print("Data loading failed. Exiting.")
        exit(1)

    key_categorical_features = [
        'OverallQual', 'Neighborhood', 'BldgType', 
        'ExterQual', 'KitchenQual', 'GarageType'
    ]
    plot_EDA(train_data, top_n_numerical=9, categorical_features_to_plot=key_categorical_features)

    X = train_data.drop(columns=['Id','SalePrice'])
    y = train_data['SalePrice']
    submission_ids = test_data['Id']
    X_submission = test_data.drop(columns=['Id'])

    X_encoded, X_submission_encoded = engineer_features(X, X_submission)

    X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

    X_submission_encoded = X_submission_encoded.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_submission_scaled = scaler.transform(X_submission_encoded)

    y_train_log = np.log1p(y_train)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "CatBoost": CatBoostRegressor(iterations=100, learning_rate=0.1, random_seed=42, verbose=0),
        "SVR": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    }

    parameters = {
        "Random Forest": {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2, 4]
        },
        "XGBoost": {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 6, 8],
            'subsample': [0.6, 0.9],
            'colsample_bytree': [0.6, 0.9]
        },
        "LightGBM": {
            'n_estimators': [100, 200, 400],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [20, 30, 55],
            'boosting_type': ['gbdt', 'dart']
        },
        "CatBoost": {
            'iterations': [100, 200, 400],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 10],
            'l2_leaf_reg': [3, 5, 7]
        },
        "SVR": {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'epsilon': [0.05, 0.1, 0.2]
        }
    }

    best_rmse = float('inf')  #using rmse to evaluate
    best_model_name = None
    best_model = None
    all_best_params = {}
    model_performance = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining and tuning {name}...")
        if name in parameters:
            random_search = RandomizedSearchCV(model, parameters[name], n_iter=10, scoring='neg_root_mean_squared_error', cv=3, verbose=1, n_jobs=-1, random_state=42)
            random_search.fit(X_train_scaled, y_train_log)

            best_model_instance = random_search.best_estimator_
            all_best_params[name] = random_search.best_params_
            print(f"Best parameters for {name}: {random_search.best_params_}")
        
        else:
            best_model_instance = model
            best_model_instance.fit(X_train_scaled, y_train_log)
            all_best_params[name] = "Default parameters used"
        
        trained_models[name] = best_model_instance
        val_preds_log = best_model_instance.predict(X_val_scaled)
        val_preds = np.expm1(val_preds_log)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        model_performance[name] = rmse
        print(f"{name} Validation RMSE: {rmse:.2f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = best_model_instance
        
    sorted_performance = sorted(model_performance.items(), key=lambda item: item[1])
    for model_name, performance in sorted_performance:
        print(f"{model_name}: RMSE = {performance:.2f}")
    
    print(f"\nBest model: {best_model_name} with RMSE: {best_rmse:.2f}")

    final_preds_log = best_model.predict(X_submission_scaled)
    final_preds = np.expm1(final_preds_log)

    submission_df = pd.DataFrame({
        'Id': submission_ids,
        'SalePrice': final_preds
    })
    submission_df.to_csv('submission.csv', index=False)

    plot_results(model_performance)
    sorted_performance_models = sorted(model_performance.items(), key=lambda item: item[1])
    top_5_models = sorted_performance_models[:5]
    plot_top_models_predictions(top_5_models, trained_models, X_val_scaled, y_val)

    final_val_preds_log = best_model.predict(X_val_scaled)
    final_val_preds = np.expm1(final_val_preds_log)
    plot_actual_predicted_best(y_val, final_val_preds, best_model_name)