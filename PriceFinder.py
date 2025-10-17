import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.base import clone

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
    
    #fill nans with median  #try with zero next 
    num_cols = combined_df.select_dtypes(include=np.number).isnull().sum()
    num_cols = num_cols[num_cols > 0].index
    for col in num_cols:
        #median_value = train_df[col].median()  #NOT FULL BC DATA LEAKAGE
        combined_df[col] = combined_df[col].fillna(0)
    print(f"Imputed {len(num_cols)} numerical columns with their median.")

    print("Starting manual feature engineering...")
    #simple mappings
    quality_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    BsmtFinType_map = {'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    quality_cols = [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond'
    ]
    for col in quality_cols:
        combined_df[col + '_mapped'] = combined_df[col].map(quality_map).fillna(0)

    # A list of the basement finish type columns
    BsmtFinType_cols = ['BsmtFinType1', 'BsmtFinType2']
    for col in BsmtFinType_cols:
        combined_df[col + '_mapped'] = combined_df[col].map(BsmtFinType_map).fillna(0)

    # Create the Basement Quality-Volume feature
    BsmtExposure_map = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    combined_df['BsmtExposure_mapped'] = combined_df['BsmtExposure'].map(BsmtExposure_map).fillna(0)
    combined_df['BsmtQual_Vol'] = (
        (combined_df['BsmtFinType1_mapped'] * combined_df['BsmtFinSF1'] * combined_df['BsmtQual_mapped'] +
        combined_df['BsmtFinType2_mapped'] * combined_df['BsmtFinSF2'] * combined_df['BsmtQual_mapped']) *
        (1 + combined_df['BsmtExposure_mapped'])
    )

    # Create total bathrooms feature
    combined_df['TotalBaths'] = (
        combined_df['FullBath'] + 0.5 * combined_df['HalfBath'] +
        combined_df['BsmtFullBath'] + 0.5 * combined_df['BsmtHalfBath']
    )

    # Create total porch square footage feature
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    combined_df['TotalPorchSF'] = combined_df[porch_cols].sum(axis=1)

    # delete used columns
    cols_to_drop = [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual',
        'FireplaceQu', 'GarageQual', 'GarageCond', 'BsmtFinType1', 'BsmtFinType2', 
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtExposure',
        'FullBath', 'HalfBath', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'
    ]
    #seeing no drop
    #cols_to_drop.extend([col + '_mapped' for col in quality_cols])
    #cols_to_drop.extend([col + '_mapped' for col in BsmtFinType_cols])
    #cols_to_drop.extend(['BsmtExposure_mapped'])
    #combined_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print("Manual feature engineering complete.")

    # use mode to fill remaining categorical nans
    cat_cols_with_nan = combined_df.select_dtypes(include='object').isnull().sum()
    cat_cols_to_impute = cat_cols_with_nan[cat_cols_with_nan > 0].index
    for col in cat_cols_to_impute:
        mode_value = train_df[col].mode()[0]
        combined_df[col] = combined_df[col].fillna(mode_value)
    print(f"Imputed {len(cat_cols_to_impute)} categorical columns with their mode.")
    
    # Convert numerical identifiers to strings before one-hot encoding
    for col in ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']:
         combined_df[col] = combined_df[col].astype(str)

    # Now, create dummy variables for the remaining object columns
    df_encoded = pd.get_dummies(combined_df, columns=combined_df.select_dtypes(include='object').columns, drop_first=True)

    # Split back into training and test sets
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

def plot_Engineered_Features(engineered_df, target, features_to_plot):
    plot_df = pd.concat([engineered_df[features_to_plot], target], axis=1)
    n_cols = 3
    n_rows = int(np.ceil(len(features_to_plot) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    fig.suptitle('Engineered Features vs. SalePrice', fontsize=18, y=1.03)

    for i, col in enumerate(features_to_plot):
        sns.scatterplot(data=plot_df, x=col, y='SalePrice', ax=axes[i], alpha=0.5)
        axes[i].set_title(f'{col} (Corr: {plot_df[col].corr(plot_df["SalePrice"]):.2f})', fontsize=12, pad=10)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('SalePrice')

    for i in range(len(features_to_plot), len(axes)):
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
    plt.savefig('engineered_features_vs_saleprice.png')
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

def plot_top_models_predictions(top_models_list, all_trained_models, X_val_data, y_val_data, lambda_param):
    plt.figure(figsize=(12, 12))

    colors = plt.cm.viridis(np.linspace(0, 1, len(top_models_list)))
    
    for i, (model_name, rmse) in enumerate(top_models_list):
        # Get the trained model from the dictionary
        model = all_trained_models[model_name]
        
        # Make predictions
        preds_boxcox = model.predict(X_val_data)
        preds = inv_boxcox(preds_boxcox, lambda_param)

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

    new_features = ['BsmtQual_Vol', 'TotalBaths', 'TotalPorchSF']
    plot_Engineered_Features(X_encoded, y, new_features)

    X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.15)
    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

    X_submission_encoded = X_submission_encoded.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_submission_scaled = scaler.transform(X_submission_encoded)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_submission_scaled = pd.DataFrame(X_submission_scaled, columns=X_submission_encoded.columns)

    y_train_boxcox, lambda_param = boxcox(y_train)
    print(f"Optimal Box-Cox lambda: {lambda_param:.4f}")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1),
        "CatBoost": CatBoostRegressor(iterations=100, learning_rate=0.1, random_seed=42, verbose=0, allow_writing_files=False),
        "SVR": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    }

    parameters = {
    "Random Forest": {
        'n_estimators': [400, 500, 700],
        'max_depth': [2, 10, None],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [2, 3, 5] 
    },
    "XGBoost": {
        'n_estimators': [400, 600, 800, 1000],
        'learning_rate': [0.01, 0.03, 0.04],
        'max_depth': [2, 3, 4], 
        'subsample': [0.5, 0.6, 0.7], 
        'colsample_bytree': [0.3, 0.5, 0.6] 
    },
    "LightGBM": {
        'n_estimators': [600, 800, 1000],
        'learning_rate': [0.03, 0.05, 0.07],
        'num_leaves': [15, 20, 25], 
        'boosting_type': ['gbdt'] 
    },
    "CatBoost": {
        'iterations': [500, 600, 700],
        'learning_rate': [0.05, .08, 0.1], 
        'depth': [4, 5, 7, 10], 
        'l2_leaf_reg': [0.3, 0.6, 1, 2] 
    },
    "SVR": {
        'kernel': ['rbf', 'poly'],
        'C': [8, 10, 12],
        'gamma': ['auto'],
        'epsilon': [0.02, 0.03, 0.05] 
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
            random_search = RandomizedSearchCV(model, parameters[name], n_iter=30, scoring='neg_root_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
            random_search.fit(X_train_scaled, y_train_boxcox)

            best_model_instance = random_search.best_estimator_
            all_best_params[name] = random_search.best_params_
            print(f"Best parameters for {name}: {random_search.best_params_}")
        
        else:
            best_model_instance = model
            best_model_instance.fit(X_train_scaled, y_train_boxcox)
            all_best_params[name] = "Default parameters used"
        
        trained_models[name] = best_model_instance
        val_preds_log = best_model_instance.predict(X_val_scaled)
        val_preds = inv_boxcox(val_preds_log, lambda_param)
        
        mse = mean_squared_error(y_val, val_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, val_preds)
        r2 = r2_score(y_val, val_preds)
        mape = mean_absolute_percentage_error(y_val, val_preds)

        model_performance[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'MSE': mse,
            'MAPE': mape,
            'R2': r2
        }

        print(f"{name} Validation RMSE: {rmse:.2f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = best_model_instance
        
    sorted_performance = sorted(model_performance.items(), key=lambda item: item[1]['RMSE'])
    for model_name, performance in sorted_performance:
        print(f"{model_name}: RMSE = {performance['RMSE']:.2f}")

    print(f"\nBest model: {best_model_name} with RMSE: {best_rmse:.2f}")

    final_preds_log = best_model.predict(X_submission_scaled)
    final_preds = inv_boxcox(final_preds_log, lambda_param)

    report_data = []
    for model_name, metrics in sorted_performance:
        params = all_best_params.get(model_name, "default")
        report_data.append({
            'Model': model_name,
            'RMSE': f"{metrics['RMSE']:.2f}",
            'MAE': f"{metrics['MAE']:.2f}",
            'MSE': f"{metrics['MSE']:.2f}",
            'MAPE': f"{metrics['MAPE']:.2f}",
            'R2': f"{metrics['R2']:.4f}",
            'Best Parameters': str(params)
        })
    report_df = pd.DataFrame(report_data)
    report_file = 'model_report.csv'
    report_df.to_csv(report_file, index=False)

    print("Creating an ensemble of the top 3 models..." )
    top_3_models = [model[0] for model in sorted_performance[:3]]
    ensemble_preds = np.zeros(X_submission_scaled.shape[0])
    val_preds_ensemble = {}
    for model_name in top_3_models:
        model_instance = trained_models[model_name]
        preds = model_instance.predict(X_val_scaled)
        val_preds_ensemble[model_name] = inv_boxcox(preds, lambda_param)
    predictions_df = pd.DataFrame(val_preds_ensemble, index=y_val.index)
    ensemble_preds = predictions_df.mean(axis=1)
    ensemble_RMSE = np.sqrt(mean_squared_error(y_val, ensemble_preds.loc[y_val.index]))
    ensemble_r2 = r2_score(y_val, ensemble_preds.loc[y_val.index])
    print(f"Ensemble Model RMSE on Validation Set: {ensemble_RMSE:.2f}, R2: {ensemble_r2:.4f}")
    plot_actual_predicted_best(y_val, ensemble_preds, "Ensemble Model")

    rmse_plot = {name: metrics['RMSE'] for name, metrics in model_performance.items()}
    plot_results(rmse_plot)
    top_5_models = sorted_performance[:5]
    top_5_plot = [(name, metrics['RMSE']) for name, metrics in top_5_models]
    plot_top_models_predictions(top_5_plot, trained_models, X_val_scaled, y_val, lambda_param)

    final_val_preds_boxcox = best_model.predict(X_val_scaled)
    final_val_preds = inv_boxcox(final_val_preds_boxcox, lambda_param)
    plot_actual_predicted_best(y_val, final_val_preds, best_model_name)

    ## combining train and validation for best model retraining
    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)
    X_full_scaled = scaler.fit_transform(X_full)
    X_full_scaled = pd.DataFrame(X_full_scaled, columns=X_full.columns)
    y_full_boxcox, lambda_param_full = boxcox(y_full)
    
    print(f"\nRetraining best model {best_model_name} on full dataset...")
    model = clone(models[best_model_name])
    best_params = all_best_params.get(best_model_name)
    if best_params != "Default parameters used":
        model.set_params(**best_params)
    model.fit(X_full_scaled, y_full_boxcox)
    final_test_preds_boxcox = model.predict(X_submission_scaled)
    final_test_preds = inv_boxcox(final_test_preds_boxcox, lambda_param_full)

    submission_df = pd.DataFrame({
        'Id': submission_ids,
        'SalePrice': final_test_preds
    })
    submission_file = 'submission(single).csv'
    submission_df.to_csv(submission_file, index=False)
    print(f"Submission file '{submission_file}' created successfully.")

    print("\nRetraining top models for an ensemble submission...")
    final_predictions = {}
    top_3_model_names = [model[0] for model in sorted_performance[:3]] 
    for model_name in top_3_model_names:
        print(f"Retraining {model_name} for the ensemble...")
        model_instance = clone(models[model_name])
        params = all_best_params.get(model_name)
        if params != "Default parameters used":
            model_instance.set_params(**params)
        model_instance.fit(X_full_scaled, y_full_boxcox)
        preds_transformed = model_instance.predict(X_submission_scaled)
        final_predictions[model_name] = inv_boxcox(preds_transformed, lambda_param)

    predictions_df = pd.DataFrame(final_predictions)
    ensemble_preds = predictions_df.mean(axis=1)
    ensemble_submission_df = pd.DataFrame({'Id': submission_ids, 'SalePrice': ensemble_preds})
    ensemble_submission_df.to_csv('final_ensemble_submission.csv', index=False)
    print("Final ensemble submission file created successfully!")