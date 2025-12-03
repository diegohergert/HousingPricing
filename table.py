import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Setup the Data
data = {
    'Model': ['Linear Regression', 'LightGBM', 'XGBoost', 'CatBoost', 'Random Forest', 'SVR'],
    'RMSE': [16841.96, 19535.22, 19942.04, 20804.86, 22038.62, 34661.68],
    'MAE': [12232.36, 13209.82, 12698.71, 12876.46, 15068.19, 18239.48],
    'MSE': [283651743.88, 381624888.01, 397684893.33, 432842267.92, 485700764.69, 1201431940.17],
    'MAPE': [0.07, 0.08, 0.07, 0.07, 0.09, 0.10],
    'R2': [0.9249, 0.8989, 0.8947, 0.8854, 0.8714, 0.6819],
    'Best Parameters': [
        'Default parameters used',
        "{'num_leaves': 15, 'n_estimators': 600, 'learning_rate': 0.03, 'boosting_type': 'gbdt'}",
        "{'subsample': 0.5, 'n_estimators': 1000, 'max_depth': 4, 'learning_rate': 0.03, 'colsample_bytree': 0.3}",
        "{'learning_rate': 0.05, 'l2_leaf_reg': 1, 'iterations': 500, 'depth': 5}",
        "{'n_estimators': 700, 'min_samples_split': 3, 'min_samples_leaf': 2, 'max_depth': None}",
        "{'kernel': 'rbf', 'gamma': 'auto', 'epsilon': 0.02, 'C': 8}"
    ]
}

df = pd.DataFrame(data)

# --- OPTION 1: Generate a Clean Image for the Report ---

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

# Create a display copy without the long parameters for the image
df_display = df.drop(columns=['Best Parameters'])

# Format numbers with commas for readability
df_display['RMSE'] = df_display['RMSE'].apply(lambda x: "{:,.2f}".format(x))
df_display['MAE'] = df_display['MAE'].apply(lambda x: "{:,.2f}".format(x))
df_display['MSE'] = df_display['MSE'].apply(lambda x: "{:,.0f}".format(x)) # No decimals for MSE, it's huge
df_display['MAPE'] = df_display['MAPE'].apply(lambda x: "{:.2%}".format(x)) # Percent format
df_display['R2'] = df_display['R2'].apply(lambda x: "{:.4f}".format(x))

# Generate and Save Image
ax = render_mpl_table(df_display, header_columns=0, col_width=2.5)
plt.title("Housing Price Prediction Model Performance", fontsize=16, y=1.05, weight='bold')
plt.savefig("results_table.png", bbox_inches='tight', dpi=300)
print("Table image saved as 'results_table.png'")


# --- OPTION 2: Print Markdown for Text Report ---
print("\n--- Markdown Table (Copy/Paste this) ---")
print(df.to_markdown(index=False))