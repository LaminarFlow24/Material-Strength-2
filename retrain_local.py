# -*- coding: utf-8 -*-
"""
Local retraining script for Material Strength models.
This script retrains BOTH tensile and flexural models locally and saves
the model + all preprocessors together, ensuring consistency.

Run this once to fix the preprocessor mismatch issue.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# --- Configuration ---
DATA_PATH = "train1.csv"
MODELS_DIR = "Models"

os.makedirs(MODELS_DIR, exist_ok=True)

# --- Load data ---
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} samples from {DATA_PATH}")
print(f"Orientations: {sorted(df['orientation'].unique())}")
print(f"Infill Patterns: {sorted(df['infill_pattern'].unique())}")
print()

# --- Hyperparameter grid (same as train.py) ---
hyperparameter_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.05, 0.1, 0.15, 0.2],
    'min_child_weight': [1, 2, 3, 4, 5],
    'booster': ['gbtree'],
    'base_score': [0.25, 0.5, 0.75, 1],
    'early_stopping_rounds': [10, 50, 100]
}


def train_and_save(target_col, model_prefix, max_depth_list=None):
    """
    Train an XGBoost model for a given target column and save
    the model + all preprocessors.
    """
    print("=" * 70)
    print(f"TRAINING: {target_col}")
    print("=" * 70)
    
    # --- Step 1: Encode categorical features ---
    encoder = OneHotEncoder()
    df_encoded = encoder.fit_transform(df[['orientation', 'infill_pattern']])
    odf = pd.DataFrame.sparse.from_spmatrix(df_encoded)
    
    print(f"Encoder categories: {encoder.categories_}")
    print(f"One-hot shape: {df_encoded.shape}")
    
    # --- Step 2: Build combined DataFrame ---
    idf = df[['layer_thick', 'infill_density', 'mwcnt', 'graphene', target_col]]
    cdf = pd.concat([odf, idf], axis=1)
    cdf.columns = cdf.columns.astype(str)
    
    # --- Step 3: Scale ALL columns (including target) ---
    scaler = StandardScaler()
    scaler.fit(cdf)
    scaled_data = scaler.transform(cdf)
    dfn = pd.DataFrame(scaled_data, columns=cdf.columns)
    
    # --- Step 4: Separate scaler for target (for inverse transform) ---
    scaler_y = StandardScaler()
    scaler_y.fit(cdf[[target_col]])
    
    # --- Step 5: Split features and target ---
    X = dfn.drop(columns=[target_col])
    y = dfn[target_col]
    
    # --- Step 6: Train/Val/Test split (same seeds as train.py) ---
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=24)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=48)
    
    print(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    
    # --- Step 7: Hyperparameter tuning ---
    grid = hyperparameter_grid.copy()
    if max_depth_list:
        grid['max_depth'] = max_depth_list
    
    XGreg = xgb.XGBRegressor(objective='reg:squarederror')
    random_cv = RandomizedSearchCV(
        estimator=XGreg,
        param_distributions=grid,
        cv=5, n_iter=50,
        scoring='neg_mean_absolute_error',
        verbose=1,
        return_train_score=True,
        random_state=42,
    )
    
    print("Running hyperparameter search...")
    random_cv.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    best_model = random_cv.best_estimator_
    print(f"Best params: {random_cv.best_params_}")
    
    # Refit on training data
    best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    # --- Step 8: Evaluate ---
    y_pred_test = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)
    
    # Convert back to original scale for reporting
    y_pred_test_orig = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1))
    y_test_orig = scaler_y.inverse_transform(pd.DataFrame(y_test).values)
    y_pred_train_orig = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1))
    y_train_orig = scaler_y.inverse_transform(pd.DataFrame(y_train).values)
    
    test_mae = mean_absolute_error(y_test_orig, y_pred_test_orig)
    train_mae = mean_absolute_error(y_train_orig, y_pred_train_orig)
    
    print(f"\nResults ({target_col}):")
    print(f"  Train MAE: {train_mae:.2f}")
    print(f"  Test MAE:  {test_mae:.2f}")
    
    # --- Step 9: Verify predictions for "On Short Edge" ---
    print(f"\nVerification - Predictions by orientation:")
    for orient in sorted(df['orientation'].unique()):
        mask = df['orientation'] == orient
        orient_indices = df[mask].index
        # Get the corresponding rows from X (which uses dfn's index)
        orient_X = X.loc[X.index.isin(orient_indices)]
        if len(orient_X) > 0:
            preds = best_model.predict(orient_X)
            preds_orig = scaler_y.inverse_transform(preds.reshape(-1, 1))
            actual = df.loc[orient_X.index, target_col].values
            print(f"  {orient:20s}: predicted={preds_orig.mean():.2f} (range {preds_orig.min():.2f}-{preds_orig.max():.2f}), "
                  f"actual={actual.mean():.2f} (range {actual.min():.2f}-{actual.max():.2f})")
    
    # --- Step 10: Save everything ---
    model_path = os.path.join(MODELS_DIR, f"{model_prefix}_orientation.pkl")
    encoder_path = os.path.join(MODELS_DIR, f"{model_prefix}_encoder.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{model_prefix}_scaler.pkl")
    scaler_y_path = os.path.join(MODELS_DIR, f"{model_prefix}_scaler_y.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(scaler_y_path, 'wb') as f:
        pickle.dump(scaler_y, f)
    
    print(f"\nSaved:")
    print(f"  Model:    {model_path}")
    print(f"  Encoder:  {encoder_path}")
    print(f"  Scaler:   {scaler_path}")
    print(f"  Scaler_y: {scaler_y_path}")
    
    return best_model, encoder, scaler, scaler_y


if __name__ == "__main__":
    # Train tensile model (same max_depth as train.py: [2, 3, 4])
    print("\n" + "=" * 70)
    print("TENSILE STRENGTH MODEL")
    print("=" * 70)
    train_and_save("tensile_str", "Tens", max_depth_list=[2, 3, 4])
    
    # Train flexural model (same max_depth as train.py: [2, 3, 4, 7])
    print("\n" + "=" * 70)
    print("FLEXURAL STRENGTH MODEL")
    print("=" * 70)
    train_and_save("flexural_str", "Flex", max_depth_list=[2, 3, 4, 7])
    
    print("\n" + "=" * 70)
    print("ALL DONE! Models and preprocessors saved to Models/ directory.")
    print("You can now run the app with consistent preprocessors.")
    print("=" * 70)
