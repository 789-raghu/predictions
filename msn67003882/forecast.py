import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Configuration Parameters ---
MSN = "67003882"
START_DATE = "2025-07-09"  # Training starts July 09, 2025
END_DATE = "2026-07-20"
PREDICTION_START_DATE = pd.Timestamp('2026-07-01 00:00:00')
PREDICTION_END_DATE = pd.Timestamp('2026-07-20 23:00:00')

OUTPUT_DIR = 'tcn_forecast_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, 'training_logs.txt')
MAE_CSV_PATH = os.path.join(OUTPUT_DIR, 'mae_errors.csv')
MAE_REPORT_PATH = os.path.join(OUTPUT_DIR, 'mae_report.txt')
BEST_PARAMS_PATH = 'best_params.json'

print(f"Results will be saved in the directory: {OUTPUT_DIR}/")

# --- 1. Data Loading Function ---
def _parse_api_data(json_data, start_date_str):
    rows = []

    # Check for API error messages first
    if "message" in json_data and json_data["message"] == "Internal Server Error":
        print(f"API returned an error for {start_date_str}: {json_data['message']}")
        return pd.DataFrame(columns=["timestamp", "consumption"])

    data_content = json_data.get("data")

    if isinstance(data_content, dict):
        for date, hourly_readings in data_content.items():
            for reading in hourly_readings:
                rows.append({
                    "timestamp": pd.to_datetime(f"{date} {reading['hour']}"),
                    "consumption": float(reading["consumption"])
                })
    elif isinstance(data_content, list):
        # Assuming start_date_str is the single date when data_content is a list
        for reading in data_content:
            rows.append({
                "timestamp": pd.to_datetime(f"{start_date_str} {reading['hour']}"),
                "consumption": float(reading["consumption"])
            })
    elif data_content is None or (isinstance(data_content, list) and not data_content):
        print(f"No data found for {start_date_str} in API response.")
        return pd.DataFrame(columns=["timestamp", "consumption"])
    else:
        raise ValueError(f"Unexpected data format from API for {start_date_str}: {json_data}")

    return pd.DataFrame(rows)

# --- 2. Feature Engineering ---
def _add_all_features(df_input, lags, windows, add_quarter_feature=True):
    df = df_input.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["weekofyear"] = df["timestamp"].dt.isocalendar().week.astype(int)
    if add_quarter_feature:
        df["quarter"] = df["timestamp"].dt.quarter
    df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)

    # Cyclical features
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"] = np.sin(2*np.pi*df["dayofweek"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dayofweek"]/7)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)

    # Lagged features
    for lag in lags:
        df[f"lag_{lag}"] = df["consumption"].shift(lag)

    # Rolling window features
    for w in windows:
        df[f"rolling_mean_{w}"] = df["consumption"].shift(1).rolling(w).mean()
        df[f"rolling_std_{w}"] = df["consumption"].shift(1).rolling(w).std()
        df[f"rolling_max_{w}"] = df["consumption"].shift(1).rolling(w).max()
        df[f"rolling_min_{w}"] = df["consumption"].shift(1).rolling(w).min()

    # Exponentially Weighted Moving Average (EWMA) features
    for alpha_val in [0.1, 0.3, 0.5]:
        df[f"ewm_{alpha_val}"] = df["consumption"].shift(1).ewm(alpha=alpha_val).mean()
    return df

# --- 3. Original Training Interface ---
def train_model(startDate, train_end_date_for_model_training, hour, add_quarter_feature=True):
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    windows = [6, 12, 24, 48, 168]
    target = "consumption"

    # Fetch Training Data
    url_train = f"https://ap.elementsenergies.com/api/fetchHConsWAvg?startdate={startDate}&enddate={train_end_date_for_model_training}&msn={MSN}"
    response_train = requests.get(url_train)
    json_data_train = response_train.json()

    df_train_raw_original = _parse_api_data(json_data_train, startDate)

    # Convert train_end_date_for_model_training to datetime.date object for comparison
    train_end_date_dt = pd.to_datetime(train_end_date_for_model_training).date()

    df_train_raw_original = df_train_raw_original[
        (
            df_train_raw_original["timestamp"].dt.date < train_end_date_dt
        )
        |
        (
            (df_train_raw_original["timestamp"].dt.date == train_end_date_dt)
            &
            (df_train_raw_original["timestamp"].dt.hour < hour)
        )
    ].reset_index(drop=True)

    # Add features to training data
    df_train_full_features = _add_all_features(df_train_raw_original, lags, windows, add_quarter_feature)
    df_train = df_train_full_features.dropna().reset_index(drop=True)

    # Define features and target
    features = [col for col in df_train.columns if col not in ["timestamp", target]]

    X_train = df_train[features]
    y_train = df_train[target]

    # Train Model
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        bagging_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    return model, df_train_raw_original, features, lags, windows

# --- 4. GPU / CUDA Availability Check ---
def detect_device_type():
    device_type = 'cpu'
    print("\n--- Checking GPU Availability for LightGBM ---")
    try:
        # Run a small training test with GPU support
        X_dummy = np.random.rand(10, 2)
        y_dummy = np.random.rand(10)
        clf = LGBMRegressor(device_type='gpu', verbose=-1)
        clf.fit(X_dummy, y_dummy)
        device_type = 'gpu'
        print("LightGBM GPU support: SUCCESS (using 'gpu')")
    except Exception as e_gpu:
        try:
            clf = LGBMRegressor(device_type='cuda', verbose=-1)
            clf.fit(X_dummy, y_dummy)
            device_type = 'cuda'
            print("LightGBM CUDA support: SUCCESS (using 'cuda')")
        except Exception as e_cuda:
            print(f"LightGBM GPU/CUDA training not available, falling back to CPU. (Errors: GPU={e_gpu}, CUDA={e_cuda})")
            device_type = 'cpu'
    return device_type

# --- 5. Hyperparameter Tuning ---
def tune_hyperparameters(X_train, y_train, device_type):
    print("\n--- Tuning Hyperparameters using RandomizedSearchCV ---")
    param_dist = {
        'n_estimators': [100, 500, 1000],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
        'max_depth': [6, 8, 10, -1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'bagging_freq': [1]  # Added to allow subsample to work without warnings
    }
    
    # 85% train, 15% validation split
    val_size = int(len(X_train) * 0.15)
    split_index = np.full(X_train.shape[0], -1)
    split_index[-val_size:] = 0
    pds = PredefinedSplit(test_fold=split_index)
    
    lgbm = LGBMRegressor(random_state=42, verbose=-1, device_type=device_type)
    random_search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_dist,
        n_iter=15,
        scoring='neg_mean_absolute_error',
        cv=pds,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    print(f"Best parameters: {best_params}")
    
    # Save the parameters to a file in the directory
    with open(BEST_PARAMS_PATH, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Saved best parameters to {BEST_PARAMS_PATH}")
    
    return best_params

# --- 6. Visualizations & Helpers ---
def clean_data(df, stricter_upper_bound):
    print("\n--- Starting Data Cleaning and Outlier Visualization ---")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df['consumption'])
    plt.title(f'Box Plot of Raw Consumption (MSN: {MSN})')
    plt.ylabel('Consumption')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'box_plot_raw_consumption.png'))
    plt.close()
    print(f"Saved box plot to {OUTPUT_DIR}/box_plot_raw_consumption.png")

    print(f"\nUsing the following threshold for outlier marking:")
    print(f"  - Upper bound (for very extreme right-tail outliers): {stricter_upper_bound:.2f}")

    extreme_right_tail_outliers = df[df['is_outlier']]
    print(f"Number of VERY extreme right-tail outliers detected: {len(extreme_right_tail_outliers)}")

    plt.figure(figsize=(15, 7))
    sns.lineplot(x='timestamp', y='consumption_cleaned', data=df)
    plt.title(f'Consumption Over Time - Cleaned/Interpolated Data (MSN: {MSN})')
    plt.xlabel('Timestamp')
    plt.ylabel('Consumption')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'cleaned_consumption_line_plot.png'))
    plt.close()
    print(f"Saved cleaned data line plot to {OUTPUT_DIR}/cleaned_consumption_line_plot.png")

def analyze_trend(df):
    print("\n--- Starting Trend Analysis ---")
    df_trend = df.copy()
    df_trend['rolling_mean_168h'] = df_trend['consumption_cleaned'].rolling(window=168).mean()

    plt.figure(figsize=(18, 8))
    sns.lineplot(x='timestamp', y='consumption_cleaned', data=df_trend, label='Cleaned Consumption', alpha=0.7)
    sns.lineplot(x='timestamp', y='rolling_mean_168h', data=df_trend, label='168-Hour Rolling Mean', color='red')
    plt.title(f'Energy Consumption Over Time with 168-Hour Rolling Mean (MSN: {MSN})')
    plt.xlabel('Timestamp')
    plt.ylabel('Consumption')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'trend_analysis.png'))
    plt.close()
    print(f"Saved trend analysis plot to {OUTPUT_DIR}/trend_analysis.png")

def analyze_seasonality(df):
    print("\n--- Starting Seasonality Analysis ---")
    df_seasonal = df.copy()
    df_seasonal = df_seasonal.set_index('timestamp')
    df_seasonal = df_seasonal.asfreq('h')
    df_seasonal['consumption_cleaned'] = df_seasonal['consumption_cleaned'].ffill().bfill()

    decomposition = seasonal_decompose(df_seasonal['consumption_cleaned'], model='additive', period=24)

    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Seasonal Decomposition (MSN: {MSN})', fontsize=16)
    plt.subplot(4, 1, 1)
    plt.plot(decomposition.observed)
    plt.title('Original Series')
    plt.ylabel('Consumption')
    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend)
    plt.title('Trend Component')
    plt.ylabel('Consumption')
    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal)
    plt.title('Seasonal Component (Daily)')
    plt.ylabel('Consumption')
    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid)
    plt.title('Residual Component')
    plt.ylabel('Consumption')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, 'seasonal_decomposition.png'))
    plt.close()
    print(f"Saved seasonal decomposition plot to {OUTPUT_DIR}/seasonal_decomposition.png")

def train_model_with_params(df_train_raw, lags, windows, best_params, device_type, add_quarter_feature=True):
    df_train_full_features = _add_all_features(df_train_raw, lags, windows, add_quarter_feature)
    df_train = df_train_full_features.dropna().reset_index(drop=True)

    target = "consumption"
    features = [col for col in df_train.columns if col not in ["timestamp", target]]

    X_train = df_train[features]
    y_train = df_train[target]

    params = best_params.copy()
    if 'subsample' in params and params['subsample'] < 1.0:
        params['bagging_freq'] = 1

    model = LGBMRegressor(
        **params,
        random_state=42,
        device_type=device_type,
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model, features

# --- Main Execution ---
if __name__ == "__main__":
    script_start_time = time.time()
    print("\n--- Starting LightGBM Energy Consumption Forecasting Script ---")
    
    # Initialize the log file
    with open(LOG_FILE_PATH, 'w') as f:
        f.write("=== LightGBM Forecast Training Logs ===\n")

    # Detect GPU availability
    start_gpu_detect = time.time()
    device_type = detect_device_type()
    duration_gpu_detect = time.time() - start_gpu_detect
    print(f"Device detection completed in {duration_gpu_detect:.2f} seconds.")
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(f"Detected and selected device type: {device_type} (took {duration_gpu_detect:.2f}s)\n")

    # 1. Fetch data from API
    start_data_fetch = time.time()
    url = f"https://ap.elementsenergies.com/api/fetchHConsWAvg?startdate={START_DATE}&enddate={END_DATE}&msn={MSN}"
    print(f"Fetching data from URL: {url}")
    response = requests.get(url)
    response.raise_for_status()
    df_raw = _parse_api_data(response.json(), START_DATE)
    if df_raw.empty:
        print("Exiting due to empty DataFrame from API call.")
        exit()
    print(f"Raw data loaded. Shape: {df_raw.shape}")

    # Reindex to continuous hourly format
    df_raw = df_raw.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    all_hours = pd.date_range(start=df_raw.index.min(), end=df_raw.index.max(), freq='h')
    df = df_raw.reindex(all_hours)
    # Fill any missing values in the consumption using interpolation
    df['consumption'] = df['consumption'].interpolate(method='linear').ffill().bfill()
    df = df.reset_index().rename(columns={'index': 'timestamp'})
    duration_data_fetch = time.time() - start_data_fetch
    print(f"Data loading and preprocessing completed in {duration_data_fetch:.2f} seconds. Shape: {df.shape}")
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(f"Data loading and preprocessing completed. Shape: {df.shape} (took {duration_data_fetch:.2f}s)\n")

    # 2. Outlier Identification (based on initial training data)
    initial_train_end_date_ts = pd.Timestamp('2026-06-30 23:00:00')
    initial_train_cutoff_idx = df[df['timestamp'] == initial_train_end_date_ts].index[0]
    
    train_subset_temp = df.iloc[:initial_train_cutoff_idx + 1]
    Q1 = train_subset_temp['consumption'].quantile(0.25)
    Q3 = train_subset_temp['consumption'].quantile(0.75)
    IQR = Q3 - Q1
    stricter_upper_bound = Q3 + 3 * IQR
    
    df['is_outlier'] = df['consumption'] > stricter_upper_bound
    
    # Create clean consumption (interpolating outliers)
    df['consumption_cleaned'] = df['consumption'].copy()
    df.loc[df['is_outlier'], 'consumption_cleaned'] = np.nan
    df['consumption_cleaned'] = df['consumption_cleaned'].interpolate(method='linear').ffill().bfill()

    # Re-slice train_subset to include the new columns
    train_subset = df.iloc[:initial_train_cutoff_idx + 1]

    # Clean & analyze data visually
    start_visualizations = time.time()
    clean_data(df, stricter_upper_bound)
    analyze_trend(df)
    analyze_seasonality(df)
    duration_visualizations = time.time() - start_visualizations
    print(f"Visualizations completed in {duration_visualizations:.2f} seconds.")
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(f"Data visualization and decomposition completed (took {duration_visualizations:.2f}s)\n")

    # Feature definitions
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    windows = [6, 12, 24, 48, 168]
    
    # Prepare features for initial training set to tune hyperparameters
    df_train_full = _add_all_features(train_subset, lags, windows)
    df_train = df_train_full.dropna().reset_index(drop=True)
    
    target = "consumption"
    features = [col for col in df_train.columns if col not in ["timestamp", target]]
    
    X_train = df_train[features]
    y_train = df_train[target]

    # 3. Hyperparameter Tuning (or load from file if it already exists)
    start_tuning = time.time()
    if os.path.exists(BEST_PARAMS_PATH):
        print(f"\nFound existing best parameters file at {BEST_PARAMS_PATH}. Loading parameters...")
        with open(BEST_PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
        print(f"Loaded parameters: {best_params}")
        duration_tuning = time.time() - start_tuning
        with open(LOG_FILE_PATH, 'a') as f:
            f.write(f"Loaded Existing Best Parameters: {json.dumps(best_params)} (took {duration_tuning:.2f}s)\n")
    else:
        best_params = tune_hyperparameters(X_train, y_train, device_type)
        duration_tuning = time.time() - start_tuning
        print(f"Hyperparameter tuning completed in {duration_tuning:.2f} seconds.")
        with open(LOG_FILE_PATH, 'a') as f:
            f.write(f"Tuned Best Parameters: {json.dumps(best_params)} (took {duration_tuning:.2f}s)\n")

    # 4. LightGBM Rolling Forecast with Retraining
    print("\n--- Starting LightGBM Rolling Forecast with Retraining ---")
    start_rolling = time.time()
    iter_count = 0
    actuals = []
    predictions = []
    timestamps_forecasted = []

    current_time = PREDICTION_START_DATE
    active_day = PREDICTION_START_DATE.date()
    UNCERTAINTY_CAP_MAE = 5.0

    # Pre-calculate total iterations for progress tracking
    total_iters = 0
    temp_time = PREDICTION_START_DATE
    while temp_time <= PREDICTION_END_DATE:
        matching_rows = df[df['timestamp'] == temp_time]
        if not matching_rows.empty:
            total_iters += 1
        temp_time += pd.Timedelta(hours=1)

    def process_completed_day(day_date):
        day_mask = [ts.date() == day_date for ts in timestamps_forecasted]
        day_actuals = [actuals[i] for i, m in enumerate(day_mask) if m]
        day_preds = [predictions[i] for i, m in enumerate(day_mask) if m]
        day_ts = [timestamps_forecasted[i] for i, m in enumerate(day_mask) if m]
        
        if len(day_actuals) > 0:
            day_mae = mean_absolute_error(day_actuals, day_preds)
            log_msg = f"Completed predictions for {day_date}. Daily MAE: {day_mae:.4f}"
            print(log_msg)
            with open(LOG_FILE_PATH, 'a') as f:
                f.write(log_msg + "\n")
            
            # Write the daily graph
            plt.figure(figsize=(15, 7))
            plt.plot(day_ts, day_actuals, label='Actual Consumption', color='blue')
            plt.plot(day_ts, day_preds, label='Predicted Consumption', color='green', alpha=0.7)
            
            uncertainty_margin = min(day_mae, UNCERTAINTY_CAP_MAE)
            predicted_lower_bound = np.array(day_preds) - uncertainty_margin
            predicted_upper_bound = np.array(day_preds) + uncertainty_margin
            
            plt.fill_between(
                day_ts,
                predicted_lower_bound,
                predicted_upper_bound,
                color='dimgray',
                alpha=0.6,
                label=f'Uncertainty Range (MAE based, capped at +/- {uncertainty_margin:.2f})'
            )
            
            plt.title(f'LightGBM Rolling Forecast vs. Actual Consumption for {day_date} (MSN: {MSN}, Daily MAE: {day_mae:.4f})')
            plt.xlabel('Time of Day')
            plt.ylabel('Consumption')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'daily_forecast_{day_date}.png'))
            plt.close()
            print(f"Saved daily forecast plot to {OUTPUT_DIR}/daily_forecast_{day_date}.png")

    while current_time <= PREDICTION_END_DATE:
        matching_rows = df[df['timestamp'] == current_time]
        if matching_rows.empty:
            current_time += pd.Timedelta(hours=1)
            continue
            
        current_time_idx = matching_rows.index[0]
        
        # Check if the day has transitioned
        if current_time.date() != active_day:
            process_completed_day(active_day)
            active_day = current_time.date()
        
        iter_start = time.time()
        iter_count += 1

        # Retrain on all data up to current_time - 1 hour
        train_df_slice = df.iloc[:current_time_idx].copy()
        
        # Train model with best params
        start_train = time.time()
        model, feat_cols = train_model_with_params(
            train_df_slice, lags, windows, best_params, device_type
        )
        duration_train = time.time() - start_train
        
        # Predict the next hour
        start_pred = time.time()
        pred_features_df = _add_all_features(df.iloc[:current_time_idx + 1], lags, windows)
        last_row = pred_features_df[pred_features_df['timestamp'] == current_time]
        
        X_pred = last_row[feat_cols]
        predicted_val = model.predict(X_pred)[0]
        duration_pred = time.time() - start_pred

        duration_iter = time.time() - iter_start
        
        # Calculate progress and ETA
        elapsed_loop_time = time.time() - start_rolling
        avg_time_per_iter = elapsed_loop_time / iter_count if iter_count > 0 else 0
        remaining_iters = total_iters - iter_count
        eta_seconds = remaining_iters * avg_time_per_iter
        
        if eta_seconds > 3600:
            eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"
        else:
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            
        progress_pct = (iter_count / total_iters) * 100

        # Log to file and console
        iter_log = (
            f"[{current_time}] Progress: {iter_count}/{total_iters} ({progress_pct:.1f}%) | "
            f"ETA: {eta_str} | "
            f"Train: {duration_train:.2f}s | Predict: {duration_pred:.2f}s | Iter Total: {duration_iter:.2f}s"
        )
        print(iter_log)
        with open(LOG_FILE_PATH, 'a') as f:
            f.write(f"[{current_time}] Progress: {iter_count}/{total_iters} ({progress_pct:.1f}%) | ETA: {eta_str} | Train: {duration_train:.2f}s, Predict: {duration_pred:.2f}s, Total: {duration_iter:.2f}s\n")
        
        predictions.append(predicted_val)
        actuals.append(df.loc[current_time_idx, 'consumption'])
        timestamps_forecasted.append(current_time)

        current_time += pd.Timedelta(hours=1)

    duration_rolling = time.time() - start_rolling
    print(f"Rolling forecast completed in {duration_rolling:.2f} seconds. Average iteration: {duration_rolling/iter_count:.2f}s")
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(f"Rolling forecast completed. Total time: {duration_rolling:.2f}s (Average iteration: {duration_rolling/iter_count:.2f}s)\n")

    # Process the final remaining day
    process_completed_day(active_day)

    print("Rolling forecast completed.")

    # --- Post-processing and Visualization ---
    df_rolling_forecast = pd.DataFrame({
        'timestamp': timestamps_forecasted,
        'actual_consumption': actuals,
        'predicted_consumption': predictions
    })

    overall_rolling_mae = mean_absolute_error(df_rolling_forecast['actual_consumption'], df_rolling_forecast['predicted_consumption'])
    print(f"\nOverall Rolling Forecast MAE: {overall_rolling_mae:.2f}")

    df_rolling_forecast['date'] = df_rolling_forecast['timestamp'].dt.date
    daily_rolling_mae = df_rolling_forecast.groupby('date').apply(
        lambda x: mean_absolute_error(x['actual_consumption'], x['predicted_consumption']), 
        include_groups=False
    ).reset_index(name='MAE')

    print("\nDaily Mean Absolute Error (MAE) for Rolling Forecast:")
    print(daily_rolling_mae.head())

    # Save the MAE errors to CSV
    df_mae_csv = daily_rolling_mae.copy()
    df_mae_csv['date'] = df_mae_csv['date'].astype(str)
    # Add a row for the combined/total MAE
    combined_row = pd.DataFrame([{'date': 'combined', 'MAE': overall_rolling_mae}])
    df_mae_csv = pd.concat([df_mae_csv, combined_row], ignore_index=True)
    df_mae_csv.to_csv(MAE_CSV_PATH, index=False)
    print(f"Saved daily and combined MAE errors to CSV: {MAE_CSV_PATH}")

    # Save the MAE errors to a readable report file
    with open(MAE_REPORT_PATH, 'w') as f:
        f.write("=== Daily and Combined Mean Absolute Error (MAE) Report ===\n\n")
        f.write(f"Generated at: {pd.Timestamp.now()}\n\n")
        f.write("Daily MAE:\n")
        f.write("-----------------------\n")
        f.write("Date       | MAE\n")
        f.write("-----------------------\n")
        for _, row in daily_rolling_mae.iterrows():
            f.write(f"{row['date']} | {row['MAE']:.4f}\n")
        f.write("-----------------------\n\n")
        f.write(f"Total MAE (All Days Combined): {overall_rolling_mae:.4f}\n")
    print(f"Saved MAE report to: {MAE_REPORT_PATH}")

    plt.figure(figsize=(18, 8))
    sns.barplot(x='date', y='MAE', data=daily_rolling_mae)
    plt.title(f'Daily Mean Absolute Error (MAE) for LightGBM Rolling Forecast (MSN: {MSN})')
    plt.xlabel('Date')
    plt.ylabel('MAE')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'daily_rolling_mae.png'))
    plt.close()
    print(f"Saved daily rolling MAE plot to {OUTPUT_DIR}/daily_rolling_mae.png")

    total_script_duration = time.time() - script_start_time
    print(f"\n--- Script execution completed in {total_script_duration:.2f} seconds (~{total_script_duration/60:.2f} minutes). ---")
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(f"\n=== Script execution completed in {total_script_duration:.2f} seconds (~{total_script_duration/60:.2f} minutes). ===\n")
