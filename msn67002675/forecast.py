import os
import sys

# Ensure CUDA libraries from the virtual environment are in LD_LIBRARY_PATH
if "LD_LIBRARY_PATH" not in os.environ or "nvidia" not in os.environ["LD_LIBRARY_PATH"]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        paths = set()
        for root, dirs, files in os.walk(os.path.join(current_dir, ".venv")):
            for file in files:
                if ".so" in file and "nvidia" in root:
                    paths.add(root)
        if paths:
            cuda_path = ":".join(paths)
            if "LD_LIBRARY_PATH" in os.environ:
                os.environ["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"] + ":" + cuda_path
            else:
                os.environ["LD_LIBRARY_PATH"] = cuda_path
            # Re-execute the current script with the updated LD_LIBRARY_PATH
            os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)
    except Exception as e:
        print(f"Warning: Failed to set LD_LIBRARY_PATH automatically: {e}")

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Add, Activation, Lambda, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

# --- Configuration Parameters ---
START_DATE = "2025-07-09"  # Training starts July 09, 2025
END_DATE = "2026-07-20"
PREDICTION_START_DATE = pd.Timestamp('2026-07-01 00:00:00')
PREDICTION_END_DATE = pd.Timestamp('2026-07-20 23:00:00')
LOOK_BACK = 24  # Look-back period for TCN
TCN_FILTERS = 64
TCN_KERNEL_SIZE = 2
TCN_DILATIONS = [1, 2, 4, 8, 16]
INITIAL_TRAINING_EPOCHS = 50
ROLLING_FORECAST_EPOCHS = 10
UNCERTAINTY_CAP_MAE = 5.0 # Cap uncertainty margin at +/- 5 units

# Sample weighting parameters
WEIGHT_MULTIPLIER = 3.0    # Give 3x weight to recent data
WEIGHT_START_DAYS = 1
WEIGHT_END_DAYS = 7

# Rolling fine-tuning parameters
FINE_TUNE_WINDOW = 30 * 24  # Fine-tune using a rolling window of 30 days (720 hours)

OUTPUT_DIR = 'tcn_forecast_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, 'training_logs.txt')
MAE_CSV_PATH = os.path.join(OUTPUT_DIR, 'mae_errors.csv')
MAE_REPORT_PATH = os.path.join(OUTPUT_DIR, 'mae_report.txt')

print(f"Results will be saved in the directory: {OUTPUT_DIR}/")

# --- 1. Data Loading Function ---
def parse_api(start_date, end_date, hour=None):
    try:
        url = (
            f"https://ap.elementsenergies.com/api/fetchHConsWAvg"
            f"?startdate={start_date}&enddate={end_date}&msn=67002675"
        )
        print(f"Fetching data from URL: {url}")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        records = []

        if isinstance(data["data"], dict):
            for date, hours in data["data"].items():
                for row in hours:
                    timestamp = pd.to_datetime(
                        f"{date} {row['hour']}",
                        format="%Y-%m-%d %H:%M"
                    )

                    records.append({
                        "timestamp": timestamp,
                        "consumption": float(row["consumption"])
                    })

        elif isinstance(data["data"], list):
            date = start_date

            for row in data["data"]:
                timestamp = pd.to_datetime(
                    f"{date} {row['hour']}",
                    format="%Y-%m-%d %H:%M"
                )

                records.append({
                    "timestamp": timestamp,
                    "consumption": float(row["consumption"])
                })

        else:
            raise ValueError("Unexpected API response format.")

        df = pd.DataFrame(records)

        if df.empty:
            return df

        df = df.sort_values("timestamp").reset_index(drop=True)

        if hour is not None:
            df = df[df["timestamp"].dt.hour == hour]
            end_timestamp = pd.to_datetime(end_date).date()
            df = df[
                ~((df["timestamp"].dt.date == end_timestamp) & (df["timestamp"].dt.hour == hour))
            ].reset_index(drop=True)

        return df

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return pd.DataFrame()
    except KeyError as e:
        print(f"Missing expected key in API response: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unable to parse API response: {e}")
        return pd.DataFrame()

# --- 2. Data Cleaning and Outlier Removal/Marking ---
def clean_data(df, stricter_upper_bound):
    print("\n--- Starting Data Cleaning and Outlier Visualization ---")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df['consumption'])
    plt.title('Box Plot of Raw Consumption')
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
    plt.title('Consumption Over Time (Cleaned/Interpolated Data)')
    plt.xlabel('Timestamp')
    plt.ylabel('Consumption')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'cleaned_consumption_line_plot.png'))
    plt.close()
    print(f"Saved cleaned data line plot to {OUTPUT_DIR}/cleaned_consumption_line_plot.png")

# --- 3. Trend Analysis ---
def analyze_trend(df):
    print("\n--- Starting Trend Analysis ---")
    df_trend = df.copy()
    df_trend['rolling_mean_168h'] = df_trend['consumption_cleaned'].rolling(window=168).mean()

    plt.figure(figsize=(18, 8))
    sns.lineplot(x='timestamp', y='consumption_cleaned', data=df_trend, label='Cleaned Consumption', alpha=0.7)
    sns.lineplot(x='timestamp', y='rolling_mean_168h', data=df_trend, label='168-Hour Rolling Mean', color='red')
    plt.title('Energy Consumption Over Time with 168-Hour Rolling Mean')
    plt.xlabel('Timestamp')
    plt.ylabel('Consumption')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'trend_analysis.png'))
    plt.close()
    print(f"Saved trend analysis plot to {OUTPUT_DIR}/trend_analysis.png")

# --- 4. Seasonality Analysis ---
def analyze_seasonality(df):
    print("\n--- Starting Seasonality Analysis ---")
    df_seasonal = df.copy()
    df_seasonal = df_seasonal.set_index('timestamp')
    df_seasonal = df_seasonal.asfreq('h') # Set frequency to hourly
    df_seasonal['consumption_cleaned'] = df_seasonal['consumption_cleaned'].ffill().bfill()

    decomposition = seasonal_decompose(df_seasonal['consumption_cleaned'], model='additive', period=24)

    plt.figure(figsize=(15, 10))
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
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'seasonal_decomposition.png'))
    plt.close()
    print(f"Saved seasonal decomposition plot to {OUTPUT_DIR}/seasonal_decomposition.png")

# --- 5. TCN Data Preparation Functions ---
def create_sequences_range(df, look_back, start_idx, end_idx, scaler):
    """
    Creates sequences for TCN training or fine-tuning.
    start_idx and end_idx represent the range of target indices (inclusive).
    Features are scaled using features from the cleaned consumption column.
    Targets are scaled using original values from the consumption column.
    """
    features = df['consumption_cleaned'].values.reshape(-1, 1)
    targets = df['consumption'].values.reshape(-1, 1)
    
    features_scaled = scaler.transform(features)
    targets_scaled = scaler.transform(targets)
    
    X, y, y_timestamps, y_is_outlier = [], [], [], []
    
    for i in range(start_idx, end_idx + 1):
        X.append(features_scaled[i - look_back : i, 0])
        y.append(targets_scaled[i, 0])
        y_timestamps.append(df.loc[i, 'timestamp'])
        y_is_outlier.append(df.loc[i, 'is_outlier'])
        
    X = np.array(X)
    if len(X) > 0:
        X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, np.array(y), pd.to_datetime(y_timestamps), np.array(y_is_outlier)

def get_sample_weights(timestamps, cutoff_time, is_outliers):
    """
    Calculates sample weights for training.
    Gives a multiplier to data from WEIGHT_START_DAYS to WEIGHT_END_DAYS (e.g. 1 to 7 days) prior to cutoff_time.
    Sets weight to 0.0 for outliers to ignore them during training.
    """
    weights = np.ones(len(timestamps), dtype=np.float32)
    deltas = (cutoff_time - timestamps) / pd.Timedelta(hours=1)
    
    # Apply multiplier for recent window
    recent_mask = (deltas >= WEIGHT_START_DAYS * 24) & (deltas <= WEIGHT_END_DAYS * 24)
    weights[recent_mask] = WEIGHT_MULTIPLIER
    
    # Ignore outliers by setting weight to 0
    weights[is_outliers] = 0.0
    return weights

# --- 6. TCN Model Definition Functions ---
def tcn_block(input_layer, filters, kernel_size, dilation_rate):
    conv = Conv1D(
        filters,
        kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        activation='relu',
        kernel_initializer='he_normal'
    )(input_layer)
    if input_layer.shape[-1] != filters:
        input_layer = Conv1D(filters, 1, padding='same')(input_layer)
    return Add()([input_layer, conv])

def build_tcn_model(input_shape, filters, kernel_size, dilations):
    input_layer = Input(shape=input_shape)
    x = input_layer

    for d in dilations:
        x = tcn_block(x, filters, kernel_size, d)
        x = Activation('relu')(x) # Activation after residual connection

    last_input_value = Lambda(lambda x: x[:, -1, :])(input_layer)
    tcn_processed_output = Lambda(lambda x: x[:, -1, :])(x)
    combined_features = Concatenate()([tcn_processed_output, last_input_value])
    output_layer = Dense(1, activation='linear')(combined_features)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

class FileLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath, log_prefix=""):
        super().__init__()
        self.filepath = filepath
        self.log_prefix = log_prefix

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_msg = f"{self.log_prefix}Epoch {epoch+1} - " + " - ".join([f"{k}: {v:.6f}" for k, v in logs.items()])
        with open(self.filepath, 'a') as f:
            f.write(log_msg + "\n")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Initialize the log file by clearing it or writing a header
    with open(LOG_FILE_PATH, 'w') as f:
        f.write("=== TCN Forecast Training Logs ===\n")

    print("\n--- Starting Energy Consumption Forecasting Script ---")

    # --- GPU Availability and Active Testing Check ---
    print("\n--- Checking GPU Availability ---")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow detected GPUs: {gpus}")
    is_actively_using_gpu = False
    if gpus:
        try:
            with tf.device('/GPU:0'):
                a = tf.random.normal([100, 100])
                b = tf.random.normal([100, 100])
                c = tf.matmul(a, b)
            is_actively_using_gpu = True
            print("GPU active usage test: SUCCESS (TensorFlow is actively using the GPU)")
        except Exception as e:
            print(f"GPU active usage test: FAILED (Error running on GPU: {e})")
    else:
        print("GPU active usage test: FAILED (No GPU devices detected by TensorFlow)")

    # 1. Load Data
    df_raw = parse_api(START_DATE, END_DATE)
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
    print(f"Data reindexed to continuous hourly frequency. Shape: {df.shape}")

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
    clean_data(df, stricter_upper_bound)
    analyze_trend(df)
    analyze_seasonality(df)

    # --- TCN Scaling & Initial Train Preparation ---
    print("\n--- Scaling data and preparing Initial Training Dataset ---")
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit scaler only on clean training data to avoid leakage
    scaler.fit(train_subset['consumption_cleaned'].values.reshape(-1, 1))

    # Create training sequences (target indices from LOOK_BACK to initial_train_cutoff_idx)
    X_train, y_train, y_ts_train, y_outlier_train = create_sequences_range(
        df, LOOK_BACK, LOOK_BACK, initial_train_cutoff_idx, scaler
    )
    
    # Calculate sample weights for initial training
    cutoff_time_initial = df.loc[initial_train_cutoff_idx, 'timestamp']
    sample_weights_initial = get_sample_weights(y_ts_train, cutoff_time_initial, y_outlier_train)

    # --- Initial TCN Model Training ---
    print("\n--- Training Initial TCN Model ---")
    input_shape = (LOOK_BACK, 1)
    model_tcn_initial = build_tcn_model(input_shape, TCN_FILTERS, TCN_KERNEL_SIZE, TCN_DILATIONS)
    model_tcn_initial.summary()

    if len(X_train) > 0:
        model_tcn_initial.fit(
            X_train, y_train,
            sample_weight=sample_weights_initial,
            epochs=INITIAL_TRAINING_EPOCHS,
            batch_size=32,
            validation_split=0.2,
            verbose=1,
            callbacks=[FileLoggerCallback(LOG_FILE_PATH, "Initial Training - ")]
        )
        print("Initial TCN model trained successfully.")
    else:
        print("Not enough data for initial TCN training. Skipping.")
        exit()

    # --- TCN Rolling Forecast with Retraining ---
    print("\n--- Starting TCN Rolling Forecast with Retraining ---")
    actuals = []
    predictions = []
    timestamps_forecasted = []

    current_time = PREDICTION_START_DATE
    while current_time <= PREDICTION_END_DATE:
        matching_rows = df[df['timestamp'] == current_time]
        if matching_rows.empty:
            current_time += pd.Timedelta(hours=1)
            continue
            
        row = matching_rows.iloc[0]
        if row['is_outlier']:
            print(f"Skipping prediction for {current_time} as it is marked as a right-tail outlier.")
            current_time += pd.Timedelta(hours=1)
            continue

        current_time_idx = matching_rows.index[0]
        
        # Retrain on all data up to current_time - 1 hour
        start_idx = LOOK_BACK
        end_idx = current_time_idx - 1
        
        if start_idx <= end_idx:
            print(f"\n--- Retraining TCN model from scratch for {current_time} ---")
            
            # Re-initialize the model from scratch
            model_tcn = build_tcn_model(input_shape, TCN_FILTERS, TCN_KERNEL_SIZE, TCN_DILATIONS)
            
            X_train_roll, y_train_roll, y_ts_roll, y_outlier_roll = create_sequences_range(
                df, LOOK_BACK, start_idx, end_idx, scaler
            )
            
            cutoff_time = df.loc[end_idx, 'timestamp']
            sample_weights_roll = get_sample_weights(y_ts_roll, cutoff_time, y_outlier_roll)
            
            # Retrain model from scratch
            model_tcn.fit(
                X_train_roll, y_train_roll,
                sample_weight=sample_weights_roll,
                epochs=ROLLING_FORECAST_EPOCHS,
                batch_size=32,
                verbose=1,
                callbacks=[FileLoggerCallback(LOG_FILE_PATH, f"[{current_time}] Retraining - ")]
            )

        # Predict the next hour
        feat = df.loc[current_time_idx - LOOK_BACK : current_time_idx - 1, 'consumption_cleaned'].values.reshape(-1, 1)
        feat_scaled = scaler.transform(feat).reshape(1, LOOK_BACK, 1)
        
        predicted_scaled = model_tcn.predict(feat_scaled, verbose=0)
        predicted_unscaled = scaler.inverse_transform(predicted_scaled)[0][0]
        
        predictions.append(predicted_unscaled)
        actuals.append(df.loc[current_time_idx, 'consumption'])
        timestamps_forecasted.append(current_time)

        current_time += pd.Timedelta(hours=1)

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
    plt.title('Daily Mean Absolute Error (MAE) for TCN Rolling Forecast')
    plt.xlabel('Date')
    plt.ylabel('MAE')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'daily_rolling_mae.png'))
    plt.close()
    print(f"Saved daily rolling MAE plot to {OUTPUT_DIR}/daily_rolling_mae.png")

    unique_forecast_dates = df_rolling_forecast['date'].unique()
    for date in unique_forecast_dates:
        daily_data = df_rolling_forecast[df_rolling_forecast['date'] == date].copy()

        plt.figure(figsize=(15, 7))
        plt.plot(daily_data['timestamp'], daily_data['actual_consumption'], label='Actual Consumption', color='blue')
        plt.plot(daily_data['timestamp'], daily_data['predicted_consumption'], label='Predicted Consumption', color='green', alpha=0.7)

        uncertainty_margin = min(overall_rolling_mae, UNCERTAINTY_CAP_MAE)
        daily_data['predicted_lower_bound'] = daily_data['predicted_consumption'] - uncertainty_margin
        daily_data['predicted_upper_bound'] = daily_data['predicted_consumption'] + uncertainty_margin

        plt.fill_between(
            daily_data['timestamp'],
            daily_data['predicted_lower_bound'],
            daily_data['predicted_upper_bound'],
            color='dimgray',
            alpha=0.6,
            label=f'Uncertainty Range (MAE based, capped at +/- {uncertainty_margin:.2f})'
        )

        plt.title(f'TCN Rolling Forecast vs. Actual Consumption for {date}')
        plt.xlabel('Time of Day')
        plt.ylabel('Consumption')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'daily_forecast_{date}.png'))
        plt.close()
    print(f"Saved daily forecast plots to {OUTPUT_DIR}/")

    print("\n--- Script execution completed. ---")
