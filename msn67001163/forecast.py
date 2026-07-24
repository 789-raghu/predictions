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
import os

# --- Configuration Parameters ---
START_DATE = "2025-05-16"
END_DATE = "2026-07-20"
PREDICTION_START_DATE = pd.Timestamp('2026-07-01 00:00:00')
PREDICTION_END_DATE = pd.Timestamp('2026-07-20 23:00:00')
LOOK_BACK = 24  # Look-back period for TCN
TCN_FILTERS = 64
TCN_KERNEL_SIZE = 2
TCN_DILATIONS = [1, 2, 4, 8, 16]
INITIAL_TRAINING_EPOCHS = 50
ROLLING_FORECAST_EPOCHS = 5
UNCERTAINTY_CAP_MAE = 5.0 # Cap uncertainty margin at +/- 5 units

OUTPUT_DIR = 'tcn_forecast_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Results will be saved in the directory: {OUTPUT_DIR}/")

# --- 1. Data Loading Function ---
def parse_api(start_date, end_date, hour=None):
    try:
        url = (
            f"https://ap.elementsenergies.com/api/fetchHConsWAvg"
            f"?startdate={start_date}&enddate={end_date}&msn=67001163"
        )
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

# --- 2. Data Cleaning and Outlier Removal ---
def clean_data(df_raw):
    print("\n--- Starting Data Cleaning and Outlier Removal ---")
    df = df_raw.copy()

    print("Descriptive statistics for 'consumption' (raw data):")
    print(df['consumption'].describe())

    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df['consumption'])
    plt.title('Box Plot of Raw Consumption')
    plt.ylabel('Consumption')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'box_plot_raw_consumption.png'))
    plt.close()
    print(f"Saved box plot to {OUTPUT_DIR}/box_plot_raw_consumption.png")

    Q1 = df['consumption'].quantile(0.25)
    Q3 = df['consumption'].quantile(0.75)
    IQR = Q3 - Q1

    # Stricter upper bound for removal (Q3 + 3 * IQR)
    stricter_upper_bound = Q3 + 3 * IQR

    print(f"\nUsing the following thresholds for outlier removal:")
    print(f"  - Lower bound (not applied for removal): No left-tail outliers will be removed.")
    print(f"  - Upper bound (for very extreme right-tail outliers): {stricter_upper_bound:.2f} (Q3 + 3 * IQR)")

    extreme_right_tail_outliers = df[df['consumption'] > stricter_upper_bound]
    print(f"Number of VERY extreme right-tail outliers detected: {len(extreme_right_tail_outliers)}")

    df_cleaned = df[df['consumption'] <= stricter_upper_bound].copy()

    print(f"Original DataFrame shape: {df.shape}")
    print(f"Cleaned DataFrame shape (only very extreme right-tail outliers removed): {df_cleaned.shape}")
    print("Descriptive statistics after removing ONLY very extreme right-tail outliers:")
    print(df_cleaned['consumption'].describe())

    plt.figure(figsize=(15, 7))
    sns.lineplot(x='timestamp', y='consumption', data=df_cleaned)
    plt.title('Consumption Over Time (Cleaned Data)')
    plt.xlabel('Timestamp')
    plt.ylabel('Consumption')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'cleaned_consumption_line_plot.png'))
    plt.close()
    print(f"Saved cleaned data line plot to {OUTPUT_DIR}/cleaned_consumption_line_plot.png")

    return df_cleaned

# --- 3. Trend Analysis ---
def analyze_trend(df_cleaned):
    print("\n--- Starting Trend Analysis ---")
    df_trend = df_cleaned.copy()
    df_trend['rolling_mean_168h'] = df_trend['consumption'].rolling(window=168).mean()

    plt.figure(figsize=(18, 8))
    sns.lineplot(x='timestamp', y='consumption', data=df_trend, label='Original Consumption', alpha=0.7)
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
def analyze_seasonality(df_cleaned):
    print("\n--- Starting Seasonality Analysis ---")
    df_seasonal = df_cleaned.copy()
    df_seasonal['timestamp'] = pd.to_datetime(df_seasonal['timestamp'])
    df_seasonal = df_seasonal.set_index('timestamp')
    df_seasonal = df_seasonal.asfreq('h') # Set frequency to hourly
    df_seasonal['consumption'] = df_seasonal['consumption'].ffill().bfill() # Fill potential NaN

    decomposition = seasonal_decompose(df_seasonal['consumption'], model='additive', period=24)

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

    return df_seasonal # Return for potential further use if needed

# --- 5. TCN Data Preparation Functions ---
def create_sequences(data, look_back):
    X, y = [], []
    if len(data) <= look_back:
        return np.array(X), np.array(y)
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

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

# --- Main Execution Block ---
if __name__ == "__main__":
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
    df = parse_api(START_DATE, END_DATE)
    if df.empty:
        print("Exiting due to empty DataFrame from API call.")
        exit()
    print(f"Raw data loaded. Shape: {df.shape}")

    # 2. Clean Data
    df_cleaned = clean_data(df)
    df_tcn = df_cleaned.reset_index() # For TCN data prep, ensure timestamp is a column

    # 3. Trend Analysis
    analyze_trend(df_cleaned.copy()) # Pass a copy to avoid modifying original df_cleaned index

    # 4. Seasonality Analysis
    # Ensure df_cleaned is not indexed by timestamp before passing it to analyze_seasonality
    if 'timestamp' not in df_cleaned.columns:
        df_temp = df_cleaned.reset_index()
    else:
        df_temp = df_cleaned.copy()
    analyze_seasonality(df_temp)

    # --- TCN Rolling Forecast Preparation ---
    print("\n--- Preparing data for TCN Rolling Forecast ---")
    data = df_tcn['consumption'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    initial_train_end_date_ts = pd.Timestamp('2026-06-30 23:00:00')
    initial_train_cutoff_idx = df_tcn[df_tcn['timestamp'] == initial_train_end_date_ts].index[0]

    data_initial_train_scaled = data_scaled[:initial_train_cutoff_idx + 1]
    X_train_initial, y_train_initial = create_sequences(data_initial_train_scaled, LOOK_BACK)
    if len(X_train_initial) > 0:
        X_train_initial = X_train_initial.reshape(X_train_initial.shape[0], X_train_initial.shape[1], 1)
    
    # --- Initial TCN Model Training ---
    print("\n--- Training Initial TCN Model ---")
    input_shape = (LOOK_BACK, 1)
    model_tcn_initial = build_tcn_model(input_shape, TCN_FILTERS, TCN_KERNEL_SIZE, TCN_DILATIONS)
    model_tcn_initial.summary()

    if len(X_train_initial) > 0:
        model_tcn_initial.fit(
            X_train_initial, y_train_initial,
            epochs=INITIAL_TRAINING_EPOCHS,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        print("Initial TCN model trained successfully.")
    else:
        print("Not enough data for initial TCN training. Skipping.")

    # --- TCN Rolling Forecast Implementation ---
    print("\n--- Starting TCN Rolling Forecast ---")
    actuals = []
    predictions = []
    timestamps_forecasted = []

    current_time = PREDICTION_START_DATE
    while current_time <= PREDICTION_END_DATE:
        # Skip prediction if the timestamp was removed as an outlier
        matching_rows = df_tcn[df_tcn['timestamp'] == current_time]
        if matching_rows.empty:
            print(f"Skipping prediction for {current_time} as it was removed as an outlier.")
            current_time += pd.Timedelta(hours=1)
            continue

        current_time_idx_in_df = matching_rows.index[0]
        current_training_data_scaled = data_scaled[:current_time_idx_in_df]

        X_train_roll, y_train_roll = create_sequences(current_training_data_scaled, LOOK_BACK)

        if len(X_train_roll) > 0:
            X_train_roll = X_train_roll.reshape(X_train_roll.shape[0], X_train_roll.shape[1], 1)

            model_tcn_roll = build_tcn_model(input_shape, TCN_FILTERS, TCN_KERNEL_SIZE, TCN_DILATIONS)
            model_tcn_roll.fit(X_train_roll, y_train_roll, epochs=ROLLING_FORECAST_EPOCHS, batch_size=32, verbose=1)

            X_current_hour = current_training_data_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
            predicted_scaled = model_tcn_roll.predict(X_current_hour, verbose=1)
            predicted_unscaled = scaler.inverse_transform(predicted_scaled)[0][0]
            predictions.append(predicted_unscaled)

            actual_unscaled = df_tcn.loc[current_time_idx_in_df, 'consumption']
            actuals.append(actual_unscaled)
            timestamps_forecasted.append(current_time)
        else:
            print(f"Skipping prediction for {current_time} as training data is insufficient.")

        if current_time.hour == 0:
            print(f"  Forecasted for {current_time.date()}")

        current_time += pd.Timedelta(hours=1)

    print("Rolling forecast completed.")

    # --- Post-processing and Visualization (Rolling Forecast) ---
    df_rolling_forecast = pd.DataFrame({
        'timestamp': timestamps_forecasted,
        'actual_consumption': actuals,
        'predicted_consumption': predictions
    })

    overall_rolling_mae = mean_absolute_error(df_rolling_forecast['actual_consumption'], df_rolling_forecast['predicted_consumption'])
    print(f"\nOverall Rolling Forecast MAE: {overall_rolling_mae:.2f}")

    df_rolling_forecast['date'] = df_rolling_forecast['timestamp'].dt.date
    daily_rolling_mae = df_rolling_forecast.groupby('date').apply(lambda x: mean_absolute_error(x['actual_consumption'], x['predicted_consumption']), include_groups=False).reset_index(name='MAE')

    print("\nDaily Mean Absolute Error (MAE) for Rolling Forecast:")
    print(daily_rolling_mae.head())

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