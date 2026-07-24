import requests
import pandas as pd


def parse_api(start_date, end_date, hour=None):
    try:
        url = (
            f"https://ap.elementsenergies.com/api/fetchHConsWAvg"
            f"?startdate={start_date}&enddate={end_date}&msn=67002675"
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
