# prompt_generator.py

import pandas as pd

def generate_system_prompt(df, df_name="df"):
    def map_dtype(dtype):
        if pd.api.types.is_string_dtype(dtype):
            return "string"
        elif pd.api.types.is_numeric_dtype(dtype):
            return "float" if pd.api.types.is_float_dtype(dtype) else "integer"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"
        else:
            return str(dtype)

    def describe_columns(df):
        descriptions = []
        for col in df.columns:
            dtype = map_dtype(df[col].dtype)
            sample_vals = df[col].dropna().unique()[:3]
            sample_str = ", ".join(repr(v) for v in sample_vals)
            descriptions.append(f"- {col} ({dtype}): e.g. {sample_str}")
        return "\n".join(descriptions)

    def identify_special_columns(df):
        time_cols = [col for col in df.columns if 'period' in col.lower() or 'date' in col.lower()]
        numeric_cols = [col for col in df.select_dtypes(include='number').columns]
        category_cols = [col for col in df.select_dtypes(include='object').columns]
        return time_cols, numeric_cols, category_cols

    time_cols, numeric_cols, category_cols = identify_special_columns(df)

    prompt = f"""
SYSTEM_PROMPT = \"\"\"
You are a data analyst with access to a pandas DataFrame called `{df_name}` containing Total Applicable Cost data.

🧾 Dataset Summary:
- Rows: {len(df):,}
- Columns: {len(df.columns)}

📅 Time Column(s):
{', '.join(time_cols) if time_cols else 'None detected'}

💰 Numeric Columns:
{', '.join(numeric_cols)}

🏷️ Categorical Columns:
{', '.join(category_cols)}

🔎 Filtering Guidance:
When filtering by supplier-like fields (e.g. 'Dell/EMC'), use partial matching:
    {df_name}['Dell/EMC'].str.contains('keyword', case=False, na=False)

📈 Instructions:
- Use only provided helper functions to analyze the data. *DO NOT redefine them**
- Use pivot tables to summarize costs over time or by category.
- For anomaly detection, compare values **within each category** across time periods, ensuring that comparisons do not cross different categories.
- Use Prophet or ARIMA for time-series forecasting without filtering out other categorical columns. **DO NOT modify Period data to datetime format**
- Apply Isolation Forest for anomaly detection within each category over time.
- Save output to Excel if reporting is needed.

⚠️ Do NOT redefine the following helper functions — use them as-is:
- detect_anomalies(df, amount_column, contamination=0.05)
- forecast(unfiltered_df, date_column, value_column, forecast_periods=3, category_columns=[]) **no need to modify Period data**
- save_to_excel(df, filename='output.xlsx')
- create_anomalies_dashboard_quarters(df, value_column='y', period_column='Period', title="Anomalies Dashboard")
- visualize_forecast(df, forecast_df, period_column='Period', value_column, title="Forecast Dashboard")

🔧 Full Column Details:
{describe_columns(df)}

⚠️ Only return executable Python code — no markdown, no explanations.
\"\"\"
"""
    return prompt

# 📈 Instructions:
# - Use pivot tables to summarize costs over time or by category.
# - Before forecasting or visualizing, rename the selected date column to `ds` and the value column to `y`.
# - Use pivot tables to summarize costs over time or categories, if requested.
# - For anomaly detection, use Isolation Forest to compare values across time within each category.
# - Forecast trends using Prophet or ARIMA on time-series data per category.

# ⚠️ Do NOT redefine the following helper functions — use them as-is:
# - detect_anomalies(df, amount_column, contamination=0.05)
# - forecast(df, date_column, value_column, forecast_periods=3)
# - save_to_excel(df, filename='output.xlsx')
# - visualize_data_and_anomalies(df, anomalies=None, value_column='y', date_column='ds', forecast_df=None, title="Visualization Dashboard")
