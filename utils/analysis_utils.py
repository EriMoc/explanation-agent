from sklearn.ensemble import IsolationForest
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output




def detect_anomalies(df, amount_column=None, contamination=0.05):
    if not amount_column:
        raise ValueError("amount_column must be specified")
    if amount_column not in df.columns:
        raise ValueError(f"{amount_column} not found in DataFrame")
    model = IsolationForest(contamination=contamination, random_state=42)
    amounts = df[[amount_column]].values
    df['anomaly_score'] = model.fit_predict(amounts)
    df['is_anomaly'] = df['anomaly_score'] == -1

    # Add explanation column
    explanations = []
    mean = df[amount_column].mean()
    std = df[amount_column].std()
    for idx, row in df.iterrows():
        if row['is_anomaly']:
            if abs(row[amount_column] - mean) > 2 * std:
                explanation = "Value is a strong outlier (>{:.1f} std dev from mean)".format(abs(row[amount_column] - mean) / std)
            else:
                explanation = "Detected as anomaly by model"
        else:
            explanation = "Normal value"
        explanations.append(explanation)
    df['anomaly_explanation'] = explanations

    return df


# def forecast(df, date_column, value_column, forecast_periods=3):
#     # Try to parse as datetime; if fails, try quarterly
#     is_quarter_format = False
#     try:
#         df['ds'] = pd.to_datetime(df[date_column])
#     except Exception:
#         is_quarter_format = True

#         def parse_quarter(qstr):
#             # Example: 'Q3 FY25' -> Timestamp('2025-09-30')
#             parts = qstr.replace('FY', '').replace(' ', '').upper()
#             quarter = parts[:2]  # 'Q3'
#             year = '20' + parts[2:]  # '25' -> '2025'
#             return pd.Period(year + quarter, freq='Q').end_time

#         df['ds'] = df[date_column].apply(parse_quarter)

#     df = df.rename(columns={value_column: 'y'})

#     model = Prophet(seasonality_mode='multiplicative')
#     model.fit(df[['ds', 'y']])

#     # Use quarterly or monthly frequency
#     freq = 'QE' if is_quarter_format else 'M'
#     future = model.make_future_dataframe(periods=forecast_periods, freq=freq)
#     forecast_df = model.predict(future)

#     last_actual_date = df['ds'].max()
#     forecast_df['is_forecasted'] = forecast_df['ds'] >= last_actual_date

#     # === Add back period string format if applicable ===
#     if is_quarter_format:
#         def datetime_to_quarter_label(dt):
#             p = pd.Period(dt, freq='Q')
#             q = f"Q{p.quarter}"
#             fy = f"FY{str(p.year)[2:]}"
#             return f"{q} {fy}"

#         forecast_df['quarter'] = forecast_df['ds'].apply(datetime_to_quarter_label)

#     return forecast_df

def forecast(unfiltered_df, date_column, value_column, forecast_periods=2, category_columns=None):
    if category_columns is None:
        category_columns = []

    def parse_quarter(qstr):
        parts = qstr.replace('FY', '').replace(' ', '').upper()
        quarter = parts[:2]
        year = '20' + parts[2:]
        return pd.Period(year + quarter, freq='Q').end_time

    def datetime_to_quarter_label(dt):
        p = pd.Period(dt, freq='Q')
        q = f"Q{p.quarter}"
        fy = f"FY{str(p.year)[2:]}"
        return f"{q} {fy}"

    all_forecasts = []

    # 🔁 Forecast pre každú kombináciu kategórií
    grouped = unfiltered_df.groupby(category_columns)
    for keys, group_df in grouped:
        subset = group_df.copy()
        subset['ds'] = subset[date_column].apply(parse_quarter)
        subset = subset.rename(columns={value_column: 'y'})

        if subset[['ds', 'y']].dropna().shape[0] < 2:
            continue  # málo dát, preskočiť

        model = Prophet(seasonality_mode='multiplicative')
        model.fit(subset[['ds', 'y']])

        future = model.make_future_dataframe(periods=forecast_periods, freq='QE')
        forecast_df = model.predict(future)

        last_actual_date = subset['ds'].max()
        forecast_df['is_forecasted'] = forecast_df['ds'] > last_actual_date
        forecast_df['quarter'] = forecast_df['ds'].apply(datetime_to_quarter_label)

        # Pridaj späť info o skupine
        if isinstance(keys, tuple):
            for col, val in zip(category_columns, keys):
                forecast_df[col] = val
        else:
            forecast_df[category_columns[0]] = keys

        all_forecasts.append(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'is_forecasted', 'quarter'] + category_columns])

    # 🔁 Forecast pre celý dataset
    df_all = unfiltered_df.copy()
    df_all['ds'] = df_all[date_column].apply(parse_quarter)
    df_all = df_all.rename(columns={value_column: 'y'})

    if df_all[['ds', 'y']].dropna().shape[0] >= 2:
        model_all = Prophet(seasonality_mode='multiplicative')
        model_all.fit(df_all[['ds', 'y']])
        future_all = model_all.make_future_dataframe(periods=forecast_periods, freq='Q')
        forecast_all = model_all.predict(future_all)
        forecast_all['is_forecasted'] = forecast_all['ds'] > df_all['ds'].max()
        forecast_all['quarter'] = forecast_all['ds'].apply(datetime_to_quarter_label)

        for col in category_columns:
            forecast_all[col] = 'TOTAL'

        all_forecasts.append(forecast_all[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'is_forecasted', 'quarter'] + category_columns])

    return pd.concat(all_forecasts, ignore_index=True)


# def visualize_anomalies(
#     df,
#     anomalies=None,
#     value_column='y',
#     date_column='ds',
#     title="Anomalies Dashboard"
# ):
#     """
#     Interactive dashboard for time series, anomalies, and forecast.
#     """

#     import numpy as np

#     fig = go.Figure()

#     # Shared customdata: every column except the index
#     # other_cols = [col for col in df.columns if col not in [date_column, value_column]]
#     # df["__hover_text"] = df.apply(lambda row: "<br>".join(f"{col}: {row[col]}" for col in other_cols), axis=1)


#     # Plot anomalies if provided
#     if anomalies is not None and not anomalies.empty:
#         # Plot points where is_anomaly is True (red 'x')
#         anomalies_true = anomalies[anomalies['is_anomaly'] == True]
#         if not anomalies_true.empty:
#             # hover_text = anomalies_true.apply(lambda row: "<br>".join(f"{col}: {row[col]}" for col in anomalies_true.columns), axis=1) shows all columns
#             fig.add_trace(go.Scatter(
#                 x=anomalies_true[date_column], y=anomalies_true[value_column],
#                 mode='markers',
#                 marker=dict(color='red', size=10, symbol='x'),
#                 name='Anomalies',
#                 text=anomalies_true['anomaly_explanation'],  # Show explanation on hover
#                 hoverinfo='text+x+y'
#             ))

#         # Plot points where is_anomaly is False (blue dot)
#         anomalies_false = anomalies[anomalies['is_anomaly'] == False]
#         if not anomalies_false.empty:
#             fig.add_trace(go.Scatter(
#                 x=anomalies_false[date_column], y=anomalies_false[value_column],
#                 mode='markers',
#                 marker=dict(color='blue', size=6, symbol='circle'),
#                 name='Normal'
#             ))


#     fig.update_layout(
#         title=title,
#         xaxis_title=date_column,
#         yaxis_title=value_column,
#         template='plotly_white'
#     )
    
#     fig.show()

def create_anomalies_dashboard_quarters(df, value_column='y', period_column='Period', title="Anomalies Dashboard"):
    app = Dash(__name__)
    df = df.copy()

    # Create datetime index from quarter strings
    df['ds'] = df[period_column].apply(quarter_to_timestamp)
    df['__period'] = df['ds'].apply(lambda ts: pd.Period(ts, freq='Q'))
    df = df.sort_values('__period')

    recent_frac = 0.2
    split_idx = int((1 - recent_frac) * len(df))
    recent_df = df.iloc[split_idx:]

    # Compute shared y-axis range
    y_min = df[value_column].min()
    y_max = df[value_column].max()

    app.layout = html.Div([
        html.H2(title),

        html.Div([
            dcc.Graph(id='subplot1', style={'width': '20vw', 'height': '800px', 'display': 'inline-block'}),
            dcc.Graph(id='subplot2', style={'width': '75vw', 'height': '800px', 'display': 'inline-block'}),
        ])
    ])

    @app.callback(
        Output('subplot2', 'figure'),
        Input('subplot1', 'clickData')
    )
    def update_subplot2(clickData):
        if clickData is None:
            return go.Figure()

        point_data = clickData['points'][0]
        clicked_label = point_data['x']
        clicked_y = point_data['y']

        # Match by both x and y to get the exact row
        selected_row = recent_df[
            (recent_df[period_column] == clicked_label) &
            (recent_df[value_column] == clicked_y)
        ].iloc[0]

        cost_center = selected_row['Cost Center']
        cost_bucket = selected_row['CostBucket']

        filtered_df = df[
            (df['Cost Center'] == cost_center) &
            (df['CostBucket'] == cost_bucket) &
            (df[period_column] != clicked_label)
        ].sort_values('__period', ascending=False)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=filtered_df[period_column],
            y=filtered_df[value_column],
            mode='lines+markers',
            name='Matching Category'
        ))

        anomalies = filtered_df[filtered_df['is_anomaly'] == True]
        fig2.add_trace(go.Scatter(
            x=anomalies[period_column],
            y=anomalies[value_column],
            mode='markers',
            marker=dict(color='red', size=10, symbol='x'),
            name='Anomalies',
            text=anomalies['anomaly_explanation'],
            hoverinfo='text+x+y'
        ))

        fig2.update_layout(
            title=f"Other Periods for {cost_center} / {cost_bucket}",
            xaxis=dict(categoryorder='array', categoryarray=filtered_df[period_column].tolist()),
            yaxis=dict(range=[y_min, y_max])
        )
        return fig2

    @app.callback(
        Output('subplot1', 'figure'),
        Input('subplot1', 'id')
    )
    def update_subplot1(_):
        fig1 = go.Figure()
        non_anomalies = recent_df[recent_df['is_anomaly'] == False]
        fig1.add_trace(go.Scatter(
            x=non_anomalies[period_column],
            y=non_anomalies[value_column],
            mode='lines+markers',
            name='Recent',
            marker=dict(size=6)
        ))

        recent_anomalies = recent_df[recent_df['is_anomaly'] == True]
        fig1.add_trace(go.Scatter(
            x=recent_anomalies[period_column],
            y=recent_anomalies[value_column],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10, symbol='x'),
            text=recent_anomalies['anomaly_explanation'],
            hoverinfo='text+x+y'
        ))

        fig1.update_layout(title="Recent Period Overview (click a point)", clickmode='event+select', yaxis=dict(range=[y_min, y_max]))
        return fig1


    app.run(debug=True)

# def visualize_forecast(df, forecast_df, value_column='y', input_date_col='Period', forecast_date_col='ds', title="Forecast Dashboard"):
#     fig = go.Figure()

#     df_sorted = df.sort_values(by='ds', ascending=True)
#     # forecast_df_sorted = forecast_df.sort_values(by=forecast_date_col, ascending=True)
#     forecast_df_sorted = forecast_df

#     fig.add_trace(go.Scatter(
#         x=df_sorted[input_date_col], y=df_sorted[value_column],
#         mode='lines+markers',
#         name='Original Data',
#         line=dict(color='blue')
#     ))
#     # Add forecasted data
#     forecast_df = forecast_df[forecast_df['is_forecasted'] == True]
#     fig.add_trace(go.Scatter(
#         x=forecast_df[forecast_date_col], y=forecast_df['yhat'],
#         mode='lines+markers',
#         name='Forecasted Data',
#         line=dict(color='orange')
#     ))

#     fig.add_traces([
#         go.Scatter(
#             x=forecast_df_sorted[forecast_date_col], y=forecast_df_sorted['yhat_lower'],
#             mode='lines',
#             name='Lower Bound',
#             line=dict(color='lightgray', dash='dash')
#         ),
#         go.Scatter(
#             x=forecast_df_sorted[forecast_date_col], y=forecast_df_sorted['yhat_upper'],
#             mode='lines',
#             name='Upper Bound',
#             line=dict(color='lightgray', dash='dash')
#         )
#     ])

#     fig.update_layout(
#         title=title,
#         xaxis_title=input_date_col,
#         yaxis_title=value_column,
#         template='plotly_white'
#     )

#     fig.show()

def sort_by_period(df: pd.DataFrame, period_col: str = 'Period') -> pd.DataFrame:
    """
    Sorts a DataFrame by a 'Period' column with format like 'Q3 FY25'.

    Args:
        df (pd.DataFrame): Input DataFrame with a period column.
        period_col (str): Name of the column containing the period strings.

    Returns:
        pd.DataFrame: Sorted DataFrame.
    """
    # Extract quarter and fiscal year
    def parse_period(period_str):
        try:
            quarter = int(period_str[1])
            year_str = period_str.split('FY')[1]
            # Convert to calendar year (assumes FY25 means 2025)
            year = int(year_str) + 2000
            return year * 10 + quarter  # or (year, quarter)
        except:
            return float('inf')  # put unparseable entries at the end

    df = df.copy()
    df['_period_sort_key'] = df[period_col].apply(parse_period)
    df_sorted = df.sort_values('_period_sort_key').drop(columns=['_period_sort_key'])
    return df_sorted

def visualize_forecast(df, forecast_df, period_column, value_column, title="Forecast Overview"):
    fig = go.Figure()

    # Rozdelenie na historické a forecastované
    forecasted = forecast_df[forecast_df['is_forecasted'] == True]
    df = sort_by_period(df, period_col=period_column)
    # Historické (modré)
    fig.add_trace(go.Scatter(
        x=df[period_column],
        y=df[value_column],
        mode='markers',
        name='Actual',
        line=dict(color='blue')
    ))

    # Forecastované (oranžové)
    fig.add_trace(go.Scatter(
        x=forecasted['quarter'],
        y=forecasted['yhat'],
        mode='markers',
        name='Forecast',
        line=dict(color='orange')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Quarter',
        yaxis_title='Value',
        template='plotly_white'
    )

    fig.show()

def quarter_to_timestamp(qstr):
    parts = qstr.replace('FY', '').replace(' ', '').upper()
    quarter = parts[:2]  # 'Q3'
    year = '20' + parts[2:]  # '25' -> '2025'
    return pd.Period(year + quarter, freq='Q').end_time


