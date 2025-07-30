import pandas as pd
from openpyxl import Workbook
import os

OUTPUT_DIR = "outputs"  # Set your desired output folder

def load_single_dataset(file_path):
    dataset_name = file_path.split("\\")[-1].split(".")[0]
    df = pd.read_csv(file_path)
    return df, dataset_name

def save_to_excel(df, filename='output.xlsx'):
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
        df.describe(include='all').reset_index().to_excel(writer, index=False, sheet_name='Summary')

        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            pivot = pd.pivot_table(
                df,
                values=numeric_columns,
                index=[col for col in df.columns if col not in numeric_columns],
                aggfunc='sum'
            )
            pivot.to_excel(writer, sheet_name='Pivot Table')
    return filepath
