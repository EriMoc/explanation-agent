from utils.io_utils import load_single_dataset
from utils.analysis_utils import detect_anomalies, create_anomalies_dashboard_quarters

# Načítať CSV zo súboru
df, _ = load_single_dataset("inputs/Total_Applicable_Cost_dataset.csv")

# Detekovať anomálie
df = detect_anomalies(df, amount_column="MKT Total Cost")

# Spustiť interaktívny Dash dashboard
create_anomalies_dashboard_quarters(
    df,
    value_column="MKT Total Cost",
    period_column="Period",
    title="📉 MKT Total Cost Anomalies Dashboard"
)
