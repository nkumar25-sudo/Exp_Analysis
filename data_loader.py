from pathlib import Path
import pandas as pd
from metrics import build_derived_columns

_DATA_DIR = Path(__file__).parent


def load_experiment_data(experiment_id: str, treatment_id: str, control_id: str):
    """
    Load experiment data for the given IDs.

    Currently reads from the two local CSVs. Replace the body of this function
    with a real Snowflake / Touchstone API call when ready — the return contract
    (main_df, listings_df with TREATMENT_GROUP == 'CONTROL' | 'TREATMENT') stays the same.

    Returns: (main_df, listings_df)
    """
    main_df = pd.read_csv(_DATA_DIR / "seller_results.csv")
    listings_df = pd.read_csv(_DATA_DIR / "seller_results1.csv")
    listings_df = build_derived_columns(listings_df)
    return main_df, listings_df
