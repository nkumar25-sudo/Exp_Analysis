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

    # Merge QTY_TOTAL into listings_df so qty_per_listing can be computed.
    # In production (Snowflake), both columns will come from one query and this merge is not needed.
    listings_df = listings_df.merge(
        main_df[["USER_ID", "QTY_TOTAL"]], on="USER_ID", how="left"
    )
    listings_df = build_derived_columns(listings_df)
    return main_df, listings_df
