import numpy as np

METRIC_REGISTRY = {
    "GMV per seller": {
        "column": "GMV_TOTAL",
        "source": "main",
        "type": "continuous",
        "fmt": ".2f",
    },
    "Qty sold per seller": {
        "column": "QTY_TOTAL",
        "source": "main",
        "type": "count",
        "fmt": ".2f",
    },
    "7D Listing Conversion": {
        "column": "conv_7d",
        "source": "listings",
        "type": "binary",
        "fmt": ".4f",
    },
    "14D Listing Conversion": {
        "column": "conv_14d",
        "source": "listings",
        "type": "binary",
        "fmt": ".4f",
    },
    "Listings per Seller": {
        "column": "listings_per_seller",
        "source": "listings",
        "type": "count",
        "fmt": ".2f",
    },
    "Qty per Listing": {
        "column": "qty_per_listing",
        "source": "listings",
        "type": "continuous",
        "fmt": ".4f",
    },
}

SEGMENT_REGISTRY = {
    "Vertical": "VERTICAL",
    "ASP Bucket": "ASP_BUCKET",
    "Listing Tool": "LISTING_TOOL",
}


def build_derived_columns(lstg_df):
    df = lstg_df.copy()
    df["conv_7d"] = df["LISTINGS_CONV_7D"] / df["LISTINGS_CNT"].replace(0, np.nan)
    df["conv_14d"] = df["LISTINGS_CONV_14D"] / df["LISTINGS_CNT"].replace(0, np.nan)
    df["listings_per_seller"] = df["LISTINGS_CNT"]
    if "QTY_TOTAL" in df.columns:
        df["qty_per_listing"] = df["QTY_TOTAL"] / df["LISTINGS_CNT"].replace(0, np.nan)
    return df
