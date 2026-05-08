import numpy as np
import pandas as pd


def bootstrap_mean_diff(series_t, series_c, n_boot=10000, seed=42):
    s_t = series_t.dropna().to_numpy()
    s_c = series_c.dropna().to_numpy()
    if len(s_t) == 0 or len(s_c) == 0:
        return None

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = (
            rng.choice(s_t, len(s_t), replace=True).mean()
            - rng.choice(s_c, len(s_c), replace=True).mean()
        )

    obs = s_t.mean() - s_c.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p = np.mean(np.abs(diffs - obs) >= abs(obs))

    return {
        "control_mean": s_c.mean(),
        "treatment_mean": s_t.mean(),
        "observed_diff": obs,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p,
    }


def run_overall(df, metric_col, fmt, n_boot=10000):
    c = df[df["TREATMENT_GROUP"] == "CONTROL"][metric_col]
    t = df[df["TREATMENT_GROUP"] == "TREATMENT"][metric_col]
    res = bootstrap_mean_diff(t, c, n_boot=n_boot)
    if res is None:
        return None
    lift_pct = (res["observed_diff"] / res["control_mean"] * 100) if res["control_mean"] != 0 else 0
    return {
        "Control Mean": f"{res['control_mean']:{fmt}}",
        "Treatment Mean": f"{res['treatment_mean']:{fmt}}",
        "Lift": f"{res['observed_diff']:{fmt}} ({lift_pct:+.1f}%)",
        "95% CI": f"[{res['ci_low']:{fmt}}, {res['ci_high']:{fmt}}]",
        "p-value": f"{res['p_value']:.4f}",
        "_diff": res["observed_diff"],
        "_p": res["p_value"],
    }


def run_segment(df, metric_col, segment_col, min_sellers, fmt, n_boot=10000):
    """
    Returns a wide DataFrame (metrics as rows, segment values as columns)
    and a Styler with color coding applied.
    """
    metrics_order = [
        "Control sellers",
        "Treatment sellers",
        "Control mean",
        "Treatment mean",
        "Diff (T - C)",
        "95% CI",
        "p-value",
    ]

    seg_cols = {}

    for seg_val, grp in df.groupby(segment_col):
        c = grp[grp["TREATMENT_GROUP"] == "CONTROL"][metric_col]
        t = grp[grp["TREATMENT_GROUP"] == "TREATMENT"][metric_col]

        if len(c) < min_sellers or len(t) < min_sellers:
            continue

        res = bootstrap_mean_diff(t, c, n_boot=n_boot)
        if res is None:
            continue

        seg_cols[seg_val] = {
            "Control sellers": str(len(c)),
            "Treatment sellers": str(len(t)),
            "Control mean": f"{res['control_mean']:{fmt}}",
            "Treatment mean": f"{res['treatment_mean']:{fmt}}",
            "Diff (T - C)": f"{res['observed_diff']:{fmt}}",
            "95% CI": f"[{res['ci_low']:{fmt}}, {res['ci_high']:{fmt}}]",
            "p-value": f"{res['p_value']:.4f}",
        }

    if not seg_cols:
        return None

    rows = []
    for metric in metrics_order:
        row = {f"{segment_col}:": metric}
        for seg_val, vals in seg_cols.items():
            row[seg_val] = vals[metric]
        rows.append(row)

    return pd.DataFrame(rows)


def style_table(df):
    styler = df.style.set_table_styles(
        [
            {"selector": "th", "props": [("border", "1px solid black"), ("padding", "4px"), ("text-align", "center")]},
            {"selector": "td", "props": [("border", "1px solid black"), ("padding", "4px"), ("text-align", "center")]},
            {"selector": "table", "props": [("border-collapse", "collapse")]},
            {"selector": "th.col_heading", "props": [("background-color", "#f0f0f0")]},
        ]
    )

    def highlight_diff(row):
        if row.iloc[0] != "Diff (T - C)":
            return [""] * len(row)
        out = [""]
        for val in row.iloc[1:]:
            try:
                v = float(val)
            except (ValueError, TypeError):
                v = 0
            out.append("background-color: #d0f0d0" if v > 0 else "background-color: #f8d0d0" if v < 0 else "")
        return out

    def highlight_p(row):
        if row.iloc[0] != "p-value":
            return [""] * len(row)
        out = [""]
        for val in row.iloc[1:]:
            try:
                v = float(val)
            except (ValueError, TypeError):
                v = 1.0
            out.append("background-color: #ffd27f" if v < 0.05 else "")
        return out

    styler = styler.apply(highlight_diff, axis=1)
    styler = styler.apply(highlight_p, axis=1)
    return styler
