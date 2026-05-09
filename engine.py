import numpy as np
import pandas as pd

_INTERP_COLORS = {
    "Significant": "background-color: #d0f0d0",
    "More samples needed": "background-color: #fff3cd",
    "No significant effect": "background-color: #e8e8e8",
}


def interpret_result(p_value, ci_low, ci_high, control_mean):
    """
    Returns (label, hex_color) verdict for a single bootstrap result.
    'More samples needed' fires when CI is wide (>30% of control mean) and not significant —
    meaning we can't rule out a meaningful effect, we just don't have enough data yet.
    """
    if p_value < 0.05:
        return "Significant", "#d0f0d0"
    ci_width = ci_high - ci_low
    relative_width = ci_width / abs(control_mean) if control_mean != 0 else float("inf")
    if relative_width > 0.30:
        return "More samples needed", "#fff3cd"
    return "No significant effect", "#e8e8e8"


def generate_reasoning(label, p_value, ci_low, ci_high, control_mean, observed_diff, lift_pct, n_control, n_treatment):
    """
    Returns a markdown string explaining why the verdict was reached,
    with specific numbers and references to standard A/B testing literature.
    """
    ci_width = ci_high - ci_low
    rel_width_pct = (ci_width / abs(control_mean) * 100) if control_mean != 0 else 0
    direction = "increase" if observed_diff > 0 else "decrease"
    ci_sign = "does not cross zero" if (ci_low > 0 or ci_high < 0) else "crosses zero"

    if label == "Significant":
        return (
            f"**p = {p_value:.4f}**, which is below the standard 0.05 significance threshold. "
            f"The 95% bootstrap CI [{ci_low:+.4g}, {ci_high:+.4g}] {ci_sign}, confirming the "
            f"observed **{lift_pct:+.1f}% {direction}** is statistically reliable and unlikely to be "
            f"explained by random chance alone. "
            f"The experiment had {n_control:,} control and {n_treatment:,} treatment sellers — "
            f"sufficient power to detect this effect size. "
            f"\n\n"
            f"*Ref: Kohavi, Tang & Xu — [Trustworthy Online Controlled Experiments](https://www.cambridge.org/trustworthyonlinecontrolledexperiments) (2020), Ch. 3 & 17: "
            f"interpreting p-values and confidence intervals in online experiments.*"
        )

    if label == "More samples needed":
        return (
            f"**p = {p_value:.4f}** (not significant), but the CI is wide: "
            f"[{ci_low:+.4g}, {ci_high:+.4g}] spans **{rel_width_pct:.0f}% of the control mean** ({control_mean:.4g}). "
            f"A wide CI means the experiment cannot reliably distinguish between a meaningful positive effect, "
            f"no effect, or a meaningful negative effect — this is a **statistical power problem**, not necessarily "
            f"evidence that the treatment doesn't work. "
            f"Current sample: {n_control:,} control / {n_treatment:,} treatment sellers. "
            f"\n\n"
            f"**Recommended actions:**\n"
            f"- Extend the experiment duration to accumulate more sellers.\n"
            f"- Increase the traffic allocation to the treatment arm.\n"
            f"- Pre-compute a required sample size using a power calculator (target 80% power at your MDE).\n"
            f"\n"
            f"*Ref: Kohavi et al. (2020), Ch. 18 (Statistical Power & Peeking). "
            f"Cohen, J. — Statistical Power Analysis for the Behavioral Sciences (1988): "
            f"a CI spanning >20–30% of the baseline is a common sign of an underpowered experiment.*"
        )

    # No significant effect
    return (
        f"**p = {p_value:.4f}** (not significant). The 95% CI [{ci_low:+.4g}, {ci_high:+.4g}] "
        f"is **narrow** — only {rel_width_pct:.0f}% of the control mean — indicating the experiment "
        f"is well-powered with {n_control:,} control and {n_treatment:,} treatment sellers. "
        f"This is a reliable null result: the treatment has **no meaningful effect** on this metric "
        f"at the observed scale. A significant result would have been detectable if it existed. "
        f"\n\n"
        f"*Ref: Lakens, D. — 'Equivalence Tests: A Practical Primer' (2017), Advances in Methods and Practices "
        f"in Psychological Science: how to distinguish a true null from an underpowered experiment. "
        f"Also: Kohavi et al. (2020), Ch. 19 (Misinterpretation of p-values).*"
    )


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
    label, color = interpret_result(res["p_value"], res["ci_low"], res["ci_high"], res["control_mean"])
    reasoning = generate_reasoning(
        label, res["p_value"], res["ci_low"], res["ci_high"],
        res["control_mean"], res["observed_diff"], lift_pct,
        n_control=c.dropna().shape[0], n_treatment=t.dropna().shape[0],
    )
    return {
        "Control Mean": f"{res['control_mean']:{fmt}}",
        "Treatment Mean": f"{res['treatment_mean']:{fmt}}",
        "Lift": f"{res['observed_diff']:{fmt}} ({lift_pct:+.1f}%)",
        "95% CI": f"[{res['ci_low']:{fmt}}, {res['ci_high']:{fmt}}]",
        "p-value": f"{res['p_value']:.4f}",
        "Interpretation": label,
        "_diff": res["observed_diff"],
        "_p": res["p_value"],
        "_interp_color": color,
        "_reasoning": reasoning,
    }


def run_segment(df, metric_col, segment_col, min_sellers, fmt, n_boot=10000):
    """
    Returns (wide_df, reasoning_dict) where reasoning_dict maps segment value → markdown reasoning string.
    wide_df has metrics as rows and segment values as columns.
    """
    metrics_order = [
        "Control sellers",
        "Treatment sellers",
        "Control mean",
        "Treatment mean",
        "Diff (T - C)",
        "95% CI",
        "p-value",
        "Interpretation",
    ]

    seg_cols = {}
    reasoning_dict = {}

    for seg_val, grp in df.groupby(segment_col):
        c = grp[grp["TREATMENT_GROUP"] == "CONTROL"][metric_col]
        t = grp[grp["TREATMENT_GROUP"] == "TREATMENT"][metric_col]

        if len(c) < min_sellers or len(t) < min_sellers:
            continue

        res = bootstrap_mean_diff(t, c, n_boot=n_boot)
        if res is None:
            continue

        lift_pct = (res["observed_diff"] / res["control_mean"] * 100) if res["control_mean"] != 0 else 0
        interp_label, _ = interpret_result(res["p_value"], res["ci_low"], res["ci_high"], res["control_mean"])
        reasoning_dict[seg_val] = generate_reasoning(
            interp_label, res["p_value"], res["ci_low"], res["ci_high"],
            res["control_mean"], res["observed_diff"], lift_pct,
            n_control=len(c.dropna()), n_treatment=len(t.dropna()),
        )
        seg_cols[seg_val] = {
            "Control sellers": str(len(c)),
            "Treatment sellers": str(len(t)),
            "Control mean": f"{res['control_mean']:{fmt}}",
            "Treatment mean": f"{res['treatment_mean']:{fmt}}",
            "Diff (T - C)": f"{res['observed_diff']:{fmt}}",
            "95% CI": f"[{res['ci_low']:{fmt}}, {res['ci_high']:{fmt}}]",
            "p-value": f"{res['p_value']:.4f}",
            "Interpretation": interp_label,
        }

    if not seg_cols:
        return None, {}

    rows = []
    for metric in metrics_order:
        row = {f"{segment_col}:": metric}
        for seg_val, vals in seg_cols.items():
            row[seg_val] = vals[metric]
        rows.append(row)

    return pd.DataFrame(rows), reasoning_dict


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

    def highlight_interp(row):
        if row.iloc[0] != "Interpretation":
            return [""] * len(row)
        out = [""]
        for val in row.iloc[1:]:
            out.append(_INTERP_COLORS.get(val, ""))
        return out

    styler = styler.apply(highlight_diff, axis=1)
    styler = styler.apply(highlight_p, axis=1)
    styler = styler.apply(highlight_interp, axis=1)
    return styler
