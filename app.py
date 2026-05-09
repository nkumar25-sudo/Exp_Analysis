import streamlit as st
from data_loader import load_experiment_data
from metrics import METRIC_REGISTRY, SEGMENT_REGISTRY
from engine import run_overall, run_segment, style_table

st.set_page_config(page_title="Experiment Analysis Assistant", layout="wide")

# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Experiment Setup")

    exp_id = st.text_input("Experiment ID", placeholder="e.g. 135181")
    treatment_id = st.text_input("Treatment ID", placeholder="e.g. 303266")
    control_id = st.text_input("Control ID", placeholder="e.g. 303267")

    st.divider()
    st.subheader("Analysis Options")

    selected_metrics = st.multiselect(
        "Metrics to analyze",
        options=list(METRIC_REGISTRY.keys()),
        default=list(METRIC_REGISTRY.keys()),
    )

    selected_cuts = st.multiselect(
        "Segment cuts",
        options=list(SEGMENT_REGISTRY.keys()),
        default=list(SEGMENT_REGISTRY.keys()),
    )

    min_sellers = st.slider("Min sellers per arm", min_value=10, max_value=100, value=20, step=5)
    n_boot = st.select_slider(
        "Bootstrap samples",
        options=[1000, 2000, 5000, 10000, 20000],
        value=10000,
    )

    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

# ── Main area ────────────────────────────────────────────────────────────────

st.title("Experiment Analysis Assistant")
st.caption("Prototype — data sourced from local CSVs (swap `data_loader.py` for live Snowflake query)")

if run_btn:
    if not exp_id or not treatment_id or not control_id:
        st.warning("Please enter Experiment ID, Treatment ID, and Control ID before running.")
        st.stop()
    if not selected_metrics:
        st.warning("Select at least one metric.")
        st.stop()

    with st.spinner("Loading data and running bootstrap analysis…"):
        main_df, listings_df = load_experiment_data(exp_id, treatment_id, control_id)

        overall_rows = []
        segment_results = {}  # {cut_label: {metric_label: DataFrame | None}}

        for metric_label in selected_metrics:
            cfg = METRIC_REGISTRY[metric_label]
            df = main_df if cfg["source"] == "main" else listings_df

            # Overall
            res = run_overall(df, cfg["column"], cfg["fmt"], n_boot=n_boot)
            if res:
                overall_rows.append({
                    "Metric": metric_label,
                    "Type": cfg["type"],
                    "Control Mean": res["Control Mean"],
                    "Treatment Mean": res["Treatment Mean"],
                    "Lift": res["Lift"],
                    "95% CI": res["95% CI"],
                    "p-value": res["p-value"],
                    "Interpretation": res["Interpretation"],
                    "_diff": res["_diff"],
                    "_p": res["_p"],
                    "_interp_color": res["_interp_color"],
                    "_reasoning": res["_reasoning"],
                })

            # Segments
            for cut_label in selected_cuts:
                seg_col = SEGMENT_REGISTRY[cut_label]
                seg_df, seg_reasoning = run_segment(df, cfg["column"], seg_col, min_sellers, cfg["fmt"], n_boot=n_boot)
                segment_results.setdefault(cut_label, {})[metric_label] = (seg_df, seg_reasoning)

    st.session_state["overall_rows"] = overall_rows
    st.session_state["segment_results"] = segment_results
    st.session_state["config"] = {
        "exp_id": exp_id,
        "treatment_id": treatment_id,
        "control_id": control_id,
        "n_metrics": len(selected_metrics),
        "n_cuts": len(selected_cuts),
        "min_sellers": min_sellers,
        "n_boot": n_boot,
    }

# ── Display results (persists across reruns) ─────────────────────────────────

if "overall_rows" not in st.session_state:
    st.info("Configure the experiment in the sidebar and click **Run Analysis** to start.")
    st.stop()

cfg = st.session_state["config"]
overall_rows = st.session_state["overall_rows"]
segment_results = st.session_state["segment_results"]

# Config summary card
with st.container(border=True):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Experiment ID", cfg["exp_id"])
    c2.metric("Treatment ID", cfg["treatment_id"])
    c3.metric("Control ID", cfg["control_id"])
    c4.metric("Metrics", cfg["n_metrics"])
    c5.metric("Segment cuts", cfg["n_cuts"])
    st.caption(f"Min sellers/arm: {cfg['min_sellers']} · Bootstrap samples: {cfg['n_boot']:,}")

st.divider()

# ── Overall Results ───────────────────────────────────────────────────────────

st.subheader("Overall Results")

if not overall_rows:
    st.warning("No results — check that the selected metrics have data.")
else:
    import pandas as pd

    display_cols = ["Metric", "Type", "Control Mean", "Treatment Mean", "Lift", "95% CI", "p-value", "Interpretation"]
    overall_df = pd.DataFrame(overall_rows)[display_cols + ["_diff", "_p", "_interp_color"]]

    def color_overall(row):
        diff = row["_diff"]
        p = row["_p"]
        interp_color = row["_interp_color"]
        row_styles = {col: "" for col in display_cols}
        row_styles["Lift"] = "background-color: #d0f0d0" if diff > 0 else "background-color: #f8d0d0" if diff < 0 else ""
        row_styles["p-value"] = "background-color: #ffd27f" if p < 0.05 else ""
        row_styles["Interpretation"] = f"background-color: {interp_color}"
        return [row_styles[col] for col in display_cols]

    styled_overall = (
        overall_df[display_cols]
        .style
        .set_table_styles([
            {"selector": "th", "props": [("border", "1px solid black"), ("padding", "6px"), ("text-align", "center"), ("background-color", "#f0f0f0")]},
            {"selector": "td", "props": [("border", "1px solid black"), ("padding", "6px"), ("text-align", "center")]},
            {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%")]},
        ])
        .apply(lambda row: color_overall(overall_df.loc[row.name]), axis=1)
        .hide(axis="index")
    )
    st.write(styled_overall.to_html(), unsafe_allow_html=True)

    with st.expander("Reasoning — why these verdicts?", expanded=False):
        for row in overall_rows:
            st.markdown(f"**{row['Metric']}** — *{row['Interpretation']}*")
            st.markdown(row["_reasoning"])
            st.divider()

st.divider()

# ── Segmented Results ─────────────────────────────────────────────────────────

st.subheader("Segmented Results")

if not segment_results:
    st.info("No segment cuts selected.")
else:
    tabs = st.tabs(list(segment_results.keys()))

    for tab, cut_label in zip(tabs, segment_results.keys()):
        with tab:
            metric_data = segment_results[cut_label]
            for metric_label, (seg_df, seg_reasoning) in metric_data.items():
                with st.expander(metric_label, expanded=True):
                    if seg_df is None:
                        st.warning(f"No segments met the minimum of {cfg['min_sellers']} sellers per arm.")
                    else:
                        styled = style_table(seg_df)
                        st.write(styled.to_html(), unsafe_allow_html=True)

                        if seg_reasoning:
                            with st.expander("Reasoning per segment", expanded=False):
                                for seg_val, reason in seg_reasoning.items():
                                    st.markdown(f"**{seg_val}**")
                                    st.markdown(reason)
                                    st.divider()
