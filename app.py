"""Before the Breach — Streamlit vulnerability prioritization dashboard.

Run from the project root with: python3 -m streamlit run app.py
Requires ranking_details_2026.csv and src/granite.py in the project.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

from src.granite import generate_briefing


ROOT = Path(__file__).resolve().parent
RANKINGS_PATH = ROOT / "ranking_details_2026.csv"

st.set_page_config(
    page_title="Before the Breach",
    page_icon="🛡️",
    layout="wide",
)

@st.cache_resource
def load_explainer():
    artifact = joblib.load(
        ROOT / "lightgbm_model.joblib"
    )

    model = artifact["model"]
    features = artifact["features"]

    explainer = shap.TreeExplainer(model)

    return explainer, features



@st.cache_data
def load_budget_results():
    return pd.read_csv(
        ROOT / "results" / "patch_budget_results.csv"
    )


@st.cache_data
def load_rankings() -> pd.DataFrame:
    """Load rankings and calculate EPSS comparison ranks."""

    frame = pd.read_csv(RANKINGS_PATH)

    # Rank by LightGBM.
    frame = frame.sort_values(
        ["ml_score", "epss", "cve_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    frame["rank"] = frame.index + 1

    # Rank the same vulnerabilities using EPSS alone.
    epss_order = frame.sort_values(
        ["epss", "cve_id"],
        ascending=[False, True],
        kind="stable",
    ).copy()

    epss_order["epss_rank"] = range(
        1, len(epss_order) + 1
    )

    # Merge EPSS ranks back into the LightGBM rankings.
    frame = frame.merge(
        epss_order[["cve_id", "epss_rank"]],
        on="cve_id",
        how="left",
        validate="one_to_one",
    )

    # Positive = ranked higher by LightGBM.
    frame["rank_improvement"] = (
        frame["epss_rank"] - frame["rank"]
    )

    return frame



st.title("🛡️ Before the Breach")
st.caption("Machine learning-powered vulnerability prioritization")
st.info(
    "Model scores are relative ranking signals, not calibrated probabilities "
    "of real-world exploitation. Public exploit indicators do not confirm an attack."
)

if not RANKINGS_PATH.is_file():
    st.error(
        "Missing ranking_details_2026.csv. Run "
        "`python3 src/predict_2026.py` from your project root first."
    )
    st.stop()

df = load_rankings()

required_columns = [
    "rank", "cve_id", "ml_score", "epss", "exploit_count", "age_days",
    "exploitdb", "metasploit", "nuclei", "poc_github",
]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    st.error(f"Ranking CSV is missing columns: {', '.join(missing_columns)}")
    st.stop()

# --- Overview ---
col1, col2, col3 = st.columns(3)
col1.metric("Vulnerabilities analyzed", f"{len(df):,}")
col2.metric("Priority vulnerabilities", f"{min(500, len(df)):,}")
col3.metric("Highest model score", f"{df['ml_score'].max():.4f}")

# --- Leaderboard ---
st.divider()
st.subheader("Vulnerability leaderboard")
search = st.text_input("Search by CVE ID", placeholder="CVE-2026-...").strip()
top_only = st.checkbox("Show only top 500", value=True)

filtered = df.head(500) if top_only else df
if search:
    filtered = filtered.loc[
        filtered["cve_id"].str.contains(
            search, case=False, regex=False, na=False
        )
    ]

st.dataframe(
    filtered[["rank", "cve_id", "ml_score", "epss", "exploit_count", "age_days"]],
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

st.divider()
st.header("LightGBM vs. EPSS")

st.caption(
    "Compare vulnerability priorities produced by "
    "the ML model and EPSS alone."
)

# Top-500 sets for each method.
ml_top500 = set(
    df.nsmallest(500, "rank")["cve_id"]
)

epss_top500 = set(
    df.nsmallest(500, "epss_rank")["cve_id"]
)

shared_cves = ml_top500 & epss_top500
ml_only = ml_top500 - epss_top500
epss_only = epss_top500 - ml_top500

left, middle, right = st.columns(3)

left.metric(
    "Shared top-500 CVEs",
    len(shared_cves)
)

middle.metric(
    "LightGBM only",
    len(ml_only)
)

right.metric(
    "EPSS only",
    len(epss_only)
)

# Display comparison chart.
comparison_counts = pd.DataFrame({
    "Category": [
        "Shared",
        "LightGBM only",
        "EPSS only"
    ],
    "CVEs": [
        len(shared_cves),
        len(ml_only),
        len(epss_only)
    ]
}).set_index("Category")

comparison_counts = comparison_counts.reindex(
    ["Shared", "LightGBM only", "EPSS only"]
)

st.bar_chart(
    comparison_counts,
    horizontal=True,
    sort=False
)

# Inspect which vulnerabilities differ.
st.subheader("Ranking differences")

comparison_filter = st.selectbox(
    "Choose a comparison group",
    [
        "LightGBM top 500",
        "Promoted by LightGBM",
        "EPSS only",
        "Shared top 500"
    ]
)

if comparison_filter == "LightGBM top 500":
    comparison_df = df[df["rank"] <= 500]

elif comparison_filter == "Promoted by LightGBM":
    comparison_df = df[
        df["cve_id"].isin(ml_only)
    ]

elif comparison_filter == "EPSS only":
    comparison_df = df[
        df["cve_id"].isin(epss_only)
    ]

else:
    comparison_df = df[
        df["cve_id"].isin(shared_cves)
    ]

st.dataframe(
    comparison_df[
        [
            "cve_id",
            "rank",
            "epss_rank",
            "rank_improvement",
            "ml_score",
            "epss",
            "exploit_count",
            "age_days"
        ]
    ].sort_values("rank"),
    hide_index=True,
    use_container_width=True,
)


# --------------------------------------------------
# HISTORICAL VALIDATION
# --------------------------------------------------

st.subheader("Historical model validation")

historical_results = pd.DataFrame({
    "Method": ["EPSS", "LightGBM"],
    "Hits@500": [4, 6]
}).set_index("Method")

st.bar_chart(historical_results)

st.caption(
    "September 2025 historical experiment: "
    "34 vulnerabilities entered KEV within 90 days. "
    "LightGBM identified 6 in its top 500, "
    "compared with 4 for EPSS. "
    "This period was also used during model development, "
    "so results are exploratory rather than an "
    "independent estimate of future performance."
)



# --------------------------------------------------
# THE PATCH BUDGET CHALLENGE
# --------------------------------------------------

st.divider()
st.header("The Patch Budget Challenge")

st.write(
    "If a security team can investigate only a limited "
    "number of vulnerabilities, which ranking identifies "
    "more future KEV additions?"
)

budget_path = ROOT / "results" / "patch_budget_results.csv"

if budget_path.exists():

    budget_data = load_budget_results()

    cohort_name = st.selectbox(
        "Choose historical evaluation population",
        budget_data["cohort"].unique().tolist()
    )

    cohort_data = budget_data.loc[
        budget_data["cohort"] == cohort_name
    ].sort_values("budget")

    if not cohort_data.empty:

        budget_options = cohort_data["budget"].tolist()

        selected_budget = st.select_slider(
            "Investigation budget (number of CVEs)",
            options=budget_options,
            value=500 if 500 in budget_options
            else budget_options[-1]
        )

        selected_row = cohort_data.loc[
            cohort_data["budget"] == selected_budget
        ].iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "LightGBM hits",
            int(selected_row["lightgbm_hits"])
        )

        col2.metric(
            "EPSS hits",
            int(selected_row["epss_hits"])
        )

        col3.metric(
            "Additional hits",
            int(selected_row["difference"])
        )

        st.caption(
            f"{int(selected_row['candidates']):,} "
            "eligible historical vulnerabilities; "
            f"{int(selected_row['positive_cves'])} "
            "future KEV additions observed."
        )

        chart_data = cohort_data.set_index("budget")[
            ["lightgbm_hits", "epss_hits"]
        ]

        st.line_chart(chart_data)

        st.caption(
            "Historical September 2025 development evaluation. "
            "The same period informed feature selection; "
            "results are exploratory, not a prospective "
            "measure of 2026 model performance."
        )

        if int(selected_row["positive_cves"]) < 10:
            st.warning(
                "This cohort has fewer than 10 observed "
                "positive CVEs. Ranking comparisons are "
                "highly sensitive to individual examples."
            )

else:
    st.info(
        "Run python3 src/patch_budget.py "
        "to generate the historical budget analysis."
    )


# --------------------------------------------------
# HISTORICAL CASE STUDIES
# --------------------------------------------------

st.divider()
st.header("Hidden in Plain Sight")

st.write(
    "Historical vulnerabilities that were subsequently "
    "added to KEV but received different priorities "
    "from LightGBM and EPSS."
)

audit_path = ROOT / "results" / "recent_positive_audit.csv"

if audit_path.exists():

    audit = pd.read_csv(audit_path)

    selected_case = st.selectbox(
        "Explore a historical case",
        audit["cve_id"].tolist()
    )

    case = audit.loc[
        audit["cve_id"] == selected_case
    ].iloc[0]

    ml_rank = int(case["ml_rank"])
    epss_rank = int(case["epss_rank"])

    st.subheader(selected_case)

    left, middle, right = st.columns(3)

    left.metric("LightGBM rank", ml_rank)
    middle.metric("EPSS rank", epss_rank)

    right.metric(
        "Rank improvement",
        f"{epss_rank - ml_rank:+,}"
    )

    st.write(
        "**Subsequent KEV addition:**",
        str(case["date_added"])[:10]
    )

    chart = pd.DataFrame({
        "Ranking": ["LightGBM", "EPSS"],
        "Position": [ml_rank, epss_rank]
    }).set_index("Ranking")

    st.bar_chart(
        chart,
        horizontal=True
    )

    st.caption(
        "Lower rank numbers indicate higher priority. "
        "Ranks are within the September 2025 cohort "
        "of CVEs published within 90 days. "
        "These are retrospective development results."
    )

else:
    st.info(
        "Generate the historical audit by running "
        "python3 src/patch_budget.py"
    )


# --- Individual vulnerability inspector ---
st.divider()
st.subheader("Vulnerability inspector")

if filtered.empty:
    st.info("No vulnerabilities match your search. Try another CVE ID or turn off the top-500 filter.")
else:
    selected_cve = st.selectbox("Select a vulnerability", filtered["cve_id"].tolist())
    selected = filtered.loc[filtered["cve_id"] == selected_cve].iloc[0]

    st.markdown(f"### {selected_cve}")
    left, right = st.columns(2)
    with left:
        st.write(f"**Ranking position:** #{int(selected['rank'])}")
        st.write(f"**Model ranking score:** {selected['ml_score']:.6f}")
        st.write(f"**EPSS (snapshot):** {selected['epss']:.4%}")
    with right:
        st.write(f"**Age at snapshot:** {int(selected['age_days'])} days")
        st.write(f"**Public exploit indicators:** {int(selected['exploit_count'])}")

    st.markdown("#### Available exploit-code signals")
    signal_names = {
        "exploitdb": "ExploitDB",
        "metasploit": "Metasploit",
        "nuclei": "Nuclei",
        "poc_github": "GitHub proof of concept",
    }
    for column, label in signal_names.items():
        status = "Present" if int(selected[column]) == 1 else "Not observed"
        st.write(f"- **{label}:** {status}")


    # ----------------------------------------
    # SHAP MODEL EXPLAINABILITY
    # ----------------------------------------

    st.divider()
    st.subheader("Why did LightGBM rank this CVE?")

    if st.button("Explain model prediction"):
        try:
            explainer, features = load_explainer()

            input_row = pd.DataFrame(
                [selected[features].to_dict()],
                columns=features
            )


            # Calculate SHAP values for the selected CVE.
            explanation = explainer(input_row)

            # Handle SHAP's binary-class output.
            if explanation.values.ndim == 3:
                explanation = explanation[:, :, 1]

            # Extract the selected vulnerability.
            cve_explanation = explanation[0]

            # Show signed feature contributions.
            contributions = pd.DataFrame({
                "Feature": features,
                "SHAP value": cve_explanation.values
            })

            st.write("Signed SHAP contributions:")
            st.dataframe(
                contributions,
                hide_index=True
            )

            # Verify the SHAP decomposition.
            raw_score = (
                float(cve_explanation.base_values)
                + float(cve_explanation.values.sum())
            )

            st.write("SHAP raw model output:", raw_score)
            st.write(
                "Model ranking score:",
                float(selected["ml_score"])
            )

            # Create the waterfall plot.
            plt.figure(figsize=(10, 6))

            shap.plots.waterfall(
                cve_explanation,
                max_display=7,
                show=False
            )

            st.pyplot(plt.gcf())
            plt.close("all")

            st.caption(
                "SHAP explains how each feature moves "
                "the model output away from its baseline. "
                "For this LightGBM classifier, the raw "
                "output is in log-odds, not probabilities. "
                "SHAP values are not causal effects."
            )


        except Exception as error:
            st.error(f"SHAP explanation failed: {error}")

    st.divider()
    st.subheader("IBM Granite analyst briefing")
    st.caption(
        "Generate an explanation grounded in the selected CVE's snapshot evidence. "
        "Granite runs locally through Ollama."
    )

    if st.button("Generate IBM Granite Briefing", type="primary"):
        with st.spinner("Generating a briefing with IBM Granite..."):
            try:
                briefing = generate_briefing(selected.to_dict())
                st.session_state["briefing"] = briefing
                st.session_state["briefing_cve"] = selected_cve
            except Exception as error:
                st.error(f"Granite briefing failed: {error}")

    if st.session_state.get("briefing_cve") == selected_cve:
        st.caption(
            "AI-generated interpretation using IBM Granite. "
            "Review recommendations against the verified "
            "snapshot evidence above before taking action."
        )
        st.markdown(st.session_state["briefing"])

# --- Export ---
st.divider()
st.subheader("Export patch prioritization")
export = df.head(500)[["cve_id"]].to_csv(index=False)
st.download_button(
    label="Download top 500 CVEs",
    data=export,
    file_name="top500.csv",
    mime="text/csv",
)
