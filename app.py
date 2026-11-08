"""Before the Breach — Streamlit vulnerability prioritization dashboard.

Run from the project root with: python3 -m streamlit run app.py
Requires ranking_details_2026.csv and src/granite.py in the project.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.granite import generate_briefing


ROOT = Path(__file__).resolve().parent
RANKINGS_PATH = ROOT / "ranking_details_2026.csv"

st.set_page_config(
    page_title="Before the Breach",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def load_rankings() -> pd.DataFrame:
    """Load saved predictions without retraining LightGBM."""
    frame = pd.read_csv(RANKINGS_PATH)
    frame = frame.sort_values(
        ["ml_score", "epss", "cve_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    frame["rank"] = frame.index + 1
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
