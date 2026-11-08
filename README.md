# Before the Breach

### Predict exploitation. Prioritize smarter. Act before the breach.

**An explainable, machine learning-powered vulnerability prioritization platform built for the IBM Track at Sheridan Datathon 2026.**

Before the Breach combines historical exploitation signals, gradient-boosted machine learning, SHAP explainability, and IBM Granite to help security teams identify vulnerabilities worth investigating first.

**IBM Track | Sheridan Datathon 2026 | November 7–8, 2026**

---

## The Problem

Security teams face an overwhelming number of newly disclosed software vulnerabilities. Traditional prioritization often relies on the Common Vulnerability Scoring System (CVSS), which measures the potential severity of a vulnerability.

But severity and exploitation likelihood are different questions.

A critical-severity vulnerability may never be exploited, while a lower-severity vulnerability may attract substantial attacker interest.

The challenge is not simply identifying dangerous vulnerabilities. It is deciding **which vulnerabilities deserve attention first when remediation resources are limited**.

## Our Solution

Before the Breach is a vulnerability intelligence and prioritization platform that:

- Ranks 72,670 CVEs from the 2026 competition benchmark.
- Uses LightGBM to learn patterns from historical EPSS scores and public exploit indicators.
- Adds engineered features describing vulnerability age and exploit-code availability.
- Compares ML predictions with conventional EPSS-based prioritization.
- Explains individual predictions using SHAP.
- Uses IBM Granite to generate contextual analyst briefings.
- Exports a ranked list of 500 CVEs for investigation and remediation planning.

The system supports human decision-making; it does not replace vulnerability verification, asset assessment, or security engineering judgment.

## Key Features

### 1. Machine Learning Vulnerability Ranking

A LightGBM gradient-boosted decision tree classifier ranks vulnerabilities according to signals associated with future additions to CISA's Known Exploited Vulnerabilities (KEV) catalog.

The current model uses seven features:

| Feature | Description |
|---|---|
| `epss` | Exploit Prediction Scoring System estimate |
| `exploitdb` | ExploitDB code indicator |
| `metasploit` | Metasploit module indicator |
| `nuclei` | Nuclei scanning-template indicator |
| `poc_github` | GitHub proof-of-concept indicator |
| `age_days` | Days between publication and prediction snapshot |
| `exploit_count` | Sum of the four public exploit indicators |

Features are derived from the supplied benchmark snapshots or historical data. The model's output is used as a relative ranking score, not a calibrated probability of real-world exploitation.

### 2. Model Comparison

The dashboard compares LightGBM predictions with EPSS-only rankings.

On the 2026 benchmark:

| Comparison | CVEs |
|---|---:|
| Selected by both methods | 75 |
| Selected only by LightGBM | 425 |
| Selected only by EPSS | 425 |

Only 15% of each method's top-500 selections overlap.

This demonstrates substantial disagreement between the approaches, but does not independently establish which ranking performs better on the hidden 2026 outcomes.

### 3. SHAP Explainability

SHAP TreeExplainer helps analysts inspect how individual features contribute to a LightGBM prediction.

The dashboard includes a waterfall plot showing how feature contributions move a prediction away from the model's baseline output.

For this classifier, explanations are expressed in raw log-odds. They are not causal effects or calibrated exploitation probabilities.

### 4. IBM Granite Analyst Briefings

IBM Granite is integrated into the vulnerability inspector through Ollama's local inference API.

The language model generates analyst-oriented interpretations and investigation recommendations using the vulnerability's supplied snapshot evidence.

The application separately displays verified dataset indicators, helping analysts distinguish observed evidence from AI-generated interpretation.

Generated briefings require human review and should not be treated as confirmation of active exploitation, available patches, or technical severity.

### 5. Interactive Streamlit Dashboard

The web interface provides:

- Searchable vulnerability leaderboard
- Top-500 prioritization queue
- LightGBM-versus-EPSS ranking comparison
- Historical validation visualizations
- Individual vulnerability inspection
- SHAP feature-contribution explanations
- IBM Granite analyst briefings
- CSV export of the 500 highest-ranked CVEs

---

## System Architecture

```text
Historical vulnerability data (2025)
                 |
                 v
     Snapshot feature engineering
                 |
                 v
       Future KEV label creation
                 |
                 v
          LightGBM training
                 |
                 v
       2026 benchmark inference
                 |
                 v
        Ranked vulnerability CSV
                 |
                 v
         Streamlit dashboard
            /          \
           v            v
    SHAP explanations  IBM Granite
                       via Ollama
            \          /
                 v
        Analyst review + export
```

Training and inference are performed separately from the dashboard. The Streamlit application loads saved model artifacts and rankings instead of retraining the model on every interaction.

---

## Dataset

The project uses the official IBM Track dataset provided during Sheridan Datathon 2026.

The supplied files include:

| File | Purpose |
|---|---|
| `vulnerabilities.csv` | CVE metadata, descriptions, publication dates, severity indicators |
| `exploit_signals_2024.csv` | Historical monthly EPSS and exploit-code signals |
| `exploit_signals_2025.csv` | Historical monthly EPSS and exploit-code signals |
| `known_exploited.csv` | CISA KEV records through December 2025 |
| `scoreboard_2026.csv` | 72,670 vulnerabilities for ranking |
| `affected_products.csv` | Affected product information |
| `references.csv` | Vulnerability reference links |
| `weakness_types.csv` | CWE definitions |

The current trained model primarily uses the 2025 monthly signals, KEV addition dates, publication dates, and the 2026 scoreboard.

The complete datasets are not committed to Git because of their size. Obtain `data.zip` from the authorized Sheridan Datathon challenge materials.

### Data Leakage Considerations

The benchmark requires using information available at each vulnerability's historical prediction snapshot.

To reduce leakage, the predictive model uses monthly EPSS and public exploit-code snapshots instead of post-event NIST analysis fields such as current `nvd_status`, reference counts, or affected-product counts.

Publication age is derived from recorded publication dates and snapshot months.

The current NIST metadata export was collected later than the historical snapshots, so complete point-in-time reproducibility of all publication metadata has not been independently established.

---

## Machine Learning Methodology

### Problem Formulation

We formulate vulnerability prioritization as a supervised ranking problem using a binary classifier.

For each historical monthly snapshot:

**Features:** Information recorded at the prediction date.

**Target:** Whether the CVE is subsequently added to the KEV catalog within a 90-day observation window.

CVEs already on KEV at prediction time are excluded from the historical candidate population.

A negative label means no KEV addition was observed during the defined window. It does not prove the vulnerability was never exploited.

### Models Explored

We experimented with:

1. EPSS-only ranking
2. Public exploit-code indicators followed by EPSS
3. Logistic regression
4. LightGBM with five original features
5. LightGBM with engineered age and exploit-count features

### Feature Ablation Study

All four feature configurations were evaluated on the September 2025 historical snapshot.

| Feature set | Hits@500 |
|---|---:|
| Original five features | 4 |
| Five features + vulnerability age | 5 |
| Five features + exploit count | 3 |
| All seven features | 6 |

The seven-feature model achieved the highest result in this experiment.

### Historical Validation

In the September 2025 validation snapshot, 34 eligible vulnerabilities were observed to enter KEV within the following 90 days.

| Method | Hits@500 | Precision@500 |
|---|---:|---:|
| EPSS | 4 | 0.8% |
| Seven-feature LightGBM | 6 | 1.2% |

The LightGBM model found 50% more observed positive cases in this particular historical comparison.

**Important:** The September validation period was also used during model development and feature selection. These exploratory results are not an independent test of generalization and do not establish performance on the hidden 2026 benchmark.

### Final Model

The final model was trained using historical snapshots from April through September 2025.

- 1,708,174 training observations
- 174 positive monthly observations
- 85 unique positive CVEs
- Seven predictive features
- LightGBM binary classification objective

The trained model was then applied to all 72,670 CVEs in the 2026 benchmark.

The submission contains the 500 highest-ranked CVE identifiers.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Programming | Python 3.11 |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, LightGBM |
| Explainability | SHAP, Matplotlib |
| Foundation model | IBM Granite |
| Local model serving | Ollama |
| Dashboard | Streamlit |
| Model serialization | joblib |
| Version control | Git, GitHub |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/zaourid1/before-the-breach.git
cd before-the-breach
```

### 2. Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows, activate the environment with:

```powershell
venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

On macOS, LightGBM may require the OpenMP runtime:

```bash
brew install libomp
```

### 4. Prepare the data

Download the challenge dataset ZIP from the authorized Sheridan Datathon materials and extract the CSV files into a `data/` directory at the repository root.

Expected structure:

```text
before-the-breach/
├── data/
│   ├── vulnerabilities.csv
│   ├── known_exploited.csv
│   ├── exploit_signals_2024.csv
│   ├── exploit_signals_2025.csv
│   ├── scoreboard_2026.csv
│   ├── affected_products.csv
│   ├── references.csv
│   └── weakness_types.csv
├── src/
├── app.py
└── README.md
```

### 5. Train and generate predictions

From the project root:

```bash
python3 src/predict_2026.py
```

This trains the final LightGBM model and generates the following artifacts:

| File | Purpose |
|---|---|
| `top500.csv` | Competition submission |
| `ranking_details_2026.csv` | Detailed rankings for Streamlit |
| `lightgbm_model.joblib` | Saved model and feature order for SHAP |

The prediction pipeline checks that the submission contains exactly 500 unique CVE IDs and one `cve_id` column.

### 6. Install IBM Granite

Install Ollama from:

https://ollama.com/download

Then download the IBM Granite model:

```bash
ollama pull granite4:micro
```

Ensure Ollama is running and verify the installation:

```bash
ollama run granite4:micro "Explain vulnerability prioritization in one sentence."
```

The application uses Ollama's local API on `http://localhost:11434`.

Granite model weights are downloaded separately and are not stored in this repository.

### 7. Launch the dashboard

```bash
python3 -m streamlit run app.py
```

Open the local Streamlit URL displayed in your terminal, typically:

http://localhost:8501

Select a vulnerability to inspect its ranking, generate a SHAP explanation, or request an IBM Granite analyst briefing.

---

## Project Structure

```text
before-the-breach/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/                     # Local challenge datasets
├── src/
│   ├── explore.py
│   ├── explore_history.py
│   ├── baseline.py
│   ├── features.py
│   ├── train.py
│   ├── train_advanced.py
│   ├── train_multimonth.py
│   ├── ablation.py
│   ├── predict_2026.py
│   └── granite.py
├── top500.csv
├── ranking_details_2026.csv
└── lightgbm_model.joblib
```

Generated artifacts may need to be recreated locally using the prediction pipeline if they are not distributed with the repository.

---

## Limitations

**Sparse exploitation labels:** Confirmed KEV additions are rare, creating extreme class imbalance. The final training observations include repeated appearances of some CVEs across months.

**Label coverage:** CISA KEV represents confirmed exploitation entries, not an exhaustive record of all exploitation activity.

**Validation limitations:** The September 2025 period was used during development. Results from this period may be optimistic.

**Distribution shift:** Historical training snapshots contain older vulnerabilities, whereas the 2026 benchmark focuses on newly published vulnerabilities.

**Historical metadata:** Certain current NIST fields may reflect information added after an exploitation event. The model avoids known high-risk leakage fields, but full historical reconstruction has not been completed.

**Ranking-score calibration:** LightGBM outputs have not been calibrated to real-world exploitation probabilities.

**Foundation-model reliability:** IBM Granite generates advisory interpretations and may produce unsupported statements. Its outputs require human review.

**Deployment:** The current application is a local research prototype. A production version would require stronger data provenance, asset-specific context, monitoring, access controls, and operational validation.

---

## Future Improvements

- Evaluate on additional strictly held-out historical periods.
- Improve the match between training candidates and newly published CVEs.
- Investigate temporal EPSS trends and time-aware feature engineering.
- Add richer vulnerability description features with verified historical availability.
- Calibrate model scores where suitable labels are available.
- Add organization-specific asset exposure and business impact.
- Implement structured verification of IBM Granite outputs.
- Develop an operational remediation queue with analyst feedback.

---

## Project Context

Built as part of the **Sheridan Datathon 2026 — IBM Track: Before the Breach**.

The project explores how historical vulnerability intelligence, supervised machine learning, explainability, and IBM foundation models can support proactive cybersecurity prioritization.

The goal is not to eliminate uncertainty, but to make prioritization more transparent, evidence-based, and actionable.

---

## Data and Technology Acknowledgments

- IBM — Granite foundation models and IBM Track challenge
- Sheridan Datathon 2026 — Challenge materials and benchmark
- NIST — National Vulnerability Database
- CISA — Known Exploited Vulnerabilities catalog
- FIRST — Exploit Prediction Scoring System
- MITRE — Common Weakness Enumeration
- LightGBM, SHAP, Streamlit, scikit-learn, pandas, and Ollama open-source communities

Dataset sources and usage are subject to their original terms and the competition rules.