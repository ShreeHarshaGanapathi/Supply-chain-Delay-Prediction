"""
Classifier Picker -- an interactive MCDM demo (Streamlit).

Takes the evaluation metrics of four trained classifiers, lets the viewer set
how much each criterion matters (sliders / presets), and ranks the classifiers
with four Multi-Criteria Decision-Making methods: TOPSIS, MARCOS, COCOSO, MABAC.

Needs only a metrics file (model_evaluation_metrics.json from the Colab run).
A built-in reference run is used when nothing is uploaded.

Run:  streamlit run app.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pymcdm.methods import COCOSO, MABAC, MARCOS, TOPSIS

st.set_page_config(page_title="Which classifier should we pick?", layout="wide")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# (name, direction): +1 = higher is better (stimulant), -1 = lower is better
CRITERIA = [
    ("Accuracy", 1),
    ("Precision", 1),
    ("Recall", 1),
    ("Log Loss", -1),
    ("MCC", 1),
    ("Training Time (s)", -1),
    ("Interpretability", -1),   # encoded: 1 = fully, 2 = partially interpretable
    ("Model Complexity", -1),
]

HELP = {
    "Accuracy": "Share of all predictions that were correct.",
    "Precision": "When the model says 'delayed', how often is it right? (few false alarms)",
    "Recall": "Of all deliveries that really were late, how many did the model catch?",
    "Log Loss": "How well-calibrated the model's confidence is. Lower is better.",
    "MCC": "One balanced score that stays fair when classes are uneven.",
    "Training Time (s)": "How long the model took to train. Lower is better.",
    "Interpretability": "Can a human follow the model's reasoning? Fully interpretable is better.",
    "Model Complexity": "Tree depth (average depth for ensembles). Lower is simpler.",
}

# Weights are on a 0-100 "importance" scale and get normalised automatically.
PRESETS = {
    "Stakeholder interview (from the paper)": dict(
        zip([c for c, _ in CRITERIA], [10, 15, 30, 10, 20, 5, 5, 5])),
    "Safety first: catch every delay": dict(
        zip([c for c, _ in CRITERIA], [10, 10, 45, 5, 20, 3, 4, 3])),
    "Fast and simple": dict(
        zip([c for c, _ in CRITERIA], [10, 10, 10, 5, 10, 25, 15, 15])),
    "Explainability first": dict(
        zip([c for c, _ in CRITERIA], [10, 10, 15, 5, 5, 5, 35, 15])),
    "Everything equal": dict(zip([c for c, _ in CRITERIA], [12] * 8)),
}
DEFAULT_PRESET = "Stakeholder interview (from the paper)"

METHODS = {"TOPSIS": TOPSIS, "MARCOS": MARCOS, "COCOSO": COCOSO, "MABAC": MABAC}

# Reference run (full DataCo dataset, order-grouped split, grouped 3-fold CV). Upload your own
# model_evaluation_metrics.json to replace these.
REFERENCE_RESULTS = {
    "AdaBoost": {
        "Accuracy": 0.698233, "Precision": 0.842613, "Recall": 0.550804,
        "Log Loss": 0.607574, "MCC": 0.443353, "Training Time (s)": 22.335,
        "Interpretability": "partially interpretable", "Model Complexity": 1.0},
    "Random Forest": {
        "Accuracy": 0.699294, "Precision": 0.832675, "Recall": 0.562997,
        "Log Loss": 0.567474, "MCC": 0.439982, "Training Time (s)": 9.554,
        "Interpretability": "partially interpretable", "Model Complexity": 10.0},
    "Decision Tree": {
        "Accuracy": 0.699128, "Precision": 0.838371, "Recall": 0.556931,
        "Log Loss": 0.565726, "MCC": 0.442535, "Training Time (s)": 1.084,
        "Interpretability": "fully interpretable", "Model Complexity": 5.0},
    "Bagging": {
        "Accuracy": 0.699261, "Precision": 0.837076, "Recall": 0.558508,
        "Log Loss": 0.560632, "MCC": 0.442089, "Training Time (s)": 23.996,
        "Interpretability": "partially interpretable", "Model Complexity": 10.0},
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def validate_results(raw):
    """Return (clean_results, error_message). Accepts the Colab metrics JSON."""
    if not isinstance(raw, dict) or len(raw) < 2:
        return None, "Need a JSON object with at least two classifiers."
    needed = [c for c, _ in CRITERIA]
    for name, row in raw.items():
        if not isinstance(row, dict):
            return None, f"Entry for '{name}' is not an object."
        missing = [c for c in needed if c not in row]
        if missing:
            return None, f"'{name}' is missing: {', '.join(missing)}"
    return raw, None


def build_matrix(results):
    names = list(results.keys())
    rows = []
    for n in names:
        r = results[n]
        interp = 1.0 if "fully" in str(r["Interpretability"]).lower() else 2.0
        rows.append([
            r["Accuracy"], r["Precision"], r["Recall"], r["Log Loss"], r["MCC"],
            r["Training Time (s)"], interp, r["Model Complexity"],
        ])
    return names, np.array(rows, dtype=float)


def run_mcdm(matrix, weights, types):
    scores = {}
    for mname, cls in METHODS.items():
        scores[mname] = np.asarray(cls()(matrix, weights, types), dtype=float)
    return scores


def robustness(matrix, weights, types, n_alt, n_trials=100, spread=0.3, seed=42):
    rng = np.random.default_rng(seed)
    wins = {m: np.zeros(n_alt) for m in METHODS}
    for _ in range(n_trials):
        w = weights * rng.uniform(1 - spread, 1 + spread, size=len(weights))
        w = w / w.sum()
        for mname, pref in run_mcdm(matrix, w, types).items():
            wins[mname][int(np.argmax(pref))] += 1
    return wins


def apply_preset(name):
    for c, _ in CRITERIA:
        st.session_state[f"w_{c}"] = PRESETS[name][c]


def highlight_best(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for c, direction in CRITERIA:
        if c in df.columns:
            best = df[c].max() if direction == 1 else df[c].min()
            styles.loc[df[c] == best, c] = "background-color: #d7f5dd; font-weight: 600"
    return styles


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
for c, _ in CRITERIA:
    if f"w_{c}" not in st.session_state:
        st.session_state[f"w_{c}"] = PRESETS[DEFAULT_PRESET][c]

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.header("1. Data")
uploaded = st.sidebar.file_uploader(
    "Upload model_evaluation_metrics.json (optional)", type="json",
    help="The file written by the training step of the Colab notebook. "
         "Without it, a built-in reference run is used.")

results, source_label = REFERENCE_RESULTS, "built-in reference run"
if uploaded is not None:
    try:
        candidate, err = validate_results(json.load(uploaded))
    except Exception as exc:  # malformed JSON
        candidate, err = None, f"Could not read the file: {exc}"
    if err:
        st.sidebar.error(err + " Using the reference run instead.")
    else:
        results, source_label = candidate, f"uploaded file ({uploaded.name})"

st.sidebar.header("2. What matters to you?")
st.sidebar.caption("Pick a preset, then fine-tune with the sliders. "
                   "Weights are normalised automatically.")
for pname in PRESETS:
    st.sidebar.button(pname, on_click=apply_preset, args=(pname,), width="stretch")

st.sidebar.markdown("---")
raw_weights = []
for c, _ in CRITERIA:
    raw_weights.append(st.sidebar.slider(c, 0, 100, key=f"w_{c}", help=HELP[c]))

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.title("Which classifier should we pick?")

names, matrix = build_matrix(results)
acc_spread = (matrix[:, 0].max() - matrix[:, 0].min()) * 100

with st.expander("What is this? (30-second version)", expanded=True):
    st.markdown(
        f"""
**The problem.** Several machine-learning models predict late deliveries, and on
this data their accuracy differs by only **{acc_spread:.2f} percentage points**.
Accuracy alone can't tell a supply-chain manager which one to use.

**The idea: MCDM (Multi-Criteria Decision-Making).** Score every model on
8 criteria (accuracy, recall, training time, interpretability, ...), let the
decision-maker say how much each criterion matters (the sliders on the left), and
let four standard ranking methods (TOPSIS, MARCOS, COCOSO, MABAC) turn that into
a ranking. A different priority can mean a different winner.

**Try it.** Click a preset or drag a slider and watch the ranking respond.
Metrics source: *{source_label}*.
"""
    )


if sum(raw_weights) == 0:
    st.warning("All sliders are at zero. Give at least one criterion some importance.")
    st.stop()

weights = np.array(raw_weights, dtype=float)
weights = weights / weights.sum()
types = np.array([t for _, t in CRITERIA])

# Criteria that are identical across all models cannot separate them and break
# some MCDM normalisations, so they are set aside (weights renormalised).
varying = [j for j in range(matrix.shape[1]) if np.ptp(matrix[:, j]) > 0 and weights[j] > 0]
dropped = [CRITERIA[j][0] for j in range(matrix.shape[1])
           if np.ptp(matrix[:, j]) == 0 and weights[j] > 0]
if dropped:
    st.info("Left out (identical for every model): " + ", ".join(dropped))
if not varying or weights[varying].sum() == 0:
    st.warning("No criterion with non-zero weight separates these models. "
               "Raise a slider for a criterion that differs between them.")
    st.stop()
m_used = matrix[:, varying]
w_used = weights[varying] / weights[varying].sum()
t_used = types[varying]

scores = run_mcdm(m_used, w_used, t_used)
if any(not np.all(np.isfinite(v)) for v in scores.values()):
    st.warning("These weights leave too few criteria to rank the models "
               "(a method could not compute). Give weight to more criteria.")
    st.stop()
alt_labels = [f"A{i + 1} ({n})" for i, n in enumerate(names)]
ranks = pd.DataFrame(
    {m: pd.Series(s, index=names).rank(ascending=False, method="min").astype(int)
     for m, s in scores.items()}
)
avg_rank = ranks.mean(axis=1)
top_by_method = {m: names[int(np.argmax(s))] for m, s in scores.items()}
counts = pd.Series(top_by_method).value_counts()
leaders = counts[counts == counts.max()].index.tolist()

if len(leaders) == 1:
    st.success(f"**{leaders[0]}** ranks #1 in **{counts.max()} of {len(METHODS)}** methods.")
else:
    tie_break = avg_rank[leaders].idxmin()
    st.info(f"Split decision: {' and '.join(leaders)} each lead in {counts.max()} "
            f"of {len(METHODS)} methods. Best average rank: **{tie_break}**.")

left, right = st.columns([3, 2])
with left:
    st.subheader("Ranking by each method (1 = best)")
    ranks_show = ranks.copy()
    ranks_show["Average"] = avg_rank.round(2)
    st.dataframe(
        ranks_show.sort_values("Average").style.highlight_min(
            subset=list(METHODS), color="#d7f5dd", axis=0),
        width="stretch")
    st.caption("Different methods can disagree. That is why four are used.")
with right:
    st.subheader("Average rank (lower is better)")
    order = avg_rank.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5, 3))
    best_name = avg_rank.idxmin()
    ax.barh(order.index, order.values,
            color=["#55a868" if n == best_name else "#9aa5b1" for n in order.index])
    ax.set_xlim(0, len(names))
    ax.set_xlabel("Average rank across the 4 methods")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

st.subheader("Weights in use")
wdf = pd.DataFrame({"Criterion": [c for c, _ in CRITERIA],
                    "Share of the decision": [f"{w * 100:.0f}%" for w in weights],
                    "Better when": ["higher" if t == 1 else "lower" for t in types]})
st.dataframe(wdf, hide_index=True, width="stretch")

with st.expander("The raw numbers (best value per criterion highlighted)"):
    matrix_df = pd.DataFrame(matrix, index=names, columns=[c for c, _ in CRITERIA])
    st.dataframe(matrix_df.style.apply(highlight_best, axis=None).format("{:.4f}"),
                 width="stretch")
    st.caption("Interpretability: 1 = fully interpretable, 2 = partially interpretable.")

with st.expander("How stable is this result? (robustness check)"):
    st.write("Nudges every weight randomly by up to ±30%, 100 times, and counts how often "
             "each classifier comes out #1 under each method.")
    wins = robustness(m_used, w_used, t_used, len(names))
    wins_df = pd.DataFrame(wins, index=names).round(0).astype(int)
    st.dataframe(wins_df.rename_axis("% of trials ranked #1"), width="stretch")
    st.caption("A column dominated by one classifier means the winner does not depend on "
               "the exact weights.")

with st.expander("Caveats"):
    st.markdown(
        """
- The scores come from the metrics source named above. Re-running training can shift
  them slightly.
- When models are nearly tied on a criterion (accuracy differs by hundredths of a point),
  the MCDM methods still rescale that criterion, so tiny gaps can tip a close ranking.
  Treat close calls as ties.
- Interpretability is a simple label (fully = 1, partially = 2), not a measured score.
- The preset "Stakeholder interview" uses the weights reported in the base paper
  (Wyrembek & Baryannis, 2024), not weights gathered from a real stakeholder for this dataset.
"""
    )
