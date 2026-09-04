"""
F1 Podium Predictor
--------------------
An interactive Streamlit app that serves a pre-trained Gradient Boosting
Classifier which predicts the probability that a driver finishes on the
podium (Top 3) in a given Formula 1 race.

Run with:
    streamlit run app.py
"""

import io

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="F1 Podium Predictor",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "f1_podium_gbm_model.pkl"

FEATURE_COLUMNS = [
    "grid",
    "position_driverStanding",
    "wins",
    "position_constructorStanding",
    "wins_constructorStanding",
    "round",
]

FEATURE_HELP = {
    "grid": "Starting grid position for the race (1 = pole position).",
    "position_driverStanding": "Driver's position in the championship standings before this race.",
    "wins": "Number of race wins the driver has this season so far.",
    "position_constructorStanding": "Constructor's (team's) position in the championship standings before this race.",
    "wins_constructorStanding": "Number of race wins the constructor has this season so far.",
    "round": "Round number of the race within the season (1 = first race).",
}

OPTIONAL_DISPLAY_COLUMNS = ["driver", "driver_name", "constructor", "constructor_name", "race", "race_name"]


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    return joblib.load(path)


def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    """Run predictions on a dataframe already restricted to FEATURE_COLUMNS."""
    X = df[FEATURE_COLUMNS].astype(float)
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    out = df.copy()
    out["podium_probability"] = proba
    out["predicted_class"] = pred
    return out


def rank_top3(df: pd.DataFrame, group_col: str | None) -> pd.DataFrame:
    """Add a rank + top-3 flag, optionally grouped by race (round)."""
    df = df.copy()
    if group_col and group_col in df.columns:
        df["prob_rank"] = (
            df.groupby(group_col)["podium_probability"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
    else:
        df["prob_rank"] = df["podium_probability"].rank(method="first", ascending=False).astype(int)
    df["predicted_top3"] = df["prob_rank"] <= 3
    return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🏁 F1 Podium Predictor")
st.sidebar.markdown(
    "Predict the probability that a driver finishes **Top 3** using a "
    "Gradient Boosting Classifier trained on historical F1 data."
)

st.sidebar.header("1. Input method")
input_mode = st.sidebar.radio(
    "Choose how to provide race data:",
    ["Upload CSV", "Manual entry"],
)

st.sidebar.header("2. Prediction settings")
threshold = st.sidebar.slider(
    "Podium probability threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="A driver is individually classified as a podium finisher if their "
    "predicted probability is at or above this value.",
)
use_top3_ranking = st.sidebar.checkbox(
    "Also rank drivers and flag Top 3 per race",
    value=True,
    help="Ranks all drivers in the uploaded file (or grouped by 'round' if "
    "present) by predicted probability and flags the 3 highest as the "
    "predicted podium — useful when you have a full grid for one race.",
)

with st.sidebar.expander("ℹ️ About the model"):
    st.write(
        "**Algorithm:** Gradient Boosting Classifier (scikit-learn)\n\n"
        "**Features used:**"
    )
    for col in FEATURE_COLUMNS:
        st.write(f"- `{col}`: {FEATURE_HELP[col]}")

# --------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find `{MODEL_PATH}`. Make sure the model file is in the "
        "same folder as app.py."
    )
    st.stop()

st.title("🏁 F1 Podium Finish Predictor")
st.caption(
    "Upload race data or enter it manually to predict podium (Top 3) "
    "probabilities using a trained Gradient Boosting model."
)

results_df = None

# --------------------------------------------------------------------------
# Input: CSV upload
# --------------------------------------------------------------------------
if input_mode == "Upload CSV":
    st.subheader("📄 Upload race data")
    st.markdown(
        "Your CSV must include these columns: "
        + ", ".join(f"`{c}`" for c in FEATURE_COLUMNS)
        + ". Optional columns like `driver`, `constructor`, or `race` will be "
        "kept for display but are not used by the model."
    )

    sample = pd.DataFrame(
        {
            "driver": ["Driver A", "Driver B", "Driver C"],
            "round": [5, 5, 5],
            "grid": [1, 3, 8],
            "position_driverStanding": [1, 2, 6],
            "wins": [3, 2, 0],
            "position_constructorStanding": [1, 1, 4],
            "wins_constructorStanding": [5, 5, 1],
        }
    )
    st.download_button(
        "⬇️ Download sample CSV template",
        data=to_csv_bytes(sample),
        file_name="sample_race_input.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
            st.stop()

        missing = [c for c in FEATURE_COLUMNS if c not in raw_df.columns]
        if missing:
            st.error(
                "The uploaded file is missing required column(s): "
                + ", ".join(f"`{c}`" for c in missing)
            )
            st.stop()

        st.success(f"Loaded {len(raw_df)} row(s).")
        with st.expander("Preview uploaded data"):
            st.dataframe(raw_df, use_container_width=True)

        results_df = predict(model, raw_df)

# --------------------------------------------------------------------------
# Input: Manual entry
# --------------------------------------------------------------------------
else:
    st.subheader("✍️ Manual entry")
    st.markdown("Enter the details for one or more drivers, then click **Predict**.")

    if "manual_rows" not in st.session_state:
        st.session_state.manual_rows = [
            {c: 0 for c in FEATURE_COLUMNS} | {"driver": "Driver 1"}
        ]

    def add_row():
        n = len(st.session_state.manual_rows) + 1
        st.session_state.manual_rows.append(
            {c: 0 for c in FEATURE_COLUMNS} | {"driver": f"Driver {n}"}
        )

    def remove_row():
        if len(st.session_state.manual_rows) > 1:
            st.session_state.manual_rows.pop()

    col_a, col_b = st.columns(2)
    col_a.button("➕ Add driver", on_click=add_row, use_container_width=True)
    col_b.button("➖ Remove last driver", on_click=remove_row, use_container_width=True)

    edited_rows = []
    for i, row in enumerate(st.session_state.manual_rows):
        with st.container(border=True):
            st.markdown(f"**Driver {i + 1}**")
            name = st.text_input("Driver name (optional)", value=row.get("driver", f"Driver {i + 1}"), key=f"name_{i}")
            c1, c2, c3 = st.columns(3)
            grid = c1.number_input("Grid position", min_value=1, max_value=30, value=int(row.get("grid", 1)) or 1, key=f"grid_{i}", help=FEATURE_HELP["grid"])
            round_ = c2.number_input("Round", min_value=1, max_value=30, value=int(row.get("round", 1)) or 1, key=f"round_{i}", help=FEATURE_HELP["round"])
            drv_wins = c3.number_input("Driver wins (season)", min_value=0, max_value=25, value=int(row.get("wins", 0)), key=f"wins_{i}", help=FEATURE_HELP["wins"])
            c4, c5 = st.columns(2)
            drv_pos = c4.number_input("Driver standing position", min_value=1, max_value=25, value=int(row.get("position_driverStanding", 1)) or 1, key=f"dpos_{i}", help=FEATURE_HELP["position_driverStanding"])
            con_pos = c5.number_input("Constructor standing position", min_value=1, max_value=15, value=int(row.get("position_constructorStanding", 1)) or 1, key=f"cpos_{i}", help=FEATURE_HELP["position_constructorStanding"])
            con_wins = st.number_input("Constructor wins (season)", min_value=0, max_value=25, value=int(row.get("wins_constructorStanding", 0)), key=f"cwins_{i}", help=FEATURE_HELP["wins_constructorStanding"])

            edited_rows.append(
                {
                    "driver": name,
                    "grid": grid,
                    "round": round_,
                    "wins": drv_wins,
                    "position_driverStanding": drv_pos,
                    "position_constructorStanding": con_pos,
                    "wins_constructorStanding": con_wins,
                }
            )

    st.session_state.manual_rows = edited_rows

    if st.button("🔮 Predict", type="primary"):
        manual_df = pd.DataFrame(edited_rows)
        results_df = predict(model, manual_df)

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
if results_df is not None:
    results_df["predicted_class"] = (results_df["podium_probability"] >= threshold).astype(int)

    display_cols = [c for c in OPTIONAL_DISPLAY_COLUMNS if c in results_df.columns]
    ordered_cols = display_cols + FEATURE_COLUMNS + ["podium_probability", "predicted_class"]

    if use_top3_ranking:
        group_col = "round" if "round" in results_df.columns else None
        results_df = rank_top3(results_df, group_col)
        ordered_cols += ["prob_rank", "predicted_top3"]

    results_df = results_df.sort_values("podium_probability", ascending=False).reset_index(drop=True)

    st.divider()
    st.header("📊 Results")

    # --- summary stats ---
    n_rows = len(results_df)
    n_predicted_podium = int(results_df["predicted_class"].sum())
    avg_prob = results_df["podium_probability"].mean()
    max_prob = results_df["podium_probability"].max()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Drivers evaluated", n_rows)
    m2.metric(f"Predicted podium (≥ {threshold:.2f})", n_predicted_podium)
    m3.metric("Average podium probability", f"{avg_prob:.1%}")
    m4.metric("Highest probability", f"{max_prob:.1%}")

    # --- final Top 3 callout ---
    label_col = "driver" if "driver" in results_df.columns else None
    top3_df = results_df.head(3)
    st.subheader("🏆 Predicted Top 3 Finishers")
    medals = ["🥇", "🥈", "🥉"]
    cols = st.columns(3)
    for i, (col, (_, row)) in enumerate(zip(cols, top3_df.iterrows())):
        name = row[label_col] if label_col else f"Row {row.name}"
        col.metric(f"{medals[i]} {name}", f"{row['podium_probability']:.1%}")

    # --- table ---
    st.subheader("Full prediction table")
    st.dataframe(
        results_df[ordered_cols].style.format({"podium_probability": "{:.1%}"}),
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Download results as CSV",
        data=to_csv_bytes(results_df[ordered_cols]),
        file_name="podium_predictions.csv",
        mime="text/csv",
    )

    # --- charts ---
    st.subheader("Charts")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        bar_df = results_df.copy()
        if label_col:
            bar_df["label"] = bar_df[label_col]
        else:
            bar_df["label"] = ["Row " + str(i) for i in bar_df.index]
        fig_bar = px.bar(
            bar_df,
            x="label",
            y="podium_probability",
            color="podium_probability",
            color_continuous_scale="Turbo",
            labels={"label": "Driver", "podium_probability": "Podium probability"},
            title="Podium probability by driver",
            text="podium_probability",
        )
        fig_bar.update_traces(
            texttemplate="%{y:.0%}",
            textposition="outside",
            marker=dict(line=dict(width=1, color="black")),
        )
        fig_bar.update_layout(
            yaxis_tickformat=".0%",
            plot_bgcolor="white",
            font=dict(size=13),
        )
        fig_bar.update_xaxes(showgrid=False)
        fig_bar.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=False)
        fig_bar.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="black",
            line_width=2,
            annotation_text=f"Threshold ({threshold:.0%})",
            annotation_position="top left",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        fig_hist = px.histogram(
            results_df,
            x="podium_probability",
            nbins=20,
            title="Distribution of predicted probabilities",
            labels={"podium_probability": "Podium probability"},
        )
        fig_hist.update_traces(
            marker=dict(color="#2E5EAA", line=dict(width=1, color="black")),
        )
        fig_hist.update_layout(
            xaxis_tickformat=".0%",
            plot_bgcolor="white",
            font=dict(size=13),
            bargap=0.05,
        )
        fig_hist.update_xaxes(showgrid=True, gridcolor="lightgray")
        fig_hist.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    if "position_driverStanding" in results_df.columns:
        st.subheader("Grid position vs. podium probability")
        fig_scatter = px.scatter(
            results_df,
            x="grid",
            y="podium_probability",
            size="wins",
            color="podium_probability",
            text=label_col if label_col else None,
            hover_data=[c for c in [label_col] if c] + FEATURE_COLUMNS,
            color_continuous_scale="Turbo",
            labels={"grid": "Starting grid position", "podium_probability": "Podium probability"},
        )
        fig_scatter.update_traces(
            marker=dict(
                sizemin=8,
                line=dict(width=1.5, color="black"),
            ),
            textposition="top center",
            textfont=dict(size=11, color="black"),
        )
        fig_scatter.update_layout(
            yaxis_tickformat=".0%",
            plot_bgcolor="white",
            font=dict(size=13),
        )
        fig_scatter.update_xaxes(showgrid=True, gridcolor="lightgray")
        fig_scatter.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=False)
        st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.info("👈 Choose an input method in the sidebar and provide race data to get started.")
