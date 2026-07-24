"""Streamlit dashboard for the Great Britain National Electricity Demand
Forecasting System (NESO data).

This dashboard never loads a machine learning model directly. It only
communicates with the FastAPI backend (``api.py``) over REST HTTP requests,
so the two services can be deployed, scaled, and restarted independently.

The API address is a fixed internal constant (``DEFAULT_API_URL``), not a
user-editable field: when it points at 127.0.0.1 / localhost and no server
is already listening there, this file automatically launches ``api.py`` as a
background subprocess inside the same container. This keeps the two files
fully decoupled (separate processes, separate code) while still letting the
whole app run as a single deployable unit on platforms that only run one
entry-point script, such as Streamlit Community Cloud.

Run with:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Great Britain Electricity Demand Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = "http://127.0.0.1:8000"
HORIZON_OPTIONS_HOURS = [12, 24, 48, 72, 168]
TEST_SET_MIN_DATE = dt.date(2025, 1, 1)
TEST_SET_MAX_DATE = dt.date(2025, 12, 31)
REQUEST_TIMEOUT_SECONDS = 60
API_STARTUP_MAX_WAIT_SECONDS = 15

MODE_FUTURE = "Future Forecast"
MODE_BACKTEST = "Accuracy Evaluation (Backtesting)"

AUTO_MODEL_LABEL = "Auto (best available model)"

CHART_THEME_LIGHT = "Light Background"
CHART_THEME_DARK = "Dark Background"

# Colour palettes chosen to stay legible against both a light and a dark
# page background: saturated enough to read on white, light enough to read
# on near-black.
CHART_PALETTES = {
    CHART_THEME_LIGHT: {
        "template": "plotly_white",
        "forecast": "#1F4E79",
        "actual": "#1B8A5A",
        "compare_forecast": "#C0392B",
        "average_line": "#8A8F98",
        "error_positive": "#1F6FB2",
        "error_negative": "#C0392B",
        "zero_line": "#5A5F66",
        "font_color": "#1A1A1A",
        "grid_color": "rgba(0, 0, 0, 0.10)",
    },
    CHART_THEME_DARK: {
        "template": "plotly_dark",
        "forecast": "#63B3ED",
        "actual": "#4ADE80",
        "compare_forecast": "#FF8A65",
        "average_line": "#B0B6BE",
        "error_positive": "#63B3ED",
        "error_negative": "#FF8A65",
        "zero_line": "#C7CCD1",
        "font_color": "#F2F2F2",
        "grid_color": "rgba(255, 255, 255, 0.12)",
    },
}


# --------------------------------------------------------------------------
# Minimal, professional styling (no emojis, no cartoon icons). Metric boxes
# use Streamlit's own theme-aware background variable, not a fixed colour,
# so they stay legible whether the user's Streamlit theme is light or dark.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 8px;
        padding: 14px 18px;
    }
    .app-header-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        margin-bottom: 0.1rem;
        line-height: 1.15;
    }
    .app-header-subtitle {
        font-size: 0.95rem;
        opacity: 0.75;
        margin-bottom: 0.6rem;
    }
    .app-header-rule {
        height: 3px;
        border: none;
        border-radius: 2px;
        background: linear-gradient(90deg, var(--primary-color) 0%, rgba(128,128,128,0.15) 100%);
        margin: 0.4rem 0 1.1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Auto-launch the local API backend if it is not already running.
#
# This only applies when the fixed DEFAULT_API_URL points at 127.0.0.1 /
# localhost. If this constant is changed to a real remote API URL (e.g. a
# separately deployed api.py on Render, Railway, Fly.io, etc.), nothing is
# launched here and the app simply calls that remote address instead.
# --------------------------------------------------------------------------
def _is_local_url(base_url: str) -> bool:
    return "127.0.0.1" in base_url or "localhost" in base_url


def ensure_local_api_running(base_url: str) -> None:
    if not _is_local_url(base_url):
        return

    try:
        requests.get(f"{base_url}/", timeout=2)
        return  # already running, nothing to do
    except Exception:
        pass

    if st.session_state.get("_api_subprocess_launched"):
        return  # already attempted once this session; do not retry every rerun
    st.session_state["_api_subprocess_launched"] = True

    app_dir = Path(__file__).resolve().parent
    port = base_url.rsplit(":", 1)[-1].split("/")[0]
    try:
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", port],
            cwd=str(app_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        st.session_state["_api_subprocess_error"] = str(exc)
        return

    waited = 0.0
    while waited < API_STARTUP_MAX_WAIT_SECONDS:
        time.sleep(0.5)
        waited += 0.5
        try:
            requests.get(f"{base_url}/", timeout=2)
            return
        except Exception:
            continue


# --------------------------------------------------------------------------
# API client helpers
# --------------------------------------------------------------------------
def check_api_health(base_url: str):
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"error": str(exc)}


def fetch_available_models(base_url: str):
    try:
        response = requests.get(f"{base_url}/models", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"error": str(exc), "real_model_available": False, "default_model_key": None, "models": []}


def call_predict_future(base_url: str, start_date: str, horizon_hours: int, model_key):
    payload = {"start_date": start_date, "horizon_hours": horizon_hours, "model_key": model_key}
    response = requests.post(
        f"{base_url}/predict/future", json=payload, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()


def call_predict_evaluate(base_url: str, start_date: str, end_date: str, model_key):
    payload = {"start_date": start_date, "end_date": end_date, "model_key": model_key}
    response = requests.post(
        f"{base_url}/predict/evaluate", json=payload, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()


# --------------------------------------------------------------------------
# Fixed internal API address (no "Connection Settings" UI). Auto-launch runs
# silently against this constant every time the app loads.
# --------------------------------------------------------------------------
api_base_url = DEFAULT_API_URL
ensure_local_api_running(api_base_url)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    if st.session_state.get("_api_subprocess_error"):
        st.caption(f"Could not auto-start local API: {st.session_state['_api_subprocess_error']}")

    models_info = fetch_available_models(api_base_url)
    model_choices = models_info.get("models", [])
    model_key_by_label = {AUTO_MODEL_LABEL: None}
    for model in model_choices:
        label = model["display_name"] + (" (default)" if model["is_default"] else "")
        model_key_by_label[label] = model["key"]

    st.markdown("### Working Mode")
    mode = st.radio(
        "Select mode",
        options=[MODE_FUTURE, MODE_BACKTEST],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Model Selection")
    if model_choices:
        selected_label = st.selectbox(
            "Prediction model",
            options=list(model_key_by_label.keys()),
        )
        selected_model_key = model_key_by_label[selected_label]
    else:
        st.caption(
            "No trained model artefact detected on the API server; "
            "requests will be served by the synthetic demand simulator "
            "regardless of model selection."
        )
        selected_model_key = None

    st.markdown("---")
    st.markdown("### Time Range")

    if mode == MODE_FUTURE:
        forecast_start_date = st.date_input(
            "Forecast start date",
            value=dt.date.today(),
        )
        horizon_hours = st.selectbox(
            "Forecast horizon (hours)",
            options=HORIZON_OPTIONS_HOURS,
            index=1,
        )
        eval_start_date = None
        eval_end_date = None
    else:
        eval_date_range = st.date_input(
            "Evaluation date range (Test Set only)",
            value=(TEST_SET_MIN_DATE, TEST_SET_MIN_DATE + dt.timedelta(days=6)),
            min_value=TEST_SET_MIN_DATE,
            max_value=TEST_SET_MAX_DATE,
        )
        if isinstance(eval_date_range, tuple) and len(eval_date_range) == 2:
            eval_start_date, eval_end_date = eval_date_range
        else:
            eval_start_date = eval_end_date = eval_date_range
        forecast_start_date = None
        horizon_hours = None

    st.markdown("---")
    st.markdown("### Chart Appearance")
    chart_theme = st.select_slider(
        "Chart colours for",
        options=[CHART_THEME_LIGHT, CHART_THEME_DARK],
        value=CHART_THEME_LIGHT,
        help=(
            "Match this to your current Streamlit theme (see the app menu, "
            "top right) so chart lines and gridlines stay clearly visible "
            "against the page background."
        ),
    )

    st.markdown("---")
    run_clicked = st.button("Run Calculation", type="primary", use_container_width=True)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    '<div class="app-header-title">National Electricity Demand Forecasting System</div>'
    '<div class="app-header-subtitle">Great Britain &middot; NESO Half-Hourly Settlement '
    "Period Data</div>"
    '<hr class="app-header-rule" />',
    unsafe_allow_html=True,
)

health = check_api_health(api_base_url)
if "error" in health:
    st.error(f"API Status: Not Connected — {health['error']}")
elif health.get("real_model_available"):
    st.success("API Status: Connected — serving forecasts from the trained model.")
else:
    st.warning("API Status: Connected — no trained model found, serving the synthetic simulator.")


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def points_to_dataframe(points: list) -> pd.DataFrame:
    df = pd.DataFrame(points)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def dataframe_download_button(df: pd.DataFrame, filename: str, label: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )


def render_source_notice(source: str, model_used: str) -> None:
    if source == "synthetic":
        st.info(
            "No trained model artefact was found on the API server; the result "
            "below was produced by the synthetic demand simulator for "
            "demonstration purposes only."
        )
    else:
        st.caption(f"Model used: {model_used}")


def base_layout_kwargs(palette: dict) -> dict:
    return dict(
        template=palette["template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=palette["font_color"]),
        xaxis=dict(gridcolor=palette["grid_color"], zerolinecolor=palette["grid_color"]),
        yaxis=dict(gridcolor=palette["grid_color"], zerolinecolor=palette["grid_color"]),
    )


# --------------------------------------------------------------------------
# Future forecast view
# --------------------------------------------------------------------------
def render_future_forecast(
    base_url: str, start_date: dt.date, horizon_hours_value: int, model_key: str, theme: str
) -> None:
    st.subheader("Future National Demand Forecast")
    palette = CHART_PALETTES[theme]

    try:
        result = call_predict_future(
            base_url, start_date.strftime("%Y-%m-%d"), horizon_hours_value, model_key
        )
    except Exception as exc:
        st.error(f"Could not retrieve forecast from the API: {exc}")
        return

    df = points_to_dataframe(result["points"])
    render_source_notice(result.get("source", "model"), result.get("model_used", "unknown"))

    peak_value = float(df["predicted_mw"].max())
    trough_value = float(df["predicted_mw"].min())
    average_value = float(df["predicted_mw"].mean())

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Peak Demand", f"{peak_value:,.0f} MW")
    metric_col_2.metric("Minimum Demand", f"{trough_value:,.0f} MW")
    metric_col_3.metric("Average Demand", f"{average_value:,.0f} MW")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["predicted_mw"],
            mode="lines",
            name="Forecast (MW)",
            line=dict(color=palette["forecast"], width=2.4),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:,.0f} MW<extra></extra>",
        )
    )
    fig.add_hline(
        y=average_value,
        line_dash="dash",
        line_color=palette["average_line"],
        annotation_text="Average",
        annotation_position="top left",
    )
    fig.update_layout(
        **base_layout_kwargs(palette),
        height=440,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Time (30-Minute Settlement Period)",
        yaxis_title="Electricity Demand (MW)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Detailed Data Table"):
        display_df = df.rename(
            columns={"datetime": "Datetime", "predicted_mw": "Forecast (MW)"}
        )
        st.dataframe(display_df, use_container_width=True, height=360)
        dataframe_download_button(
            display_df,
            filename=f"forecast_{start_date.strftime('%Y%m%d')}_{horizon_hours_value}h.csv",
            label="Download CSV",
        )


# --------------------------------------------------------------------------
# Backtest / accuracy evaluation view
# --------------------------------------------------------------------------
def render_backtest(
    base_url: str, start_date: dt.date, end_date: dt.date, model_key: str, theme: str
) -> None:
    st.subheader("Accuracy Evaluation on the Test Set")
    palette = CHART_PALETTES[theme]

    try:
        result = call_predict_evaluate(
            base_url, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), model_key
        )
    except Exception as exc:
        st.error(f"Could not retrieve evaluation data from the API: {exc}")
        return

    df = points_to_dataframe(result["points"])
    render_source_notice(result.get("source", "model"), result.get("model_used", "unknown"))

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("MAPE (%)", f"{result['mape']:.2f}")
    metric_col_2.metric("MAE (MW)", f"{result['mae']:,.1f}")
    metric_col_3.metric("RMSE (MW)", f"{result['rmse']:,.1f}")
    st.caption(f"Settlement periods evaluated: {result['n_periods']:,}")

    comparison_fig = go.Figure()
    comparison_fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["actual_mw"],
            mode="lines",
            name="Actual (NESO)",
            line=dict(color=palette["actual"], width=2.4),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Actual: %{y:,.0f} MW<extra></extra>",
        )
    )
    comparison_fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["predicted_mw"],
            mode="lines",
            name="Forecast (ML)",
            line=dict(color=palette["compare_forecast"], width=2.2, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Forecast: %{y:,.0f} MW<extra></extra>",
        )
    )
    comparison_fig.update_layout(
        **base_layout_kwargs(palette),
        height=440,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Time (30-Minute Settlement Period)",
        yaxis_title="Electricity Demand (MW)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.markdown("**Comparison Chart: Actual vs. Forecast**")
    st.plotly_chart(comparison_fig, use_container_width=True)

    residual_fig = go.Figure()
    residual_fig.add_trace(
        go.Bar(
            x=df["datetime"],
            y=df["error_mw"],
            name="Error (MW)",
            marker_color=[
                palette["error_negative"] if v < 0 else palette["error_positive"]
                for v in df["error_mw"]
            ],
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Error: %{y:,.0f} MW<extra></extra>",
        )
    )
    residual_fig.add_hline(y=0, line_color=palette["zero_line"], line_width=1)
    residual_fig.update_layout(
        **base_layout_kwargs(palette),
        height=340,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Time (30-Minute Settlement Period)",
        yaxis_title="Actual Minus Predicted (MW)",
        showlegend=False,
    )
    st.markdown("**Error Chart (Residuals)**")
    st.plotly_chart(residual_fig, use_container_width=True)

    with st.expander("View Detailed Data Table"):
        display_df = df.rename(
            columns={
                "datetime": "Datetime",
                "actual_mw": "Actual (MW)",
                "predicted_mw": "Forecast (MW)",
                "error_mw": "Error (MW)",
            }
        )
        st.dataframe(display_df, use_container_width=True, height=360)
        dataframe_download_button(
            display_df,
            filename=f"evaluation_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
            label="Download CSV",
        )


# --------------------------------------------------------------------------
# Main render logic
# --------------------------------------------------------------------------
if "last_result_mode" not in st.session_state:
    st.session_state["last_result_mode"] = None

if run_clicked:
    st.session_state["last_result_mode"] = mode

if st.session_state["last_result_mode"] == MODE_FUTURE:
    if forecast_start_date is None:
        st.warning("Please select a start date and forecast horizon, then click Run Calculation.")
    else:
        render_future_forecast(
            api_base_url, forecast_start_date, horizon_hours, selected_model_key, chart_theme
        )
elif st.session_state["last_result_mode"] == MODE_BACKTEST:
    if eval_start_date is None or eval_end_date is None:
        st.warning("Please select an evaluation date range, then click Run Calculation.")
    else:
        render_backtest(
            api_base_url, eval_start_date, eval_end_date, selected_model_key, chart_theme
        )
else:
    st.markdown(
        "Select a working mode, model, and time range in the sidebar, "
        "then click **Run Calculation** to view results."
    )
