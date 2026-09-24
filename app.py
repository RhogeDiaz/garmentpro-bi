from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget

# ============================================================
# GarmentPro BI — Shiny for Python
# ============================================================
# Put your cleaned/source UCI CSV at:
#   data/productivity.csv
#
# The app loads the CSV once at startup, applies your group's
# preprocessing, and builds reusable summary tables for the
# dashboard charts.
# ============================================================

APP_DIR = Path(__file__).parent
DATA_PATHS = [APP_DIR / "data" / "productivity.csv", APP_DIR / "productivity.csv"]

FEATURES = [
    "incentive",
    "idle_time",
    "over_time",
    "smv",
    "no_of_workers",
    "no_of_style_change",
    "wip",
    "idle_men",
]


def make_demo_data(n=240, seed=42):
    """Demo data so the UI works before the real CSV is added."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=90, freq="D")
    date = rng.choice(dates, n)

    department = rng.choice(["Sewing", "Finishing"], n, p=[0.72, 0.28])
    team = rng.integers(1, 13, n)
    targeted = np.clip(rng.normal(0.78, 0.07, n), 0.60, 0.95)
    smv = np.clip(rng.normal(16, 4, n), 5, 30)
    wip = np.clip(rng.gamma(3, 40, n), 0, 250)
    overtime = np.clip(rng.normal(700, 350, n), 0, 1600)
    incentive = np.clip(rng.normal(35, 20, n), 0, 100)
    idle_time = np.clip(rng.normal(25, 20, n), 0, 100)
    idle_men = np.clip(rng.normal(5, 5, n), 0, 25)
    style_change = rng.integers(0, 4, n)
    workers = rng.integers(30, 65, n)

    productivity = (
        0.56
        + 0.0055 * incentive
        - 0.0032 * idle_time
        - 0.00008 * overtime
        + 0.0020 * smv
        + 0.00045 * workers
        - 0.018 * style_change
        + 0.00025 * wip
        + rng.normal(0, 0.06, n)
    )
    productivity = np.clip(productivity, 0.35, 0.99)

    return pd.DataFrame(
        {
            "date": pd.to_datetime(date),
            "quarter": rng.choice(["1", "2", "3", "4"], n),
            "department": department,
            "day": pd.to_datetime(date).day_name(),
            "team": team,
            "targeted_productivity": targeted,
            "smv": smv,
            "wip": wip,
            "over_time": overtime,
            "incentive": incentive,
            "idle_time": idle_time,
            "idle_men": idle_men,
            "no_of_style_change": style_change,
            "no_of_workers": workers,
            "actual_productivity": productivity,
        }
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact preprocessing supplied by the project team."""
    df = df.copy()

    # Normalize column names only.
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    # --------------------------------------------------------
    # YOUR GROUP'S CLEANING / PREPROCESSING
    # --------------------------------------------------------
    df["department"] = df["department"].str.replace(
        "sweing", "sewing", regex=False
    )
    df["quarter"] = df["quarter"].str.replace(
        "Quarter", "", regex=False
    )
    df["wip"] = df["wip"].astype(str).replace("nan", "0").astype(float)
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")

    required = [
        "date",
        "department",
        "team",
        "targeted_productivity",
        "smv",
        "wip",
        "over_time",
        "incentive",
        "idle_time",
        "idle_men",
        "no_of_style_change",
        "no_of_workers",
        "actual_productivity",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing columns in productivity.csv: " + ", ".join(missing)
        )

    for col in required:
        if col not in {"date", "department"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["department"] = df["department"].astype(str).str.strip().str.title()
    df["team"] = pd.to_numeric(df["team"], errors="coerce").astype("Int64")

    df = (
        df.dropna(subset=["date", "actual_productivity"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df


def load_data():
    for path in DATA_PATHS:
        if path.exists():
            return clean_data(pd.read_csv(path)), False
    return make_demo_data(), True


# ============================================================
# LOAD ONCE
# ============================================================
DATA, USING_DEMO = load_data()


def pretty_feature(name):
    return {
        "incentive": "Incentive",
        "idle_time": "Idle Time",
        "over_time": "Overtime",
        "smv": "SMV",
        "no_of_workers": "Workers",
        "no_of_style_change": "Style Changes",
        "wip": "WIP",
        "idle_men": "Idle Men",
    }.get(name, str(name).replace("_", " ").title())


def feature_importance_table(df):
    """Simple association placeholder; replace with real model/SHAP later."""
    rows = []
    for feature in FEATURES:
        temp = df[[feature, "actual_productivity"]].dropna()
        if len(temp) > 2 and temp[feature].nunique() > 1:
            corr = abs(temp[feature].corr(temp["actual_productivity"]))
            value = 0.0 if pd.isna(corr) else float(corr)
        else:
            value = 0.0
        rows.append((feature, value))

    return (
        pd.DataFrame(rows, columns=["feature", "importance"])
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def fmt_pct(value):
    return f"{value * 100:.1f}%"


def fmt_num(value):
    return f"{value:,.1f}"


# ============================================================
# PRECOMPUTED SUMMARIES
# ============================================================
# These are built one time and reused. This avoids repeatedly
# grouping/scanning the entire DataFrame for each chart/page.
OVERVIEW_DAILY = (
    DATA.groupby("date", as_index=False)
    .agg(
        actual=("actual_productivity", "mean"),
        target=("targeted_productivity", "mean"),
    )
)

DEPARTMENT_SUMMARY = (
    DATA.groupby("department", as_index=False)
    .agg(
        actual=("actual_productivity", "mean"),
        target=("targeted_productivity", "mean"),
    )
)

TEAM_SUMMARY = (
    DATA.groupby("team", as_index=False)["actual_productivity"]
    .mean()
    .sort_values("actual_productivity")
)

DRIVER_SUMMARY = feature_importance_table(DATA)

_DATA_DAY = DATA.copy()
_DATA_DAY["day_name"] = pd.Categorical(
    _DATA_DAY["date"].dt.day_name(),
    categories=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    ordered=True,
)
HEATMAP_SUMMARY = _DATA_DAY.pivot_table(
    index="department",
    columns="day_name",
    values="actual_productivity",
    aggfunc="mean",
    observed=False,
)

_MEDIAN = DATA["actual_productivity"].median()
LOW_AVG = DATA.loc[DATA["actual_productivity"] < _MEDIAN, FEATURES].mean()
HIGH_AVG = DATA.loc[DATA["actual_productivity"] >= _MEDIAN, FEATURES].mean()
LOW_HIGH_SUMMARY = pd.DataFrame(
    {"Low Productivity": LOW_AVG, "High Productivity": HIGH_AVG}
).reset_index()
LOW_HIGH_SUMMARY["Feature"] = LOW_HIGH_SUMMARY["index"].map(pretty_feature)

WEEKLY_HISTORY = (
    DATA.set_index("date")["actual_productivity"]
    .resample("W")
    .mean()
    .reset_index()
    .rename(columns={"actual_productivity": "actual"})
)


# ============================================================
# STYLING
# ============================================================
CSS = r"""
:root {
  --navy: #08264d;
  --navy2: #061a35;
  --blue: #1677e8;
  --ink: #142033;
  --muted: #6b778c;
  --line: #dfe7f1;
  --bg: #f4f7fb;
  --card: #ffffff;
  --green: #20a56a;
  --red: #e24c4c;
}

html, body {
  margin: 0;
  min-height: 100%;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system,
               BlinkMacSystemFont, "Segoe UI", sans-serif;
  overflow-x: hidden;
}

.dashboard-shell {
  display: grid;
  grid-template-columns: 250px minmax(0, 1fr);
  min-height: 100vh;
  padding: 0 !important;
  align-items: stretch;
}

.nav-panel {
  background: linear-gradient(180deg, var(--navy) 0%, var(--navy2) 100%);
  color: white;
  min-height: 100vh;
  height: 100%
  box-sizing: border-box;
  padding: 24px 14px;
  position: sticky;
  top: 0;
  overflow-y: auto;
  overflow-x: hidden;
}

.brand {
  font-size: 20px;
  line-height: 1.05;
  font-weight: 800;
  letter-spacing: -0.4px;
  margin: 0 8px 5px;
  white-space: nowrap;
}

.brand-sub {
  font-size: 10px;
  line-height: 1.35;
  opacity: 0.78;
  margin: 0 8px 24px;
  max-width: 205px;
}

.nav-btn {
  width: 100%;
  display: block;
  text-align: left !important;
  border: 0 !important;
  border-radius: 9px !important;
  color: #dce8f7 !important;
  background: transparent !important;
  margin-bottom: 6px !important;
  padding: 10px 12px !important;
  font-size: 13px !important;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.nav-btn:hover {
  background: rgba(255,255,255,.08) !important;
}

.nav-btn:focus {
  box-shadow: 0 0 0 2px rgba(255,255,255,.15) !important;
}

.content-panel {
  min-width: 0;
  overflow-x: hidden;
  padding: 28px;
}

.page-title {
  font-size: 29px;
  font-weight: 800;
  letter-spacing: -0.6px;
  margin: 0;
}

.page-subtitle {
  color: var(--muted);
  font-size: 13px;
  margin: 4px 0 20px;
}

.kpi {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 18px;
  box-shadow: 0 2px 10px rgba(21,45,74,.04);
  height: 100%;
  box-sizing: border-box;
}

.kpi-label { color: var(--muted); font-size: 12px; font-weight: 600; }
.kpi-value { font-size: 27px; font-weight: 800; margin-top: 2px; }
.kpi-foot { font-size: 11px; margin-top: 6px; }

.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px;
  margin-bottom: 14px;
  box-shadow: 0 2px 10px rgba(21,45,74,.04);
  box-sizing: border-box;
  min-width: 0;
  overflow: hidden;
}

.card-title { font-size: 13px; font-weight: 800; margin-bottom: 9px; }
.big-number { font-size: 48px; line-height: 1; font-weight: 900; }

.question-chip {
  display: block;
  width: 100%;
  margin: 6px 0;
  text-align: left;
  border: 1px solid #dbe7f5;
  background: #f8fbff;
  color: #22466d;
  border-radius: 9px;
  padding: 8px 10px;
  font-size: 11px;
  box-sizing: border-box;
}

.chatbox {
  min-height: 350px;
  max-height: 450px;
  overflow-y: auto;
  background: #f8fbff;
  border: 1px solid #e0ebf7;
  border-radius: 12px;
  padding: 12px;
}

.chat-user, .chat-ai {
  padding: 10px 12px;
  border-radius: 10px;
  margin: 7px 0;
  font-size: 12px;
  line-height: 1.45;
}

.chat-user { background: #e8f2ff; margin-left: 15%; }
.chat-ai { background: white; border: 1px solid #dde7f2; margin-right: 10%; }

.badge {
  display: inline-block;
  font-size: 11px;
  padding: 4px 7px;
  border-radius: 99px;
  background: #eef6ff;
  color: #1677e8;
  font-weight: 700;
}

.good { color: var(--green); }
.bad { color: var(--red); }
.about-box {
  background: white;
  border: 1px solid var(--line);
  border-radius: 15px;
  padding: 22px;
  margin-bottom: 14px;
}
.footer-note { color: #9aabc0; font-size: 10px; margin-top: 30px; }

@media (max-width: 900px) {
  .dashboard-shell { grid-template-columns: 190px minmax(0,1fr); }
  .content-panel { padding: 18px; }
  .brand { white-space: normal; }
}

@media (max-width: 700px) {
  .dashboard-shell { grid-template-columns: 1fr; }
  .nav-panel { position: relative; height: auto; min-height: unset; }
  .content-panel { padding: 14px; }
  .page-title { font-size: 23px; }
}
"""


# ============================================================
# UI HELPERS / PAGES
# ============================================================
def kpi_card(label, value, foot="", foot_class=""):
    return ui.div(
        ui.div(label, class_="kpi-label"),
        ui.div(value, class_="kpi-value"),
        ui.div(foot, class_=f"kpi-foot {foot_class}"),
        class_="kpi",
    )


def page_shell(title, subtitle, *children):
    return ui.div(
        ui.div(title, class_="page-title"),
        ui.div(subtitle, class_="page-subtitle"),
        *children,
    )


NAV_ITEMS = [
    ("overview", "▣  Overview"),
    ("drivers", "◈  Drivers"),
    ("diagnostics", "⌁  Diagnostics"),
    ("forecasting", "◔  Forecasting"),
    ("simulator", "⚙  Simulator"),
    ("assistant", "☁  AI Assistant"),
    ("about", "ⓘ  About"),
]


def nav_button(key, label):
    return ui.input_action_button(f"nav_{key}", label, class_="nav-btn")


sidebar = ui.div(
    ui.div("GarmentPro BI", class_="brand"),
    ui.div("Productivity Analytics & Insights", class_="brand-sub"),
    *[nav_button(k, label) for k, label in NAV_ITEMS],
    ui.div("Last updated", ui.br(), "UCI / demo data", class_="footer-note"),
    class_="nav-panel",
)


OVERVIEW_UI = page_shell(
    "1. Executive Overview",
    "How productive are we? A high-level view of performance, targets and key drivers.",
    ui.layout_columns(
        ui.output_ui("ov_kpi_1"),
        ui.output_ui("ov_kpi_2"),
        ui.output_ui("ov_kpi_3"),
        ui.output_ui("ov_kpi_4"),
        col_widths=[3, 3, 3, 3],
    ),
    ui.layout_columns(
        ui.div(
            ui.div(
                ui.div("Actual vs Target Productivity Over Time", class_="card-title"),
                output_widget("overview_time"),
                class_="card",
            ),
            ui.layout_columns(
                ui.div(
                    ui.div("Productivity by Department", class_="card-title"),
                    output_widget("department_bar"),
                    class_="card",
                ),
                ui.div(
                    ui.div("Avg. Productivity by Team", class_="card-title"),
                    output_widget("team_bar"),
                    class_="card",
                ),
                col_widths=[6, 6],
            ),
            width=9,
        ),
        ui.div(
            ui.div(
                ui.div("Top Drivers Associated with Productivity", class_="card-title"),
                output_widget("driver_bar_small"),
                ui.div(
                    "Simple association proxy — replace with model/SHAP values.",
                    style="font-size:10px;color:#8a97a9",
                ),
                class_="card",
            ),
            ui.div(
                ui.div("Productivity Assistant", class_="card-title"),
                *[
                    ui.div(q, class_="question-chip")
                    for q in [
                        "Why did productivity fall last week?",
                        "Which teams are below target?",
                        "What factors affect productivity most?",
                        "What is the forecast for next week?",
                    ]
                ],
                class_="card",
            ),
            width=3,
        ),
    ),
)


DRIVERS_UI = page_shell(
    "2. Productivity Drivers",
    "Which factors are most associated with productivity?",
    ui.layout_columns(
        ui.div(
            ui.div(
                ui.div("Feature Importance (Global)", class_="card-title"),
                output_widget("feature_importance"),
                class_="card",
            ),
            ui.layout_columns(
                ui.div(
                    ui.div("Incentive vs Productivity", class_="card-title"),
                    output_widget("scatter_incentive"),
                    class_="card",
                ),
                ui.div(
                    ui.div("Idle Time vs Productivity", class_="card-title"),
                    output_widget("scatter_idle"),
                    class_="card",
                ),
                ui.div(
                    ui.div("Overtime vs Productivity", class_="card-title"),
                    output_widget("scatter_overtime"),
                    class_="card",
                ),
                col_widths=[4, 4, 4],
            ),
            width=9,
        ),
        ui.div(
            ui.div(
                ui.div("SHAP-ready design", class_="card-title"),
                ui.p(
                    "The driver chart currently uses absolute correlation as a "
                    "transparent placeholder. Replace it with SHAP values from "
                    "your trained model for the final submission."
                ),
                ui.div("1. Train model", class_="badge"),
                ui.br(),
                ui.div("2. Calculate SHAP values", class_="badge"),
                ui.br(),
                ui.div("3. Feed values into this page", class_="badge"),
                class_="card",
            ),
            width=3,
        ),
    ),
)


DIAGNOSTICS_UI = page_shell(
    "3. Productivity Diagnostic",
    "Why was productivity low or high? Explore the key reasons behind performance changes.",
    ui.layout_columns(
        ui.input_date("diag_date", "Date", value=DATA["date"].min().date()),
        ui.input_select(
            "diag_department",
            "Department",
            choices=["All"] + sorted(DATA.department.dropna().unique().tolist()),
            selected="All",
        ),
        ui.input_select(
            "diag_team",
            "Team",
            choices=["All"] + sorted([str(int(x)) for x in DATA.team.dropna().unique()]),
            selected="All",
        ),
        ui.input_action_button("diag_analyze", "Analyze", class_="btn-primary"),
        col_widths=[3, 3, 3, 3],
    ),
    ui.output_ui("diag_kpis"),
    ui.layout_columns(
        ui.div(
            ui.div(
                ui.div("Productivity Heatmap", class_="card-title"),
                output_widget("heatmap"),
                class_="card",
            ),
            ui.div(
                ui.div("Low vs High Productivity Comparison", class_="card-title"),
                output_widget("low_high"),
                class_="card",
            ),
            width=8,
        ),
        ui.div(
            ui.div(
                ui.div("Possible Contributors", class_="card-title"),
                ui.output_ui("possible_contributors"),
                class_="card",
            ),
            ui.div(
                ui.div("Key Takeaway", class_="card-title"),
                ui.output_ui("diag_takeaway"),
                class_="card",
            ),
        ),
    ),
)


FORECAST_UI = page_shell(
    "4. Forecasting",
    "What is expected to happen next? Weekly productivity forecast with an uncertainty band.",
    ui.layout_columns(
        ui.output_ui("fc_kpi_1"),
        ui.output_ui("fc_kpi_2"),
        ui.output_ui("fc_kpi_3"),
        col_widths=[4, 4, 4],
    ),
    ui.div(
        ui.div("Weekly Productivity Forecast (Next 4 Weeks)", class_="card-title"),
        output_widget("forecast_chart"),
        class_="card",
    ),
    ui.layout_columns(
        ui.div(
            ui.div("Forecast Summary", class_="card-title"),
            ui.output_ui("forecast_summary"),
            class_="card",
        ),
        ui.div(
            ui.div("Forecast by Department", class_="card-title"),
            ui.output_data_frame("forecast_dept_table"),
            class_="card",
        ),
        col_widths=[6, 6],
    ),
)


SIMULATOR_UI = page_shell(
    "5. Productivity Simulator",
    "What happens if we change the inputs? Connect this page to your trained ML model.",
    ui.layout_columns(
        ui.div(
            ui.div(
                ui.div("Input Parameters", class_="card-title"),
                ui.input_select("sim_department", "Department", ["Sewing", "Finishing"]),
                ui.input_numeric("sim_team", "Team", 5, min=1, max=100),
                ui.input_numeric("sim_workers", "Workers", 55, min=1),
                ui.input_numeric("sim_style", "Style Changes", 1, min=0),
                ui.input_numeric("sim_target", "Targeted Productivity", 0.80, min=0, max=1, step=0.01),
                ui.input_numeric("sim_smv", "SMV", 15.5, min=0),
                ui.input_numeric("sim_wip", "WIP", 130, min=0),
                ui.input_numeric("sim_overtime", "Overtime", 500, min=0),
                ui.input_numeric("sim_incentive", "Incentive", 30, min=0),
                ui.input_numeric("sim_idle", "Idle Time", 0, min=0),
                ui.input_numeric("sim_idle_men", "Idle Men", 0, min=0),
                ui.input_action_button("run_prediction", "Run Prediction", class_="btn-primary"),
                class_="card",
            ),
            width=4,
        ),
        ui.div(
            ui.div(
                ui.div("Predicted Productivity", class_="card-title"),
                ui.output_ui("sim_big_prediction"),
                class_="card",
            ),
            ui.div(
                ui.div("What Influenced This Prediction?", class_="card-title"),
                output_widget("sim_contribution"),
                class_="card",
            ),
            ui.output_ui("sim_note"),
            width=8,
        ),
    ),
)


ASSISTANT_UI = page_shell(
    "6. AI Productivity Assistant",
    "Ask questions about the data, trends and model outputs.",
    ui.layout_columns(
        ui.div(
            ui.div(
                ui.div("Productivity Assistant", class_="card-title"),
                ui.div(ui.output_ui("chat_history"), class_="chatbox"),
                ui.div(
                    ui.input_text(
                        "chat_input",
                        "",
                        placeholder="Ask: Why did productivity fall last week?",
                    ),
                    ui.input_action_button("chat_send", "➤", class_="btn-primary"),
                    class_="mt-2",
                ),
                class_="card",
            ),
            width=9,
        ),
        ui.div(
            ui.div(
                ui.div("Quick Suggestions", class_="card-title"),
                *[
                    ui.input_action_button(f"quick_{i}", q, class_="question-chip")
                    for i, q in enumerate(
                        [
                            "Why did productivity fall last week?",
                            "Which teams are below target?",
                            "What factors affect productivity most?",
                            "What is the forecast for next week?",
                            "What if we increase overtime by 10%?",
                        ]
                    )
                ],
                class_="card",
            ),
            width=3,
        ),
    ),
)


ABOUT_UI = page_shell(
    "About the Project",
    "Business Intelligence final project — Productivity Management for a Bangladesh-based garment company.",
    ui.layout_columns(
        ui.div(
            ui.div(
                ui.div("Project Information", class_="card-title"),
                ui.h4("GarmentPro BI"),
                ui.p(
                    "An interactive BI dashboard designed to turn the UCI garment "
                    "employee productivity dataset into executive-friendly insights."
                ),
                ui.tags.hr(),
                ui.p(ui.tags.b("Dataset: "), "UCI Productivity Prediction of Garment Employees"),
                ui.p(ui.tags.b("Analytics: "), "Descriptive, diagnostic, forecasting and predictive analytics"),
                ui.p(ui.tags.b("Tools: "), "Python, Shiny for Python, Pandas, Plotly and your ML model"),
                class_="about-box",
            ),
            width=6,
        ),
        ui.div(
            ui.div(
                ui.div("Team Members", class_="card-title"),
                ui.p("Member 1 — Your Name"),
                ui.p("Member 2 — Your Name"),
                ui.p("Member 3 — Your Name"),
                ui.p("Member 4 — Your Name"),
                ui.tags.hr(),
                ui.p("Replace the names with your team information, roles, section, course and instructor."),
                class_="about-box",
            ),
            width=6,
        ),
    ),
    ui.layout_columns(
        ui.div(
            ui.div(
                ui.div("Our Objective", class_="card-title"),
                ui.p(
                    "Answer six management questions: What is happening? What factors are "
                    "associated with productivity? Why did performance change? What happens next? "
                    "What if a manager changes the inputs? And what can the AI assistant explain?"
                ),
                class_="about-box",
            )
        ),
        ui.div(
            ui.div(
                ui.div("Important Note", class_="card-title"),
                ui.p(
                    "The forecasting and predictive components in this starter app are placeholders "
                    "for your validated models. Replace them before presenting final results."
                ),
                class_="about-box",
            )
        ),
        col_widths=[6, 6],
    ),
)


PAGE_MAP = {
    "overview": OVERVIEW_UI,
    "drivers": DRIVERS_UI,
    "diagnostics": DIAGNOSTICS_UI,
    "forecasting": FORECAST_UI,
    "simulator": SIMULATOR_UI,
    "assistant": ASSISTANT_UI,
    "about": ABOUT_UI,
}

app_ui = ui.page_fluid(
    ui.tags.head(ui.tags.style(CSS)),
    ui.div(
        sidebar,
        ui.div(ui.output_ui("page_content"), class_="content-panel"),
        class_="dashboard-shell",
    ),
)


# ============================================================
# SERVER
# ============================================================
def server(input, output, session):
    page_state = reactive.Value("overview")

    # Correct navigation: each button changes the page state directly.
    # This avoids the old issue where an earlier click remained > 0 and
    # could override later clicks — especially the About page.
    def make_nav_handler(key):
        @reactive.effect
        @reactive.event(getattr(input, f"nav_{key}"))
        def _handler():
            page_state.set(key)

        return _handler

    for _key, _label in NAV_ITEMS:
        make_nav_handler(_key)

    @render.ui
    def page_content():
        return PAGE_MAP[page_state()]

    # ---------------- Overview ----------------
    @render.ui
    def ov_kpi_1():
        return kpi_card(
            "Average Productivity",
            fmt_pct(DATA.actual_productivity.mean()),
            "available observations",
        )

    @render.ui
    def ov_kpi_2():
        gap = (DATA.actual_productivity - DATA.targeted_productivity).mean()
        ratio = DATA.actual_productivity.mean() / DATA.targeted_productivity.mean()
        return kpi_card(
            "Target Achievement",
            fmt_pct(ratio),
            f"{gap * 100:+.1f} pp vs target",
            "good" if gap >= 0 else "bad",
        )

    @render.ui
    def ov_kpi_3():
        below = (DATA.actual_productivity < DATA.targeted_productivity).mean()
        return kpi_card(
            "Below-Target Rate",
            fmt_pct(below),
            "observations below target",
            "bad",
        )

    @render.ui
    def ov_kpi_4():
        return kpi_card(
            "Avg. Incentive (team-day)",
            fmt_num(DATA.incentive.mean()),
            "dataset average",
        )

    @render_widget
    def overview_time():
        d = OVERVIEW_DAILY
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=d.date, y=d.actual * 100, name="Actual", mode="lines+markers"))
        fig.add_trace(go.Scattergl(x=d.date, y=d.target * 100, name="Target", mode="lines", line=dict(dash="dash")))
        fig.update_layout(
            height=300,
            margin=dict(l=30, r=10, t=10, b=30),
            yaxis_title="Productivity (%)",
            xaxis_title="",
            hovermode="x unified",
            uirevision="overview-time",
        )
        return fig

    @render_widget
    def department_bar():
        long = DEPARTMENT_SUMMARY.melt("department", var_name="metric", value_name="value")
        long["metric"] = long["metric"].map({"actual": "Actual", "target": "Target"})
        fig = px.bar(long, x="department", y="value", color="metric", barmode="group", text_auto=".0%")
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=25), yaxis_tickformat=".0%", uirevision="department")
        return fig

    @render_widget
    def team_bar():
        d = TEAM_SUMMARY.copy()
        d["team"] = "Team " + d.team.astype(int).astype(str)
        fig = px.bar(d, x="actual_productivity", y="team", orientation="h", text_auto=".0%")
        fig.update_layout(height=260, margin=dict(l=20, r=10, t=5, b=15), xaxis_tickformat=".0%", uirevision="team")
        return fig

    @render_widget
    def driver_bar_small():
        d = DRIVER_SUMMARY.head(7).copy()
        d["feature"] = d.feature.map(pretty_feature)
        fig = px.bar(d, x="importance", y="feature", orientation="h", text_auto=".2f")
        fig.update_layout(height=250, margin=dict(l=5, r=15, t=5, b=10), yaxis=dict(categoryorder="total ascending"), uirevision="driver-small")
        return fig

    # ---------------- Drivers ----------------
    @render_widget
    def feature_importance():
        d = DRIVER_SUMMARY.copy()
        d["feature"] = d.feature.map(pretty_feature)
        fig = px.bar(d, x="importance", y="feature", orientation="h", text_auto=".2f")
        fig.update_layout(
            height=330,
            margin=dict(l=20, r=15, t=10, b=20),
            xaxis_title="Absolute association with actual productivity",
            yaxis_title="",
            yaxis=dict(categoryorder="total ascending"),
            uirevision="driver-global",
        )
        return fig

    def make_scatter(x_col, x_label):
        d = DATA[[x_col, "actual_productivity"]].dropna()
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=d[x_col],
                y=d.actual_productivity * 100,
                mode="markers",
                name="Observations",
                opacity=0.55,
                marker=dict(size=6),
            )
        )
        if len(d) >= 2 and d[x_col].nunique() > 1:
            coeff = np.polyfit(
                d[x_col].to_numpy(dtype=float),
                d.actual_productivity.to_numpy(dtype=float) * 100,
                1,
            )
            xs = np.linspace(d[x_col].min(), d[x_col].max(), 50)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=coeff[0] * xs + coeff[1],
                    mode="lines",
                    name="Trend",
                )
            )
        fig.update_layout(
            height=260,
            margin=dict(l=20, r=10, t=5, b=30),
            xaxis_title=x_label,
            yaxis_title="Actual Productivity (%)",
            uirevision=x_col,
        )
        return fig

    @render_widget
    def scatter_incentive():
        return make_scatter("incentive", "Incentive")

    @render_widget
    def scatter_idle():
        return make_scatter("idle_time", "Idle Time")

    @render_widget
    def scatter_overtime():
        return make_scatter("over_time", "Overtime")

    # ---------------- Diagnostics ----------------
    @reactive.calc
    def diagnostic_view():
        input.diag_analyze()
        d = DATA
        selected_date = input.diag_date()
        selected_department = input.diag_department()
        selected_team = input.diag_team()

        if selected_date:
            target = pd.Timestamp(selected_date).normalize()
            d = d.loc[d.date.dt.normalize() == target]
        if selected_department != "All":
            d = d.loc[d.department == selected_department]
        if selected_team != "All":
            d = d.loc[d.team.astype(str) == str(selected_team)]
        return d

    @render.ui
    def diag_kpis():
        d = diagnostic_view()
        if d.empty:
            return ui.div("No matching observations for the selected filters.", class_="card")

        actual = d.actual_productivity.mean()
        target = d.targeted_productivity.mean()
        gap = actual - target
        return ui.layout_columns(
            kpi_card("Actual Productivity", fmt_pct(actual)),
            kpi_card("Target Productivity", fmt_pct(target)),
            kpi_card("Gap", f"{gap * 100:+.1f} pp", "Actual − Target", "good" if gap >= 0 else "bad"),
            col_widths=[4, 4, 4],
        )

    @render_widget
    def heatmap():
        fig = px.imshow(
            HEATMAP_SUMMARY * 100,
            text_auto=".0f",
            aspect="auto",
            labels={"x": "Day", "y": "Department", "color": "Productivity (%)"},
        )
        fig.update_layout(height=300, margin=dict(l=20, r=10, t=10, b=30), uirevision="heatmap")
        return fig

    @render_widget
    def low_high():
        long = LOW_HIGH_SUMMARY.melt(
            id_vars="Feature",
            value_vars=["Low Productivity", "High Productivity"],
            var_name="Group",
            value_name="Value",
        )
        fig = px.bar(long, x="Feature", y="Value", color="Group", barmode="group")
        fig.update_layout(height=300, margin=dict(l=20, r=10, t=10, b=80), uirevision="low-high")
        return fig

    @render.ui
    def possible_contributors():
        d = diagnostic_view()
        if d.empty:
            return ui.p("No data available for this filter.")

        rows = []
        for _, row in feature_importance_table(d).head(5).iterrows():
            rows.append(
                ui.div(
                    ui.tags.b(pretty_feature(row.feature), style="font-size:12px;"),
                    ui.span(
                        f"  avg={d[row.feature].mean():,.2f}",
                        style="font-size:11px;color:#7d8b9d;",
                    ),
                    style="padding:5px 0;border-bottom:1px solid #edf1f6;",
                )
            )
        return ui.div(*rows)

    @render.ui
    def diag_takeaway():
        d = diagnostic_view()
        if d.empty:
            return ui.p("No takeaway available.")

        actual = d.actual_productivity.mean()
        target = d.targeted_productivity.mean()
        gap = actual - target
        top = feature_importance_table(d).iloc[0].feature
        return ui.p(
            f"The selected slice has average productivity of {fmt_pct(actual)} against "
            f"a target of {fmt_pct(target)} ({gap * 100:+.1f} pp). The strongest simple "
            f"association in this slice is {pretty_feature(top)}."
        )

    # ---------------- Forecasting ----------------
    @reactive.calc
    def weekly_forecast_data():
        w = WEEKLY_HISTORY.copy()
        if w.empty:
            return pd.DataFrame(columns=["date", "actual", "forecast", "lower", "upper"])

        y = w.actual.to_numpy(dtype=float)
        window = min(4, len(y))
        baseline = float(y[-window:].mean())
        slope = float(np.polyfit(np.arange(len(y)), y, 1)[0]) if len(y) >= 3 else 0.0
        future_dates = pd.date_range(w.date.max() + pd.Timedelta(days=7), periods=4, freq="W")

        future_rows = []
        for i, dt in enumerate(future_dates, start=1):
            value = float(np.clip(baseline + slope * i, 0, 1))
            spread = max(0.025, float(np.std(y[-window:]) * 0.8))
            future_rows.append(
                {
                    "date": dt,
                    "actual": np.nan,
                    "forecast": value,
                    "lower": max(0, value - spread),
                    "upper": min(1, value + spread),
                }
            )

        hist = w[["date", "actual"]].copy()
        hist["forecast"] = np.nan
        hist["lower"] = np.nan
        hist["upper"] = np.nan
        return pd.concat([hist, pd.DataFrame(future_rows)], ignore_index=True)

    @render.ui
    def fc_kpi_1():
        f = weekly_forecast_data().forecast.dropna()
        return kpi_card("Next Week Forecast", fmt_pct(f.iloc[0]), "rolling-trend demo")

    @render.ui
    def fc_kpi_2():
        f = weekly_forecast_data().forecast.dropna()
        gap = f.iloc[0] - DATA.targeted_productivity.mean()
        return kpi_card("Forecast vs Target", f"{gap * 100:+.1f} pp", "next-week forecast", "good" if gap >= 0 else "bad")

    @render.ui
    def fc_kpi_3():
        f = weekly_forecast_data().forecast.dropna().to_numpy()
        trend = "Improving" if len(f) > 1 and f[-1] > f[0] else "Declining"
        return kpi_card("Forecast Trend", trend, "next 4 weeks", "good" if trend == "Improving" else "bad")

    @render_widget
    def forecast_chart():
        d = weekly_forecast_data()
        hist = d[d.actual.notna()]
        future = d[d.forecast.notna()]

        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=hist.date, y=hist.actual * 100, name="Actual", mode="lines+markers"))
        fig.add_trace(go.Scattergl(x=future.date, y=future.forecast * 100, name="Forecast", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=future.date, y=future.upper * 100, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future.date, y=future.lower * 100, fill="tonexty", fillcolor="rgba(22,119,232,.12)", line=dict(width=0), name="Prediction Interval", hoverinfo="skip"))
        fig.add_hline(y=DATA.targeted_productivity.mean() * 100, line_dash="dash", annotation_text="Target")
        fig.update_layout(height=360, margin=dict(l=30, r=15, t=15, b=35), yaxis_title="Productivity (%)", hovermode="x unified", uirevision="forecast")
        return fig

    @render.ui
    def forecast_summary():
        f = weekly_forecast_data().forecast.dropna()
        return ui.tags.ul(
            ui.tags.li(f"Forecasted productivity for next week: {fmt_pct(f.iloc[0])}"),
            ui.tags.li(f"Expected gap to target: {(f.iloc[0] - DATA.targeted_productivity.mean()) * 100:+.1f} pp"),
            ui.tags.li(f"Week 4 forecast: {fmt_pct(f.iloc[-1])}"),
            ui.tags.li("Replace this rolling-trend demo with your validated forecasting model."),
        )

    @render.data_frame
    def forecast_dept_table():
        overall = DATA.actual_productivity.mean()
        rows = []
        for dep in sorted(DATA.department.unique()):
            dep_avg = DATA.loc[DATA.department == dep, "actual_productivity"].mean()
            target = DATA.loc[DATA.department == dep, "targeted_productivity"].mean()
            forecast = float(np.clip(dep_avg + (overall - dep_avg) * 0.15, 0, 1))
            rows.append([dep, forecast, forecast - target])

        out = pd.DataFrame(rows, columns=["Department", "Forecast", "Gap vs Target"])
        out["Forecast"] = out["Forecast"].map(lambda x: f"{x * 100:.1f}%")
        out["Gap vs Target"] = out["Gap vs Target"].map(lambda x: f"{x * 100:+.1f} pp")
        return out

    # ---------------- Simulator ----------------
    @reactive.calc
    def simulated_prediction():
        x = {
            "incentive": input.sim_incentive(),
            "idle_time": input.sim_idle(),
            "over_time": input.sim_overtime(),
            "smv": input.sim_smv(),
            "no_of_workers": input.sim_workers(),
            "no_of_style_change": input.sim_style(),
            "wip": input.sim_wip(),
            "idle_men": input.sim_idle_men(),
        }

        # Placeholder model formula. Replace with your trained model.
        pred = (
            0.55
            + 0.005 * x["incentive"]
            - 0.0027 * x["idle_time"]
            - 0.00006 * x["over_time"]
            + 0.0021 * x["smv"]
            + 0.0005 * x["no_of_workers"]
            - 0.018 * x["no_of_style_change"]
            + 0.00025 * x["wip"]
            - 0.003 * x["idle_men"]
        )
        return x, float(np.clip(pred, 0, 1))

    @render.ui
    def sim_big_prediction():
        _, prediction = simulated_prediction()
        target = input.sim_target()
        gap = prediction - target
        return ui.div(
            ui.div(fmt_pct(prediction), class_="big-number"),
            ui.div(
                f"Target: {fmt_pct(target)}  |  Predicted gap: {gap * 100:+.1f} pp",
                class_="kpi-foot " + ("good" if gap >= 0 else "bad"),
            ),
            style="padding:10px 4px 15px",
        )

    @render_widget
    def sim_contribution():
        x, _ = simulated_prediction()
        baseline = DATA[FEATURES].mean(numeric_only=True)
        coef = {
            "incentive": 0.005,
            "idle_time": -0.0027,
            "over_time": -0.00006,
            "smv": 0.0021,
            "no_of_workers": 0.0005,
            "no_of_style_change": -0.018,
            "wip": 0.00025,
            "idle_men": -0.003,
        }
        rows = [
            (pretty_feature(f), (x[f] - baseline[f]) * coef[f])
            for f in FEATURES
        ]
        d = pd.DataFrame(rows, columns=["feature", "contribution"])
        d = d.reindex(d.contribution.abs().sort_values(ascending=False).index).head(7).sort_values("contribution")
        fig = px.bar(d, x="contribution", y="feature", orientation="h", text_auto=".2f")
        fig.add_vline(x=0)
        fig.update_layout(height=310, margin=dict(l=30, r=15, t=5, b=20), xaxis_title="Contribution to prediction (demo proxy)", yaxis_title="", uirevision="sim-contribution")
        return fig

    @render.ui
    def sim_note():
        return ui.div(
            ui.div("Model integration point", class_="card-title"),
            ui.p(
                "Replace simulated_prediction() with your trained model and add "
                "real SHAP/local explanations for the final project."
            ),
            class_="card",
        )

    # ---------------- AI Assistant ----------------
    chat_messages = reactive.Value(
        [
            {
                "role": "assistant",
                "text": (
                    "Hi! I can summarize the current productivity data. This starter "
                    "version uses a rule-based insight engine; connect your LLM/API here later."
                ),
            }
        ]
    )

    def assistant_answer(question):
        q = question.lower().strip()
        avg = DATA.actual_productivity.mean()
        target = DATA.targeted_productivity.mean()
        below = (DATA.actual_productivity < DATA.targeted_productivity).mean()
        top = DRIVER_SUMMARY.iloc[0].feature

        if "below target" in q or "under target" in q:
            teams = DATA.groupby("team").actual_productivity.mean().sort_values().head(3)
            team_text = ", ".join(
                f"Team {int(i)} ({v * 100:.1f}%)" for i, v in teams.items()
            )
            return f"{below * 100:.1f}% of observations are below target. The three lowest team averages are {team_text}."

        if any(word in q for word in ["factor", "driver", "affect"]):
            factors = ", ".join(pretty_feature(x) for x in DRIVER_SUMMARY.head(5).feature)
            return f"The strongest simple associations in this dashboard are: {factors}."

        if "forecast" in q or "next week" in q:
            forecast = weekly_forecast_data().forecast.dropna()
            return f"The current rolling-trend demo forecasts {fmt_pct(forecast.iloc[0])} for next week. Replace it with your validated forecasting model before the final presentation."

        if "fall" in q or "low" in q:
            return f"Average productivity is {fmt_pct(avg)} versus an average target of {fmt_pct(target)}. The largest simple association is {pretty_feature(top)}."

        if "overtime" in q:
            return "Overtime is included in the driver analysis and productivity simulator so you can explore its relationship with productivity."

        return f"Current average productivity is {fmt_pct(avg)}, with an average target of {fmt_pct(target)}. Try asking about drivers, teams below target, forecasting, or overtime."

    @render.ui
    def chat_history():
        return ui.div(
            *[
                ui.div(
                    message["text"],
                    class_="chat-user" if message["role"] == "user" else "chat-ai",
                )
                for message in chat_messages()
            ]
        )

    @reactive.effect
    @reactive.event(input.chat_send)
    def _send_chat():
        question = input.chat_input().strip()
        if question:
            chat_messages.set(
                chat_messages()
                + [
                    {"role": "user", "text": question},
                    {"role": "assistant", "text": assistant_answer(question)},
                ]
            )

    QUICK_QUESTIONS = [
        "Why did productivity fall last week?",
        "Which teams are below target?",
        "What factors affect productivity most?",
        "What is the forecast for next week?",
        "What if we increase overtime by 10%?",
    ]

    def make_quick_handler(index):
        @reactive.effect
        @reactive.event(getattr(input, f"quick_{index}"))
        def _quick():
            question = QUICK_QUESTIONS[index]
            chat_messages.set(
                chat_messages()
                + [
                    {"role": "user", "text": question},
                    {"role": "assistant", "text": assistant_answer(question)},
                ]
            )

        return _quick

    for _index in range(len(QUICK_QUESTIONS)):
        make_quick_handler(_index)


app = App(app_ui, server)
