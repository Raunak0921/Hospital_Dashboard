"""
Hospital Data Dashboard
------------------------
Upload raw data (CSV/Excel/JSON/TSV/PDF) -> it gets cleaned -> KPIs and
charts are generated automatically -> edit anything, add your own columns
and charts. Your data and edits are saved locally so you never lose work.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import re
import os
import json
import textwrap
import unicodedata
import hashlib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# LOCAL PERSISTENCE
# ----------------------------------------------------------------------
STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_store")
os.makedirs(STORE_DIR, exist_ok=True)
DATA_PATH = os.path.join(STORE_DIR, "last_data.csv")  # legacy single-file store, kept for migration
CONFIG_PATH = os.path.join(STORE_DIR, "config.json")
FILES_DIR = os.path.join(STORE_DIR, "files")
MANIFEST_PATH = os.path.join(STORE_DIR, "manifest.json")
os.makedirs(FILES_DIR, exist_ok=True)

def load_local_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_local_config(cfg: dict) -> None:
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f)
    except Exception:
        pass


def load_manifest() -> dict:
    """The registry of every file the person has uploaded: {file_id: {filename, rows, included}}.
    Lets the dashboard remember multiple uploads across sessions, and lets
    the person pick which ones are actually combined into the working
    dataset right now."""
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_manifest(manifest: dict) -> None:
    try:
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f)
    except Exception:
        pass


# ----------------------------------------------------------------------
# MULTI-HOSPITAL SUPPORT
# ----------------------------------------------------------------------
# Everything above (FILES_DIR / MANIFEST_PATH / CONFIG_PATH) is a MODULE-
# LEVEL variable, and every function that reads it (load_manifest,
# save_manifest, load_local_config, save_local_config, plus the file-
# combining logic further down) looks it up fresh at call time rather than
# once at import time. That means the simplest way to make the WHOLE
# dashboard hospital-aware is to just reassign these three names to point
# at the currently-selected hospital's own folder before any of that code
# runs each rerun (done near the top of the UI section below) — no other
# function needs to change. Each hospital's uploaded files, manifest,
# calculated columns, and saved charts live in a completely separate
# folder, so switching hospitals can never mix one hospital's patient
# data into another's.
HOSPITALS_DIR = os.path.join(STORE_DIR, "hospitals")
HOSPITALS_REGISTRY_PATH = os.path.join(STORE_DIR, "hospitals_registry.json")
os.makedirs(HOSPITALS_DIR, exist_ok=True)


def _slugify_hospital_id(name: str, existing_ids) -> str:
    """Turns a display name like 'City Care Hospital' into a filesystem-
    safe, unique folder id ('city_care_hospital', or '_2' etc. suffixed if
    that id is already taken by a different hospital)."""
    base = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not base:
        base = "hospital_" + hashlib.md5(name.encode("utf-8")).hexdigest()[:6]
    candidate = base
    n = 2
    while candidate in existing_ids:
        candidate = f"{base}_{n}"
        n += 1
    return candidate


def load_hospitals_registry() -> dict:
    """{"hospitals": {hospital_id: {"name": display_name}}, "last_active":
    hospital_id}. The ordered set of every hospital this dashboard has
    been set up for, plus which one was open last (so re-opening the
    dashboard defaults back to it instead of always picking the first
    hospital in the list)."""
    if os.path.exists(HOSPITALS_REGISTRY_PATH):
        try:
            with open(HOSPITALS_REGISTRY_PATH) as f:
                data = json.load(f)
                if "hospitals" in data:
                    return data
        except Exception:
            pass
    return {"hospitals": {}, "last_active": None}


def save_hospitals_registry(registry: dict) -> None:
    try:
        with open(HOSPITALS_REGISTRY_PATH, "w") as f:
            json.dump(registry, f)
    except Exception:
        pass


def hospital_paths(hospital_id: str) -> dict:
    base = os.path.join(HOSPITALS_DIR, hospital_id)
    files_dir = os.path.join(base, "files")
    os.makedirs(files_dir, exist_ok=True)
    return {
        "files_dir": files_dir,
        "manifest_path": os.path.join(base, "manifest.json"),
        "config_path": os.path.join(base, "config.json"),
    }


def ensure_hospital_registry_initialized() -> dict:
    """One-time setup: if this dashboard has never had a hospital
    registered before, create a first hospital and migrate any pre-
    existing single-hospital data — the legacy DATA_PATH single-file
    store, or files/manifest/config sitting at the OLD top-level paths
    from before multi-hospital support existed — into it, so upgrading to
    multi-hospital support never loses data that was already uploaded."""
    registry = load_hospitals_registry()
    if registry["hospitals"]:
        return registry

    default_id = "hospital_1"
    paths = hospital_paths(default_id)

    # Migrate the legacy single-CSV store (oldest format) into this
    # hospital's manifest, the same way the old single-hospital migration
    # used to work.
    if os.path.exists(DATA_PATH):
        try:
            legacy_cleaned = clean_data(pd.read_csv(DATA_PATH))
            legacy_hash = dataframe_content_hash(legacy_cleaned)
            legacy_cleaned["_Source File"] = "Previously uploaded data"
            legacy_cleaned.to_csv(os.path.join(paths["files_dir"], f"{legacy_hash}.csv"), index=False)
            legacy_manifest = {legacy_hash: {"filename": "Previously uploaded data", "rows": len(legacy_cleaned), "included": True}}
            save_manifest_at(paths["manifest_path"], legacy_manifest)
            os.remove(DATA_PATH)
        except Exception:
            pass

    # Migrate the pre-multi-hospital top-level manifest/files/config, if any.
    if os.path.exists(MANIFEST_PATH) and not os.path.exists(paths["manifest_path"]):
        try:
            os.replace(MANIFEST_PATH, paths["manifest_path"])
        except Exception:
            pass
    if os.path.isdir(FILES_DIR):
        for fname in os.listdir(FILES_DIR):
            src, dst = os.path.join(FILES_DIR, fname), os.path.join(paths["files_dir"], fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                try:
                    os.replace(src, dst)
                except Exception:
                    pass
    if os.path.exists(CONFIG_PATH) and not os.path.exists(paths["config_path"]):
        try:
            os.replace(CONFIG_PATH, paths["config_path"])
        except Exception:
            pass

    registry = {"hospitals": {default_id: {"name": "Hospital 1"}}, "last_active": default_id}
    save_hospitals_registry(registry)
    return registry


def save_manifest_at(path: str, manifest: dict) -> None:
    """Same as save_manifest, but for an explicit path — used only during
    one-time migration, before a hospital's paths are the active globals."""
    try:
        with open(path, "w") as f:
            json.dump(manifest, f)
    except Exception:
        pass


def _canonicalize_for_hash(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes a cleaned DataFrame into one fixed, format-independent
    shape before hashing. Without this, the exact same records uploaded
    as two different file types (e.g. an .xlsx export and a .pdf export
    of the same day's data) can hash as "different files" purely because
    of how each format happened to get parsed — PDF table extraction
    yields plain text, so a date can come through as a string instead of
    a real date, and a whole number can pick up a trailing ".0" or not,
    even though the underlying value is identical. Column ORDER can also
    differ (a PDF's table columns don't always come out in the same
    order as an Excel sheet's), which alone changes a plain to_csv()
    dump even when every cell is otherwise the same. This function fixes
    all three: columns are sorted into a stable order, every date-like
    column is rendered as a plain YYYY-MM-DD string, and every numeric
    column is cast to float and rounded to 2 decimal places — so ₹23,400
    parsed as an int from Excel and ₹23400.00 parsed as text from a PDF
    both collapse to the exact same representation. ROW order is also
    normalized (sorted by every column's value) for the same reason: a
    PDF's table extraction very often returns rows in a different order
    than the source Excel/CSV sheet did (e.g. page-by-page instead of the
    sheet's original row order), and a plain to_csv() dump is sensitive to
    row order even when the set of rows is identical."""
    out = df.reindex(sorted(df.columns), axis=1)
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(float).round(2)
        else:
            out[col] = out[col].astype(str)
    try:
        out = out.sort_values(by=list(out.columns), kind="stable").reset_index(drop=True)
    except Exception:
        pass  # fall back to original row order if sorting ever fails (e.g. unhashable column contents)
    return out


def dataframe_content_hash(df: pd.DataFrame) -> str:
    """Content-based id for a cleaned DataFrame, ignoring the '_Source
    File' tag column. Two uploads of the SAME underlying data hash the
    same, whether they arrived as different filenames, different formats
    (e.g. .xlsx vs .pdf export of the same data — see
    _canonicalize_for_hash for why that needs extra normalizing), or one
    came through the old single-file migration path and the other
    through a fresh upload — so they get recognized as duplicates instead
    of both being kept and double-counted in every chart."""
    compare_cols = [c for c in df.columns if c != "_Source File"]
    try:
        canon = _canonicalize_for_hash(df[compare_cols])
        payload = canon.to_csv(index=False).encode("utf-8")
    except Exception:
        payload = str(df[compare_cols].values.tolist()).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:16]


def fuzzy_signature(df: pd.DataFrame) -> tuple:
    """A rough, format-independent fingerprint used only to WARN about a
    likely duplicate upload that the exact content hash (dataframe_content_
    hash) didn't catch — e.g. a badly-extracted PDF that split or merged a
    column, changing the exact hash without changing the substance of the
    data. Two files with the SAME row count and the SAME total across
    every numeric column are almost certainly the same underlying data,
    even if some text column parsed differently between formats."""
    compare_cols = [c for c in df.columns if c != "_Source File"]
    numeric_total = 0.0
    for col in compare_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                numeric_total += float(df[col].sum())
            except Exception:
                pass
    return (len(df), round(numeric_total, 2))


def _manifest_signature(manifest: dict, files_dir: str) -> tuple:
    """Cheap fingerprint of the current file set — just ids and mtimes, no
    file reads, no hashing. Comparing this against the last run's fingerprint
    is how we know it's safe to skip the expensive content-hash dedupe scan
    below, which otherwise re-reads and re-hashes every stored CSV on every
    single widget interaction (every filter click, every checkbox)."""
    sig = []
    for fid in sorted(manifest.keys()):
        fpath = os.path.join(files_dir, f"{fid}.csv")
        try:
            sig.append((fid, os.path.getmtime(fpath)))
        except OSError:
            sig.append((fid, None))
    return tuple(sig)


def dedupe_manifest_by_content(manifest: dict) -> tuple:
    """Scans every currently-registered file and removes any whose data is
    byte-for-byte identical (ignoring the source-file tag) to one already
    kept — fixes an existing double-counted dashboard automatically, not
    just prevents new duplicates going forward. Returns (manifest, changed)."""
    seen_hashes = {}
    changed = False
    for fid in list(manifest.keys()):
        fpath = os.path.join(FILES_DIR, f"{fid}.csv")
        if not os.path.exists(fpath):
            manifest.pop(fid, None)
            changed = True
            continue
        try:
            stored = pd.read_csv(fpath)
        except Exception:
            continue
        h = dataframe_content_hash(stored)
        if h in seen_hashes:
            try:
                os.remove(fpath)
            except Exception:
                pass
            manifest.pop(fid, None)
            changed = True
        else:
            seen_hashes[h] = fid
    return manifest, changed


# ----------------------------------------------------------------------
# PAGE SETUP / STYLE
# ----------------------------------------------------------------------
st.set_page_config(page_title="Hospital Data Dashboard", layout="wide")

PALETTE = ["#0B5394", "#1C7C54", "#3D5A80", "#EE8434", "#5B8C5A",
           "#3A6EA5", "#C1666B", "#4A7C82", "#8D6A9F", "#D4A017"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE
CHART_HEIGHT = 260

# A single fixed color for every "Total" bar anywhere in the dashboard —
# distinct from the regular entity palette so a total always reads as a
# total, in every chart that has one.
TOTAL_COLOR = "#3B4652"


def entity_color(label) -> str:
    """Deterministic color for a category label (doctor, department,
    panel, day, etc.), so the same label always renders in the same color
    across every chart in the dashboard — not just within one chart's own
    sort order, which changes from chart to chart. "Total" (any casing/
    spacing) always gets the fixed TOTAL_COLOR instead, so every total bar
    anywhere shares one color too."""
    key = str(label).strip()
    if key.lower() == "total":
        return TOTAL_COLOR
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(PALETTE)
    return PALETTE[idx]

# A single modebar style used everywhere: a fixed dark, semi-opaque chip
# with light icons (and a blue hover/active state). This is intentionally
# NOT theme-dependent — giving the toolbar its own consistent background
# keeps its icons readable whether the surrounding app is in light or dark
# mode, instead of the near-invisible "white on white"/"gray on dark" look
# a theme-matched, low-opacity background produced before.
MODEBAR_STYLE = dict(
    orientation="v",
    bgcolor="rgba(30,41,59,0.85)",
    color="#CBD5E1",
    activecolor="#2E86DE",
)



st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #F4F7F9 !important;
        border: 1px solid #DCE3E8;
        border-radius: 8px;
        padding: 8px 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
        min-height: 64px;
    }
    /* Blanket rule first: every bit of text inside a metric card is dark
       and visible, regardless of which tag Streamlit renders it in. */
    div[data-testid="stMetric"] * {
        color: #1B2A41 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #1B2A41 !important;
    }
    /* Then make the big number stand out in blue */
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #0B5394 !important;
        -webkit-text-fill-color: #0B5394 !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
    }
    div[data-testid="stMetricLabel"] p {
        font-weight: 600 !important;
        font-size: 0.72rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        word-break: break-word !important;
        line-height: 1.1 !important;
    }
    h1 { color: #2E86DE; margin-bottom: 0.6rem !important; padding-bottom: 0 !important; font-size: 1.9rem !important; }
    h5 { margin-top: 0.2rem !important; margin-bottom: 0.2rem !important; }
    div.block-container {
        padding-top: 2.6rem !important;
        padding-bottom: 1rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FBFCFD;
        border-radius: 12px;
        padding: 4px 4px;
        margin-bottom: 0.6rem;
        overflow: hidden;
    }
    div[data-testid="stHorizontalBlock"] { gap: 0.8rem !important; }
    div[data-testid="stElementContainer"] { margin-bottom: 0.3rem !important; }
    div[data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
    div[data-testid="stExpander"] { margin-bottom: 0.4rem !important; }
    div[data-testid="stMarkdown"] p { margin-bottom: 0.2rem !important; }

    /* --- Chart-overlap fixes ---
       Plotly titles/traces can render slightly outside their own SVG's
       box on narrow columns; clipping that overflow stops one chart's
       heading from bleeding onto the chart next to it. */
    div[data-testid="stPlotlyChart"] { overflow: hidden !important; }
    div[data-testid="stPlotlyChart"] > div { overflow: hidden !important; }
    /* Shrink + tuck the modebar into the corner so zoom/pan controls stay
       available but never sit on top of a title or another chart. Gets its
       own rounded, dark chip (matching MODEBAR_STYLE in Python) so the
       icons stay clearly visible regardless of whether the surrounding
       app is in light or dark theme. */
    .js-plotly-plot .modebar {
        transform: scale(0.78);
        transform-origin: top right;
        border-radius: 6px;
        overflow: hidden;
    }
    .js-plotly-plot .modebar-btn svg path {
        fill: #CBD5E1 !important;
    }
    .js-plotly-plot .modebar-btn:hover svg path {
        fill: #2E86DE !important;
    }
    .js-plotly-plot .modebar-btn.active svg path {
        fill: #2E86DE !important;
    }

    /* --- Dashboard heading: never wrap to a second line ---
       Keeps "Hospital Executive Dashboard" on one row next to the range
       selector even in a narrow column; long text truncates with an
       ellipsis instead of pushing onto a second line. */
    .dashboard-heading {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin: 0 !important;
        line-height: 1.3 !important;
    }

    /* --- Average stat line under a chart ---
       Sits below the chart's own x-axis, inside the same bordered panel,
       so it never overlaps the bars/labels or bleeds into the next card. */
    .panel-avg-stat {
        text-align: center;
        font-size: 0.78rem;
        font-weight: 600;
        color: #1B2A41;
        margin-top: -10px !important;
        padding-bottom: 4px;
    }
    .panel-avg-stat span {
        color: #0B5394;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# NUMBER / CURRENCY FORMATTING (Indian numbering system)
# ----------------------------------------------------------------------
CURRENCY_HINTS = ("amount", "revenue", "price", "cost", "fee", "charge",
                   "bill", "due", "arpp", "target")
_CURRENCY_HINT_PATTERN = re.compile(
    r"\b(" + "|".join(CURRENCY_HINTS) + r")\b", re.IGNORECASE
)

def col_is_currency(name) -> bool:
    return bool(_CURRENCY_HINT_PATTERN.search(str(name)))

def kpi_is_currency(key: str) -> bool:
    return bool(_CURRENCY_HINT_PATTERN.search(key))

def format_number(value, currency: bool = False) -> str:
    """Formats using the Indian numbering system:
    1,000 -> '1K', 1,00,000 -> '1 L', 1,00,00,000 -> '1 Cr'.
    Pass currency=True to prefix the value with the rupee sign."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    sign = "-" if value < 0 else ""
    v = abs(value)
    symbol = "\u20b9" if currency else ""
    if v >= 1_00_00_000:
        s = f"{v/1_00_00_000:.2f}".rstrip("0").rstrip(".")
        return f"{sign}{symbol}{s} Cr"
    if v >= 1_00_000:
        s = f"{v/1_00_000:.2f}".rstrip("0").rstrip(".")
        return f"{sign}{symbol}{s} L"
    if v >= 1_000:
        s = f"{v/1_000:.2f}".rstrip("0").rstrip(".")
        return f"{sign}{symbol}{s}K"
    if v == int(v):
        return f"{sign}{symbol}{int(v)}"
    return f"{sign}{symbol}{v:.2f}"


def indian_axis_ticks(max_val, currency: bool = True, n: int = 5):
    """Builds n evenly spaced y-axis tick values/labels in Indian units,
    since Plotly has no built-in 'K / L / Cr' tick formatter."""
    if not max_val or max_val <= 0:
        return [0], [format_number(0, currency)]
    step = max_val / (n - 1)
    vals = [round(step * i, 2) for i in range(n)]
    text = [format_number(v, currency) for v in vals]
    return vals, text


def wrap_title(title: str, width: int = 24) -> str:
    """Wraps long chart titles onto multiple lines so they never overflow
    their own chart's width and bleed into a neighboring chart."""
    if not title:
        return title
    lines = textwrap.wrap(title, width=width)
    return "<br>".join(lines) if lines else title


def apply_date_ticks(fig, dates, step: int = 5):
    """Forces a clean, evenly spaced set of x-axis date labels (every Nth
    day) instead of leaving tick selection to Plotly, which otherwise
    shows an inconsistent, seemingly random subset of dates."""
    try:
        clean = pd.to_datetime(pd.Series(list(dates))).dropna().sort_values().unique()
    except Exception:
        return
    if len(clean) == 0:
        return
    n = len(clean)
    use_step = 1 if n <= 10 else step
    tickvals = list(clean[::use_step])
    if len(tickvals) == 0 or tickvals[-1] != clean[-1]:
        tickvals.append(clean[-1])
    ticktext = [pd.Timestamp(d).strftime("%d %b") for d in tickvals]
    # Steeper angle + smaller font + automargin: date labels stay legible
    # and non-overlapping even when the chart's own column gets narrow
    # (e.g. two charts side by side), the same treatment already used for
    # category-name x-axes in colorful_bar.
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext,
                      tickangle=-45, tickfont=dict(size=10), automargin=True)


# Shared Plotly config: keeps zoom/pan/reset/download but drops the
# rarely-used buttons that most often crowd into a chart's title area.
PLOTLY_CONFIG = {
    "displayModeBar": "hover",  # only shows the toolbar while hovering the chart
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "select2d", "lasso2d", "toggleSpikelines",
        "hoverCompareCartesian", "hoverClosestCartesian",
    ],
    "responsive": True,
}


def styled_fig(fig, title="", hide_yaxis=True, wrap_width=24):
    wrapped_title = wrap_title(title, width=wrap_width) if title else ""
    n_lines = wrapped_title.count("<br>") + 1 if wrapped_title else 0
    top_margin = 15 if not title else (36 + 18 * n_lines)
    layout_kwargs = dict(
        showlegend=False,
        margin=dict(t=top_margin, b=42, l=15, r=15),
        height=CHART_HEIGHT,
        # Vertical modebar = a slim column of icons at the top-right corner
        # instead of a wide horizontal strip, so it never overlaps a
        # left-aligned title.
        modebar=MODEBAR_STYLE,
    )
    if title:
        layout_kwargs["title"] = dict(
            text=wrapped_title, x=0.02, xanchor="left", y=0.95, yanchor="top",
            font=dict(size=13),
        )
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None, visible=not hide_yaxis)
    return fig

def _split_total_row(data: pd.DataFrame, x: str):
    """If the last row of `data` is the "Total" bar the hospital panels
    append (see the `pd.concat([..., Total row])` pattern used throughout),
    split it out from the rest. Returns (main_df, total_row) — total_row is
    None when the last row isn't a Total bar, or there's nothing to compare
    it against (fewer than 2 rows)."""
    if len(data) < 2:
        return data, None
    if str(data[x].iloc[-1]).strip().lower() != "total":
        return data, None
    return data.iloc[:-1], data.iloc[-1]


# A category breakdown (departments, doctors, panels, periods) is very
# often long-tailed — one or two big categories next to a long list of
# much smaller ones, or a "No department"/"Total" bucket that dwarfs every
# real entry. On a plain linear scale that makes every smaller bar shrink
# down to an invisible sliver. Raising each value to this power compresses
# that range for the bar's on-screen HEIGHT only — the number printed on
# top of every bar is always the real, uncompressed value (see
# format_number calls below), and the y-axis itself is always hidden, so
# nothing about this is visible to the reader except "small bars are
# actually easy to see now". Only applied to category axes, not real
# day-by-day date trends, so a genuine time series still reads linearly.
BAR_HEIGHT_POWER = 0.4


def _compress_heights(values):
    """Applies BAR_HEIGHT_POWER to a value or a Series/array of
    non-negative values, for use as bar HEIGHT — never as a displayed
    number."""
    if isinstance(values, pd.Series):
        return np.power(values.clip(lower=0).astype(float), BAR_HEIGHT_POWER)
    return float(max(values, 0)) ** BAR_HEIGHT_POWER


def colorful_bar(data, x, y, title="", currency=False, date_axis=False, wrap_width=24):
    """Bar chart with a distinct color per bar, value shown big and clear on
    top of each bar (in Indian units), no axis titles, and the y-axis
    hidden since the value is already printed on the bar. Colors come from
    entity_color(), so the same label (doctor, department, panel, etc.)
    always renders in the same color across every chart, and every "Total"
    bar anywhere in the dashboard shares one fixed color.

    On a category axis (date_axis=False), every bar's HEIGHT is compressed
    with _compress_heights() so a long tail of small categories next to
    one or two big ones (or an appended "Total") doesn't squash the small
    ones down to nothing — see BAR_HEIGHT_POWER above. When the data ends
    with an appended "Total" bar, that bar is additionally drawn on its
    own secondary y-axis, ranged so it lands a touch taller than the
    tallest individual bar (a visual cue that it's the sum) rather than
    towering over everything else. The printed number on top of a bar is
    always its real value either way, and the y-axis itself stays hidden,
    so none of this rescaling is visible as an axis discontinuity."""
    main, total_row = _split_total_row(data, x)
    color_keys_all = data[x].astype(str)
    color_map = {key: entity_color(key) for key in color_keys_all.unique()}

    if total_row is not None and main[y].max() > 0:
        total_val = total_row[y]
        main_text = main[y].apply(lambda v: format_number(v, currency))
        main_heights = main[y] if date_axis else _compress_heights(main[y])
        total_height = total_val if date_axis else _compress_heights(total_val)
        max_main_h = main_heights.max()
        fig = go.Figure()
        fig.add_bar(
            x=main[x].astype(str), y=main_heights,
            marker_color=[color_map[str(k)] for k in main[x]],
            text=main_text, textposition="outside", cliponaxis=False, textfont_size=15,
            hovertemplate="%{x}<br>" + str(y) + ": %{text}<extra></extra>",
        )
        # Secondary axis just for the Total bar, ranged so it lands a touch
        # taller than the tallest individual bar (a visual cue that it's
        # the sum) instead of towering over it at its true linear scale.
        fig.add_bar(
            x=[str(total_row[x])], y=[total_height], yaxis="y2",
            marker_color=TOTAL_COLOR,
            text=[format_number(total_val, currency)], textposition="outside",
            cliponaxis=False, textfont_size=15,
            hovertemplate="%{x}<br>" + str(y) + ": %{text}<extra></extra>",
        )
        fig.update_layout(
            yaxis=dict(range=[0, max_main_h * 1.22]),
            yaxis2=dict(range=[0, (total_height / 0.95) if total_height > 0 else 1], overlaying="y"),
        )
    else:
        text_vals = data[y].apply(lambda v: format_number(v, currency))
        heights = data[y] if date_axis else _compress_heights(data[y])
        fig = px.bar(data, x=x, y=heights, color=color_keys_all, color_discrete_map=color_map, text=text_vals)
        fig.update_traces(
            textposition="outside", cliponaxis=False, textfont_size=15,
            hovertemplate="%{x}<br>" + str(y) + ": %{text}<extra></extra>",
        )
    fig.update_layout(bargap=0.15)
    fig = styled_fig(fig, title, wrap_width=wrap_width)
    if date_axis:
        apply_date_ticks(fig, data[x])
    else:
        # Steep rotation + small font + automargin: category names (e.g.
        # doctor/department names) never overlap each other, no matter how
        # narrow the chart column gets.
        fig.update_xaxes(tickangle=-60, tickfont=dict(size=10), automargin=True)
    return fig


def line_chart(data, x, y, title="", currency=False, date_axis=False, wrap_width=24,
               always_show_labels=False, show_stat_lines=False):
    fig = px.line(data, x=x, y=y, markers=True, color_discrete_sequence=[PALETTE[0]])
    fig.update_traces(line=dict(width=3), marker=dict(size=7))
    # With many points (e.g. a full month of daily data), per-point value
    # labels collide into an unreadable mess. Normally we fall back to an
    # axis scale in that case; passing always_show_labels=True instead keeps
    # per-point labels but staggers them above/below each point so dense
    # data doesn't overlap.
    show_labels = always_show_labels or len(data) <= 15
    if show_labels:
        text_vals = data[y].apply(lambda v: format_number(v, currency))
        dense = always_show_labels and len(data) > 15
        if dense:
            # Alternate top/bottom placement point-by-point: adjacent labels
            # never sit at the same height, so they don't collide even when
            # points are packed close together.
            positions = ["top center" if i % 2 == 0 else "bottom center" for i in range(len(data))]
            label_font_size = 10
        else:
            positions = "top center"
            label_font_size = 13
        fig.update_traces(text=text_vals, textposition=positions, textfont_size=label_font_size,
                           mode="lines+markers+text", cliponaxis=False)
        fig = styled_fig(fig, title, hide_yaxis=True, wrap_width=wrap_width)
    else:
        fig = styled_fig(fig, title, hide_yaxis=False, wrap_width=wrap_width)
        max_val = data[y].max()
        tickvals, ticktext = indian_axis_ticks(max_val, currency=currency, n=5)
        fig.update_yaxes(tickvals=tickvals, ticktext=ticktext)
    fig.update_traces(hovertemplate="%{x}<br>" + str(y) + ": %{y:,.0f}<extra></extra>")
    if date_axis:
        apply_date_ticks(fig, data[x])

    if show_stat_lines and len(data):
        y_min, y_max, y_avg = data[y].min(), data[y].max(), data[y].mean()
        # Dashed reference lines with their value in the label. Anchored
        # explicitly at the right edge of the plotting area (xref="paper",
        # x=1) with xanchor="left", so each label grows rightward into the
        # margin instead of being centered on the plot's edge — which is
        # what was causing "Avg"/"Max"/"Min" to render half off-screen and
        # get sliced by the chart container's overflow:hidden.
        stat_lines = [
            (y_avg, f"Avg {format_number(y_avg, currency)}", PALETTE[3]),
            (y_max, f"Max {format_number(y_max, currency)}", PALETTE[1]),
            (y_min, f"Min {format_number(y_min, currency)}", PALETTE[6]),
        ]
        for y_val, label, color in stat_lines:
            fig.add_hline(
                y=y_val, line_dash="dash", line_width=1.5, line_color=color,
                annotation_text=label, annotation_font_size=12, annotation_font_color=color,
                annotation_xref="paper", annotation_x=1.0,
                annotation_xanchor="left", annotation_yanchor="middle",
            )
        # A little headroom above/below so the max/min lines and their
        # point labels never get clipped at the very edge of the chart.
        pad = (y_max - y_min) * 0.18 or max(y_max, 1) * 0.1
        fig.update_yaxes(range=[y_min - pad, y_max + pad])
        # Extra right margin so the "Avg/Max/Min <value>" labels anchored
        # just outside the plot area have room to fully render instead of
        # being cut off by the chart container's overflow:hidden.
        current_margin = fig.layout.margin
        fig.update_layout(margin=dict(
            t=current_margin.t, b=current_margin.b, l=current_margin.l, r=72,
        ))
    return fig



# ----------------------------------------------------------------------
# 1. LOAD  (cached so re-running the app or tweaking a filter never
#    re-parses the file — this is the main speed fix)
# ----------------------------------------------------------------------
def _read_by_extension(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    import io
    name = file_name.lower()
    buf = io.BytesIO(file_bytes)
    if name.endswith(".csv"):
        return pd.read_csv(buf)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    if name.endswith(".json"):
        return pd.read_json(buf)
    if name.endswith(".tsv"):
        return pd.read_csv(buf, sep="\t")
    if name.endswith(".pdf"):
        return _load_pdf_tables(buf)
    raise ValueError("Unsupported file type. Use CSV, Excel, JSON, TSV, or PDF.")


def _clean_pdf_cell(value) -> str:
    """One PDF table cell -> clean single-line text. pdfplumber returns None
    for an empty cell and keeps the newline wherever the PDF wrapped a long
    value onto a second line inside its own cell — both of which turn into
    junk category names once they reach the charts."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\n", " ")).strip()


def _load_pdf_tables(buf) -> pd.DataFrame:
    """Extract tables from a PDF (e.g. a sheet exported to PDF) and stitch
    same-shaped tables from every page into one DataFrame.

    Every cell is cleaned before anything downstream sees it, and a repeated
    header row is recognised by its comparison key rather than exact
    equality, so a header whose spacing came out slightly different on a
    later page is skipped instead of being kept as a data row (which is one
    way a header word can end up glued onto a real value)."""
    import pdfplumber

    header = None
    header_key = None
    rows = []
    with pdfplumber.open(buf) as pdf:
        for page in pdf.pages:
            for raw_table in page.extract_tables():
                if not raw_table:
                    continue
                table = [[_clean_pdf_cell(cell) for cell in row] for row in raw_table]
                table = [row for row in table if any(cell for cell in row)]
                if not table:
                    continue
                if header is None:
                    header = table[0]
                    header_key = [_label_key(cell) for cell in header]
                    body = table[1:]
                elif [_label_key(cell) for cell in table[0]] == header_key:
                    body = table[1:]  # repeated header on this page, skip it
                else:
                    body = table
                for row in body:
                    if [_label_key(cell) for cell in row] == header_key:
                        continue  # a header repeated mid-page
                    if len(row) < len(header):
                        row = row + [""] * (len(header) - len(row))
                    elif len(row) > len(header):
                        row = row[: len(header)]
                    rows.append(row)
        if header is None:
            raise ValueError("No tables could be found in this PDF.")
    return pd.DataFrame(rows, columns=header).replace("", np.nan)


@st.cache_data(show_spinner="Reading and cleaning your file…")
def load_and_clean(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    raw_df = _read_by_extension(file_bytes, file_name)
    cleaned = clean_data(raw_df)
    # Tag every row with its source file — lets multiple uploads be told
    # apart later (visible in "View cleaned data", and usable as an
    # ordinary entity filter via the sidebar's "Filter by" dropdown).
    cleaned["_Source File"] = file_name
    return cleaned


@st.cache_data(show_spinner="Loading your saved data…")
def load_local_data(path: str, mtime: float) -> pd.DataFrame:
    raw_df = pd.read_csv(path)
    return clean_data(raw_df)


@st.cache_data(show_spinner="Combining your uploaded files…")
def load_combined_stored_files(file_specs: tuple) -> tuple:
    """file_specs: tuple of (path, mtime) pairs for every currently-included
    stored file (mtime included purely so the cache invalidates if a file
    changes). Each stored file was already cleaned once before saving, but
    round-tripping through CSV loses dtype info (dates/numbers come back
    as plain strings) — so the combined result is run through clean_data()
    once more here. That also means text categories get deduped
    CONSISTENTLY ACROSS every uploaded file (e.g. a department name
    spelled slightly differently in two different monthly uploads still
    merges into a single bar in the charts, not two).

    Also collapses rows that are identical across two different files once
    the "_Source File" tag is ignored — e.g. the same file re-uploaded
    under a different name, or a legacy-migrated copy sitting alongside a
    fresh re-upload of that same data. Without this, the exact same
    patient record counted once per matching file, silently doubling
    every KPI and chart. Returns (combined_df, duplicate_rows_removed).

    HANDLING UPDATED DUES ACROSS MONTHS: a patient's Cash Due / Credit Due
    at discharge is only what was still owed AT THAT TIME. When the person
    later settles that due, the hospital's own export for that later pull
    can re-include the SAME record with an updated (usually lower, now
    partly/fully paid) Cash Due / Credit Due. That is not a duplicate row
    to just drop — it is the current status of an existing record.

    IMPORTANT: a single visit is often made up of MULTIPLE rows (e.g. an
    OPD encounter with one row per service — consultation, lab test, etc.
    — all sharing the same UHID/Doa/DOD/Type). So matching "is this the
    same record" on UHID+Doa+DOD+Type alone is NOT safe — it would treat
    every one of a patient's distinct same-day service rows as duplicates
    of each other and silently delete real rows/revenue. Instead, two rows
    (whether from the same file or two different files) are only treated
    as "the same record, due amount updated" when EVERY column matches
    except Cash Due / Credit Due themselves — i.e. same UHID, Doa, DOD,
    Type, doctor, department, panel, revenue, service name, bed days, etc.
    Only then is it safe to assume they're the same billing line with a
    payment update, and the version from the MOST RECENTLY UPLOADED file
    is kept (files are read oldest to newest, later rows win). Because
    Doa/DOD/Month/Period stay untouched, the reduced due still shows up in
    the record's original month everywhere in the dashboard. Non-hospital
    data (no due columns detected) falls back to the original whole-row
    duplicate check.
    """
    # Oldest-uploaded first, so when two files describe the same record,
    # "keep the last one" means "keep the most recently uploaded one".
    ordered_specs = sorted(file_specs, key=lambda spec: spec[1])
    frames = []
    for path, _mtime in ordered_specs:
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(), 0
    combined_raw = pd.concat(frames, ignore_index=True)
    rows_before = len(combined_raw)

    visit_id_present = any(c in combined_raw.columns for c in ("UHID", "Doa", "DOD", "Type"))
    cash_due_col = find_col(combined_raw, "Cash Due", "Cash Amount", "Cash")
    credit_due_col = find_col(combined_raw, "Credit Due", "Credit Amount", "Credit", "Outstanding")
    due_cols = [c for c in (cash_due_col, credit_due_col) if c]

    if visit_id_present and due_cols:
        # Every column EXCEPT the due amounts (and the source-file tag)
        # must match for two rows to be considered the same record — this
        # is deliberately narrow so distinct same-day service rows for one
        # patient are never mistaken for duplicates of each other.
        key_cols = [c for c in combined_raw.columns if c not in due_cols and c != "_Source File"]
        combined_raw = combined_raw.drop_duplicates(subset=key_cols, keep="last")
    else:
        dedup_cols = [c for c in combined_raw.columns if c != "_Source File"]
        combined_raw = combined_raw.drop_duplicates(subset=dedup_cols if dedup_cols else None, keep="first")

    duplicates_removed = rows_before - len(combined_raw)
    return clean_data(combined_raw), duplicates_removed


# ----------------------------------------------------------------------
# 2. CLEAN  (generic, works on any tabular file)
# ----------------------------------------------------------------------
DATE_PATTERN = re.compile(
    r"^\s*("
    r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}(\s+\d{1,2}:\d{2}(:\d{2})?)?"   # 01-08-2026, 2026/08/01, with optional time
    r"|\d{1,2}[-\s][A-Za-z]{3,9}[-\s]\d{2,4}"                      # 01-Aug-2026, 01 August 2026 (common PDF export style)
    r")\s*$",
    re.IGNORECASE,
)


def _label_key(value) -> str:
    """Format-independent comparison key for a label: lowercase, every space
    and punctuation mark removed. 'Obs & Gyne Te' and 'obs&gyne  te' both
    become 'obsgynete', so a spacing difference between an Excel cell and a
    PDF cell can't split one category into two bars."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _squeeze(text: str) -> str:
    """Collapses runs of repeated characters: 'departmentt' -> 'department'."""
    return re.sub(r"(.)\1+", r"\1", text)


def _is_typo_variant(a: str, b: str) -> bool:
    """True when two comparison keys are almost certainly the same label, one
    of them mangled by PDF table extraction.

    Catches the two things that actually happen:
      * a repeated letter  ('no departmentt' vs 'no department')
      * 1-2 stray ALPHABETIC characters stuck on the front or the back, which
        is what pdfplumber produces when a column boundary is slightly off or
        a cell's text wrapped — 'hmedicine' vs 'medicine',
        'aobsgynete' vs 'obsgynete', 'gorthopedictea' vs 'orthopedictea'

    Deliberately strict so real categories are never merged:
      * both labels must be at least 4 characters
      * the extra characters must be LETTERS, so 'ward1' and 'ward11' — or
        any genuinely numbered category — are left alone
      * a gap of 3+ characters is never treated as a typo, so 'Medicine'
        ('medicine') and 'Medicine Team' ('medicineteam') stay separate bars
    """
    if not a or not b or a == b:
        return False
    if min(len(a), len(b)) < 4:
        return False
    if _squeeze(a) == _squeeze(b):
        return True
    long_, short_ = (a, b) if len(a) > len(b) else (b, a)
    diff = len(long_) - len(short_)
    if not 1 <= diff <= 2:
        return False
    if long_.endswith(short_):
        extra = long_[:diff]      # junk glued to the FRONT ('hMedicine')
    elif long_.startswith(short_):
        extra = long_[-diff:]     # junk glued to the BACK
    else:
        return False
    return extra.isalpha()


def _normalize_text_value(v):
    """Collapses cosmetic differences that would otherwise create duplicate
    categories (e.g. two 'No Department' bars): normalizes unicode (so a
    non-breaking space or curly quote matches its plain equivalent), strips
    invisible formatting/control characters (zero-width spaces, soft
    hyphens, direction marks, stray NUL bytes, etc. — anything in Unicode
    category 'C') that render as nothing but still make two visually
    identical strings compare as different, then squashes any run of
    whitespace (tabs, double spaces, stray newlines) down to a single
    regular space and trims the ends."""
    if pd.isna(v):
        return v
    text = unicodedata.normalize("NFKC", str(v))
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _dedupe_case_insensitive(series: pd.Series) -> pd.Series:
    """Folds every spelling of the same category into one label, so each real
    department/doctor/panel draws exactly one bar.

    Handles case-only differences and doubled letters (as before), plus the
    stray-leading/trailing-character variants a PDF extraction produces:
    'hMedicine' into 'Medicine', 'aObs & Gyne Te' into 'Obs & Gyne Te',
    'MDr. Madhuri' into 'Dr. Madhuri'. Comparison is done on `_label_key`
    (spacing/punctuation-insensitive), and the surviving display label is
    always the spelling that occurs most often — the clean one — so a
    PDF-sourced hospital ends up with the same bars as an Excel-sourced one.

    Guards against merging two genuinely different categories:
      * `_is_typo_variant` only allows a 1-2 letter difference
      * a spelling is only folded into one that's clearly dominant — at
        least 3x more common, or simply more common when the rarer spelling
        appears no more than twice (what a one-off extraction glitch looks
        like)
    """
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return series

    keys = non_null.map(_label_key)
    n_unique = keys.nunique()
    # Free-text columns (patient names, IDs, service notes) are mostly
    # unique values — folding those would be slow and could quietly merge
    # two real people, so they're skipped entirely.
    if n_unique == len(non_null) or (n_unique > 50 and n_unique > len(non_null) * 0.5):
        return series

    counts = non_null.groupby(keys).size().to_dict()
    display = non_null.groupby(keys).agg(lambda vals: vals.value_counts().idxmax()).to_dict()

    by_frequency = sorted(counts, key=lambda k: (-counts[k], k))
    canonical_of = {k: k for k in by_frequency}

    def resolve(key: str) -> str:
        seen = set()
        while canonical_of[key] != key and key not in seen:
            seen.add(key)
            key = canonical_of[key]
        return key

    def dominates(common: str, rare: str) -> bool:
        if counts[common] >= counts[rare] * 3:
            return True
        # A one-off extraction glitch shows up once or twice; anything more
        # common than it is a safe home for it.
        return counts[rare] <= 2 and counts[common] > counts[rare]

    # Rarest spellings first, each folded into the most common spelling it
    # could plausibly be a mangled copy of.
    for rare in sorted(by_frequency, key=lambda k: counts[k]):
        for common in by_frequency:          # already ordered most-common first
            if common == rare or not dominates(common, rare):
                continue
            if _is_typo_variant(rare, common):
                canonical_of[rare] = common
                break

    final_map = {key: display[resolve(key)] for key in by_frequency}
    return series.map(lambda v: final_map.get(_label_key(v), v) if pd.notna(v) else v)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Column HEADERS get the same full normalization as cell values (not
    # just a trim of the outer edges): PDF table extraction very often
    # leaves internal formatting artifacts in header text that a plain
    # .strip() never touches — a doubled internal space ("Dept.  Name"),
    # a non-breaking space, or an invisible control character copied from
    # the PDF's fixed-width layout. Left alone, that turns "Dept. Name"
    # from an Excel/CSV upload and "Dept.  Name" from a PDF upload of the
    # exact same data into two DIFFERENT column names — which means every
    # later same-data check (the content hash used to catch a duplicate
    # upload, cross-file column matching, etc.) silently fails even though
    # every underlying value is identical.
    df.columns = [_normalize_text_value(str(c)) for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")

    for col in df.columns:
        is_text_like = not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col])
        if is_text_like:
            df[col] = df[col].map(_normalize_text_value)
            df[col] = df[col].replace({"nan": np.nan, "": np.nan, "None": np.nan})

            sample = df[col].dropna().astype(str).head(20)
            if len(sample) and sample.apply(lambda x: bool(DATE_PATTERN.match(x))).mean() > 0.7:
                # Try the unambiguous default parse first, then fall back to
                # dayfirst — a PDF-exported date string (e.g. "01-Aug-2026")
                # and the same date from an Excel cell should both land on
                # the exact same Timestamp, whichever parse gets there.
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() <= 0.7:
                    parsed_dayfirst = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                    if parsed_dayfirst.notna().mean() > parsed.notna().mean():
                        parsed = parsed_dayfirst
                if parsed.notna().mean() > 0.7:
                    df[col] = parsed
                    continue

            cleaned_numeric = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(r"[₹$€%]", "", regex=True)
                .str.strip()
                .replace("", np.nan)
            )
            numeric_candidate = pd.to_numeric(cleaned_numeric, errors="coerce")
            if numeric_candidate.notna().mean() > 0.9 and df[col].notna().any():
                df[col] = numeric_candidate
            else:
                # Still text after all conversions were tried — fold
                # case-only duplicates (e.g. "No Department" / "no department")
                # into one category so charts don't split them into two bars.
                df[col] = _dedupe_case_insensitive(df[col])

    df = df.drop_duplicates()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            pass
        else:
            df[col] = df[col].fillna("Unknown")

    return df


def find_col(df: pd.DataFrame, *candidates) -> str | None:
    """Exact match first, then loose substring match ignoring case/punctuation."""
    cols_norm = {c: re.sub(r"[\s._\-/]+", "", c).lower() for c in df.columns}
    for cand in candidates:
        cand_norm = re.sub(r"[\s._\-/]+", "", cand).lower()
        for original, norm in cols_norm.items():
            if norm == cand_norm:
                return original
    for cand in candidates:
        cand_norm = re.sub(r"[\s._\-/]+", "", cand).lower()
        for original, norm in cols_norm.items():
            if cand_norm in norm or norm in cand_norm:
                return original
    return None


ID_LIKE_PATTERN = re.compile(r"(^s\.?\s*no\.?$|^sno$|\bid$|\bids$|uhid|patient\s*name|^name$)", re.IGNORECASE)

def _is_id_like(col: str) -> bool:
    return bool(ID_LIKE_PATTERN.search(col.strip()))

def detect_columns(df: pd.DataFrame):
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [
        c for c in df.columns
        if c not in date_cols and c not in numeric_cols and df[c].nunique() <= max(50, len(df) * 0.5)
    ]
    numeric_cols = sorted(numeric_cols, key=lambda c: (_is_id_like(c), df[c].nunique()))
    categorical_cols = sorted(categorical_cols, key=lambda c: (_is_id_like(c), df[c].nunique()))
    return date_cols, numeric_cols, categorical_cols


# ----------------------------------------------------------------------
# 3. HOSPITAL SCHEMA
# ----------------------------------------------------------------------
def resolve_hospital_columns(df: pd.DataFrame) -> dict:
    return {
        "type": find_col(df, "Type"),
        "uhid": find_col(df, "UHID", "Patient ID", "MRN", "Registration No"),
        "doa": find_col(df, "Doa", "Date of Admission", "Admission Date"),
        "dod": find_col(df, "DOD", "Date of Discharge", "Discharge Date"),
        "dept": find_col(df, "Dept. Name", "Dept Name", "Department", "Department Name"),
        "doctor": find_col(df, "Doctor Name", "Doctor", "Consultant Name", "Consultant"),
        "panel": find_col(df, "Panel Name", "Panel", "Payer", "Payer Name"),
        "revenue": find_col(df, "Service Final Amount", "Service Final Amt", "Final Amount",
                             "Net Amount", "Total Amount", "Bill Amount", "Revenue"),
        "cash_due": find_col(df, "Cash Due", "Cash Amount", "Cash"),
        "credit_due": find_col(df, "Credit Due", "Credit Amount", "Credit", "Outstanding"),
        "bed_days": find_col(df, "Bed Days", "Length of Stay", "LOS", "No of Days", "Stay Days"),
        "service_name": find_col(df, "Service Name"),
        "service_type": find_col(df, "Service Type", "Category"),
        "period": find_col(df, "PERIOD", "Period"),
    }


def hospital_schema_detected(df: pd.DataFrame) -> bool:
    expected = {"Type", "Doa", "DOD", "Dept. Name", "Doctor Name", "Panel Name"}
    return expected.issubset(set(df.columns))


def coerce_to_datetime(series: pd.Series) -> pd.Series:
    """Make sure a column is real datetime64, no matter how it arrived:
    already-parsed dates, Excel serial-number dates (e.g. 46236), text
    formats the generic cleaner didn't recognize (e.g. '01-Aug-2026'), or
    an ISO timestamp string from a CSV round-trip (e.g. '2026-08-01 00:00:00').
    Tries the unambiguous default parse first — dayfirst=True is only used
    as a fallback, since applying it to an already-ISO string would wrongly
    swap day/month (e.g. turning 01-Aug into 08-Jan)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        # Excel stores dates as day counts from 1899-12-30
        return pd.to_datetime(series, unit="D", origin="1899-12-30", errors="coerce")
    default_parse = pd.to_datetime(series, errors="coerce")
    if default_parse.notna().mean() > 0.9:
        return default_parse
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def add_hospital_derived_columns(df: pd.DataFrame, c: dict) -> pd.DataFrame:
    """Adds columns used across KPIs/charts, visible in the data table too.
    The Type column has three literal categories — 'Admission', 'Discharge',
    and 'OPD' — each a distinct row, not a status derived from other columns."""
    df = df.copy()
    if c["type"]:
        type_clean = df[c["type"]].astype(str).str.strip().str.lower()
        df["Is Admission Row"] = type_clean == "admission"
        df["Is Discharge Row"] = type_clean == "discharge"
        df["Is OPD Row"] = type_clean == "opd"
    if c["doa"]:
        df[c["doa"]] = coerce_to_datetime(df[c["doa"]])
        df["Admission Day"] = df[c["doa"]].dt.date
    if c["dod"]:
        df[c["dod"]] = coerce_to_datetime(df[c["dod"]])
        df["Discharge Day"] = df[c["dod"]].dt.date
    return df


PERIOD_ORDER = ["1-10 Days", "11-20 Days", "21-31 Days"]

def ordered_period(df: pd.DataFrame, period_col: str) -> list:
    present = list(df[period_col].dropna().unique())
    ordered = [p for p in PERIOD_ORDER if p in present]
    remaining = sorted(p for p in present if p not in ordered)
    return ordered + remaining


def build_hospital_subsets(df: pd.DataFrame, c: dict, start_date=None, end_date=None) -> dict:
    """Builds the exact row-subset each KPI/chart needs, using the CORRECT
    date field per Type (Admission rows use Doa, Discharge/OPD rows use their
    own event date):
      - admission: Type == 'Admission' rows, filtered by Admission Date (Doa)
      - discharge: Type == 'Discharge' rows, filtered by Discharge Date (DOD)
      - opd:       Type == 'OPD' rows, filtered by encounter date (Doa)
      - opd_consult: opd narrowed to Service Type == 'Consultation'
    """
    admission = df[df["Is Admission Row"]] if "Is Admission Row" in df.columns else df.iloc[0:0]
    discharge = df[df["Is Discharge Row"]] if "Is Discharge Row" in df.columns else df.iloc[0:0]
    opd = df[df["Is OPD Row"]] if "Is OPD Row" in df.columns else df.iloc[0:0]

    if c["doa"] and start_date and end_date:
        admission = admission[
            (admission[c["doa"]] >= pd.Timestamp(start_date)) & (admission[c["doa"]] <= pd.Timestamp(end_date))
        ]
        opd = opd[(opd[c["doa"]] >= pd.Timestamp(start_date)) & (opd[c["doa"]] <= pd.Timestamp(end_date))]
    if c["dod"] and start_date and end_date:
        discharge = discharge[
            (discharge[c["dod"]] >= pd.Timestamp(start_date)) & (discharge[c["dod"]] <= pd.Timestamp(end_date))
        ]

    opd_consult = opd[opd[c["service_type"]] == "Consultation"] if c["service_type"] else opd

    return {"admission": admission, "discharge": discharge, "opd": opd, "opd_consult": opd_consult}


def build_hospital_kpis(subsets: dict, c: dict, n_days: int = 1) -> dict:
    admission, discharge, opd, opd_consult = subsets["admission"], subsets["discharge"], subsets["opd"], subsets["opd_consult"]

    kpis = {
        "IPD Admissions": len(admission),
        "IPD Discharges": len(discharge),
        "OPD Consultations": len(opd_consult),
    }

    if c["revenue"]:
        ipd_revenue = discharge[c["revenue"]].sum()
        opd_revenue = opd[c["revenue"]].sum()
        total_revenue = ipd_revenue + opd_revenue
        kpis["Total Revenue"] = round(total_revenue, 2)
        kpis["IPD Revenue"] = round(ipd_revenue, 2)
        kpis["IPD ARPP"] = round(ipd_revenue / max(len(discharge), 1), 2)
        # Revenue earned per calendar day across the range currently being
        # shown (Total Revenue = IPD + OPD, divided by the number of days
        # in the selected/confirmed date range).
        kpis["Per-Day Revenue"] = round(total_revenue / max(n_days, 1), 2)
    if c["cash_due"]:
        kpis["Cash Due"] = round(discharge[c["cash_due"]].sum(), 2)
    if c["credit_due"]:
        kpis["Credit Due"] = round(discharge[c["credit_due"]].sum(), 2)
    if c["bed_days"]:
        kpis["ALOS (days)"] = round(discharge[c["bed_days"]].sum() / max(len(discharge), 1), 2)
        if c["revenue"]:
            # Average charge generated per patient per day of stay: IPD
            # revenue (from discharged patients) divided by the total
            # patient-days (sum of Bed Days) those same patients accounted
            # for — not just revenue-per-patient (that's IPD ARPP already).
            total_bed_days = discharge[c["bed_days"]].sum()
            kpis["Per-Day Per-Patient Charge"] = round(ipd_revenue / max(total_bed_days, 1), 2)
    return kpis


def build_generic_kpis(df, numeric_cols) -> dict:
    kpis = {"Total Rows": len(df)}
    for col in numeric_cols[:4]:
        kpis[f"Sum of {col}"] = round(df[col].sum(), 2)
        kpis[f"Avg {col}"] = round(df[col].mean(), 2)
    return kpis


# ----------------------------------------------------------------------
# 4. STREAMLIT UI
# ----------------------------------------------------------------------

# ---------------- Hospital selector (must run before anything else) ----------------
# Reassigns FILES_DIR / MANIFEST_PATH / CONFIG_PATH to the selected
# hospital's own folder — every function below (load_manifest,
# save_manifest, load_and_clean, the file-combining logic, calculated
# columns, saved custom charts...) reads these as plain module-level
# globals at call time, so this one reassignment is all it takes to make
# the entire rest of the dashboard operate on just this hospital's data.
_hospital_registry = ensure_hospital_registry_initialized()

if "active_hospital_id" not in st.session_state:
    st.session_state.active_hospital_id = (
        _hospital_registry.get("last_active") or next(iter(_hospital_registry["hospitals"]))
    )

with st.sidebar.expander("🏥 Switch hospital", expanded=True):
    hospital_ids = list(_hospital_registry["hospitals"].keys())
    hospital_names = [_hospital_registry["hospitals"][hid]["name"] for hid in hospital_ids]
    current_idx = hospital_ids.index(st.session_state.active_hospital_id) if st.session_state.active_hospital_id in hospital_ids else 0
    picked_name = st.selectbox("Currently viewing", hospital_names, index=current_idx, key="hospital_picker")
    picked_id = hospital_ids[hospital_names.index(picked_name)]

    if picked_id != st.session_state.active_hospital_id:
        # Switching hospitals: wipe every bit of session state (filters,
        # uploader box, calculated-column/chart editors, quick-range
        # picks...) so nothing from the previous hospital's data bleeds
        # into the new one — each hospital's own saved settings get
        # reloaded fresh from its own config file below instead.
        st.session_state.clear()
        st.session_state.active_hospital_id = picked_id
        _hospital_registry["last_active"] = picked_id
        save_hospitals_registry(_hospital_registry)
        st.rerun()

    with st.expander("➕ Add a new hospital"):
        new_hosp_name = st.text_input("Hospital name", key="new_hospital_name")
        if st.button("Add hospital", key="add_hospital_btn"):
            if new_hosp_name.strip():
                new_id = _slugify_hospital_id(new_hosp_name, hospital_ids)
                _hospital_registry["hospitals"][new_id] = {"name": new_hosp_name.strip()}
                _hospital_registry["last_active"] = new_id
                save_hospitals_registry(_hospital_registry)
                hospital_paths(new_id)  # create its empty folder up front
                st.session_state.clear()
                st.session_state.active_hospital_id = new_id
                st.rerun()

    if len(hospital_ids) > 1:
        with st.expander("🗑 Remove a hospital"):
            st.caption("This permanently deletes that hospital's uploaded files and settings.")
            remove_name = st.selectbox("Hospital to remove", hospital_names, key="remove_hospital_picker")
            remove_id = hospital_ids[hospital_names.index(remove_name)]
            if st.button(f"Delete '{remove_name}'", key="remove_hospital_btn"):
                import shutil
                shutil.rmtree(os.path.join(HOSPITALS_DIR, remove_id), ignore_errors=True)
                _hospital_registry["hospitals"].pop(remove_id, None)
                remaining = list(_hospital_registry["hospitals"].keys())
                new_active = remaining[0]
                _hospital_registry["last_active"] = new_active
                save_hospitals_registry(_hospital_registry)
                st.session_state.clear()
                st.session_state.active_hospital_id = new_active
                st.rerun()

active_hospital_id = st.session_state.active_hospital_id
active_hospital_name = _hospital_registry["hospitals"].get(active_hospital_id, {}).get("name", active_hospital_id)
_paths = hospital_paths(active_hospital_id)
FILES_DIR = _paths["files_dir"]
MANIFEST_PATH = _paths["manifest_path"]
CONFIG_PATH = _paths["config_path"]

st.title(f"Hospital Data Dashboard — {active_hospital_name}")


def hkey(name: str) -> str:
    """Suffixes a widget/session-state key with the ACTIVE hospital id.

    st.session_state.clear() on a hospital switch empties the underlying
    Python-side value, but Streamlit's frontend can still keep an
    already-rendered widget's own last-shown value visible for one more
    render unless the widget's `key` itself changes — the exact same
    class of issue the file uploader above already works around with its
    own hospital-suffixed key. Without this, a dropdown like "Range type"
    could keep visually showing a previous hospital's pick (e.g. "Month")
    even though the data underneath had already reverted to Full Range —
    a dashboard that LOOKS filtered but silently isn't. Suffixing every
    filter/quick-range/calculated-column/custom-chart key with the active
    hospital id guarantees each hospital's widgets are entirely distinct
    components, so switching hospitals can never visually leak one
    hospital's selection onto another's dashboard."""
    return f"{name}__{active_hospital_id}"


# The uploader's key includes both a counter we control AND the active
# hospital id, so switching hospitals always shows an empty uploader box
# (rather than momentarily showing whatever was mid-upload for the
# previous hospital), on top of the existing reset-after-upload behavior.
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_files = st.sidebar.file_uploader(
    "Import data file(s)", type=["csv", "xlsx", "xls", "json", "tsv", "pdf"],
    accept_multiple_files=True,
    key=f"file_uploader_{active_hospital_id}_{st.session_state.uploader_key}",
)

_local_cfg = load_local_config()
if hkey("calc_columns") not in st.session_state:
    st.session_state[hkey("calc_columns")] = [tuple(x) for x in _local_cfg.get("calc_columns", [])]
if hkey("custom_charts") not in st.session_state:
    st.session_state[hkey("custom_charts")] = _local_cfg.get("custom_charts", [])

# Base (semantic) names for every filter/quick-range widget. Used as: (a)
# the JSON keys this hospital's own config.json saves/restores these
# values under (see save_current_filter_state below), and (b) the base
# that hkey() suffixes into the ACTUAL widget/session-state key, so one
# hospital's picks are never visible to another's widgets.
FILTER_AND_RANGE_KEYS = [
    "filter_start_date", "filter_end_date", "filter_date_col", "filter_entity_col", "filter_entity_val",
    "quick_range_type", "quick_range_start", "quick_range_end",
    "quick_date", "quick_week", "quick_period", "quick_month", "quick_year",
    "monthly_revenue_target",
]
_FILTER_DATE_KEYS = {"filter_start_date", "filter_end_date", "quick_range_start", "quick_range_end", "quick_date", "quick_week"}


def _saved_filter_value(name: str, fallback=None):
    """Reads this hospital's own PERSISTED value for a filter/range field
    (saved to its config.json the last time filters were applied /
    confirmed), for use as a widget's `value=`/`index=` default — this is
    what lets returning to a hospital resume exactly where you left off,
    instead of resetting to the smart computed default every time. Falls
    back to `fallback` if nothing was ever saved, or the saved value
    doesn't parse (e.g. a date string that's no longer valid)."""
    raw = _local_cfg.get("filter_state", {}).get(name, None)
    if raw is None:
        return fallback
    if name in _FILTER_DATE_KEYS:
        try:
            import datetime as _dt
            return _dt.date.fromisoformat(raw)
        except Exception:
            return fallback
    return raw


def save_current_filter_state() -> None:
    """Snapshots every currently-applied filter/quick-range value into
    THIS hospital's own config.json, so switching to a different hospital
    and back — or simply reopening the dashboard later — restores the
    exact view you left, instead of resetting to defaults. Called right
    after "Apply Filters" is submitted and right after a quick-range
    "Confirm" click (see both further down)."""
    import datetime as _dt
    filter_state = {}
    for name in FILTER_AND_RANGE_KEYS:
        if hkey(name) in st.session_state:
            val = st.session_state[hkey(name)]
            if isinstance(val, _dt.date):
                val = val.isoformat()
            filter_state[name] = val
    save_local_config({"filter_state": filter_state})



# quick_range_start / quick_range_end aren't tied to any widget (they're
# only ever set programmatically by the "Confirm" button below) — so,
# unlike the widget-bound keys above, it's always safe to pre-populate
# them directly from this hospital's saved state (no Streamlit "value set
# both via session_state and a widget parameter" conflict is possible
# here). Without this, switching back to a hospital that had a range
# confirmed would still show the right pick in the dropdown but the
# ACTUAL applied data would silently be Full Range until Confirm was
# clicked again — exactly the mismatch that was reported.
for _k in ("quick_range_start", "quick_range_end"):
    if hkey(_k) not in st.session_state:
        _sv = _saved_filter_value(_k)
        if _sv is not None:
            st.session_state[hkey(_k)] = _sv

manifest = load_manifest()

# Fix any duplicates already sitting in the registry (e.g. the same data
# stored once via the old migration path and once via a fresh re-upload
# before this content-based check existed) — this repairs an
# already-double-counted dashboard, not just prevents new duplicates.
#
# This scan reads and hashes every stored CSV, so it's only worth paying for
# when the file set has actually changed since it last ran (upload, delete,
# or the first load this session) — otherwise it would silently redo that
# full disk-read on every single widget interaction (every filter click,
# every checkbox toggle), which is the main reason things felt slow.
_dedupe_sig_key = hkey("manifest_dedupe_sig")
_current_manifest_sig = _manifest_signature(manifest, FILES_DIR)
if st.session_state.get(_dedupe_sig_key) != _current_manifest_sig:
    manifest, _deduped = dedupe_manifest_by_content(manifest)
    if _deduped:
        save_manifest(manifest)
        st.toast("ℹ️ Removed duplicate data that was counted twice.", icon="ℹ️")
    st.session_state[_dedupe_sig_key] = _manifest_signature(manifest, FILES_DIR)

# Persist every newly uploaded file into the registry, keyed by the
# content of its CLEANED data — so re-uploading the same underlying data
# (even under a different filename or file format) is recognized as
# already-stored instead of being added again and double-counted. As soon
# as anything new is safely stored, the uploader widget is reset (see
# `uploader_key` above) so the file disappears from "Import data file(s)"
# instead of sitting there indefinitely — including when the upload
# turned out to be a duplicate, so the box empties either way.
newly_added = False
for uf in uploaded_files or []:
    file_bytes = uf.getvalue()
    cleaned = load_and_clean(file_bytes, uf.name)  # cached — instant on reruns
    content_id = dataframe_content_hash(cleaned)
    if content_id in manifest:
        if manifest[content_id].get("filename") != uf.name:
            st.toast(
                f"ℹ️ '{uf.name}' has the same data as already-uploaded "
                f"'{manifest[content_id]['filename']}' — skipped to avoid double-counting.",
                icon="ℹ️",
            )
        newly_added = True  # still reset the uploader box even though nothing new was stored
        continue
    # SAFETY NET: the exact hash above is designed to catch the same data
    # arriving as a different file format (see dataframe_content_hash /
    # _canonicalize_for_hash), but a badly-extracted PDF can still mangle a
    # column just enough to slip past it (a split/merged column, a column
    # that failed to parse as numeric, etc.). Before treating this as
    # genuinely new data, do a rough format-independent comparison — same
    # row count AND the same total across every numeric column — against
    # every file already in the registry. That combination matching is a
    # strong signal of "same data, didn't clean identically" even when the
    # exact hash didn't line up, so warn instead of silently double-
    # counting it.
    possible_dupe_of = None
    new_sig = fuzzy_signature(cleaned)
    for other_id, other_meta in manifest.items():
        other_path = os.path.join(FILES_DIR, f"{other_id}.csv")
        if not os.path.exists(other_path):
            continue
        try:
            other_df = pd.read_csv(other_path)
        except Exception:
            continue
        if fuzzy_signature(other_df) == new_sig:
            possible_dupe_of = other_meta.get("filename", other_id)
            break
    if possible_dupe_of:
        st.toast(
            f"⚠️ '{uf.name}' has the same row count and the same numeric totals as "
            f"already-uploaded '{possible_dupe_of}', but its data didn't clean "
            f"identically (likely a column that a PDF extraction parsed differently) "
            f"so it's being kept as a separate file rather than skipped automatically. "
            f"If this is really the same data, uncheck one copy in '📁 Uploaded files' "
            f"below to avoid double-counting.",
            icon="⚠️",
        )
    try:
        cleaned.to_csv(os.path.join(FILES_DIR, f"{content_id}.csv"), index=False)
        manifest[content_id] = {"filename": uf.name, "rows": len(cleaned), "included": True}
        newly_added = True
    except Exception:
        st.toast(f"⚠️ Couldn't save {uf.name}.", icon="⚠️")
if newly_added:
    save_manifest(manifest)
    # A new file changes the combined dataset's date range/columns — clear
    # stale filter/range selections so the dashboard picks sensible fresh
    # defaults instead of carrying over a window that suited the old data.
    for k in FILTER_AND_RANGE_KEYS:
        st.session_state.pop(hkey(k), None)
    save_local_config({"filter_state": {}})
    st.session_state.uploader_key += 1  # empty the uploader box now that the file(s) are safely stored
    st.rerun()

# ---------------- Sidebar: manage uploaded files ----------------
if manifest:
    with st.sidebar.expander(f"📁 Uploaded files ({len(manifest)})", expanded=False):
        st.caption("Uncheck a file to leave it out of the dashboard without deleting it.")
        for fid, meta in list(manifest.items()):
            fcol1, fcol2, fcol3 = st.columns([3, 1, 1])
            fcol1.markdown(f"**{meta.get('filename', fid)}**  \n{meta.get('rows', '?')} rows")
            is_included = fcol2.checkbox(
                "Use", value=meta.get("included", True), key=f"include_file_{fid}",
                label_visibility="visible",
            )
            if is_included != meta.get("included", True):
                manifest[fid]["included"] = is_included
                save_manifest(manifest)
                st.rerun()
            if fcol3.button("🗑", key=f"delete_file_{fid}", help="Remove this file"):
                manifest.pop(fid, None)
                save_manifest(manifest)
                try:
                    os.remove(os.path.join(FILES_DIR, f"{fid}.csv"))
                except Exception:
                    pass
                # NOTE: no st.cache_data.clear() here on purpose — it used to
                # wipe every OTHER hospital's cached, already-parsed data too
                # (including slow PDF extraction), forcing a full re-read of
                # everything the next time you opened them. It isn't needed:
                # load_combined_stored_files is cached by (path, mtime), so a
                # deleted file's path simply stops being picked up on the
                # very next rerun.
                # Also reset the uploader box, in case this file is still
                # sitting there — otherwise it would get read back in and
                # re-added on the very next rerun, making delete look broken.
                st.session_state.uploader_key += 1
                st.rerun()
        if st.button("🗑 Clear ALL uploaded files", use_container_width=True):
            for fid in list(manifest.keys()):
                try:
                    os.remove(os.path.join(FILES_DIR, f"{fid}.csv"))
                except Exception:
                    pass
            manifest = {}
            save_manifest(manifest)
            # (same reasoning as the single-file delete above — no global
            # cache clear needed, and it would have slowed down every other
            # hospital too)
            st.session_state.uploader_key += 1
            st.rerun()


# ---------------- Combine every included file into one working dataset ----------------
included_specs = []
for fid, meta in manifest.items():
    if not meta.get("included", True):
        continue
    fpath = os.path.join(FILES_DIR, f"{fid}.csv")
    if os.path.exists(fpath):
        included_specs.append((fpath, os.path.getmtime(fpath)))

if included_specs:
    df, cross_file_duplicates_removed = load_combined_stored_files(tuple(sorted(included_specs)))
    if df.empty:
        st.info("👈 Upload a CSV / Excel / JSON / TSV / PDF file from the sidebar to get started.")
        st.stop()
    if cross_file_duplicates_removed > 0:
        st.toast(
            f"ℹ️ Combined your included files: {cross_file_duplicates_removed} record(s) were "
            f"superseded by a newer upload of the same patient visit (e.g. a due that's since "
            f"been paid) or were exact duplicates — the most recent version of each is kept.",
            icon="ℹ️",
        )
else:
    st.info("👈 Upload one or more CSV / Excel / JSON / TSV / PDF files from the sidebar to get started.")
    st.stop()

is_hospital = hospital_schema_detected(df)
hcols = resolve_hospital_columns(df) if is_hospital else {}
if is_hospital:
    df = add_hospital_derived_columns(df, hcols)

# Apply any user-added calculated columns
for name, formula in st.session_state[hkey("calc_columns")]:
    try:
        df[name] = df.eval(formula)
    except Exception as e:
        st.toast(f"⚠️ Column '{name}' has a problem: {e}", icon="⚠️")

date_cols, numeric_cols, categorical_cols = detect_columns(df)

# ---------------- Sidebar: Add calculated column ----------------
with st.sidebar.expander("➕ Add a calculated column"):
    new_col_name = st.text_input("New column name", key=hkey("new_col_name"))
    new_col_formula = st.text_input(
        "Formula (use existing column names)",
        placeholder="e.g. `Service Final Amount` - `Cash Due`",
        key=hkey("new_col_formula"),
    )
    if st.button("Add column", key=hkey("add_col_btn")):
        if new_col_name and new_col_formula:
            st.session_state[hkey("calc_columns")].append((new_col_name, new_col_formula))
            save_local_config({"calc_columns": st.session_state[hkey("calc_columns")], "custom_charts": st.session_state[hkey("custom_charts")]})
            st.rerun()

    if st.session_state[hkey("calc_columns")]:
        st.write("Active calculated columns:")
        for i, (n, f) in enumerate(st.session_state[hkey("calc_columns")]):
            col_a, col_b = st.columns([4, 1])
            col_a.write(f"`{n}` = {f}")
            if col_b.button("✕", key=hkey(f"del_{i}")):
                st.session_state[hkey("calc_columns")].pop(i)
                save_local_config({"calc_columns": st.session_state[hkey("calc_columns")], "custom_charts": st.session_state[hkey("custom_charts")]})
                st.rerun()

date_cols, numeric_cols, categorical_cols = detect_columns(df)

# ---------------- Sidebar: filters ----------------
# Filter widgets live inside a form, so picking a new date or entity does
# NOT touch the dashboard until "Apply Filters" is clicked. "Refresh"
# clears the saved filter values entirely, putting the dashboard back to
# the full, unfiltered view.
start_date, end_date = None, None
active_entity_col, active_entity_val = None, None
active_generic_date_desc = None
with st.sidebar.expander("🔎 Filters", expanded=True):
    with st.form(hkey("filters_form")):
        filtered_df = df

        if is_hospital and hcols["doa"]:
            # This date range only SETS start_date/end_date — the actual
            # row filtering happens later in build_hospital_subsets(),
            # which applies the correct date field per Type (Admission rows
            # -> Doa, Discharge rows -> DOD, OPD -> Doa). We deliberately do
            # NOT filter `filtered_df` here by Doa alone: doing so used to
            # silently delete legitimate Discharge rows for patients
            # admitted before this range but discharged inside it (e.g.
            # admitted 22 Jul, discharged 5 Aug is a real August discharge
            # that has nothing to do with July admissions).
            doa_series = df[hcols["doa"]]
            # Only pull DOD from actual Discharge rows — for OPD rows this
            # column is meaningless/leftover junk data (e.g. stray dates
            # from 2022-2025 in some exports) and including it here blew
            # the widget's date bounds out by years, which in turn made
            # "Per-Day Revenue" divide by a huge, wrong day count.
            if hcols["dod"] and "Is Discharge Row" in df.columns:
                dod_series = df.loc[df["Is Discharge Row"], hcols["dod"]]
            elif hcols["dod"]:
                dod_series = df[hcols["dod"]]
            else:
                dod_series = None
            combined_dates = pd.concat([doa_series, dod_series]) if dod_series is not None else doa_series
            min_date, max_date = combined_dates.min(), combined_dates.max()
            if pd.notna(min_date) and pd.notna(max_date):
                min_date, max_date = min_date.date(), max_date.date()
                # Default the pickers to the calendar month containing the
                # MOST RECENT date in the file (e.g. an "August" upload
                # defaults to 1-31 Aug) instead of the absolute earliest
                # date across the whole file. A few patients admitted in
                # the tail end of the previous month and discharged in the
                # target month are real, legitimate rows — they still get
                # included correctly by every chart when relevant — but
                # they shouldn't drag the DEFAULT view back to a month you
                # didn't intend to look at. The full min/max stays
                # selectable via the picker's own bounds if you want it.
                # This hospital's own last-applied From/To (see
                # save_current_filter_state) takes priority when it's
                # still within the current data's bounds; otherwise fall
                # back to the smart "most recent month" default.
                smart_from = max(min_date, max_date.replace(day=1))
                smart_to = max_date
                saved_from = _saved_filter_value("filter_start_date")
                saved_to = _saved_filter_value("filter_end_date")
                default_from = saved_from if saved_from and min_date <= saved_from <= max_date else smart_from
                default_to = saved_to if saved_to and min_date <= saved_to <= max_date else smart_to
                fc1, fc2 = st.columns(2)
                start_date = fc1.date_input("From", value=default_from, min_value=min_date, max_value=max_date, key=hkey("filter_start_date"))
                end_date = fc2.date_input("To", value=default_to, min_value=min_date, max_value=max_date, key=hkey("filter_end_date"))
                if start_date > end_date:
                    st.toast("⚠️ 'From' date is after 'To' date — showing all data instead.", icon="⚠️")
                    start_date, end_date = None, None
        elif date_cols:
            saved_date_col = _saved_filter_value("filter_date_col")
            date_col_index = date_cols.index(saved_date_col) if saved_date_col in date_cols else 0
            date_col_for_filter = st.selectbox("Filter by date column", date_cols, index=date_col_index, key=hkey("filter_date_col"))
            min_d, max_d = df[date_col_for_filter].min(), df[date_col_for_filter].max()
            if pd.notna(min_d) and pd.notna(max_d):
                min_date, max_date = min_d.date(), max_d.date()
                saved_from = _saved_filter_value("filter_start_date")
                saved_to = _saved_filter_value("filter_end_date")
                default_from = saved_from if saved_from and min_date <= saved_from <= max_date else min_date
                default_to = saved_to if saved_to and min_date <= saved_to <= max_date else max_date
                fc1, fc2 = st.columns(2)
                fstart = fc1.date_input("From", value=default_from, min_value=min_date, max_value=max_date, key=hkey("filter_start_date"))
                fend = fc2.date_input("To", value=default_to, min_value=min_date, max_value=max_date, key=hkey("filter_end_date"))
                if fstart > fend:
                    st.toast("⚠️ 'From' date is after 'To' date — showing all data instead.", icon="⚠️")
                else:
                    mask = (filtered_df[date_col_for_filter] >= pd.Timestamp(fstart)) & (filtered_df[date_col_for_filter] <= pd.Timestamp(fend))
                    filtered_df = filtered_df[mask]
                    active_generic_date_desc = f"{date_col_for_filter}: {fstart.strftime('%d %b %Y')} – {fend.strftime('%d %b %Y')}"

        # Filter everything down to one entity, e.g. one doctor
        priority_cols = [hcols.get("doctor"), find_col(df, "Patient Name")] if is_hospital else []
        entity_candidates = [c for c in priority_cols if c] + [c for c in categorical_cols if c not in priority_cols]
        if entity_candidates:
            entity_options = ["None"] + entity_candidates
            saved_entity_col = _saved_filter_value("filter_entity_col")
            entity_col_index = entity_options.index(saved_entity_col) if saved_entity_col in entity_options else 0
            entity_col = st.selectbox("Filter by", entity_options, index=entity_col_index, key=hkey("filter_entity_col"))
            if entity_col != "None":
                entity_values = ["All"] + sorted(df[entity_col].dropna().astype(str).unique())
                saved_entity_val = _saved_filter_value("filter_entity_val")
                entity_val_index = entity_values.index(saved_entity_val) if saved_entity_val in entity_values else 0
                entity_value = st.selectbox(f"{entity_col} value", entity_values, index=entity_val_index, key=hkey("filter_entity_val"))
                if entity_value != "All":
                    filtered_df = filtered_df[filtered_df[entity_col].astype(str) == entity_value]
                    active_entity_col, active_entity_val = entity_col, entity_value

        filters_submitted = st.form_submit_button("✅ Apply Filters", use_container_width=True)
        if filters_submitted:
            save_current_filter_state()

    if st.button("🔄 Refresh (reset filters)", use_container_width=True):
        for k in FILTER_AND_RANGE_KEYS:
            st.session_state.pop(hkey(k), None)
        save_local_config({"filter_state": {}})
        st.rerun()

df = filtered_df

# ---------------- Quick range selector, next to the dashboard heading ----------------
# A faster alternative to the sidebar's From/To pickers: pick a range TYPE
# (a specific date, a week, a 10-day period, a month, or a year), then hit
# Confirm. Confirming here overrides the sidebar's date range until you
# switch back to "Full Range" and confirm again.
if is_hospital and hcols["doa"]:
    doa_col = hcols["doa"]
    all_doa_dates = df[doa_col].dropna()
    min_date_overall = all_doa_dates.min().date() if len(all_doa_dates) else None
    max_date_overall = all_doa_dates.max().date() if len(all_doa_dates) else None

    head_col, type_col, val_col, btn_col = st.columns([2.4, 1.3, 1.6, 1])
    # Rendered as raw HTML (not st.markdown's default) with a nowrap class,
    # so the heading always stays on a single line next to the range
    # controls — even at the narrowest column width — truncating with an
    # ellipsis instead of pushing onto a second row.
    head_col.markdown(
        "<h5 class='dashboard-heading'>🏥 Hospital Executive Dashboard</h5>",
        unsafe_allow_html=True,
    )

    _range_type_options = ["Full Range", "Date", "Week", "10-Day Period", "Month", "Year"]
    _saved_range_type = _saved_filter_value("quick_range_type")
    _range_type_index = _range_type_options.index(_saved_range_type) if _saved_range_type in _range_type_options else 0
    range_type = type_col.selectbox(
        "Range type", _range_type_options, index=_range_type_index,
        key=hkey("quick_range_type"), label_visibility="collapsed",
    )

    quick_start, quick_end = None, None
    if range_type == "Date" and min_date_overall:
        _saved_quick_date = _saved_filter_value("quick_date")
        _default_quick_date = _saved_quick_date if _saved_quick_date and min_date_overall <= _saved_quick_date <= max_date_overall else max_date_overall
        picked = val_col.date_input(
            "Pick date", value=_default_quick_date, min_value=min_date_overall,
            max_value=max_date_overall, key=hkey("quick_date"), label_visibility="collapsed",
        )
        quick_start, quick_end = picked, picked
    elif range_type == "Week" and min_date_overall:
        _smart_week_start = max(max_date_overall - pd.Timedelta(days=6), min_date_overall)
        _saved_week_start = _saved_filter_value("quick_week")
        _default_week_start = _saved_week_start if _saved_week_start and min_date_overall <= _saved_week_start <= max_date_overall else _smart_week_start
        week_start = val_col.date_input(
            "Week starting", value=_default_week_start,
            min_value=min_date_overall, max_value=max_date_overall,
            key=hkey("quick_week"), label_visibility="collapsed",
        )
        quick_start = week_start
        quick_end = min(week_start + pd.Timedelta(days=6), max_date_overall)
    elif range_type == "10-Day Period" and hcols["period"]:
        periods = ordered_period(df, hcols["period"])
        _saved_period = _saved_filter_value("quick_period")
        _period_index = periods.index(_saved_period) if periods and _saved_period in periods else 0
        chosen_period = val_col.selectbox("Period", periods, index=_period_index, key=hkey("quick_period"), label_visibility="collapsed") if periods else None
        if chosen_period:
            p_dates = df[df[hcols["period"]] == chosen_period][doa_col].dropna()
            if len(p_dates):
                quick_start, quick_end = p_dates.min().date(), p_dates.max().date()
    elif range_type == "Month" and min_date_overall:
        month_periods = sorted(df[doa_col].dropna().dt.to_period("M").unique())
        month_labels = [m.strftime("%B %Y") for m in month_periods]
        _saved_month = _saved_filter_value("quick_month")
        _month_index = month_labels.index(_saved_month) if month_labels and _saved_month in month_labels else (len(month_labels) - 1 if month_labels else 0)
        chosen_month = val_col.selectbox(
            "Month", month_labels, index=_month_index, key=hkey("quick_month"), label_visibility="collapsed",
        ) if month_labels else None
        if chosen_month:
            chosen_p = month_periods[month_labels.index(chosen_month)]
            quick_start = chosen_p.start_time.date()
            quick_end = min(chosen_p.end_time.date(), max_date_overall)
    elif range_type == "Year" and min_date_overall:
        years = sorted(df[doa_col].dropna().dt.year.unique())
        _saved_year = _saved_filter_value("quick_year")
        _year_index = years.index(_saved_year) if years and _saved_year in years else (len(years) - 1 if years else 0)
        chosen_year = val_col.selectbox(
            "Year", years, index=_year_index, key=hkey("quick_year"), label_visibility="collapsed",
        ) if years else None
        if chosen_year:
            quick_start = pd.Timestamp(year=chosen_year, month=1, day=1).date()
            quick_end = min(pd.Timestamp(year=chosen_year, month=12, day=31).date(), max_date_overall)
    else:
        val_col.write("")  # keeps the row aligned when "Full Range" is picked

    # Live preview of what the confirmed range will actually cover, shown
    # right under the controls, before you even click Confirm.
    if range_type == "Week" and quick_start and quick_end:
        val_col.caption(f"📅 Covers {quick_start.strftime('%d %b')} – {quick_end.strftime('%d %b')} (7 days)")
    elif range_type in ("10-Day Period", "Month", "Year") and quick_start and quick_end:
        val_col.caption(f"📅 Covers {quick_start.strftime('%d %b %Y')} – {quick_end.strftime('%d %b %Y')}")

    if btn_col.button("✅ Confirm", key=hkey("quick_range_confirm"), use_container_width=True):
        if range_type == "Full Range":
            st.session_state[hkey("quick_range_start")] = None
            st.session_state[hkey("quick_range_end")] = None
        else:
            st.session_state[hkey("quick_range_start")] = quick_start
            st.session_state[hkey("quick_range_end")] = quick_end
        save_current_filter_state()
        st.toast("✅ Dashboard range updated.", icon="✅")
        st.rerun()

    # A confirmed quick range takes over from the sidebar's From/To until
    # reset back to "Full Range". Always show what's currently applied so
    # there's no ambiguity about what data the dashboard below reflects —
    # including the entity filter (e.g. "Filter by: Doctor = Dr. Uttam"),
    # not just the date range, so a filtered dashboard never gets mistaken
    # for the full unfiltered view.
    applied_start = st.session_state.get(hkey("quick_range_start"))
    applied_end = st.session_state.get(hkey("quick_range_end"))
    if applied_start and applied_end:
        start_date = applied_start
        end_date = applied_end
        range_text = f"{start_date.strftime('%d %b %Y')} – {end_date.strftime('%d %b %Y')}"
    else:
        range_text = "Full Range (all data)"
    summary_parts = [range_text]
    if active_entity_col:
        summary_parts.append(f"{active_entity_col} = {active_entity_val}")
    st.caption("📊 Currently showing: **" + "  |  ".join(summary_parts) + "**")
else:
    st.markdown("<h5 class='dashboard-heading'>🏥 Hospital Executive Dashboard</h5>", unsafe_allow_html=True)
    summary_parts = [active_generic_date_desc] if active_generic_date_desc else ["Full Range (all data)"]
    if active_entity_col:
        summary_parts.append(f"{active_entity_col} = {active_entity_val}")
    st.caption("📊 Currently showing: **" + "  |  ".join(summary_parts) + "**")

# ---------------- Build KPIs ----------------
if is_hospital:
    subsets = build_hospital_subsets(df, hcols, start_date, end_date)
    if start_date and end_date:
        n_days = (end_date - start_date).days + 1
    else:
        # No range confirmed — fall back to the full span of admission
        # dates present in the (entity-filtered) data.
        all_doa_span = df[hcols["doa"]].dropna() if hcols.get("doa") else pd.Series(dtype="datetime64[ns]")
        n_days = (all_doa_span.max().date() - all_doa_span.min().date()).days + 1 if len(all_doa_span) else 1
    all_kpis = build_hospital_kpis(subsets, hcols, n_days=n_days)
    # Exactly the rows currently powering the charts/KPIs above — each row
    # Type filtered by its own relevant date field (Admission -> Doa,
    # Discharge -> DOD, OPD -> Doa), same as build_hospital_subsets — not
    # just the entity-filtered `df`, which still spans every date. Any row
    # that isn't Admission/Discharge/OPD (unrecognized Type, if any) is
    # kept as-is since no date logic applies to it.
    known_mask = df.get("Is Admission Row", False) | df.get("Is Discharge Row", False) | df.get("Is OPD Row", False)
    uncategorized = df[~known_mask] if isinstance(known_mask, pd.Series) else df.iloc[0:0]
    display_df = pd.concat(
        [subsets["admission"], subsets["discharge"], subsets["opd"], uncategorized]
    ).sort_index()
else:
    all_kpis = build_generic_kpis(df, numeric_cols)
    display_df = df

# ---------------- Goal pace: Total Revenue vs monthly target ----------------
# Computed once here (before the KPI row) and reused later by the
# "Cumulative Revenue vs Target" chart, so the KPI card and the chart's own
# coloring/status always agree. The target the user enters is a PER-MONTH
# rate (e.g. ₹2.5 Cr/month); the actual goal for "on pace" and the chart is
# that rate scaled to however many days the currently selected range
# covers (`n_days`, already computed above) — so viewing one month targets
# ~₹2.5 Cr, two months targets ~₹5 Cr, a half month targets ~₹1.25 Cr, and
# so on, instead of always comparing against a flat single-month figure
# regardless of how much of the calendar the current view actually spans.
DEFAULT_MONTHLY_TARGET = 2.5 * 1_00_00_000  # ₹2.5 Cr default, per month
AVG_DAYS_PER_MONTH = 30.44  # 365.25 / 12 — used to convert a day-count into "months"
GOAL_MET_COLOR = "#1C7C54"
GOAL_MISSED_COLOR = "#C1666B"
on_pace = None
if is_hospital and hcols.get("revenue"):
    st.session_state.setdefault(hkey("monthly_revenue_target"), _saved_filter_value("monthly_revenue_target", DEFAULT_MONTHLY_TARGET))
    monthly_target = st.session_state[hkey("monthly_revenue_target")]
    # Total goal for the range currently in view = per-month rate x how
    # many months' worth of days that range covers.
    target_prorated_to_date = monthly_target * (n_days / AVG_DAYS_PER_MONTH) if monthly_target else 0
    achieved_revenue = all_kpis.get("Total Revenue", 0)
    on_pace = achieved_revenue >= target_prorated_to_date

# ---------------- Sidebar: choose KPIs & charts ----------------
with st.sidebar.expander("🛠 Customize KPIs", expanded=False):
    selected_kpis = st.multiselect("KPIs to show", list(all_kpis.keys()), default=list(all_kpis.keys()))

chart_options = ["Trend over time", "Breakdown by category", "Top values by category"]
HOSPITAL_CHART_NAMES = [
    "IPD Admission & Discharge by Day",
    "Dept & Doctor wise IPD Admission",
    "Dept & Doctor wise IPD Discharge",
    "Daily OPD Consultation + Doctor wise",
    "Dept & Doctor wise IPD Revenue",
    "10-Day Patient & Revenue Performance",
    "IPD Revenue Trend",
    "Panel Performance & Outstanding",
    "Cumulative Revenue vs Target",
    "OPD Services",
]

if is_hospital:
    with st.sidebar.expander("🛠 Customize charts", expanded=False):
        selected_hospital_charts = st.multiselect("Dashboard panels to show", HOSPITAL_CHART_NAMES, default=HOSPITAL_CHART_NAMES)
    selected_charts, groupby_col, metric_col, trend_date_col = [], None, None, None
else:
    with st.sidebar.expander("🛠 Customize charts", expanded=False):
        selected_charts = st.multiselect("Charts to show", chart_options, default=chart_options)
        groupby_col = st.selectbox("Group bar charts by", categorical_cols) if categorical_cols else None
        metric_col = st.selectbox("Metric for bar/trend charts", numeric_cols) if numeric_cols else None
        trend_date_col = st.selectbox("Date column for trend chart", date_cols) if date_cols else None
    selected_hospital_charts = []

# ---------------- KPI ROW ----------------
if selected_kpis:
    row_cols = st.columns(len(selected_kpis))
    for i, k in enumerate(selected_kpis):
        if k == "Total Revenue" and on_pace is not None:
            # Goal-based coloring: green once we're on pace toward the
            # monthly target, red/pink while behind — recomputed every
            # time the date range, filters, or target value change.
            goal_color = GOAL_MET_COLOR if on_pace else GOAL_MISSED_COLOR
            goal_icon = "✅" if on_pace else "⚠️"
            row_cols[i].markdown(
                f"""<div style='background:{goal_color}22;border:1px solid {goal_color};
                border-radius:8px;padding:8px 6px;min-height:64px;
                box-shadow:0 2px 6px rgba(0,0,0,0.12);'>
                <div style='font-weight:600;font-size:0.72rem;color:{goal_color};line-height:1.1;'>
                {k} {goal_icon}</div>
                <div style='font-weight:700;font-size:1.05rem;color:{goal_color};'>
                {format_number(all_kpis[k], currency=True)}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            row_cols[i].metric(k, format_number(all_kpis[k], currency=kpi_is_currency(k)))

# ---------------- GENERIC AUTO-CHARTS (non-hospital files only) ----------------
if not is_hospital:
    c1, c2 = st.columns(2)
    if "Trend over time" in selected_charts and trend_date_col and metric_col:
        trend = df.groupby(df[trend_date_col].dt.normalize())[metric_col].sum().reset_index()
        c1.plotly_chart(
            line_chart(trend, trend_date_col, metric_col, f"{metric_col} over time",
                       currency=col_is_currency(metric_col), date_axis=True, wrap_width=40),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
    if "Breakdown by category" in selected_charts and groupby_col and metric_col:
        breakdown = df.groupby(groupby_col)[metric_col].sum().sort_values(ascending=False).reset_index()
        c2.plotly_chart(
            colorful_bar(breakdown, groupby_col, metric_col, f"{metric_col} by {groupby_col}",
                         currency=col_is_currency(metric_col)),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
    if "Top values by category" in selected_charts and groupby_col:
        counts = df[groupby_col].value_counts().head(15).reset_index()
        counts.columns = [groupby_col, "count"]
        c1.plotly_chart(
            colorful_bar(counts, groupby_col, "count", f"Record count by {groupby_col}"),
            use_container_width=True, config=PLOTLY_CONFIG,
        )

# ---------------- HOSPITAL DASHBOARD PANELS ----------------
if is_hospital:
    c = hcols
    admission, discharge, opd, opd_consult = subsets["admission"], subsets["discharge"], subsets["opd"], subsets["opd_consult"]
    S = selected_hospital_charts

    def panel(render_fn, condition):
        """One bordered card, like the boxed panels in the reference dashboard.
        No outer title here — each chart inside carries its own full title,
        so there's exactly one heading per chart, not two."""
        if not condition:
            return
        with card_slot():
            render_fn()

    # Panels are placed two per row (in bordered cards), matching the
    # reference dashboard's layout, so related charts stay grouped together.
    slot_queue = []
    def make_row_slots():
        cols = st.columns(2)
        slot_queue.extend(cols)

    def card_slot():
        if not slot_queue:
            make_row_slots()
        return slot_queue.pop(0).container(border=True)

    # 1. Admission & Discharge trend by day
    def render_1():
        t1, t2 = st.columns(2)
        adm_trend = admission.groupby(admission[c["doa"]].dt.normalize()).size().reset_index(name="Admissions")
        adm_trend.columns = ["Day", "Admissions"]
        t1.plotly_chart(
            colorful_bar(adm_trend, "Day", "Admissions", "IPD Admissions by Day", date_axis=True),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
        avg_adm = adm_trend["Admissions"].mean() if len(adm_trend) else 0
        t1.markdown(
            f"<div class='panel-avg-stat'>Avg Admissions/Day: <span>{avg_adm:.1f}</span></div>",
            unsafe_allow_html=True,
        )
        if c["dod"]:
            dis_trend = discharge.groupby(discharge[c["dod"]].dt.normalize()).size().reset_index(name="Discharges")
            dis_trend.columns = ["Day", "Discharges"]
            t2.plotly_chart(
                colorful_bar(dis_trend, "Day", "Discharges", "IPD Discharges by Day", date_axis=True),
                use_container_width=True, config=PLOTLY_CONFIG,
            )
            avg_dis = dis_trend["Discharges"].mean() if len(dis_trend) else 0
            t2.markdown(
                f"<div class='panel-avg-stat'>Avg Discharges/Day: <span>{avg_dis:.1f}</span></div>",
                unsafe_allow_html=True,
            )
    panel(render_1, "IPD Admission & Discharge by Day" in S and c["doa"])

    # 2. Dept & Doctor wise IPD Admission
    def render_2():
        d1, d2 = st.columns(2)
        dept_admit = admission.groupby(c["dept"]).size().sort_values(ascending=False).reset_index(name="Admissions")
        d1.plotly_chart(
            colorful_bar(dept_admit, c["dept"], "Admissions", "IPD Admissions by Department"),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
        if c["doctor"]:
            doc_admit = admission.groupby(c["doctor"]).size().sort_values(ascending=False).reset_index(name="Admissions")
            d2.plotly_chart(
                colorful_bar(doc_admit, c["doctor"], "Admissions", "IPD Admissions by Doctor"),
                use_container_width=True, config=PLOTLY_CONFIG,
            )
    panel(render_2, "Dept & Doctor wise IPD Admission" in S and len(admission) and c["dept"])

    # 3. Dept & Doctor wise IPD Discharge
    def render_3():
        e1, e2 = st.columns(2)
        dept_dis = discharge.groupby(c["dept"]).size().sort_values(ascending=False).reset_index(name="Discharges")
        e1.plotly_chart(
            colorful_bar(dept_dis, c["dept"], "Discharges", "IPD Discharges by Department"),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
        if c["doctor"]:
            doc_dis = discharge.groupby(c["doctor"]).size().sort_values(ascending=False).reset_index(name="Discharges")
            e2.plotly_chart(
                colorful_bar(doc_dis, c["doctor"], "Discharges", "IPD Discharges by Doctor"),
                use_container_width=True, config=PLOTLY_CONFIG,
            )
    panel(render_3, "Dept & Doctor wise IPD Discharge" in S and len(discharge) and c["dept"])

    # 4. Daily OPD Consultation + Doctor wise
    def render_4():
        f1, f2 = st.columns(2)
        daily_consult = opd_consult.groupby(opd_consult[c["doa"]].dt.normalize()).size().reset_index(name="Consultations")
        daily_consult.columns = ["Day", "Consultations"]
        f1.plotly_chart(
            line_chart(daily_consult, "Day", "Consultations", "OPD Consultations - Daily Trend", date_axis=True,
                       always_show_labels=True, show_stat_lines=True),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
        if c["doctor"]:
            doc_consult = opd_consult.groupby(c["doctor"]).size().sort_values(ascending=False).reset_index(name="Consultations")
            f2.plotly_chart(
                colorful_bar(doc_consult, c["doctor"], "Consultations", "OPD Consultations by Doctor"),
                use_container_width=True, config=PLOTLY_CONFIG,
            )
    panel(render_4, "Daily OPD Consultation + Doctor wise" in S and len(opd_consult) and c["doa"])

    # 5. Dept & Doctor wise IPD Revenue (discharge-based)
    def render_5():
        g1, g2 = st.columns(2)
        dept_rev = discharge.groupby(c["dept"])[c["revenue"]].sum().sort_values(ascending=False).reset_index()
        # Append a "Total" bar summing every department's revenue, so the
        # overall figure is visible right alongside the breakdown.
        dept_rev = pd.concat([
            dept_rev,
            pd.DataFrame({c["dept"]: ["Total"], c["revenue"]: [dept_rev[c["revenue"]].sum()]}),
        ], ignore_index=True)
        g1.plotly_chart(
            colorful_bar(dept_rev, c["dept"], c["revenue"], "IPD Revenue by Department", currency=True),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
        if c["doctor"]:
            doc_rev = discharge.groupby(c["doctor"])[c["revenue"]].sum().sort_values(ascending=False).reset_index()
            doc_rev = pd.concat([
                doc_rev,
                pd.DataFrame({c["doctor"]: ["Total"], c["revenue"]: [doc_rev[c["revenue"]].sum()]}),
            ], ignore_index=True)
            g2.plotly_chart(
                colorful_bar(doc_rev, c["doctor"], c["revenue"], "IPD Revenue by Doctor", currency=True),
                use_container_width=True, config=PLOTLY_CONFIG,
            )
    panel(render_5, "Dept & Doctor wise IPD Revenue" in S and len(discharge) and c["dept"] and c["revenue"])

    # 6. 10-day Patient Discharge & Revenue Performance
    def render_6():
        order = ordered_period(df, c["period"])
        h1, h2 = st.columns(2)

        # Each chart gets an extra "Total" bar summing the periods shown,
        # so the overall count/amount is visible alongside the breakdown.
        period_discharge = discharge.groupby(c["period"]).size().reindex(order).fillna(0).reset_index(name="Discharges")
        total_discharge_row = pd.DataFrame({c["period"]: ["Total"], "Discharges": [period_discharge["Discharges"].sum()]})
        period_discharge = pd.concat([period_discharge, total_discharge_row], ignore_index=True)
        h1.plotly_chart(
            colorful_bar(period_discharge, c["period"], "Discharges", "Patient Discharges by 10-Day Period"),
            use_container_width=True, config=PLOTLY_CONFIG,
        )

        period_rev = discharge.groupby(c["period"])[c["revenue"]].sum().reindex(order).fillna(0).reset_index()
        total_rev_row = pd.DataFrame({c["period"]: ["Total"], c["revenue"]: [period_rev[c["revenue"]].sum()]})
        period_rev = pd.concat([period_rev, total_rev_row], ignore_index=True)
        h2.plotly_chart(
            colorful_bar(period_rev, c["period"], c["revenue"], "IPD Revenue by 10-Day Period", currency=True),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
    panel(render_6, "10-Day Patient & Revenue Performance" in S and c["period"] and c["revenue"])

    # 7. IPD Revenue Trend by discharge day
    def render_7():
        rev_trend = discharge.groupby(discharge[c["dod"]].dt.normalize())[c["revenue"]].sum().reset_index()
        rev_trend.columns = ["Day", c["revenue"]]
        total_rev = rev_trend[c["revenue"]].sum()
        # Format the dates as short labels first, then append a "Total"
        # bar — mixing a real Timestamp column with a text label breaks
        # the date-axis tick logic, so once "Total" is added the x-axis is
        # plotted as text categories instead.
        day_labels = rev_trend["Day"].dt.strftime("%d %b")
        rev_trend["Day"] = day_labels
        rev_trend = pd.concat([
            rev_trend, pd.DataFrame({"Day": ["Total"], c["revenue"]: [total_rev]}),
        ], ignore_index=True)
        fig = colorful_bar(rev_trend, "Day", c["revenue"], "IPD Revenue Trend (by Discharge Day)",
                            currency=True, date_axis=False)
        # Recreate the original sparse date-tick spacing (every 5th day,
        # always including the last day) on top of the category axis, and
        # always keep the "Total" label visible.
        n_days_shown = len(day_labels)
        step = 1 if n_days_shown <= 10 else 5
        shown_labels = list(day_labels.iloc[::step])
        if day_labels.iloc[-1] not in shown_labels:
            shown_labels.append(day_labels.iloc[-1])
        shown_labels.append("Total")
        fig.update_xaxes(tickmode="array", tickvals=shown_labels, ticktext=shown_labels,
                          tickangle=-45, tickfont=dict(size=10), automargin=True)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        # Average revenue per day across the exact date range this chart
        # covers (e.g. a month of data -> average daily revenue for that
        # month), computed from the real daily totals, not the Total bar.
        avg_rev_per_day = total_rev / n_days_shown if n_days_shown else 0
        st.markdown(
            f"<div class='panel-avg-stat'>Avg Revenue/Day: <span>{format_number(avg_rev_per_day, currency=True)}</span></div>",
            unsafe_allow_html=True,
        )
    panel(render_7, "IPD Revenue Trend" in S and c["dod"] and c["revenue"] and len(discharge))

    # 8. Panel Performance vs Outstanding (discharge-based)
    def render_8():
        panel_df = discharge.groupby(c["panel"]).agg(Revenue=(c["revenue"], "sum"), Credit_Due=(c["credit_due"], "sum")).reset_index()
        # Collected = Revenue actually received = Revenue minus the portion
        # still outstanding (Credit_Due). Clipped at 0 so a panel where
        # Credit_Due happens to exceed Revenue doesn't show a negative bar.
        panel_df["Collected"] = (panel_df["Revenue"] - panel_df["Credit_Due"]).clip(lower=0)
        # Append a "Total" group summing Revenue, Credit_Due, and Collected
        # across all panels, so the overall figures sit right next to the
        # per-panel breakdown.
        panel_df = pd.concat([
            panel_df,
            pd.DataFrame({
                c["panel"]: ["Total"],
                "Revenue": [panel_df["Revenue"].sum()],
                "Credit_Due": [panel_df["Credit_Due"].sum()],
                "Collected": [panel_df["Collected"].sum()],
            }),
        ], ignore_index=True)
        categories = list(panel_df[c["panel"]].astype(str))
        metrics = ["Revenue", "Credit_Due", "Collected"]
        metric_colors = {"Revenue": PALETTE[0], "Credit_Due": PALETTE[1], "Collected": PALETTE[3]}
        is_total_row = panel_df[c["panel"]].astype(str).str.strip().str.lower() == "total"
        main_panels, total_panel = panel_df[~is_total_row], panel_df[is_total_row]
        # Compressed heights (see BAR_HEIGHT_POWER) so a panel with much
        # smaller Revenue/Credit_Due/Collected than the biggest payer still
        # renders as a visible bar, not a sliver. Real amounts still show
        # via the printed labels below, this only changes bar HEIGHT.
        main_heights = main_panels[metrics].apply(_compress_heights) if len(main_panels) else main_panels[metrics]
        max_main = main_heights.to_numpy().max() if len(main_panels) else 0

        fig = go.Figure()
        for metric in metrics:
            labels = main_panels[metric].apply(lambda v: format_number(v, currency=True))
            fig.add_bar(
                name=metric, x=main_panels[c["panel"]].astype(str), y=main_heights[metric],
                marker_color=metric_colors[metric], text=labels, textposition="outside",
                cliponaxis=False, textfont_size=13, offsetgroup=metric, legendgroup=metric,
                hovertemplate="%{x}<br>" + metric + ": %{text}<extra></extra>",
            )
        # The "Total" panel's three bars sit on their own secondary y-axis,
        # scaled so they land a touch taller than the tallest individual
        # panel's bars — the same fix as colorful_bar's Total handling —
        # instead of dwarfing every real panel at their true, much larger
        # scale. Falls back to the shared axis if there's nothing to
        # compare against (e.g. only one panel in the data).
        use_secondary = len(total_panel) and max_main > 0
        total_heights = total_panel[metrics].apply(_compress_heights) if len(total_panel) else total_panel[metrics]
        for metric in metrics:
            labels = total_panel[metric].apply(lambda v: format_number(v, currency=True))
            fig.add_bar(
                name=metric, x=total_panel[c["panel"]].astype(str), y=total_heights[metric],
                yaxis="y2" if use_secondary else "y",
                marker_color=metric_colors[metric], text=labels, textposition="outside",
                cliponaxis=False, textfont_size=13, offsetgroup=metric, legendgroup=metric,
                showlegend=False,
                hovertemplate="%{x}<br>" + metric + ": %{text}<extra></extra>",
            )

        layout_kwargs = dict(
            title=dict(text=wrap_title("Revenue vs Outstanding by Panel", 30),
                       x=0.02, xanchor="left", y=0.95, yanchor="top", font=dict(size=13)),
            height=CHART_HEIGHT + 25,
            margin=dict(t=45, b=68, l=15, r=15),
            bargap=0.15, bargroupgap=0.05,
            legend_title_text="",
            legend=dict(orientation="h", yanchor="bottom", y=-0.42, xanchor="center", x=0.5),
            modebar=MODEBAR_STYLE,
            barmode="group",
        )
        if max_main > 0:
            layout_kwargs["yaxis"] = dict(range=[0, max_main * 1.22])
        if use_secondary:
            total_max = total_heights.to_numpy().max()
            layout_kwargs["yaxis2"] = dict(range=[0, (total_max / 0.95) if total_max > 0 else 1], overlaying="y")
        fig.update_layout(**layout_kwargs)
        # Force every panel name to render as its own tick, so none get
        # silently dropped the way "auto" tick spacing was doing before.
        fig.update_xaxes(title=None, type="category", tickmode="array",
                          tickvals=categories, ticktext=categories, tickangle=-60,
                          tickfont=dict(size=10), automargin=True)
        fig.update_yaxes(title=None, visible=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    panel(render_8, "Panel Performance & Outstanding" in S and c["panel"] and c["revenue"] and c["credit_due"])

    # Flush any half-filled row before the two full-width panels below
    slot_queue.clear()

    # 9. Cumulative Revenue vs Target (Total Revenue = IPD + OPD, matching
    # the "Total Revenue" KPI) — full width
    if "Cumulative Revenue vs Target" in S and c["period"] and c["revenue"] and (len(discharge) or len(opd)):
        with st.container(border=True):
            with st.expander("🎯 Set monthly revenue target", expanded=False):
                monthly_target = st.number_input(
                    "Monthly revenue target (₹)",
                    min_value=0.0,
                    value=float(st.session_state.get(hkey("monthly_revenue_target"), DEFAULT_MONTHLY_TARGET)),
                    step=1_00_000.0,
                    key=hkey("monthly_revenue_target"),
                )
                save_current_filter_state()  # persist the target immediately so it survives a hospital switch
                # The entered figure is a PER-MONTH rate. The goal line
                # below always targets that rate scaled to however many
                # days the currently selected range covers — so a 2-month
                # view targets ~2x this figure, a half-month view targets
                # ~0.5x, etc. — recomputed here from the same n_days used
                # for the "on pace" check, so this stays in sync with the
                # active date range/filters.
                target_total = monthly_target * (n_days / AVG_DAYS_PER_MONTH) if monthly_target else 0
                months_covered = n_days / AVG_DAYS_PER_MONTH
                st.caption(
                    f"Per month: {format_number(monthly_target, currency=True)}  •  "
                    f"Target for the current {n_days}-day range (~{months_covered:.1f} "
                    f"month{'s' if abs(months_covered - 1) > 0.05 else ''}): "
                    f"{format_number(target_total, currency=True)}"
                )
            order = ordered_period(df, c["period"])
            n_buckets = len(order)
            # Total Revenue = IPD revenue (discharge rows) + OPD revenue
            # (opd rows), summed per period — the same definition used by
            # the "Total Revenue" KPI — so the chart's final point matches
            # the KPI card instead of showing IPD-only revenue.
            ipd_by_period = discharge.groupby(c["period"])[c["revenue"]].sum().reindex(order).fillna(0)
            opd_by_period = opd.groupby(c["period"])[c["revenue"]].sum().reindex(order).fillna(0) if len(opd) else pd.Series(0, index=order)
            total_by_period = ipd_by_period.add(opd_by_period, fill_value=0)
            cum_rev = total_by_period.cumsum().reset_index()
            cum_rev.columns = [c["period"], "Cumulative Revenue"]
            cum_rev["Target Revenue"] = [target_total * (i + 1) / n_buckets for i in range(n_buckets)]
            cum_long = cum_rev.melt(id_vars=c["period"], value_vars=["Cumulative Revenue", "Target Revenue"], var_name="Series", value_name="Amount")
            cum_long["Label"] = cum_long["Amount"].apply(lambda v: format_number(v, currency=True))
            # Goal-based coloring, using the same "on pace" status computed
            # earlier (achieved vs the monthly target prorated to today) —
            # green once we're on pace, red/pink while behind. The dashed
            # Target Revenue line stays a neutral orange as a reference.
            actual_color = GOAL_MET_COLOR if on_pace else GOAL_MISSED_COLOR
            fig = px.line(cum_long, x=c["period"], y="Amount", color="Series", markers=True,
                          color_discrete_map={"Cumulative Revenue": actual_color, "Target Revenue": PALETTE[3]})
            fig.update_traces(mode="lines+markers+text", textfont_size=11, cliponaxis=False,
                               hovertemplate="%{x}<br>%{fullData.name}: %{customdata}<extra></extra>")
            # Print the value on every point. The two series track close
            # together, so put "Cumulative Revenue" labels below their
            # markers and "Target Revenue" labels above theirs — that way
            # the two label sets never land on top of each other.
            series_order = list(cum_long["Series"].unique())
            label_positions = {"Cumulative Revenue": "bottom center", "Target Revenue": "top center"}
            for trace, series_name in zip(fig.data, series_order):
                label_series = cum_long[cum_long["Series"] == series_name]["Label"]
                trace.customdata = label_series.values
                trace.text = label_series.values
                trace.textposition = label_positions.get(series_name, "top center")
            max_amt = cum_long["Amount"].max()
            tickvals, ticktext = indian_axis_ticks(max_amt, currency=True, n=5)
            fig.update_layout(
                title=dict(text="Cumulative Revenue vs Target", x=0.02, xanchor="left", font=dict(size=13)),
                height=CHART_HEIGHT + 25,
                margin=dict(t=55, b=55, l=15, r=15),
                legend_title_text="",
                legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="center", x=0.5),
                modebar=MODEBAR_STYLE,
            )
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None, tickvals=tickvals, ticktext=ticktext)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            gap = achieved_revenue - target_prorated_to_date
            if on_pace:
                st.markdown(
                    f"<div class='panel-avg-stat' style='color:{GOAL_MET_COLOR};'>✅ On pace — "
                    f"<span style='color:{GOAL_MET_COLOR};'>{format_number(gap, currency=True)}</span> ahead of target-to-date</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='panel-avg-stat' style='color:{GOAL_MISSED_COLOR};'>⚠️ Behind pace — "
                    f"<span style='color:{GOAL_MISSED_COLOR};'>{format_number(abs(gap), currency=True)}</span> short of target-to-date</div>",
                    unsafe_allow_html=True,
                )


    # 10. OPD Services breakdown — full width (Consultation excluded, it's
    # already covered by its own "OPD Consultations" panel above)
    if "OPD Services" in S and len(opd) and c["service_name"]:
        with st.container(border=True):
            opd_non_consult = opd[opd[c["service_name"]].astype(str).str.strip().str.lower() != "consultation"]
            top_services = opd_non_consult[c["service_name"]].value_counts().head(20).reset_index()
            top_services.columns = [c["service_name"], "Count"]
            st.plotly_chart(
                colorful_bar(top_services, c["service_name"], "Count", "OPD Services", wrap_width=40),
                use_container_width=True, config=PLOTLY_CONFIG,
            )

# ---------------- CUSTOM CHARTS ----------------
st.markdown("##### Custom charts")

with st.expander("➕ Add a chart"):
    all_cols = list(df.columns)
    cc1, cc2, cc3 = st.columns(3)
    chart_type = cc1.selectbox("Chart type", ["Bar", "Line", "Scatter", "Pie"], key=hkey("cc_type"))
    x_col = cc2.selectbox("X axis / category", all_cols, key=hkey("cc_x"))
    y_col = cc3.selectbox("Y axis / value (numeric)", ["(count)"] + numeric_cols, key=hkey("cc_y"))
    agg = st.selectbox("Aggregation", ["sum", "mean", "count", "max", "min"], key=hkey("cc_agg"))
    chart_title = st.text_input("Chart title (optional)", key=hkey("cc_title"))
    if st.button("Add chart", key=hkey("add_chart_btn")):
        st.session_state[hkey("custom_charts")].append({"type": chart_type, "x": x_col, "y": y_col, "agg": agg, "title": chart_title})
        save_local_config({"calc_columns": st.session_state[hkey("calc_columns")], "custom_charts": st.session_state[hkey("custom_charts")]})
        st.rerun()

chart_cols = st.columns(2)
for idx, spec in enumerate(st.session_state[hkey("custom_charts")]):
    try:
        x, y, agg, ctype = spec["x"], spec["y"], spec["agg"], spec["type"]
        title = spec["title"] or f"{ctype}: {x}" + (f" vs {y}" if y != "(count)" else " (count)")

        if y == "(count)":
            plot_df = df.groupby(x).size().reset_index(name="count")
            y_field = "count"
        else:
            plot_df = df.groupby(x)[y].agg(agg).reset_index()
            y_field = y

        is_date_x = pd.api.types.is_datetime64_any_dtype(df[x]) if x in df.columns else False
        currency_flag = col_is_currency(y) if y != "(count)" else False

        if ctype == "Bar":
            fig = colorful_bar(plot_df, x, y_field, title, currency=currency_flag, date_axis=is_date_x)
        elif ctype == "Line":
            fig = line_chart(plot_df, x, y_field, title, currency=currency_flag, date_axis=is_date_x)
        elif ctype == "Scatter":
            fig = px.scatter(df, x=x, y=(y if y != "(count)" else x), title=title)
            fig.update_layout(height=CHART_HEIGHT, margin=dict(t=45, b=30, l=15, r=15),
                               modebar=MODEBAR_STYLE)
        else:
            fig = px.pie(plot_df, names=x, values=y_field, title=title)
            fig.update_layout(height=CHART_HEIGHT, margin=dict(t=45, b=30, l=15, r=15),
                               modebar=MODEBAR_STYLE)

        target_col = chart_cols[idx % 2]
        target_col.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=hkey(f"custom_chart_{idx}"))
        if target_col.button("Remove", key=hkey(f"del_chart_{idx}")):
            st.session_state[hkey("custom_charts")].pop(idx)
            save_local_config({"calc_columns": st.session_state[hkey("calc_columns")], "custom_charts": st.session_state[hkey("custom_charts")]})
            st.rerun()
    except Exception as e:
        st.toast(f"⚠️ Couldn't build chart {idx + 1}: {e}", icon="⚠️")

with st.expander("🧾 View cleaned data"):
    st.caption("Showing exactly the rows currently powering the charts/KPIs above — reflects the selected date range and any entity filter.")
    st.dataframe(display_df, use_container_width=True)
    st.download_button("Download cleaned data as CSV", display_df.to_csv(index=False).encode(), "cleaned_data.csv")
    # Ignore local storage and confidential files

