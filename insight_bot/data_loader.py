from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SUPPORTED_SUFFIXES = {".csv", ".parquet", ".xlsx"}


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_excel(path)

    frame.columns = (
        frame.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    if "department" in frame:
        frame["department"] = (
            frame["department"].astype("string").str.strip()
            .str.replace("sweing", "sewing", regex=False).str.title()
        )
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    if "wip" in frame:
        frame["wip"] = pd.to_numeric(frame["wip"], errors="coerce").fillna(0)
    if "quarter" in frame:
        frame["quarter"] = frame["quarter"].astype("string").str.replace(
            "Quarter", "", regex=False
        )

    return frame


def load_dataset() -> pd.DataFrame:
    candidates = sorted(
        (path for path in DATA_DIR.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: (path.name != "productivity.csv", path.name.lower()),
    ) if DATA_DIR.exists() else []
    if not candidates:
        raise FileNotFoundError(
            f"No CSV, Parquet, or Excel dataset found in {DATA_DIR}"
        )

    frame = _read_dataset(candidates[0])
    if "date" in frame:
        frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return frame


# Loaded once for LlamaIndex; app.py continues to own its dashboard-specific cleaning.
df = load_dataset()