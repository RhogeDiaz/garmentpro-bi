from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, int)):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _top_points(points: Sequence[Sequence[Any]]) -> list[Sequence[Any]]:
    points = list(points)
    if len(points) <= 10:
        return points
    numeric = [
        (index, point)
        for index, point in enumerate(points)
        if len(point) > 1 and isinstance(point[1], (int, float))
    ]
    if len(numeric) == len(points):
        return [point for _, point in sorted(numeric, key=lambda item: item[1][1], reverse=True)[:10]]
    return points[-10:]


def build_chart_context(
    charts: Sequence[Mapping[str, Any]],
    *,
    page: str,
    filters: Mapping[str, Any] | None = None,
    date_range: tuple[Any, Any] | None = None,
    kpis: Mapping[str, Any] | None = None,
) -> str:
    """Serialize the active dashboard view into a compact, auditable prompt context."""
    if not charts:
        return "No charts are currently visible. Open a dashboard page with charts first."

    lines = [f"Visible dashboard page: {page}"]
    if filters:
        lines.append("Applied filters: " + ", ".join(
            f"{key}={_format_value(value)}" for key, value in filters.items()
        ))
    if date_range:
        lines.append(
            f"Visible date range: {_format_value(date_range[0])} to {_format_value(date_range[1])}"
        )
    if kpis:
        lines.append("KPI cards:")
        lines.extend(f"- {key}: {_format_value(value)}" for key, value in kpis.items())

    for chart in charts:
        lines.extend([
            "",
            f"### {chart['title']}",
            f"Chart type: {chart['type']}",
            f"X axis: {chart.get('x_label') or '(not labeled)'}",
            f"Y axis: {chart.get('y_label') or '(not labeled)'}",
        ])
        for series in chart.get("series", []):
            lines.extend([
                f"Series: {series['name']}",
                "| Category / x | Value / y |",
                "| --- | ---: |",
            ])
            lines.extend(
                f"| {_format_value(point[0])} | {_format_value(point[1])} |"
                for point in _top_points(series.get("points", []))
            )
    return "\n".join(lines)