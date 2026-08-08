"""
export_utils.py

Pure CSV / JSON export helpers for Feature 2. No web-framework
dependency — `dashboard/web_dashboard/blueprints/api.py` wraps these in
a plain `flask.Response` (`mimetype="text/csv"` /
`mimetype="application/json"`) for the dashboard's "Export CSV" /
"Export JSON" buttons.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence, Union

import pandas as pd


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to CSV bytes (UTF-8, no index column)."""
    return df.to_csv(index=False).encode("utf-8")


def records_to_json_bytes(records: Union[Sequence[Mapping[str, Any]], Mapping[str, Any]]) -> bytes:
    """Serialize a list of dict records (or a single dict) to indented JSON bytes."""
    return json.dumps(records, indent=2, default=str).encode("utf-8")
