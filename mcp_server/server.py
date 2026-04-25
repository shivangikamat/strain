"""
EmotiScan MCP server (stdio) — emotiscan-tools subset.

Run: python -m mcp_server.server
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from emotiscan.features.extract import extract_features
from emotiscan.io.emotions_csv import load_dataset, load_emotions_csv
from emotiscan.models.classifier import (
    classify_emotion,
    explain_decision,
    load_classifier_pipeline,
)
from emotiscan.screening.mental_health import screen_mental_health

mcp = FastMCP("emotiscan-tools")


def _json(data: Any) -> str:
    return json.dumps(data, indent=2)


@mcp.tool()
def load_dataset_tool(
    dataset: str = "eeg_brainwave",
    csv_path: str | None = None,
    max_rows: int | None = None,
) -> str:
    """Load dataset metadata (Kaggle emotions CSV). csv_path overrides default data/emotions.csv."""
    p = Path(csv_path) if csv_path else None
    return _json(load_dataset(dataset, csv_path=p, max_rows=max_rows))


@mcp.tool()
def extract_features_tool(
    csv_path: str | None = None,
    row_index: int = 0,
) -> str:
    """Extract proxy band features for one row of the emotions CSV."""
    ds = load_emotions_csv(csv_path)
    idx = row_index % ds.X.shape[0]
    row = ds.X[idx]
    return _json(extract_features(row, ds.feature_names))


@mcp.tool()
def classify_emotion_tool(csv_path: str | None = None, row_index: int = 0) -> str:
    """Classify emotion (NEGATIVE/NEUTRAL/POSITIVE) for one CSV row."""
    ds = load_emotions_csv(csv_path)
    idx = row_index % ds.X.shape[0]
    bundle = load_classifier_pipeline()
    return _json(classify_emotion(ds.X[idx], feature_names=ds.feature_names, bundle=bundle))


@mcp.tool()
def explain_decision_tool(csv_path: str | None = None, row_index: int = 0) -> str:
    """Linear-model attribution for the predicted class."""
    ds = load_emotions_csv(csv_path)
    idx = row_index % ds.X.shape[0]
    row = ds.X[idx]
    bundle = load_classifier_pipeline()
    pred = classify_emotion(row, feature_names=ds.feature_names, bundle=bundle)
    return _json(explain_decision(row, ds.feature_names, pred, bundle=bundle))


@mcp.tool()
def screen_mental_health_tool(csv_path: str | None = None, row_index: int = 0) -> str:
    """Demo screening scores (not clinical)."""
    ds = load_emotions_csv(csv_path)
    idx = row_index % ds.X.shape[0]
    row = ds.X[idx]
    bundle = load_classifier_pipeline()
    feats = extract_features(row, ds.feature_names)
    cls = classify_emotion(row, feature_names=ds.feature_names, bundle=bundle)
    return _json(screen_mental_health(cls, feats))


import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="EmotiScan MCP Server")
    parser.add_argument("--sse", action="store_true", help="Run over SSE for promptopinion.ai integration")
    parser.add_argument("--port", type=int, default=8001, help="Port to run the SSE server on")
    args = parser.parse_args()

    if args.sse:
        print(f"Starting EmotiScan FastMCP Server on Server-Sent Events (SSE) mode... [Port {args.port}]")
        mcp.run(transport="sse", host="127.0.0.1", port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
