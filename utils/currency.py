"""
Currency conversion utility.

This project's prices (``base_price`` and everything derived from it —
min/max price, revenue, rewards) were originally denominated in US
Dollars. This module is the single source of truth for converting them
to Indian Rupees, so the whole codebase uses one real, traceable rate
instead of each file guessing or hardcoding its own number.

USD_TO_INR_RATE is a snapshot of the USD/INR mid-market exchange rate
(source: aggregated FX quotes — Yahoo Finance, XE, Investing.com,
BookMyForex — as of 2026-08-03, ~95.2-95.5 range; 95.39 used here).
Exchange rates move daily; this is a fixed snapshot for reproducible
training runs, not a live feed. If you want it pinned to a different
date's rate, change USD_TO_INR_RATE here — every price in the project
derives from this one constant, nothing else needs to change.
"""

USD_TO_INR_RATE = 95.39


def usd_to_inr(usd_amount: float) -> float:
    """Convert a US Dollar amount to Indian Rupees at USD_TO_INR_RATE."""
    return usd_amount * USD_TO_INR_RATE