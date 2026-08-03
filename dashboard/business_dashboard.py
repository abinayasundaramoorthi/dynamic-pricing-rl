"""
Business KPI Dashboard

Displays business performance metrics generated after
policy evaluation in a dashboard-friendly format.
"""

import json
import os


def load_dashboard_metrics(filepath="evaluation/dashboard_metrics.json"):
    """
    Load dashboard metrics from JSON file.
    """
    with open(filepath, "r") as file:
        return json.load(file)


def display_dashboard(metrics):
    """
    Display Business KPI Dashboard.
    """

    print("=" * 60)
    print("        BUSINESS KPI DASHBOARD")
    print("=" * 60)

    for strategy in metrics:

        print(f"\nStrategy : {strategy['Strategy']}")
        print("-" * 40)

        print(f"Total Revenue           : ₹{strategy['Total Revenue']:.2f}")
        print(f"Average Revenue         : ₹{strategy['Average Revenue']:.2f}")
        print(f"Inventory Utilization   : {strategy['Inventory Utilization']}%")
        print(f"Sell-through Rate       : {strategy['Sell-through Rate']}%")
        print(f"Average Selling Price   : ₹{strategy['Average Selling Price']:.2f}")
        print(f"Revenue Growth          : {strategy['Revenue Growth']}%")
        print(f"Policy Rank             : {strategy['Policy Rank']}")

    print("\n")
    print("=" * 60)
    print("Dashboard Ready for Integration")
    print("=" * 60)


if __name__ == "__main__":

    metrics = load_dashboard_metrics()

    display_dashboard(metrics)