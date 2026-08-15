"""
demo.py

Runs the complete Digital Twin Market Simulator.
"""

from .simulation_runner import SimulationRunner
from .twin_dashboard import TwinDashboard
from .simulation_report import SimulationReport
from .policy_comparison import PolicyComparison
from .risk_analyzer import RiskAnalyzer
from .export_results import ExportResults
from .config import SIMULATION_EPISODES


def main():

    print("=" * 70)
    print("        DIGITAL TWIN MARKET SIMULATOR")
    print("=" * 70)

    # --------------------------------------------------------
    # RUN SIMULATION
    # --------------------------------------------------------

    runner = SimulationRunner()

    results = runner.run(
        SIMULATION_EPISODES
    )

    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    dashboard = TwinDashboard()

    dashboard.display(results)

    # --------------------------------------------------------
    # POLICY COMPARISON
    # --------------------------------------------------------

    comparison = PolicyComparison()
    risk = RiskAnalyzer()

    print("\n")
    print("=" * 70)
    print(" POLICY COMPARISON & RISK ANALYSIS ")
    print("=" * 70)

    for index, scenario in enumerate(results, start=1):

        print(f"\nScenario {index}")
        print("-" * 50)

        comparison_result = comparison.compare(
            scenario
        )

        for strategy, revenue in comparison_result["strategies"].items():

            print(f"{strategy:<22} : ₹{revenue}")

        print(
            f"Best Strategy           : "
            f"{comparison_result['best_strategy']}"
        )

        risk_result = risk.analyze(
            scenario
        )

        print(
            f"Risk Level              : "
            f"{risk_result['risk']}"
        )

        print(
            f"Reason                  : "
            f"{risk_result['reason']}"
        )

        print(
            f"Recommendation          : "
            f"{risk_result['recommendation']}"
        )

    # --------------------------------------------------------
    # BUSINESS REPORT
    # --------------------------------------------------------

    report = SimulationReport()

    report.generate(results)

    # --------------------------------------------------------
    # DASHBOARD SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print(" DASHBOARD SUMMARY ")
    print("=" * 60)

    best_count = {}

    high_risk = 0
    low_risk = 0

    for scenario in results:

        winner = comparison.compare(
            scenario
        )["best_strategy"]

        best_count[winner] = best_count.get(
            winner,
            0
        ) + 1

        level = risk.analyze(
            scenario
        )["risk"]

        if level == "HIGH":
            high_risk += 1
        else:
            low_risk += 1

    best_strategy = max(
        best_count,
        key=best_count.get
    )

    highest = max(
        results,
        key=lambda x: x["revenue"]
    )

    total_revenue = sum(
        r["revenue"]
        for r in results
    )

    average_revenue = total_revenue / len(results)

    occupancy = (
        sum(r["rooms_sold"] for r in results)
        /
        (len(results) * 100)
    ) * 100

    print(f"Best Strategy       : {best_strategy}")
    print(f"Highest Revenue     : ₹{highest['revenue']}")
    print(f"Average Revenue     : ₹{average_revenue:.2f}")
    print(f"Average Occupancy   : {occupancy:.2f}%")
    print(f"Total Revenue       : ₹{total_revenue}")
    print(f"High Risk Scenarios : {high_risk}")
    print(f"Low Risk Scenarios  : {low_risk}")

    print("=" * 60)

    # --------------------------------------------------------
    # EXPORT RESULTS
    # --------------------------------------------------------

    exporter = ExportResults()

    exporter.export(results)

    print("\n")
    print("=" * 70)
    print(" DIGITAL TWIN SIMULATION COMPLETED SUCCESSFULLY ")
    print("=" * 70)


if __name__ == "__main__":

    main()