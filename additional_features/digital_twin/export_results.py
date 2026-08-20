import csv
import os

from .policy_comparison import PolicyComparison
from .risk_analyzer import RiskAnalyzer


class ExportResults:

    def export(self, results):

        os.makedirs("evaluation", exist_ok=True)

        comparison = PolicyComparison()
        risk = RiskAnalyzer()

        with open(
            "evaluation/digital_twin_results.csv",
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Event",
                "Weather",
                "Competitor Price",
                "Demand",
                "RL Price",
                "Rooms Sold",
                "Revenue",
                "Policy Winner",
                "Risk Level"
            ])

            for row in results:

                winner = comparison.compare(row)[
                    "best_strategy"
                ]

                risk_level = risk.analyze(row)[
                    "risk"
                ]

                writer.writerow([
                    row["event"],
                    row["weather"],
                    row["competitor_price"],
                    row["simulated_demand"],
                    row["rl_price"],
                    row["rooms_sold"],
                    row["revenue"],
                    winner,
                    risk_level
                ])

        print("\nResults exported to evaluation/digital_twin_results.csv")