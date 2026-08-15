class SimulationReport:

    def generate(self, results):

        print("\n")
        print("=" * 60)
        print(" DIGITAL TWIN BUSINESS REPORT ")
        print("=" * 60)

        total = len(results)

        total_revenue = sum(r["revenue"] for r in results)

        average_revenue = total_revenue / total

        average_demand = sum(
            r["simulated_demand"]
            for r in results
        ) / total

        average_price = sum(
            r["rl_price"]
            for r in results
        ) / total

        average_occupancy = (
            sum(r["rooms_sold"] for r in results)
            /
            (total * 100)
        ) * 100

        sell_through = (
            sum(r["rooms_sold"] for r in results)
            /
            sum(r["simulated_demand"] for r in results)
        ) * 100

        highest = max(
            results,
            key=lambda x: x["revenue"]
        )

        lowest = min(
            results,
            key=lambda x: x["revenue"]
        )

        print(f"Total Simulations : {total}")
        print(f"Total Revenue     : ₹{total_revenue}")
        print(f"Average Revenue   : ₹{average_revenue:.2f}")
        print(f"Average Demand    : {average_demand:.2f}")
        print(f"Average RL Price  : ₹{average_price:.2f}")
        print(f"Average Occupancy : {average_occupancy:.2f}%")
        print(f"Sell-through Rate : {sell_through:.2f}%")

        print("\nHighest Revenue Scenario")
        print("---------------------------")
        print(f"Event      : {highest['event']}")
        print(f"Weather    : {highest['weather']}")
        print(f"Demand     : {highest['simulated_demand']}")
        print(f"RL Price   : ₹{highest['rl_price']}")
        print(f"Revenue    : ₹{highest['revenue']}")

        print("\nLowest Revenue Scenario")
        print("---------------------------")
        print(f"Event      : {lowest['event']}")
        print(f"Weather    : {lowest['weather']}")
        print(f"Revenue    : ₹{lowest['revenue']}")

        print("\nAI Recommendation")
        print("---------------------------")

        print("• Increase prices during Festivals.")
        print("• Maintain prices during Normal Days.")
        print("• Offer discounts during Rainy weather.")
        print("• Continuously monitor competitor pricing.")
        print("• Run Digital Twin simulations before applying new pricing.")

        print("=" * 60)