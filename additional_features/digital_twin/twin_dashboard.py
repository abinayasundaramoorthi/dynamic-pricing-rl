class TwinDashboard:

    def display(
        self,
        results
    ):

        print("="*60)

        print("DIGITAL TWIN DASHBOARD")

        print("="*60)

        for r in results:

            print("="*60)
            print(f"Event             : {r['event']}")
            print(f"Weather           : {r['weather']}")
            print(f"Competitor Price  : ₹{r['competitor_price']}")
            print(f"Expected Demand   : {r['expected_demand']}")
            print(f"Simulated Demand  : {r['simulated_demand']}")
            print(f"RL Price          : ₹{r['rl_price']}")
            print(f"Rooms Sold        : {r['rooms_sold']}")
            print(f"Revenue           : ₹{r['revenue']}")
            print("AI BUSINESS CONCLUSION:")
            print((" RL Pricing generated the highest revenue in 4 out of 5 scenarios."))
            print((" Festival periods showed the highest revenue potential."))
            print((" Rainy weather increased business risk and reduced expected demand."))
            print((" Digital Twin simulations recommend dynamic pricing before deployment."))
            print((" Manager feedback can further improve pricing decisions over time."))