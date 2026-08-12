"""
Human Feedback Learning Demo
"""
from .feedback_dashboard import FeedbackDashboard
from .feedback_trend import FeedbackTrend
from .feedback_export import FeedbackExporter
from .manager_feedback import ManagerFeedback
from .feedback_statistics import FeedbackStatistics
from .preference_learning import PreferenceLearning

def run_demo():

    print("=" * 70)
    print("        HUMAN-IN-THE-LOOP LEARNING DEMO")
    print("=" * 70)

    manager = ManagerFeedback()

    scenarios = [

        {
            "name": "Festival Weekend",
            "ai_price": 260,
            "manager_price": 275
        },

        {
            "name": "Normal Weekday",
            "ai_price": 230,
            "manager_price": 225
        },

        {
            "name": "Heavy Rain",
            "ai_price": 240,
            "manager_price": 210
        },

        {
            "name": "Conference Event",
            "ai_price": 280,
            "manager_price": 280
        },

        {
            "name": "Holiday Season",
            "ai_price": 300,
            "manager_price": 315
        }

    ]

    for scenario in scenarios:

        decision = manager.submit_feedback(
            scenario["ai_price"],
            scenario["manager_price"],
            scenario["name"]
        )

        print()

        print("=" * 60)
        print(f"Scenario : {scenario['name']}")
        print("=" * 60)

        print(f"AI Recommended Price : ₹{scenario['ai_price']}")
        print(f"Manager Price        : ₹{scenario['manager_price']}")
        print(f"Decision             : {decision}")

    stats = FeedbackStatistics(manager.memory)

    print("\n")
    print("=" * 70)
    print("BUSINESS SUMMARY")
    print("=" * 70)

    print(f"Acceptance Rate : {stats.acceptance_rate():.2f}%")
    print(f"Average Difference : {stats.average_difference():.2f}")
    print(f"Total Decisions : {stats.total_feedback()}")

    print("=" * 70)

    learning = PreferenceLearning(manager.memory)

    print("\n")
    print("=" * 60)
    print(" LEARNING SUMMARY ")
    print("=" * 60)

    print(
        f"Average Adjustment : "
        f"{learning.preferred_adjustment():.2f}"
    )

    print(
        f"Acceptance Rate    : "
        f"{learning.acceptance_percentage()}%"
    )

    print(
        f"Recommended Price for ₹250 : "
        f"₹{learning.recommend_price(250)}"
    )

    print(
        f"Manager Preference : "
        f"{learning.manager_preference()}"
    )

    print(
        f"Business Insight   : "
        f"{learning.business_insight()}"
    )

    dashboard = FeedbackDashboard()
    dashboard.display(manager.memory)
    trend = FeedbackTrend()
    trend.display(manager.memory)
    FeedbackExporter.export(manager.memory)

    print("\n")
    print("=" * 60)
    print(" NEXT AI PRICE PREDICTION ")
    print("=" * 60)

    new_ai_price = 250

    recommended = learning.recommend_price(new_ai_price)

    print(f"Original AI Price : ₹{new_ai_price}")
    print(f"Updated AI Price  : ₹{recommended}")
    print("Reason            : Learned from previous manager decisions.")

    

if __name__ == "__main__":
    run_demo()