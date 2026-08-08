from innovation.manager_feedback import ManagerFeedback
from innovation.feedback_statistics import FeedbackStatistics
from innovation.feedback_analyzer import FeedbackAnalyzer
from innovation.preference_learning import PreferenceLearning

manager = ManagerFeedback()

manager.submit_feedback(220, 210, "Festival")
manager.submit_feedback(180, 180, "Weekend")
manager.submit_feedback(250, 230, "Holiday")
manager.submit_feedback(200, 205, "Rain")

manager.memory.export_csv()

stats = FeedbackStatistics(manager.memory)

FeedbackAnalyzer(stats).generate_report()

model = PreferenceLearning(manager.memory)

print("\nSuggested price after learning:")
print(model.recommend_price(240))