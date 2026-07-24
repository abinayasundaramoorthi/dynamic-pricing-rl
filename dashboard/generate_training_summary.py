"""
Training Summary Generator

Combines the generated training graphs into a single summary image.
"""

import os
from PIL import Image


def generate_training_summary():

    evaluation_dir = "evaluation"
    dashboard_dir = "dashboard"

    os.makedirs(dashboard_dir, exist_ok=True)

    reward = Image.open(os.path.join(evaluation_dir, "reward_trend.png"))
    revenue = Image.open(os.path.join(evaluation_dir, "revenue_trend.png"))
    inventory = Image.open(os.path.join(evaluation_dir, "inventory_trend.png"))

    width = max(
        reward.width,
        revenue.width,
        inventory.width
    )

    total_height = (
        reward.height +
        revenue.height +
        inventory.height
    )

    summary = Image.new(
        "RGB",
        (width, total_height),
        color="white"
    )

    y = 0

    summary.paste(reward, (0, y))
    y += reward.height

    summary.paste(revenue, (0, y))
    y += revenue.height

    summary.paste(inventory, (0, y))

    output = os.path.join(
        dashboard_dir,
        "training_summary.png"
    )

    summary.save(output)

    print("Training summary generated successfully!")
    print(output)


if __name__ == "__main__":
    generate_training_summary()