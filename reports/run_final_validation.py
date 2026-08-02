"""
run_final_validation.py

Final end-to-end technical validation for Issue #111. Runs a lightweight
but genuine pass through every major component of the project, checking
for runtime errors, before the project's final submission.

Uses SMALL episode counts (fast smoke tests) purely to confirm every
piece still runs correctly end-to-end without crashing -- this is
separate from the FULL evaluation results already produced and
documented in earlier issues (#80, #84, #91, #99, #103), which this
report references as the actual performance evidence.
"""

import sys
import os
import traceback
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

results = {}


def check(name, fn):
    """Run a validation check, catching and recording any error."""
    print(f"\n--- Checking: {name} ---")
    try:
        fn()
        results[name] = {"status": "PASS", "error": None}
        print(f"PASS: {name}")
    except Exception as e:
        results[name] = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}
        print(f"FAIL: {name}")
        print(traceback.format_exc())


# ------------------------------------------------------------
# 1. Environment Pipeline
# ------------------------------------------------------------
def check_environment():
    from pricing_env.pricing_env import PricingEnvironment, PricingEnvConfig

    env = PricingEnvironment(PricingEnvConfig())
    observation, info = env.reset(seed=1)
    for _ in range(5):
        action = 3  # hold price
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    assert observation is not None
    assert isinstance(reward, float)


# ------------------------------------------------------------
# 2. Complete DQN Pipeline (small smoke-test training run)
# ------------------------------------------------------------
def check_dqn_pipeline():
    from pricing_env.pricing_env import PricingEnvironment, PricingEnvConfig
    from agents.dqn_agent import DQNAgent

    env = PricingEnvironment(PricingEnvConfig())
    agent = DQNAgent(state_dim=2, action_dim=env.action_space.n)
    agent.train(env, num_episodes=20, verbose=False)

    assert len(agent.episode_rewards) == 20
    assert len(agent.episode_avg_loss) == 20

    metrics = agent.evaluate(env, num_episodes=5)
    assert "avg_reward" in metrics
    assert not np.isnan(metrics["avg_reward"])


# ------------------------------------------------------------
# 3. Trained Model Loading
# ------------------------------------------------------------
def check_model_loading():
    from pricing_env.pricing_env import PricingEnvironment, PricingEnvConfig
    from agents.dqn_agent import DQNAgent

    env = PricingEnvironment(PricingEnvConfig())
    agent = DQNAgent(state_dim=2, action_dim=env.action_space.n)
    agent.load_model("models/dqn_weights.pth")

    test_state = np.array([50, 15], dtype=np.float32)
    action_1 = agent.choose_action(test_state, greedy=True)
    action_2 = agent.choose_action(test_state, greedy=True)
    assert action_1 == action_2, "Loaded model is not deterministic!"


# ------------------------------------------------------------
# 4. Policy Evaluation Results
# ------------------------------------------------------------
def check_policy_evaluation():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'evaluation'))
    from pricing_env.pricing_env import PricingEnvironment, PricingEnvConfig
    from policy_evaluator import PolicyEvaluator, make_fixed_price_policy

    env = PricingEnvironment(PricingEnvConfig())
    evaluator = PolicyEvaluator(env)
    policy_fn = make_fixed_price_policy(env)

    summary, records = evaluator.evaluate_policy(policy_fn, num_episodes=5, base_seed=42)

    assert len(records) == 5
    assert not np.isnan(summary["avg_episode_reward"])
    assert summary["avg_inventory_sold"] >= 0

    # Also confirm previously exported results file exists on disk
    assert os.path.exists("evaluation/policy_ranking.csv"), \
        "policy_ranking.csv from Issue #99 not found"


# ------------------------------------------------------------
# 5. Dashboard Functionality
# ------------------------------------------------------------
def check_dashboard_files():
    expected_files = [
        "dashboard/performance_summary.png",
    ]
    for filepath in expected_files:
        assert os.path.exists(filepath), f"Missing dashboard file: {filepath}"
        assert os.path.getsize(filepath) > 0, f"Dashboard file is empty: {filepath}"


# ------------------------------------------------------------
# 6. Generated Visualizations
# ------------------------------------------------------------
def check_visualizations():
    expected_charts = [
        "evaluation/pricing_charts/price_trajectory.png",
        "evaluation/pricing_charts/revenue_trend.png",
        "evaluation/pricing_charts/inventory_remaining.png",
        "evaluation/pricing_charts/daily_price_changes.png",
        "evaluation/pricing_charts/policy_performance_comparison.png",
    ]
    for filepath in expected_charts:
        assert os.path.exists(filepath), f"Missing chart: {filepath}"
        assert os.path.getsize(filepath) > 0, f"Chart file is empty: {filepath}"


if __name__ == "__main__":
    check("Environment Pipeline", check_environment)
    check("Complete DQN Pipeline", check_dqn_pipeline)
    check("Trained Model Loading", check_model_loading)
    check("Policy Evaluation Results", check_policy_evaluation)
    check("Dashboard Functionality", check_dashboard_files)
    check("Generated Visualizations", check_visualizations)

    print("\n\n========== VALIDATION SUMMARY ==========")
    all_passed = True
    for name, result in results.items():
        status = result["status"]
        print(f"{status:6} | {name}")
        if status == "FAIL":
            all_passed = False
            print(f"        Error: {result['error']}")

    print("=========================================")
    if all_passed:
        print("\nAll checks PASSED. No critical runtime issues found.")
    else:
        print("\nSome checks FAILED. Review errors above before submission.")

    # Save results for use in the written report
    os.makedirs("reports", exist_ok=True)
    with open("reports/validation_results.txt", "w", encoding="utf-8") as f:
        for name, result in results.items():
            f.write(f"{result['status']} | {name} | {result['error']}\n")