# Exploration Strategy Comparison — Issue #51

**Training episodes:** 2000
**Evaluation episodes:** 200 (greedy, no exploration)
**Common base hyperparameters:** alpha=0.1, gamma=0.95, epsilon_decay=0.999 (best config identified in Issue #47)

## Strategies Compared

**Original — plain epsilon-greedy:** when exploring, picks a completely random action. Epsilon decay begins immediately from episode 1.

**Improved — Boltzmann + exploration warm-up:** when exploring, picks an action using softmax over current Q-values (still explores, but leans toward more promising actions instead of wasting attempts on actions with clear evidence of being bad). Additionally, epsilon is held at its starting value for the first 200 episodes (warm-up) before decay begins, giving the agent a guaranteed period of broad exploration before it starts narrowing its strategy down — intended to avoid settling into a poor local pricing habit too early.

## Performance Comparison

| Metric | Original (epsilon-greedy) | Improved (Boltzmann + warm-up) | Change |
|---|---|---|---|
| Avg Reward | 14555.42 | 12261.69 | -15.8% |
| Avg Revenue | 15832.34 | 13969.05 | -11.8% |
| Inventory Utilization | 94.7% | 95.1% | +0.4pp |
| States Explored (breadth) | 1948 | 1731 | -217 |

## Convergence Behavior

| Metric | Original | Improved |
|---|---|---|
| Early training avg reward (first 100 episodes) | 11981.35 | 11250.49 |
| Late training avg reward (last 100 episodes) | 13786.19 | 12213.37 |
| Episode where performance stabilized (within 5% of final) | 483 | 119 |

## Interpretation

In this run, the improved exploration strategy did not outperform plain epsilon-greedy exploration (-15.8% change). This may indicate the warm-up period or temperature schedule needs further tuning, or that this environment's state space is small enough that plain epsilon-greedy already explores it adequately.

States explored increased from 1948 to 1731, indicating narrower coverage of the state space under the improved strategy.

## Recommendation

Continue using plain epsilon-greedy exploration for now, and revisit Boltzmann/warm-up tuning (e.g. different temperature decay rates or warm-up lengths) in a future iteration.
