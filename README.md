# IMPLEMENTATION OF ON-POLICY MONTE CARLO CONTROL USING GYMNASIUM FROZENLAKE-V1

## Aim

To implement **On-Policy Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment contains frozen tiles, holes, a starting position, and a goal position. The agent must learn a policy that allows it to reach the goal while avoiding the holes.

The objectives of this experiment are:

1. To generate complete episodes using the Gymnasium environment.
2. To estimate the action-value function `Q(s,a)` using Monte Carlo returns.
3. To use an epsilon-greedy policy for exploration and exploitation.
4. To improve the policy based on the learned Q-values.
5. To display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description

**FrozenLake-v1** is a grid-based reinforcement learning environment.

* The environment consists of a **4 × 4 grid**.
* The agent starts from the starting tile.
* The agent must move across frozen tiles.
* Some tiles contain holes, which terminate the episode.
* The goal tile provides a reward of `1`.
* Reaching a hole provides a reward of `0`.
* Other movements normally provide a reward of `0`.
* `is_slippery=False` is used so that the agent moves deterministically in the selected direction.

### Action Mapping

| Action | Direction |
| ------ | --------- |
| 0      | Left      |
| 1      | Down      |
| 2      | Right     |
| 3      | Up        |

---

## Theory

### Monte Carlo Method

Monte Carlo methods learn from **complete episodes**. Instead of updating the value function after every individual step, the agent waits until the episode finishes and then calculates the return obtained from each state-action pair.

An episode can be represented as:
