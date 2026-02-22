# Unit 3 — Reinforcement Learning

## Topics

### 1. The RL Framework
- Agent interacts with an Environment over time
- At each step: observe state sₜ, take action aₜ, receive reward rₜ, transition to sₜ₊₁
- Goal: learn a policy π(s) that maximizes cumulative discounted reward: G = Σ γᵗ rₜ
- Discount factor γ ∈ [0,1]: trades off immediate vs future reward

### 2. Key Concepts
- **Policy** π(a|s): probability of taking action a in state s
- **Value function** V^π(s): expected return starting from state s following policy π
- **Action-value function** Q^π(s,a): expected return taking action a in state s, then following π
- **Bellman equation**: V(s) = Σₐ π(a|s) [R(s,a) + γ · V(s')]

### 3. Markov Decision Processes (MDPs)
- Markov property: future depends only on current state, not history
- Defined by: (S, A, P, R, γ) — states, actions, transition probabilities, reward function, discount

### 4. Q-Learning
- Model-free, off-policy algorithm
- Update: Q(s,a) ← Q(s,a) + α[r + γ·max_{a'} Q(s',a') − Q(s,a)]
- Converges to optimal Q* if all state-action pairs visited infinitely often
- Tabular: Q-table for discrete state/action spaces

### 5. Deep Q-Network (DQN)
- Use a neural network to approximate Q(s,a; θ)
- Two key tricks: **experience replay** (store transitions, sample random mini-batches) and **target network** (separate, slowly-updated network for Q targets)
- Achieved superhuman Atari game performance (DeepMind, 2015)

### 6. Policy Gradient Methods
- Directly optimize the policy π(a|s; θ)
- REINFORCE: gradient = ∑ ∇ log π(aₜ|sₜ; θ) · Gₜ
- Actor-Critic: separate networks for policy (actor) and value function (critic)
- PPO (Proximal Policy Optimization): stable, widely used; standard in RL fine-tuning of LLMs

---

## Exercises

### Exercise 1 — Q-Learning on FrozenLake (Tabular)
```python
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)
n_states  = env.observation_space.n  # 16
n_actions = env.action_space.n       # 4
Q = np.zeros((n_states, n_actions))

alpha, gamma, epsilon = 0.8, 0.95, 0.1
n_episodes = 2000
rewards = []

for ep in range(n_episodes):
    state, _ = env.reset()
    total_r = 0
    for _ in range(100):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = Q[state].argmax()
        next_state, reward, terminated, truncated, _ = env.step(action)
        # Q-Learning update
        Q[state, action] += alpha * (reward + gamma * Q[next_state].max() - Q[state, action])
        state = next_state
        total_r += reward
        if terminated or truncated:
            break
    rewards.append(total_r)

print(f"Success rate (last 200 episodes): {np.mean(rewards[-200:]):.2f}")
```

### Exercise 2 — DQN Architecture Sketch
```python
import torch
import torch.nn as nn
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(list, zip(*batch))
    def __len__(self):
        return len(self.buffer)

# Example: CartPole-v1 has 4-dim state and 2 actions
dqn = DQN(state_dim=4, action_dim=2)
buffer = ReplayBuffer()
print("DQN parameters:", sum(p.numel() for p in dqn.parameters()))
print("Architecture:"); print(dqn)
```

---

## Questions & Answers

**Q1: What is the difference between model-based and model-free RL?**
> Model-free algorithms (Q-Learning, DQN, PPO) learn directly from experience without explicitly modeling the environment's dynamics. Model-based algorithms (Dyna-Q, AlphaZero, MuZero) learn a model of the environment (transitions and rewards) and use it to plan. Model-based approaches are more sample-efficient but require a good model; model-free is more general.

**Q2: What is the exploration-exploitation tradeoff?**
> The agent must balance exploring new actions (to discover better strategies) vs exploiting known good actions (to accumulate reward). Pure exploitation gets stuck in local optima; pure exploration collects little reward. ε-greedy, UCB, and Thompson sampling are common strategies to balance this tradeoff.

**Q3: Why does DQN use experience replay?**
> Raw sequential experience has high temporal correlation — consecutive transitions are highly similar, causing the network to overfit to recent experience. Experience replay stores transitions in a buffer and samples random mini-batches for training, breaking correlations and reusing experience multiple times for efficiency.

**Q4: What is the target network in DQN and why is it needed?**
> The Q-network is used to both compute predictions and generate targets Q(s') — the same changing network for both creates a moving target and leads to instability. The target network is a separate, periodically-updated copy of the Q-network. By keeping the targets stable for many steps, training is much more stable.

**Q5: How is RL used in LLM fine-tuning (RLHF)?**
> Reinforcement Learning from Human Feedback (RLHF) fine-tunes language models to align with human preferences. A reward model is trained on human preference data. The LLM is then fine-tuned with PPO to maximize the reward model's score while staying close to the original model (KL divergence penalty). This is how ChatGPT, Claude, and other aligned LLMs are trained.
