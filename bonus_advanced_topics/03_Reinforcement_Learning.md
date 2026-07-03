# Reinforcement Learning

## Overview

**Reinforcement Learning (RL)**: agent learns optimal behavior through interaction with environment.

**Key elements**: agent, environment, state, action, reward, policy.

**Core problem**: maximize cumulative reward over time (balance exploration vs exploitation).

---

## Markov Decision Process (MDP)

**State (s)**: current situation; summarizes history
**Action (a)**: agent's choice
**Transition (P)**: P(s'|s,a) = probability of next state given current state and action
**Reward (r)**: immediate reward r(s,a,s')
**Discount factor (γ)**: future rewards matter less (0 < γ < 1)

**Markov property**: future depends only on current state, not history.

---

## Value Functions

### State Value Function V(s)
Expected cumulative discounted reward starting from state s under policy π:

V^π(s) = E[R_t | s_t = s]

where R_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...

### Action Value Function Q(s,a)
Expected cumulative reward after taking action a in state s:

Q^π(s,a) = E[R_t | s_t = s, a_t = a]

**Relationship**: V(s) = Σ_a π(a|s) · Q(s,a)

### Bellman Equation
Value function satisfies recursive relation:

V(s) = E[r + γ·V(s') | s]
Q(s,a) = E[r + γ·max_a' Q(s',a') | s,a]

---

## Model-Free RL

Agent learns without knowing environment dynamics.

### Value-Based: Q-Learning

Learn Q-function (action values) using temporal difference (TD) learning.

**Update rule**:
```
Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
```

where α = learning rate

**Algorithm**:
```
1. Initialize Q randomly
2. For each episode:
   - For each step:
     - Take action a (ε-greedy: mostly argmax_a Q(s,a), occasionally random)
     - Observe reward r, next state s'
     - Update: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
```

**Convergence**: guaranteed for finite state/action spaces (Watkins & Dayan)

**Limitations**: assumes discrete, small state/action spaces (table doesn't scale)

### Deep Q-Networks (DQN)

Use neural network to approximate Q-function: Q(s,a) ≈ NN(s)[a]

**Challenges**:
1. **Instability**: correlation between updates (experiences not i.i.d.)
2. **Divergence**: positive feedback (overestimate Q-values → bad actions)

**Solutions**:
- **Experience Replay**: store transitions in memory; sample batches for training (breaks correlation)
- **Target Network**: separate network for computing target (updated periodically)

```python
class DQN:
    def __init__(self, state_dim, action_dim):
        self.q_network = NN(state_dim, action_dim)
        self.target_network = copy(self.q_network)
    
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        # Current Q-values
        q_values = self.q_network(states)[actions]
        
        # Target Q-values (using target network)
        target_q = rewards + gamma * self.target_network(next_states).max(dim=1)[0] * (1 - dones)
        
        # Loss and backprop
        loss = MSE(q_values, target_q.detach())
        optimizer.step()
        
        # Update target network periodically
        if step % update_freq == 0:
            self.target_network = copy(self.q_network)
```

**Extensions**:
- **Double DQN**: use main network to select action, target network to evaluate (reduce overestimation)
- **Dueling DQN**: split network into value + advantage streams (better learning)
- **Prioritized Replay**: sample high-TD-error transitions more often

---

## Policy-Based Methods

Learn policy π(a|s) directly (not value function).

### Policy Gradient

Optimize policy by taking gradient of expected return.

**Objective**: maximize J(θ) = E[return under π_θ]

**Policy Gradient Theorem**:
∇_θ J(θ) = E[∇_θ log π_θ(a|s) · Q(s,a)]

**REINFORCE** (Monte Carlo):
```
gradient ← ∇_θ log π_θ(a|s) · G_t
```
where G_t = actual cumulative return (unbiased but high variance)

```python
class PolicyGradient:
    def update(self, trajectory):
        states, actions, rewards = trajectory
        returns = compute_returns(rewards, gamma)
        
        loss = 0
        for s, a, G in zip(states, actions, returns):
            log_prob = log_probability(s, a)
            loss -= log_prob * G  # negative for gradient ascent
        
        loss.backward()
        optimizer.step()
```

### Actor-Critic

Combine policy (actor) + value function (critic):
- **Actor**: learns policy π(a|s)
- **Critic**: learns value V(s) to estimate G_t (reduces variance)

```python
class ActorCritic:
    def __init__(self):
        self.actor = NN(state_dim, action_dim)  # π
        self.critic = NN(state_dim, 1)  # V
    
    def update(self, state, action, reward, next_state):
        # TD error
        v_current = self.critic(state)
        v_next = self.critic(next_state)
        td_error = reward + gamma * v_next - v_current
        
        # Update critic (value function)
        critic_loss = td_error.detach() ** 2
        self.critic.backward(critic_loss)
        
        # Update actor (policy)
        log_prob = log_probability(state, action)
        actor_loss = -log_prob * td_error.detach()  # use TD error as baseline
        self.actor.backward(actor_loss)
```

**Advantage**: lower variance than REINFORCE; better sample efficiency

### Policy Optimization Methods

#### Proximal Policy Optimization (PPO)

Clips gradient to prevent huge policy updates (stability).

**Objective**:
```
L(θ) = E[min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]
```

where r_t = π_θ(a|s) / π_old(a|s) (probability ratio)

**Effect**: if ratio > 1+ε or < 1-ε, stop gradient (trust region)

```python
def ppo_update(states, actions, old_probs, rewards):
    returns = compute_returns(rewards)
    advantages = returns - critic(states)
    
    for epoch in range(K):
        new_probs = actor(states).gather(1, actions)
        ratio = new_probs / old_probs
        
        unclipped = ratio * advantages
        clipped = clip(ratio, 1-eps, 1+eps) * advantages
        
        loss = -torch.min(unclipped, clipped).mean()
        loss.backward()
        optimizer.step()
```

**Advantages**: stable, sample-efficient, robust

---

## Exploration vs Exploitation

### ε-Greedy
With probability ε, take random action; otherwise take argmax_a Q(s,a)

```python
if random() < epsilon:
    action = random_action()
else:
    action = argmax_a Q(s, a)
```

**Simple**: easy to implement
**Limitation**: ε uniform over actions; wastes time on clearly bad actions

### Upper Confidence Bound (UCB)
Balance: favor high Q-values + actions with high uncertainty.

```
action = argmax_a [Q(s,a) + c * sqrt(log(t) / N(s,a))]
```

N(s,a) = times action taken; explores uncertain actions

### Boltzmann Exploration
Sample action proportional to exponential of Q-value.

```python
temperatures = Q(s, :) / T
probabilities = softmax(temperatures)
action = sample(probabilities)
```

T = temperature; high T → uniform; low T → greedy

---

## Practical Considerations

### Reward Shaping
Design rewards carefully; sparse rewards (only at goal) = slow learning.

Add shaping: r_shaped = r + γ·Φ(s') - Φ(s) (potential-based shaping)

### Off-policy vs On-policy
- **On-policy**: learn from actions taken by current policy (REINFORCE, A3C)
- **Off-policy**: learn from actions taken by other policy (Q-learning, DQN)

Off-policy can reuse data; on-policy simpler but sample-inefficient.

### Convergence & Stability
- Q-learning: guaranteed for finite MDPs; unstable with function approximation
- Policy gradient: guaranteed convergence to local optimum; may be slow
- Actor-critic: faster than policy gradient; requires two networks

---

## Interview Key Points

- **MDP vs Bandit?** MDP: state matters (sequential decisions). Bandit: state doesn't matter (independent decisions).
- **Q-learning: on or off-policy?** Off-policy (learns optimal policy while following exploratory policy).
- **Why experience replay in DQN?** Breaks correlation between updates (experiences nearly i.i.d.; improves stability).
- **Value-based vs Policy-based?** Value: estimate V/Q first, then act greedily. Policy: directly optimize policy (can be stochastic).
- **Actor-Critic: why both networks?** Actor (policy) learns actions; Critic (value) estimates advantage (reduces variance).
- **PPO vs TRPO?** PPO: simpler, easier to tune (clip gradient). TRPO: theoretically sound trust region (more complex).
- **Exploration: why needed?** Find better strategies; without exploration, stick to initial random policy.
- **Reward shaping: impact?** Can speed up learning or hide optimal policy (if shaped poorly).
- **Sample efficiency: which method best?** Off-policy (DQN, Q-learning) > on-policy (REINFORCE). Actor-critic middle ground.
