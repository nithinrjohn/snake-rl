# snake-rl

A Deep Q-Network (DQN) agent that learns to play Snake, built with [Gymnasium](https://gymnasium.farama.org/) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

## Setup

```bash
# go to project folder
cd /path/to/project/

# create virtual environment
python -m venv .venv

# activate virtual environment
source .venv/bin/activate

# install dependencies
pip install gymnasium numpy pygame stable-baselines3
```

## Quick Start


### Train the agent

```bash
python src/train.py
```

Trains a DQN agent for 500,000 steps and saves it to `models/snake_dqn.zip`. Training progress is printed to the terminal.

### Watch the trained agent

```bash
python src/run_agent.py
```

Loads the saved model and plays Snake in a Pygame window. Each episode's total reward is printed to the terminal. Close the window to stop.

### Watch a random agent (optional)

```bash
python src/run_random_agent.py
```

Plays 10 episodes with random actions. Useful as a baseline to compare against the trained agent.

## How It Works

### Environment

`SnakeEnv` is a Gymnasium environment on a square grid (10×10 by default). The snake starts with length 1 in the center and grows each time it eats food.

**Actions** (4, discrete): up, down, left, right. Reversing directly into the snake's own body is ignored.

**Observation** (11 binary values):

| Group             | Values                                          |
| ----------------- | ----------------------------------------------- |
| Danger            | Collision straight ahead, to the right, to the left |
| Current direction | Moving up, down, left, right                    |
| Food location     | Food is up, down, left, right of the head       |

**Rewards:**

| Event                          | Reward  |
| ------------------------------ | ------- |
| Eating food                    | `+10`   |
| Hitting a wall or itself       | `-10`   |
| Each step                      | `-0.01` |
| Moving closer to the food      | `+0.1`  |
| Moving farther from the food   | `-0.1`  |

An episode ends when the snake dies, or is cut off if it goes `grid_size × grid_size × 4` steps without eating.

### Agent

The agent is Stable-Baselines3's `DQN` with an MLP policy. It explores with epsilon-greedy action selection, with epsilon decaying from `1.0` to `0.05` over the first 20% of training.

## Configuration

Grid size is set when the environment is created, in each script:

```python
env = SnakeEnv(grid_size=10)
```

Training settings are in **`src/train.py`**:

```python
total_timesteps = 500_000   # passed to model.learn()
learning_rate = 1e-4
gamma = 0.99                # discount factor
buffer_size = 100_000       # replay buffer size
exploration_fraction = 0.2  # fraction of training spent decaying epsilon
```

The observation size doesn't depend on grid size, so a model trained on one size can be run on another.

Playback speed is set by `render_fps` in the `metadata` of `src/snake.py` (default `8`).