from snake import SnakeEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# Create environment
env = SnakeEnv(grid_size=10)

# Optional: verify Gymnasium compatibility
check_env(env)

# Create DQN agent
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=5_000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1_000,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1,
)


# Train
model.learn(total_timesteps=500_000)

# Save model
model.save("models/snake_dqn")

env.close()