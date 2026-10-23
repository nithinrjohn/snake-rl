from snake import SnakeEnv
from stable_baselines3 import DQN

env = SnakeEnv(grid_size=10)

model = DQN.load("models/snake_dqn")

episode = 1

while True:

    observation, info = env.reset()

    done = False
    total_reward = 0

    while not done:

        action, _ = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        env.render()

        done = terminated or truncated

        # Stop if the user closed the window
        if env.closed:
            break

    if env.closed:
        break

    print(
        f"Episode {episode}: reward = {total_reward:.2f}"
    )

    episode += 1


env.close()