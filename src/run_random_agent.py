from snake import SnakeEnv

env = SnakeEnv(grid_size=10)

episodes = 10

for episode in range(episodes):

    observation, info = env.reset()

    done = False
    total_reward = 0

    while not done:

        # Pick a random action
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        env.render()

        done = terminated or truncated

    print(f"Episode {episode + 1}: reward = {total_reward:.2f}")

env.close()