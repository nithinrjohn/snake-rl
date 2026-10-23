import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class SnakeEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 8,
    }

    def __init__(self, grid_size=10):
        super().__init__()

        self.grid_size = grid_size

        self.cell_size = 40

        # Actions:
        # 0 = up
        # 1 = down
        # 2 = left
        # 3 = right
        self.action_space = spaces.Discrete(4)

        # Observation:
        # Danger straight
        # Danger right
        # Danger left
        # Moving up
        # Moving down
        # Moving left
        # Moving right
        # Food up
        # Food down
        # Food left
        # Food right
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(11,),
            dtype=np.float32,
        )

        self.screen = None
        self.clock = None
        self.closed = False

        self.reset()

    def _get_obs(self):
        head_x, head_y = self.snake[0]
        dx, dy = self.direction

        # Directions
        left = (0, -1)
        right = (0, 1)
        up = (-1, 0)
        down = (1, 0)

        # Position after moving straight
        straight = (head_x + dx, head_y + dy)

        # Position after turning right
        if self.direction == up:
            right_pos = (head_x, head_y + 1)
            left_pos = (head_x, head_y - 1)
        elif self.direction == down:
            right_pos = (head_x, head_y - 1)
            left_pos = (head_x, head_y + 1)
        elif self.direction == left:
            right_pos = (head_x - 1, head_y)
            left_pos = (head_x + 1, head_y)
        else:  # right
            right_pos = (head_x + 1, head_y)
            left_pos = (head_x - 1, head_y)

        def collision(pos):
            x, y = pos

            if x < 0 or x >= self.grid_size:
                return True
            if y < 0 or y >= self.grid_size:
                return True

            # Ignore the tail since it moves away unless food is eaten
            body = self.snake[:-1]
            return pos in body

        fx, fy = self.food

        obs = np.array([
            # Danger
            collision(straight),
            collision(right_pos),
            collision(left_pos),

            # Current direction
            self.direction == up,
            self.direction == down,
            self.direction == left,
            self.direction == right,

            # Food location
            fy < head_y,   # food left
            fy > head_y,   # food right
            fx < head_x,   # food up
            fx > head_x,   # food down
        ], dtype=np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        start_pos = int(self.grid_size / 2)
        self.snake = [(start_pos, start_pos)]
        self.direction = (0, 1)

        self.spawn_food()

        observation = self._get_obs()
        info = {}

        self.steps = 0

        return observation, info

    def spawn_food(self):
        while True:
            pos = (
                np.random.randint(self.grid_size),
                np.random.randint(self.grid_size),
            )

            if pos not in self.snake:
                self.food = pos
                return

    def step(self, action):
        # Increment step counter
        self.steps += 1

        # Movement directions
        directions = [
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1),   # right
        ]

        new_direction = directions[action]

        opposites = {
            (1, 0): (-1, 0),
            (-1, 0): (1, 0),
            (0, 1): (0, -1),
            (0, -1): (0, 1),
        }

        # Prevent reversing direction
        if new_direction != opposites.get(self.direction):
            self.direction = new_direction

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        food_x, food_y = self.food
        old_distance = abs(head_x - food_x) + abs(head_y - food_y)
        new_distance = abs(new_head[0] - food_x) + abs(new_head[1] - food_y)

        reward = -0.01
        terminated = False
        truncated = False
        info = {}

        # Wall collision
        if (
            new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size
        ):
            return self._get_obs(), -10, True, truncated, info

        # Determine whether the tail will move this turn
        eating = (new_head == self.food)

        # Allow moving into the current tail position if it will move away
        body = self.snake if eating else self.snake[:-1]

        # Self collision
        if new_head in body:
            return self._get_obs(), -10, True, truncated, info

        # Move snake
        self.snake.insert(0, new_head)

        if eating:
            reward = 10
            self.spawn_food()
            self.steps = 0  # Reset starvation timer
        else:
            self.snake.pop()

            # Reward shaping
            if new_distance < old_distance:
                reward += 0.1
            elif new_distance > old_distance:
                reward -= 0.1

        # End episode if too many moves without food
        max_steps = self.grid_size * self.grid_size * 4
        if self.steps >= max_steps:
            truncated = True

        observation = self._get_obs()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.closed:
            return

        if self.metadata.get("render_mode", "human") != "human":
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (
                    self.grid_size * self.cell_size,
                    self.grid_size * self.cell_size,
                )
            )
            pygame.display.set_caption("Snake RL")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
                pygame.quit()
                return

        # Colors
        BG1 = (40, 40, 40)
        BG2 = (30, 30, 30)
        GREEN1 = (80, 200, 80)
        GREEN2 = (50, 160, 50)
        RED = (220, 60, 60)

        # Draw grid
        self.draw_grid(BG1, BG2)

        # Draw snake
        self.draw_snake(GREEN1, GREEN2)

        # Draw food
        self.draw_food(RED)

        pygame.display.flip()

        self.clock.tick(self.metadata.get("render_fps"))

    def draw_grid(self, color1, color2):
        for y in range(0, self.grid_size * self.cell_size, self.cell_size):
            for x in range(0, self.grid_size * self.cell_size, self.cell_size):
                # Determine color based on position for checker pattern
                if ((x // self.cell_size) + (y // self.cell_size)) % 2 == 0:
                    color = color1
                else:
                    color = color2
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)

    def draw_snake(self, color1, color2):
        for i, (x, y) in enumerate(self.snake):
            # Alternate colors based on segment index
            color = color1 if i % 2 == 0 else color2

            rect = pygame.Rect(
                y * self.cell_size,
                x * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
            pygame.draw.rect(self.screen, color, rect)

    def draw_food(self, color):
        fx, fy = self.food

        rect = pygame.Rect(
            fy * self.cell_size,
            fx * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

        pygame.draw.rect(self.screen, color, rect)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()