import gymnasium as gym
import numpy as np
from constants import HIDDEN_SIZE, OUTPUT_SIZE

# Define a simple neural network as a genome
def policy(observation, genes: np.ndarray) -> float:
    # genes: [input_size * hidden + hidden * output]
    input_size = observation.shape[0]

    w1 = genes[:input_size * HIDDEN_SIZE].reshape(input_size, HIDDEN_SIZE)
    w2 = genes[input_size * HIDDEN_SIZE:].reshape(HIDDEN_SIZE, OUTPUT_SIZE)

    hidden = np.tanh(np.dot(observation, w1))
    output = np.tanh(np.dot(hidden, w2))  # Action range is [-1, 1]
    return output * 2.0  # Scale to [-2, 2] for Pendulum

def calculate_fitness(env: gym.Env, genes: np.ndarray, episodes=3) -> float:
    total_reward = 0.0
    for _ in range(episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            action = policy(observation, genes)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward # type: ignore
            done = terminated or truncated
    return total_reward / episodes
