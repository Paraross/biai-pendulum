import numpy as np

# Define a simple neural network as a genome
def policy(observation, weights):
    # weights: [input_size * hidden + hidden * output]
    input_size = observation.shape[0]
    hidden_size = 8
    output_size = 1

    w1 = weights[:input_size * hidden_size].reshape(input_size, hidden_size)
    w2 = weights[input_size * hidden_size:].reshape(hidden_size, output_size)

    hidden = np.tanh(np.dot(observation, w1))
    output = np.tanh(np.dot(hidden, w2))  # Action range is [-1, 1]
    return output * 2.0  # Scale to [-2, 2] for Pendulum

def evaluate(env, weights, episodes=3):
    total_reward = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy(obs, weights)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
    return total_reward / episodes
