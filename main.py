import gymnasium as gym
import numpy as np
import genetic
import train
import constants

def read_weights_from_file():
    weights_file = open(constants.WEIGHTS_FILE_PATH)
    content = weights_file.read()
    weights = [float(line) for line in content.splitlines()]
    return np.array(weights)

def main():
    weights = read_weights_from_file()
    
    env_test = gym.make("Pendulum-v1", render_mode="human")

    obs, _ = env_test.reset()
    while (True):
        action = genetic.policy(obs, weights)
        obs, reward, terminated, truncated, _ = env_test.step(action)
        env_test.render()
        if terminated or truncated:
            obs, _ = env_test.reset()
            break

    env_test.close()


if __name__ == "__main__":
    main()