import gymnasium as gym
import numpy as np
import genetic
import train
import constants

def read_genes_from_file():
    genes_file = open(constants.GENES_FILE_PATH)
    content = genes_file.read()
    genes = [float(line) for line in content.splitlines()]
    return np.array(genes)

def main():
    genes = read_genes_from_file()
    
    env_test = gym.make("Pendulum-v1", render_mode="human")

    obs, _ = env_test.reset()
    while (True):
        action = genetic.policy(obs, genes)
        obs, reward, terminated, truncated, _ = env_test.step(action)
        env_test.render()
        if terminated or truncated:
            obs, _ = env_test.reset()
            break

    env_test.close()


if __name__ == "__main__":
    main()