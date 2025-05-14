import gymnasium as gym

def main():
    env = gym.make("Pendulum-v1", render_mode="human", g=9.81)

    observation, info = env.reset()
    
    max_steps = 500
    current_step = 0
    episode_over = False
    while not episode_over and current_step < max_steps:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        current_step += 1

    episode_over = terminated or truncated

    env.close()

if __name__ == "__main__":
    main()