import gymnasium as gym
import utils

def main():
    print_to_console = False
    write_to_file = False

    env = gym.make("Pendulum-v1", render_mode="human", g=9.81)

    observation, info = env.reset()

    file = open("output.txt", "w" if write_to_file else "a")
    
    max_steps = 100
    current_step = 0
    episode_over = False
    while not episode_over and current_step < max_steps:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        if print_to_console:
            utils.print_info_to_console(current_step, action, observation, reward)
        if write_to_file:
            utils.write_info_to_file(file, current_step, action, observation, reward)
        
        if reward > -1.0:
            print(f"Reward is good, step: {current_step}\n")
        

        episode_over = terminated or truncated
        current_step += 1

    file.close()
    env.close()

if __name__ == "__main__":
    main()