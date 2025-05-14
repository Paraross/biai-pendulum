import gymnasium as gym

print_to_console = False
write_to_file = True

def print_info_to_console(current_step, action, observation, reward):
    print(f"step        : {current_step}")
    print(f"action      : {action}")
    print(f"observation : {observation}")
    print(f"reward      : {reward}\n")

def write_info_to_file(file, current_step, action, observation, reward):
    file.write(f"step        : {current_step}\n")
    file.write(f"action      : {action}\n")
    file.write(f"observation : {observation}\n")
    file.write(f"reward      : {reward}\n\n")

def main():
    env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)

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
            print_info_to_console(current_step, action, observation, reward)
        if write_to_file:
            write_info_to_file(file, current_step, action, observation, reward)
        
        if reward > -1.0:
            print(f"Reward is good, step: {current_step}\n")
        

        episode_over = terminated or truncated
        current_step += 1

    file.close()
    env.close()

if __name__ == "__main__":
    main()