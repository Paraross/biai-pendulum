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