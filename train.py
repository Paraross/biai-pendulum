import gymnasium as gym
import numpy as np
import genetic
import constants

def train(
    population_size: int,
    generations: int,
    mutation_rate: float,
    elite_fraction: float,
    write_to_file: bool = False
):
    env = gym.make("Pendulum-v1")
    input_size = env.observation_space.shape[0] # type: ignore
    hidden_size = 8
    output_size = 1
    weights_dim = input_size * hidden_size + hidden_size * output_size

    # Initialize population with random weights
    population = np.random.randn(population_size, weights_dim)

    for generation in range(generations):
        # Evaluate fitness
        fitness = np.array([genetic.evaluate(env, ind) for ind in population])
        print(f"Generation {generation} - Best Fitness: {np.max(fitness):.2f}")

        # Select elites
        elite_count = int(elite_fraction * population_size)
        elite_indices = fitness.argsort()[-elite_count:]
        elites = population[elite_indices]

        # Generate new population
        new_population = []
        while len(new_population) < population_size:
            # Select two parents
            parents = elites[np.random.choice(elite_count, 2, replace=False)]
            crossover_point = np.random.randint(weights_dim)
            child = np.concatenate([
                parents[0][:crossover_point],
                parents[1][crossover_point:]
            ])

            # Mutation
            if np.random.rand() < mutation_rate:
                mutation = np.random.randn(weights_dim) * 0.1
                child += mutation

            new_population.append(child)

        population = np.array(new_population)

    # Test best individual
    best_index = np.argmax([genetic.evaluate(env, ind) for ind in population])
    best_weights = population[best_index]
    
    env.close()
    
    if write_to_file:
        with open(constants.WEIGHTS_FILE_PATH, "w") as weights_file:
            lines = [f"{str(weight)}\n" for weight in best_weights]
            weights_file.writelines(lines)
    
    return best_weights

if __name__ == "__main__":
    train(50, 50, 0.1, 0.2, True)