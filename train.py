import gymnasium as gym
import numpy as np
import genetic
import constants

def train(
    population_size: int,
    generations: int,
    mutation_rate: float,
    mutation_strength: float,
    elite_fraction: float,
):
    env = gym.make("Pendulum-v1")
    # observation space: [x = cos(theta), y = sin(theta), angular velocity]
    input_size = env.observation_space.shape[0] # type: ignore
    hidden_size = 8
    output_size = 1
    genes_dim = hidden_size * (input_size + output_size)

    # Initialize population with random genes
    population = np.random.randn(population_size, genes_dim)

    for generation in range(generations):
        # Evaluate fitness
        fitnesses = np.array([genetic.calculate_fitness(env, individual) for individual in population])
        print(f"Generation {generation} - Best Fitness: {np.max(fitnesses):.2f}")

        # Select elites
        elite_count = int(elite_fraction * population_size)
        elite_indices = fitnesses.argsort()[-elite_count:]
        elites = population[elite_indices]

        # Generate new population
        new_population = []
        while len(new_population) < population_size:
            # Select two parents
            parents = elites[np.random.choice(elite_count, 2, replace=False)]
            crossover_point = np.random.randint(genes_dim)
            child = np.concatenate([
                parents[0][:crossover_point],
                parents[1][crossover_point:]
            ])

            # Mutation
            if np.random.rand() < mutation_rate:
                mutation = np.random.randn(genes_dim) * mutation_strength
                child += mutation

            new_population.append(child)

        population = np.array(new_population)

    # Test best individual
    best_individual_index = np.argmax([genetic.calculate_fitness(env, individual) for individual in population])
    best_individual = population[best_individual_index]
    
    env.close()
    
    with open(constants.GENES_FILE_PATH, "w") as genes_file:
        lines = [f"{str(gene)}\n" for gene in best_individual]
        genes_file.writelines(lines)
    
    return best_individual

if __name__ == "__main__":
    train(50, 50, 0.1, 0.1, 0.2)