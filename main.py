import gymnasium as gym
import numpy as np
import genetic

def main():
    # Genetic Algorithm Parameters
    population_size = 50
    generations = 50
    mutation_rate = 0.1
    elite_fraction = 0.2

    # env = gym.make("Pendulum-v1", render_mode="human")
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    input_size = env.observation_space.shape[0]
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
    print(f"Best index: {best_index}")
    print(f"Best weights: {best_weights}")

    env.close()

    env_test = gym.make("Pendulum-v1", render_mode="human")

    obs, _ = env_test.reset()
    for _ in range(100):
        action = genetic.policy(obs, best_weights)
        obs, reward, terminated, truncated, _ = env_test.step(action)
        env_test.render()
        if terminated or truncated:
            obs, _ = env_test.reset()

    env_test.close()


if __name__ == "__main__":
    main()