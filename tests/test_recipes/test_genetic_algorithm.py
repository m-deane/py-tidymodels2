"""
Tests for genetic algorithm utilities.
"""

import pytest
import numpy as np
from py_recipes.utils.genetic_algorithm import GeneticAlgorithm, GAConfig


class TestGAConfig:
    """Tests for GAConfig validation."""

    def test_default_config(self):
        """Test default configuration is valid."""
        config = GAConfig()
        assert config.population_size == 50
        assert config.generations == 100
        assert config.mutation_rate == 0.1
        assert config.crossover_rate == 0.8
        assert config.elitism == 0.1

    def test_invalid_population_size(self):
        """Test that invalid population_size raises error."""
        with pytest.raises(ValueError, match="population_size must be >= 2"):
            GAConfig(population_size=1)

    def test_invalid_mutation_rate(self):
        """Test that mutation_rate out of range raises error."""
        with pytest.raises(ValueError, match="mutation_rate must be in"):
            GAConfig(mutation_rate=1.5)

        with pytest.raises(ValueError, match="mutation_rate must be in"):
            GAConfig(mutation_rate=-0.1)

    def test_invalid_elitism(self):
        """Test that elitism out of range raises error."""
        with pytest.raises(ValueError, match="elitism must be in"):
            GAConfig(elitism=1.5)


class TestGeneticAlgorithm:
    """Tests for GeneticAlgorithm class."""

    def test_initialization(self):
        """Test GA initialization."""
        def dummy_fitness(chromosome):
            return np.sum(chromosome)

        ga = GeneticAlgorithm(n_features=10, fitness_function=dummy_fitness)
        assert ga.n_features == 10
        assert ga.config.population_size == 50
        assert ga.best_fitness_ == -np.inf

    def test_invalid_n_features(self):
        """Test that n_features < 1 raises error."""
        def dummy_fitness(chromosome):
            return 0.0

        with pytest.raises(ValueError, match="n_features must be >= 1"):
            GeneticAlgorithm(n_features=0, fitness_function=dummy_fitness)

    def test_initialize_population(self):
        """Test population initialization."""
        def dummy_fitness(chromosome):
            return np.sum(chromosome)

        config = GAConfig(population_size=20, random_state=42)
        ga = GeneticAlgorithm(n_features=10, fitness_function=dummy_fitness, config=config)

        population = ga.initialize_population()

        # Check shape
        assert population.shape == (20, 10)

        # Check all values are 0 or 1
        assert np.all(np.isin(population, [0, 1]))

        # Check each chromosome has at least one feature
        assert np.all(np.sum(population, axis=1) >= 1)

    def test_evaluate_population(self):
        """Test population evaluation."""
        def count_features(chromosome):
            return float(np.sum(chromosome))

        config = GAConfig(population_size=10, random_state=42)
        ga = GeneticAlgorithm(n_features=5, fitness_function=count_features, config=config)

        population = ga.initialize_population()
        fitness = ga.evaluate_population(population)

        # Check fitness shape
        assert fitness.shape == (10,)

        # Check fitness values match chromosome sums
        for i in range(10):
            assert fitness[i] == np.sum(population[i])

    def test_tournament_selection(self):
        """Test tournament selection."""
        def dummy_fitness(chromosome):
            return float(np.sum(chromosome))

        config = GAConfig(tournament_size=3, random_state=42)
        ga = GeneticAlgorithm(n_features=5, fitness_function=dummy_fitness, config=config)

        # Create population with known fitness
        population = np.array([
            [1, 0, 0, 0, 0],  # fitness = 1
            [1, 1, 0, 0, 0],  # fitness = 2
            [1, 1, 1, 0, 0],  # fitness = 3
            [1, 1, 1, 1, 0],  # fitness = 4
            [1, 1, 1, 1, 1],  # fitness = 5
        ])
        fitness = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Run tournament multiple times
        selected_fitness = []
        for _ in range(20):
            parent = ga.tournament_selection(population, fitness)
            selected_fitness.append(np.sum(parent))

        # Tournament should favor higher fitness
        mean_selected = np.mean(selected_fitness)
        assert mean_selected > 2.5  # Should be above random selection mean

    def test_crossover(self):
        """Test single-point crossover."""
        def dummy_fitness(chromosome):
            return 0.0

        config = GAConfig(crossover_rate=1.0, random_state=42)
        ga = GeneticAlgorithm(n_features=10, fitness_function=dummy_fitness, config=config)

        parent1 = np.ones(10)
        parent2 = np.zeros(10)

        offspring1, offspring2 = ga.crossover(parent1, parent2)

        # Check offspring are different from parents (crossover happened)
        assert not np.array_equal(offspring1, parent1) or not np.array_equal(offspring2, parent2)

        # Check offspring are valid (all 0s and 1s)
        assert np.all(np.isin(offspring1, [0, 1]))
        assert np.all(np.isin(offspring2, [0, 1]))

        # Check offspring contain mix of parent genes
        assert np.sum(offspring1) > 0 and np.sum(offspring1) < 10

    def test_crossover_no_crossover(self):
        """Test crossover with rate=0 returns parent copies."""
        def dummy_fitness(chromosome):
            return 0.0

        config = GAConfig(crossover_rate=0.0, random_state=42)
        ga = GeneticAlgorithm(n_features=10, fitness_function=dummy_fitness, config=config)

        parent1 = np.ones(10)
        parent2 = np.zeros(10)

        offspring1, offspring2 = ga.crossover(parent1, parent2)

        # Should return copies of parents
        assert np.array_equal(offspring1, parent1)
        assert np.array_equal(offspring2, parent2)

    def test_mutate(self):
        """Test bit-flip mutation."""
        def dummy_fitness(chromosome):
            return 0.0

        config = GAConfig(mutation_rate=0.5, random_state=42)
        ga = GeneticAlgorithm(n_features=20, fitness_function=dummy_fitness, config=config)

        chromosome = np.zeros(20)
        mutated = ga.mutate(chromosome)

        # Check mutation occurred (some bits flipped)
        assert not np.array_equal(mutated, chromosome)

        # Check at least one feature is selected
        assert np.sum(mutated) >= 1

        # Check all values are 0 or 1
        assert np.all(np.isin(mutated, [0, 1]))

    def test_mutate_ensures_one_feature(self):
        """Test mutation ensures at least one feature selected."""
        def dummy_fitness(chromosome):
            return 0.0

        config = GAConfig(mutation_rate=0.0, random_state=42)
        ga = GeneticAlgorithm(n_features=10, fitness_function=dummy_fitness, config=config)

        # Start with all zeros (after hypothetical mutation)
        chromosome = np.zeros(10)
        mutated = ga.mutate(chromosome)

        # Should have at least one feature
        assert np.sum(mutated) >= 1

    def test_elitism(self):
        """Test elitism preserves top individuals."""
        def dummy_fitness(chromosome):
            return 0.0

        config = GAConfig(population_size=10, elitism=0.2, random_state=42)
        ga = GeneticAlgorithm(n_features=5, fitness_function=dummy_fitness, config=config)

        # Old population with high fitness
        old_population = np.ones((10, 5))
        old_fitness = np.array([10.0] * 10)

        # New population with low fitness
        new_population = np.zeros((10, 5))
        new_fitness = np.array([1.0] * 10)

        final_pop, final_fitness = ga.apply_elitism(
            old_population, old_fitness,
            new_population, new_fitness
        )

        # Top 20% (2 individuals) should be from old population
        n_elite = 2
        top_indices = np.argsort(final_fitness)[-n_elite:]
        assert np.all(final_fitness[top_indices] == 10.0)

    def test_convergence_detection(self):
        """Test convergence detection."""
        def dummy_fitness(chromosome):
            return 0.0

        config = GAConfig(convergence_threshold=0.01, convergence_patience=3)
        ga = GeneticAlgorithm(n_features=5, fitness_function=dummy_fitness, config=config)

        # Not enough history
        ga.fitness_history_ = [1.0, 1.01]
        assert not ga.check_convergence()

        # Improving fitness (no convergence)
        ga.fitness_history_ = [1.0, 1.1, 1.2, 1.3, 1.4]
        assert not ga.check_convergence()

        # Plateaued fitness (convergence)
        ga.fitness_history_ = [1.0, 1.005, 1.008, 1.009, 1.0095]
        assert ga.check_convergence()

    def test_evolve_simple_problem(self):
        """Test evolution on simple maximization problem."""
        # Maximize number of selected features (trivial problem)
        def count_features(chromosome):
            return float(np.sum(chromosome))

        config = GAConfig(
            population_size=20,
            generations=50,
            random_state=42,
            verbose=False
        )
        ga = GeneticAlgorithm(n_features=10, fitness_function=count_features, config=config)

        best_chromosome, best_fitness, history = ga.evolve()

        # Should find solution with all features selected
        assert best_fitness == 10.0
        assert np.sum(best_chromosome) == 10

        # Fitness should improve over time
        assert history[-1] >= history[0]

    def test_evolve_with_convergence(self):
        """Test evolution with early convergence."""
        def simple_fitness(chromosome):
            # Favor chromosomes with 5 features
            return 10.0 - abs(np.sum(chromosome) - 5)

        config = GAConfig(
            population_size=30,
            generations=100,
            convergence_threshold=0.001,
            convergence_patience=5,
            random_state=42,
            verbose=False
        )
        ga = GeneticAlgorithm(n_features=10, fitness_function=simple_fitness, config=config)

        best_chromosome, best_fitness, history = ga.evolve()

        # Should converge before max generations
        assert ga.converged_
        assert ga.n_generations_ < 100

        # Should find near-optimal solution (5 features)
        assert abs(np.sum(best_chromosome) - 5) <= 1

    def test_random_state_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        def dummy_fitness(chromosome):
            return float(np.sum(chromosome))

        config1 = GAConfig(population_size=10, generations=10, random_state=42)
        ga1 = GeneticAlgorithm(n_features=5, fitness_function=dummy_fitness, config=config1)
        best1, fitness1, _ = ga1.evolve()

        config2 = GAConfig(population_size=10, generations=10, random_state=42)
        ga2 = GeneticAlgorithm(n_features=5, fitness_function=dummy_fitness, config=config2)
        best2, fitness2, _ = ga2.evolve()

        # Should produce identical results
        assert np.array_equal(best1, best2)
        assert fitness1 == fitness2

    def test_evolve_with_verbose(self, capsys):
        """Test verbose output during evolution."""
        def dummy_fitness(chromosome):
            return float(np.sum(chromosome))

        config = GAConfig(
            population_size=10,
            generations=15,
            random_state=42,
            verbose=True
        )
        ga = GeneticAlgorithm(n_features=5, fitness_function=dummy_fitness, config=config)

        ga.evolve()

        captured = capsys.readouterr()
        assert "Generation 0" in captured.out
        assert "Generation 10" in captured.out
        assert "Best fitness" in captured.out
