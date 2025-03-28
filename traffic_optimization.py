import numpy as np
import matplotlib.pyplot as plt

def fitness_function(C, g, x, c):
    a = (1 - (g / C)) ** 2
    p = 1 - ((g / C) * x)
    d1i = (0.38 * C * a) / p

    a2 = 173 * (x ** 2)
    ri1 = np.sqrt(np.maximum(0, (x - 1) + (x - 1) ** 2 + ((16 * x) / c)))

    d2i = a2 * ri1

    return d1i + d2i

def initialize_population(pop_size, num_lights, green_min, green_max, cycle_time, cars):
    population = []
    road_capacity = [20] * num_lights
    road_congestion = np.array(road_capacity) - np.array(cars)
    road_congestion = road_congestion / np.array(road_capacity)

    while len(population) < pop_size:
        green_times = np.random.randint(green_min, green_max + 1, num_lights)
        if np.sum(green_times) <= cycle_time:
            total_delay = np.sum([fitness_function(cycle_time, green_times[i], road_congestion[i], road_capacity[i]) for i in range(num_lights)])
            population.append((green_times, total_delay))
    return sorted(population, key=lambda x: x[1])

def roulette_wheel_selection(population, total_delays, beta, debug=False):
    probabilities = np.exp(-beta * total_delays)

    
    if len(probabilities) != len(population):
        if debug:  
            print(f"Error: Length mismatch -> Population: {len(population)}, Probabilities: {len(probabilities)}")
        return np.random.randint(0, len(population))  

    
    if np.isnan(probabilities).any() or probabilities.sum() == 0:
        probabilities = np.ones(len(population)) / len(population)  

    probabilities /= probabilities.sum() 

    return np.random.choice(len(population), p=probabilities)


def crossover(parent1, parent2, num_lights):
    point = np.random.randint(1, num_lights)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def mutate(individual, mutation_rate, green_min, green_max):
    num_lights = len(individual)
    mutated = individual.copy()
    for _ in range(int(mutation_rate * num_lights)):
        idx = np.random.randint(0, num_lights)
        sigma = np.random.choice([-1, 1]) * 0.02 * (green_max - green_min)
        mutated[idx] = np.clip(individual[idx] + sigma, green_min, green_max)
    return mutated

def inversion(individual, num_lights):
    idx1, idx2 = np.random.randint(0, num_lights, 2)
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    individual[idx1:idx2+1] = individual[idx1:idx2+1][::-1]
    return individual

def genetic_algorithm(pop_size, num_lights, max_iter, green_min, green_max, cycle_time, mutation_rate, pinv, beta, cars):
    population = initialize_population(pop_size, num_lights, green_min, green_max, cycle_time, cars)
    best_sol = population[0]
    best_delays = [best_sol[1]]

    road_capacity = [20] * num_lights
    road_congestion = np.array(road_capacity) - np.array(cars)
    road_congestion = road_congestion / np.array(road_capacity)

    for _ in range(max_iter):
        total_delays = [ind[1] for ind in population]
        new_population = []

        while len(new_population) < pop_size:
            i1 = roulette_wheel_selection(population, total_delays, beta)
            i2 = roulette_wheel_selection(population, total_delays, beta)

            parent1, parent2 = population[i1][0], population[i2][0]
            child1, child2 = crossover(parent1, parent2, num_lights)

            if np.sum(child1) <= cycle_time:
                child1 = mutate(child1, mutation_rate, green_min, green_max)
                child1 = np.clip(child1, green_min, green_max)
                total_delay = np.sum([fitness_function(cycle_time, child1[i], road_congestion[i], road_capacity[i]) for i in range(num_lights)])
                new_population.append((child1, total_delay))

            if np.sum(child2) <= cycle_time:
                child2 = mutate(child2, mutation_rate, green_min, green_max)
                child2 = np.clip(child2, green_min, green_max)
                total_delay = np.sum([fitness_function(cycle_time, child2[i], road_congestion[i], road_capacity[i]) for i in range(num_lights)])
                new_population.append((child2, total_delay))

        
        while len(new_population) < pop_size:
            i = np.random.randint(0, len(population))
            individual = inversion(population[i][0], num_lights)
            if np.sum(individual) <= cycle_time:
                individual = mutate(individual, mutation_rate, green_min, green_max)
                total_delay = np.sum([fitness_function(cycle_time, individual[i], road_congestion[i], road_capacity[i]) for i in range(num_lights)])
                new_population.append((individual, total_delay))

       
        population += new_population
        population = sorted(population, key=lambda x: x[1])[:pop_size]
        
        if population[0][1] < best_sol[1]:
            best_sol = population[0]
        
        best_delays.append(best_sol[1])
        print(f"Iteration: Best Total Delay = {best_sol[1]}")
        print(f"Green Times: Intersection_1 = {best_sol[0][0]}, Intersection_2 = {best_sol[0][1]}, Intersection_3 = {best_sol[0][2]}, Intersection_4 = {best_sol[0][3]}")
    
    return best_sol, best_delays

def optimize_traffic(cars):
    
    pop_size = 400
    num_lights = 4
    max_iter = 25
    green_min = 10
    green_max = 60
    cycle_time = 160 - 12
    mutation_rate = 0.02
    pinv = 0.2
    beta = 8

    
    best_sol, best_delays = genetic_algorithm(pop_size, num_lights, max_iter, green_min, green_max, cycle_time, mutation_rate, pinv, beta, cars)

    
    result = {
        'Intersection_1': int(best_sol[0][0]),
        'Intersection_2': int(best_sol[0][1]),
        'Intersection_3': int(best_sol[0][2]),
        'Intersection_4': int(best_sol[0][3])
    }

    print('Optimal Solution:')
    print(f'Intersection_1 Green Time = {result["Intersection_1"]} seconds')
    print(f'Intersection_2 Green Time = {result["Intersection_2"]} seconds')
    print(f'Intersection_3 Green Time = {result["Intersection_3"]} seconds')
    print(f'Intersection_4 Green Time = {result["Intersection_4"]} seconds')

    return result
