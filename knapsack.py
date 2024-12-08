import random

def generate_population(itemsSize: int, populationSize: int) -> list[list[int]]:
    # generate population of len populationSize, with each individual of len itemsSize
    chanceToSelect = 0.01
    return [[1 if chanceToSelect >= random.random() else 0 for _ in range(itemsSize)] for _ in range(populationSize)]

def load_data(filePath: str):
    data: list[tuple[int, int]] = []
    with open(filePath, 'r') as file:
        firstLine = file.readline().strip()
        populationSize, capacity = map(int, firstLine.split())
        for line in file:
            value, weight = map(int, line.split())
            data.append((value, weight))
    return populationSize, capacity, data

def calculate_fitness(individual: list[int], items: list[tuple[int, int]], capacity: int) -> int:
    totalValue = 0
    totalWeight = 0
    for selection, (value, weight) in zip(individual, items):
        if selection == 1:
            totalValue += value
            totalWeight += weight
        if totalWeight > capacity:
            return 0
    return totalValue

def single_point_crossover(parent1: list[int], parent2: list[int]):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def two_point_crossover(parent1: list[int], parent2: list[int]):
    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2

def performCrossover(parent1: list[int], parent2: list[int], crossoverType: int):
    if(crossoverType == 1):
        return single_point_crossover(parent1, parent2)
    elif(crossoverType == 2):
        return two_point_crossover(parent1, parent2)
    else: 
        raise Exception(f"Illegal crossover type selected ({crossoverType})!!!")

def mutation(individual: list[int]):
    idxToMutate = random.randint(0, len(individual) - 1)
    mutated = individual.copy()
    value = mutated[idxToMutate]
    mutated[idxToMutate] = 0 if value == 1 else 1
    return mutated

def rouletteSelection(population: list[list[int]], populationSize: int, fitnessScores: list[int]) -> list[list[int]]:
    totalFitness = sum(fitnessScores)
    updatedPopulation: list[list[int]] = []

    while(len(updatedPopulation) < populationSize):
        for individual, fitness in zip(population, fitnessScores):
            if totalFitness == 0:
                randomIndex = random.randint(0, len(population) - 1)
                updatedPopulation.append(population[randomIndex])
            else:
                rouletteChance = fitness / totalFitness
                threshold = random.random()
                if(rouletteChance >= threshold):
                    updatedPopulation.append(individual)

            if(len(updatedPopulation) == populationSize):
                return updatedPopulation

    return updatedPopulation

def rankSelection(population: list[list[int]], populationSize: int, fitnessScores: list[int]) -> list[list[int]]:
    updatedPopulation: list[list[int]] = []
    ordered = [ind for ind, _ in sorted(zip(population, fitnessScores), key=lambda tup : tup[1])]
    while(len(updatedPopulation) < populationSize):
        for index in range(populationSize, 0, -1):
            rank = index / populationSize
            individualsNumToSelect = int((1/3) * populationSize * rank)
            individualToTake = ordered[-(index-populationSize+1)]
            updatedPopulation.extend([individualToTake] * individualsNumToSelect)
            
            if(len(updatedPopulation) >= populationSize):
                return updatedPopulation[:populationSize]

    return updatedPopulation[:populationSize]

def tournamentSelection(population: list[list[int]], populationSize: int, fitnessScores: list[int]) -> list[list[int]]:
    updatedPopulation: list[list[int]] = []
    k = int(populationSize * 0.6)
    for _ in range(populationSize):       
        start = random.randint(0, len(population) - k)
        group = list(zip(population, fitnessScores))[start:start+k]
        bestIndividual, _ = max(group, key=lambda x: x[1])
        updatedPopulation.append(bestIndividual)

    return updatedPopulation[:populationSize]

def performSelection(population: list[list[int]], populationSize: int, fitnessScores: list[int], selectionType: int) -> list[list[int]]:
    if(selectionType == 1):
        return rouletteSelection(population, populationSize, fitnessScores)
    elif(selectionType == 2):
        return rankSelection(population, populationSize, fitnessScores)
    elif(selectionType == 3):
        return tournamentSelection(population, populationSize, fitnessScores)
    else:
        raise Exception(f"Illegal seelction type selected ({selectionType})!!!")

def run_genetic(
        populationSize: int,
        capacity: int,
        iterationsNum: int,
        mutationRate: float,
        crossoverType: int,
        crossoverProbability: float,
        selectionType: int,
        items: list[tuple[int, int]]
):
    population = generate_population(len(items), populationSize)

    for i in range(iterationsNum):
        crossoverPopulation = population.copy()
        # to make sure we are creating new popultion of the required size
        for individual in population:
            if(crossoverProbability >= random.random()):
                randomParent2Idx = random.randint(0, len(population) - 1)
                child1, child2 = performCrossover(individual, population[randomParent2Idx], crossoverType)
                crossoverPopulation.append(child1)
                crossoverPopulation.append(child2)

        mutatedPopulation = crossoverPopulation.copy()
        for individual in mutatedPopulation:
            if(mutationRate >= random.random()):
                mutatedPopulation.append(mutation(individual))

        fitnessScores = [calculate_fitness(individual, items, capacity) for individual in mutatedPopulation]
        population = performSelection(mutatedPopulation, populationSize, fitnessScores, selectionType)

        bestIndividualFitnessScore = max(fitnessScores)
        print(f"{bestIndividualFitnessScore}")

if __name__ == "__main__":
    # filePath = input("Type file path: ") #dane_AG/low-dimensional/f1_l-d_kp_10_269 #dane_AG/large_scale/knapPI_1_10000_1000_1
    populationSize, capacity, items = load_data("dane_AG/low-dimensional/f10_l-d_kp_20_879")
    crossoverType = int(input("Choose crossover type (1 - one-point, 2 - double-point): "))
    crossoverProbability = 0.65#float(input("Choose crossover probability (0.5-1.0): "))
    mutationRate = float(input("Choose mutation rate (0.0-0.1): "))
    iterationsNum = 500 #int(input("Type iterations number: "))
    selectionType = int(input("Choose selection type (1-roulette, 2-ranking, 3-tournament): "))

    run_genetic(
        populationSize=populationSize,
        capacity=capacity,
        iterationsNum=iterationsNum,
        mutationRate=mutationRate,
        crossoverType=crossoverType,
        crossoverProbability=crossoverProbability,
        selectionType=selectionType,
        items=items
    )