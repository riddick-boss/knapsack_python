import random

def generate_population(itemsSize: int, populationSize: int) -> list[list[int]]:
    # generate population of len populationSize, with each individual of len itemsSize
    return [[random.randint(0, 1) for _ in range(itemsSize)] for _ in range(populationSize)]

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

def performCrossover(parent1: list[int], parent2: list[int], crossoverProbability: float, crossoverType: int):
    if(random.random() < crossoverProbability):
        return parent1, parent2
    if(crossoverType == 1):
        return single_point_crossover(parent1, parent2)
    elif(crossoverType == 2):
        return two_point_crossover(parent1, parent2)
    else: 
        raise Exception(f"Illegal crossover type selected ({crossoverType})!!!")

def mutation(individual: list[int], mutationRate: float):
    mutated = []
    for gene in individual:
        mGene = 1 - gene if random.random() < mutationRate else gene
        mutated.append(mGene)
    return mutated

def rouletteSelection(population: list[list[int]], fitnessScores: list[int]) -> list[int]:
    totalFitness = sum(fitnessScores)
    threshold = random.randint(0, totalFitness)
    currentFitness = 0
    for individual, fitness in zip(population, fitnessScores):
        currentFitness += fitness
        if currentFitness >= threshold:
            return individual
        
def rankSelection(population: list[list[int]], fitnessScores: list[int]) -> list[int]:
    scored = zip(population, fitnessScores)
    sortedInidviduals = [ind for ind, _ in sorted(scored, key=lambda tup : tup[1])]

    #generate ranks [0.1, 0.3, 0.6]
    ranks = range(1, len(population) + 1)
    totalRank = sum(ranks)
    probabilities = [rank / totalRank for rank in ranks]

    # Cumulate probabilities [0.1, 0.4, 1.0]
    cumulativeProbabilities = []
    cumulative = 0
    for probability in probabilities:
        cumulative += probability
        cumulativeProbabilities.append(cumulative)

    threshold = random.uniform(0.15, 1)
    for individual, cumulativeProb in zip(sortedInidviduals, cumulativeProbabilities):
        if cumulativeProb >= threshold:
            return individual

def tournamentSelection(population: list[list[int]], fitnessScores: list[int]) -> list[int]:
    k = 50
    start = random.randint(0, len(population) - k)
    
    group = list(zip(population, fitnessScores))[start:start+k]
    bestInd, _ = max(group, key=lambda x: x[1])
    return bestInd

def performSelection(population: list[list[int]], fitnessScores: list[int], selectionType: int) -> list[int]:
    if(selectionType == 1):
        return rouletteSelection(population, fitnessScores)
    elif(selectionType == 2):
        return rankSelection(population, fitnessScores)
    elif(selectionType == 3):
        return tournamentSelection(population, fitnessScores)
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
    fitnessScores = [calculate_fitness(individual, items, capacity) for individual in population]

    for i in range(iterationsNum):
        newPopulation = []
        # to make sure we are creating new popultion of the required size
        while(len(newPopulation) < populationSize):
            parent1 = performSelection(population, fitnessScores, selectionType)
            parent2 = performSelection(population, fitnessScores, selectionType)
            child1, child2 = performCrossover(parent1, parent2, crossoverProbability, crossoverType)
            newPopulation.append(mutation(child1, mutationRate))
            newPopulation.append(mutation(child2, mutationRate))
        # to make sure we are replacing with same size
        population = newPopulation[:populationSize]
        fitnessScores = [calculate_fitness(individual, items, capacity) for individual in population]
        bestIndividualFitnessScore = max(fitnessScores)
        # bestIndividual = population[fitnessScores.index(bestIndividualFitnessScore)]
        # print(f"Iteration {i}: FS = {bestIndividualFitnessScore}, BI: {bestIndividual}")
        # print(f"{i+1} {bestIndividualFitnessScore}")
        print(f"{bestIndividualFitnessScore}")

if __name__ == "__main__":
    # filePath = input("Type file path: ") #dane_AG/low-dimensional/f1_l-d_kp_10_269
    populationSize, capacity, items = load_data("dane_AG/large_scale/knapPI_1_10000_1000_1")
    crossoverType = 1#int(input("Choose crossover type (1 - one-point, 2 - double-point): "))
    crossoverProbability = 0.65#float(input("Choose crossover probability (0.5-1.0): "))
    mutationRate = float(input("Choose mutation rate (0.0-0.1): "))
    iterationsNum = 500 #int(input("Type iterations number: "))
    selectionType = int(input("Choose selection type (1-roulette, 2-ranking, 3-tournament): "))

    # print(f"Population size: {populationSize}")
    # print(f"Capacity: {capacity}")
    # print(f"Items: {items}")
    # print(f"Crossover type: {crossoverType}")
    # print("--------------------------------------")

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