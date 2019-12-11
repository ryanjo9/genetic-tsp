#!/usr/bin/python3
from copy import deepcopy

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    pass
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import numpy as np
from TSPClasses import *
import heapq
import random


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None
        self.populationSize = 100
        self.population = []

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    # O(n^2) Time and Space because we call the create matrix and create route functions
    def greedy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        bssf = None
        start_time = time.time()
        costMatrix = self.createCostMatrix(ncities, cities)  # initialize cost matrix
        bssf = self.createGreedyRoute(ncities, cities, costMatrix)  # find route solution using the cost matrix
        end_time = time.time()
        results['cost'] = bssf.cost
        results['count'] = 1
        results['time'] = end_time - start_time
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    # Function to create cost matrix
    # Time O(n^2) and space O(n^2)
    def createCostMatrix(self, ncities, cities):
        costMatrix = np.full([ncities, ncities], math.inf)
        for index1 in range(ncities):
            for index2 in range(ncities):
                edgeCost = cities[index1].costTo(cities[index2])  # calculate cost of city at index1 to city at index2
                if edgeCost != 0:
                    costMatrix[index1][index2] = edgeCost
        return costMatrix

    # Function that uses the costMatrix that was just created to find a greedy route.
    # O(n^2) Time and space because we check the values and choose the lowest one for each current value
    def createGreedyRoute(self, ncities, cities, costMatrix):
        currentIndex = 0
        visitedCities = []
        route = []
        for index1 in range(ncities):
            minValue = math.inf
            minIndex = 0
            for index2 in range(ncities):  # check the cost current city to next city and replace min if necessary
                if index2 not in visitedCities and costMatrix[currentIndex, index2] < minValue:
                    minValue = costMatrix[currentIndex, index2]
                    minIndex = index2
            if len(visitedCities) < ncities:  # While we haven't visited all cities, append next city
                currentIndex = minIndex
                visitedCities.append(minIndex)
                route.append(cities[minIndex])
        return TSPSolution(route)

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

    # O() Time and O() Space complexity
    def branchAndBound(self, time_allowance=60.0):
        try:
            cities = self._scenario.getCities()
            ncities = len(cities)
            bssf = self.greedy(time_allowance)['soln']  # O(n^2) because we use a matrix and find the route
            results = {}
            start_time = time.time()
            queueMax = 0  # keeps track of the largest size the queue ever reaches
            count = 1  # keeps count for number of valid solutions found
            prunedCount = 0  # keeps count for number of states pruned
            stateCount = 0  # keeps count for total number of states
            queue = []  # queue that will keep track of the states that still need to be investigated
            currentState = State()
            currentState.initializeStartState(cities)  # O(n^2) because we initialize the matrix
            heapq.heappush(queue, (currentState.lowerBound, currentState))  # push an item onto the priority queue
            stateCount += 1  # sort queue based on lowerBound O(n*log(n))
            while queue:  # O((n-1)!) in worst case. Worst case being every branch needs to be investigated
                if (time.time() - start_time) > time_allowance:  # if we are over time limit, stop
                    prunedCount += len(queue)
                    break
                if len(queue) > queueMax:  # update maxQueue size
                    queueMax = len(queue)
                currentState = heapq.heappop(queue)[1]  # get next state out of the queue O(n*log(n))
                if currentState.lowerBound > bssf.cost:  # if lowerBound is greater than cost, prune this state
                    prunedCount += 1
                    continue
                for nextCity in range(currentState.costMatrix.shape[0]):  # O(n)
                    if (time.time() - start_time) > time_allowance:  # if we are over time limit, stop
                        prunedCount += len(queue)
                        queue.clear()
                        break
                    if currentState.costMatrix[
                        currentState.cityID, nextCity] != math.inf:  # only enter if valid path exists
                        nextState = State()
                        nextState.initializeNextState(currentState, nextCity, cities[nextCity])  # O(n^2)
                        stateCount += 1
                        if nextState.lowerBound < bssf.cost:  # only enter if lowerBound is better than current solution
                            if len(nextState.path) == nextState.costMatrix.shape[0]:
                                route = []
                                for i in range(nextState.costMatrix.shape[0]):
                                    route.append(cities[nextState.route[i]])
                                bssf = TSPSolution(route)
                                bssf.setPath(nextState.path)
                                count += 1
                            else:
                                heapq.heappush(queue, (nextState.lowerBound, nextState))
                        else:
                            prunedCount += 1
            end_time = time.time()
            results['cities'] = ncities
            results['soln'] = bssf
            results['cost'] = bssf.cost
            results['time'] = end_time - start_time
            results['count'] = count
            results['max'] = queueMax
            results['pruned'] = prunedCount
            results['path'] = bssf.path
            results['total'] = stateCount
            return results
        except Exception as e:
            print(e)

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

    def fancy(self, time_allowance=60.0):
        start_time = time.time()
        results = {}
        no_improv_count = 0
        TERMINATION_LIMIT = 100
        currentBssf = self.greedy(time_allowance)['soln']
        while time.time() - start_time < time_allowance and no_improv_count < TERMINATION_LIMIT:
            # fill population
            self.fillPopulationWithRandom()
            # prune population
            new_pop, bssf = self.prune(self.population, currentBssf)
            if bssf.cost < currentBssf.cost:
                currentBssf = bssf
                no_improv_count = 0
            else:
                no_improv_count += 1
            # perform crossover and add them to the population
            children = self.crossOver(new_pop)
            self.population = np.append(new_pop, children)
            # perform mutation and add them to the population
            # for genome in self.population:
            #     self.population = np.append(self.population, self.mutate(genome))
            self.population = [self.mutate(g) for g in self.population]
        end_time = time.time()
        # since we only have to report on time and cost, the other results are unnecessary
        results['cost'] = currentBssf.cost
        results['count'] = 1
        results['time'] = end_time - start_time
        results['soln'] = TSPSolution(currentBssf.get_list())
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    # Function to cross over a list of genomes
    # Time O(n^3)
    def crossOver(self, genomes):
        genome_list = genomes.copy()
        children_genomes = []

        # shuffle the list to randomize the cross over
        random.shuffle(genome_list)
        genome_it = iter(genome_list)

        # loop through grabbing two genomes at a time
        for genome1 in genome_it:
            genome2 = next(genome_it, None)
            if genome2 is None:
                break

            # grab a random index to be the cross over line
            idx = random.randrange(0, len(genome1.genome_list))
            # for each element to the left cross over
            for i in range(idx):
                swap_idx = genome1.genome_list.index(genome2.genome_list[i])
                hold_val = genome1.genome_list[i]
                genome1.genome_list[i] = genome1.genome_list[swap_idx]
                genome1.genome_list[swap_idx] = hold_val
            children_genomes.append(genome1)
        return children_genomes

    def fillPopulationWithRandom(self):
        bssf = []
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        while len(self.population) < self.populationSize:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            genome = Genome(route)
            genome.get_cost()
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                self.population.append(genome)  # or push bssf if we want population to be TSPSolution objects

    """<summary>
		Takes in a population of genomes and uses weighted probabilities to determine a subset.
		Returns a list of genomes and bssf
		</summary>
	"""

    def prune(self, genomes, bssf):
        percent_to_keep = 0.67
        keep_size = math.floor(percent_to_keep * len(genomes))

        # Check if there is a new bssf
        minimum = min(genomes, key=lambda g: g.get_cost())

        if minimum.get_cost() < bssf.cost:
            bssf = minimum

        # Prune the genomes
        total_cost = sum(g.get_cost() if g.get_cost() != np.inf else 1 for g in genomes)

        # each items probability of survival is its cost / sum of all costs
        # subtracted from 1 so the lower costs have higher probabilities of being kepts
        pdf = [(g.get_cost() / total_cost) if g.get_cost() != np.inf else 1 / total_cost for g in genomes]
        subset = np.random.choice(genomes, keep_size, replace=False, p=pdf)

        return subset.tolist(), bssf

    def mutate(self, genome):
        mutation_rate = 0.1
        mutate_chance = random.uniform(0.0, 1.0)

        genome_list = genome.get_list()
        if mutate_chance < mutation_rate:
            val1, val2 = random.randrange(0, len(genome_list)), random.randrange(0, len(genome_list))
            genome_list[val1], genome_list[val2] = genome_list[val2], genome_list[val1]

        return Genome(genome_list)

    def fitness(self, genome):
        return genome.get_cost()


"""
This class is the object that represents a node in the tree. It has a matrix that holds values of cost, and it
keeps record of the paths it has done so far.
"""


class State:
    def __init__(self):
        self.route = list()  # list to keep track of the cityIDs in our current path. Used in TSPsolution
        self.path = list()  # list to keep track of names of cities we've been to so far

    # Function to create initial distance cost matrix
    # Time O(n^2), space O(n^2)
    def createMatrix(self, cities):
        ncities = len(cities)
        matrix = np.full([ncities, ncities], math.inf)  # initialize matrix to all infinity values
        for index1 in range(ncities):  # O(n^2)
            for index2 in range(ncities):
                edgeCost = cities[index1].costTo(
                    cities[index2])  # replace cost of going from city at index1 to city at index2
                matrix[index1, index2] = edgeCost
        return matrix

    # Function to reduce matrix based on what city we are coming from
    # Time O(n^2)
    def reduceMatrix(self, matrix, prevCityID):
        if prevCityID is not None:  # O(n)
            matrix[prevCityID, :] = math.inf  # Infinity out row
            matrix[:, self.cityID] = math.inf  # Infinity out column
            matrix[self.cityID, prevCityID] = math.inf  # Infinity out path back to where I was
        cost = 0
        for row in range(matrix.shape[0]):  # O(n^2)
            minimum = min(matrix[row, :])  # find the minimum in the row
            if minimum != math.inf:
                cost += minimum  # update cost
                matrix[row, :] -= minimum  # subtract minimum
        for col in range(matrix.shape[1]):  # O(n^2)
            minimum = min(matrix[:, col])  # find the minimum in the col
            if minimum != math.inf:
                cost += minimum  # update cost
                matrix[:, col] -= minimum  # subtract minimum
        return matrix, cost

    # Function for initializing the start state of the tree.
    # Time O(n^2) and space O(n^2) because it calls the create and reduceMatrix functions
    def initializeStartState(self, cities):
        self.cityID = 0
        self.cityName = cities[0].getName()
        self.route.append(0)  # append starting cityID to route
        self.path.append(self.cityName)  # append starting city name to path
        matrix = self.createMatrix(cities)  # O(n^2)
        matrixAndCost = self.reduceMatrix(matrix, None)  # O(n^2)
        self.costMatrix = matrixAndCost[0]
        self.lowerBound = 0 + 0 + matrixAndCost[1]

    # Function to initialize the next state. i.e. expanding the tree.
    # Time O(n^2) and space O(n^2) because it calls the reduceMatrix function
    def initializeNextState(self, prevState, cityID, city):
        self.cityID = cityID
        self.cityName = city.getName()
        self.route = deepcopy(prevState.route)  # make a copy of the route so far
        self.route.append(cityID)  # append current city to route
        self.path = deepcopy(prevState.path)  # make a copy of the path so far
        self.path.append(city.getName())  # append current city name to path
        prevMatrix = prevState.costMatrix.copy()
        matrixAndCost = self.reduceMatrix(prevMatrix, prevState.cityID)  # O(n^2)
        self.costMatrix = matrixAndCost[0]
        self.lowerBound = prevState.lowerBound + prevState.costMatrix[prevState.cityID][self.cityID] + matrixAndCost[1]

    # O(1)
    def __lt__(self, other):
        return self.path[-1] < other.path[-1]


class Genome:
    def __init__(self, genome_list):
        self.cost = None
        self.genome_list = []
        self.genome_list = genome_list

    def get_list(self):
        return self.genome_list

    def get_cost(self):
        if self.cost is None:
            self.cost = TSPSolution(self.genome_list).cost
        return self.cost
